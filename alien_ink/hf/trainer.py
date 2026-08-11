"""Hugging Face Trainer helpers for causal language modeling."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from alien_ink.com.device import (
    AcceleratorInfo,
    device_info,
    distributed_world_size,
)
from alien_ink.hf.metrics import (
    ModelSize,
    RunSummary,
    build_run_summary,
    log_run_summary,
    push_summary_to_wandb,
    save_run_summary,
)
from alien_ink.com.log import blank, detail, get_logger, step

log = get_logger("hf.trainer")

__all__ = [
    "CausalLmTrainerConfig",
    "DEFAULT_EPOCH_EVALS_PER_EPOCH",
    "apply_epoch_cadence",
    "best_model_metric",
    "build_causal_lm_trainer",
    "build_lm_data_collator",
    "build_trainer_callbacks",
    "build_training_arguments",
    "epoch_cadence_steps",
    "has_eval_examples",
    "optimizer_steps_per_epoch",
    "reporting_disabled",
    "save_model_and_tokenizer",
    "tokens_per_optimizer_step",
    "train_and_save",
]

# Subset / epoch runs: evaluate this many times per epoch (last tick = epoch end).
DEFAULT_EPOCH_EVALS_PER_EPOCH = 5
# Match the streamed reference ratio (50 log / 1000 eval ≈ 20 logs per eval).
_LOGS_PER_EVAL_INTERVAL = 20


@dataclass(frozen=True)
class CausalLmTrainerConfig:
    """Hyperparameters and bookkeeping for a causal-LM ``Trainer`` run.

    Length is either step-capped (``max_steps >= 1``) or epoch-based
    (``max_steps == -1`` and ``num_train_epochs > 0``). Positive ``max_steps``
    wins over epochs, matching Hugging Face ``TrainingArguments``.
    """

    output_dir: Path
    max_steps: int = 50_000
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 6e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int | None = 2_000
    warmup_ratio: float | None = None
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    logging_steps: int = 50
    eval_steps: int = 1_000
    save_steps: int = 1_000
    save_total_limit: int = 2
    early_stopping_patience: int = 0
    seed: int = 101
    data_seed: int = 101
    prefer_fp16: bool = True
    prefer_bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    dataloader_prefetch_factor: int | None = None
    dataloader_persistent_workers: bool = False
    tf32: bool | None = None
    torch_compile: bool = False
    optim: str = "adamw_torch"
    run_name: str = "causal-lm"
    report_to: str = "wandb"
    resume_from_checkpoint: str | Path | bool | None = None

    def uses_epochs(self) -> bool:
        """True when length is controlled by ``num_train_epochs``."""
        return self.max_steps < 0

    def validate(self) -> None:
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError(
                f"max_steps must be >= 1 or -1 (epoch mode), got {self.max_steps}"
            )
        if self.uses_epochs() and self.num_train_epochs <= 0:
            raise ValueError(
                "num_train_epochs must be > 0 when max_steps=-1, "
                f"got {self.num_train_epochs}"
            )
        if not self.uses_epochs() and self.num_train_epochs <= 0:
            raise ValueError(f"num_train_epochs must be > 0, got {self.num_train_epochs}")
        if self.per_device_train_batch_size < 1:
            raise ValueError(
                "per_device_train_batch_size must be >= 1, "
                f"got {self.per_device_train_batch_size}"
            )
        if self.per_device_eval_batch_size < 1:
            raise ValueError(
                "per_device_eval_batch_size must be >= 1, "
                f"got {self.per_device_eval_batch_size}"
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps must be >= 1, "
                f"got {self.gradient_accumulation_steps}"
            )
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.warmup_steps is not None and self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.warmup_ratio is not None and not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(
                f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}"
            )
        if self.warmup_steps is not None and self.warmup_ratio is not None:
            raise ValueError("set only one of warmup_steps and warmup_ratio")
        if (
            not self.uses_epochs()
            and self.warmup_steps is not None
            and self.warmup_steps > self.max_steps
        ):
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) cannot exceed "
                f"max_steps ({self.max_steps})"
            )
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be > 0, got {self.max_grad_norm}")
        for name, value in (
            ("adam_beta1", self.adam_beta1),
            ("adam_beta2", self.adam_beta2),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")
        for name, value in (
            ("logging_steps", self.logging_steps),
            ("eval_steps", self.eval_steps),
            ("save_steps", self.save_steps),
            ("save_total_limit", self.save_total_limit),
        ):
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.save_steps % self.eval_steps != 0:
            raise ValueError("save_steps must be a multiple of eval_steps")
        if self.early_stopping_patience < 0:
            raise ValueError(
                "early_stopping_patience must be >= 0, "
                f"got {self.early_stopping_patience}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.data_seed < 0:
            raise ValueError(f"data_seed must be >= 0, got {self.data_seed}")
        if self.dataloader_num_workers < 0:
            raise ValueError(
                f"dataloader_num_workers must be >= 0, got {self.dataloader_num_workers}"
            )
        if (
            self.dataloader_prefetch_factor is not None
            and self.dataloader_prefetch_factor < 1
        ):
            raise ValueError(
                "dataloader_prefetch_factor must be >= 1 when set, "
                f"got {self.dataloader_prefetch_factor}"
            )
        if self.dataloader_persistent_workers and self.dataloader_num_workers < 1:
            raise ValueError(
                "dataloader_persistent_workers requires dataloader_num_workers >= 1"
            )
        if not self.optim.strip():
            raise ValueError("optim must be a non-empty string")


def tokens_per_optimizer_step(
    *,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    block_size: int,
    world_size: int | None = None,
) -> int:
    """Tokens consumed per optimizer step across the process group."""
    ws = distributed_world_size() if world_size is None else max(1, world_size)
    return (
        per_device_train_batch_size
        * gradient_accumulation_steps
        * block_size
        * ws
    )


def optimizer_steps_per_epoch(
    num_examples: int,
    *,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int | None = None,
) -> int:
    """Optimizer steps in one epoch (matches HF ``Trainer`` dataloader math)."""
    if num_examples < 1:
        raise ValueError(f"num_examples must be >= 1, got {num_examples}")
    if per_device_train_batch_size < 1:
        raise ValueError(
            "per_device_train_batch_size must be >= 1, "
            f"got {per_device_train_batch_size}"
        )
    if gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps must be >= 1, "
            f"got {gradient_accumulation_steps}"
        )
    ws = distributed_world_size() if world_size is None else max(1, world_size)
    len_dataloader = math.ceil(num_examples / (per_device_train_batch_size * ws))
    return max(
        len_dataloader // gradient_accumulation_steps
        + int(len_dataloader % gradient_accumulation_steps > 0),
        1,
    )


def epoch_cadence_steps(
    steps_per_epoch: int,
    *,
    evals_per_epoch: int = DEFAULT_EPOCH_EVALS_PER_EPOCH,
) -> dict[str, int]:
    """Log / eval / save cadence for epoch-based runs.

    Targets ``evals_per_epoch`` evenly spaced step evals (``steps // N``). The
    last scheduled eval lands on epoch end when ``steps`` divides evenly;
    otherwise :class:`_EpochEndEvalCallback` covers the epoch boundary.
    Checkpoints once per eval-group (≈ once per epoch); logging tracks the
    streamed reference density (~20 logs / eval).
    """
    if steps_per_epoch < 1:
        raise ValueError(f"steps_per_epoch must be >= 1, got {steps_per_epoch}")
    if evals_per_epoch < 1:
        raise ValueError(f"evals_per_epoch must be >= 1, got {evals_per_epoch}")
    n = min(evals_per_epoch, steps_per_epoch)
    eval_steps = max(1, steps_per_epoch // n)
    return {
        "logging_steps": max(1, eval_steps // _LOGS_PER_EVAL_INTERVAL),
        "eval_steps": eval_steps,
        "save_steps": eval_steps * n,
    }


def apply_epoch_cadence(
    config: CausalLmTrainerConfig,
    *,
    num_train_examples: int,
    world_size: int | None = None,
    evals_per_epoch: int = DEFAULT_EPOCH_EVALS_PER_EPOCH,
) -> CausalLmTrainerConfig:
    """Set step-based log/eval/save cadence for an epoch-length run."""
    if not config.uses_epochs():
        return config
    steps = optimizer_steps_per_epoch(
        num_train_examples,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        world_size=world_size,
    )
    cadence = epoch_cadence_steps(steps, evals_per_epoch=evals_per_epoch)
    detail(
        f"epoch cadence: {steps:,} steps/epoch → "
        f"log every {cadence['logging_steps']}, "
        f"eval every {cadence['eval_steps']} "
        f"(~{evals_per_epoch}×/epoch incl. epoch end), "
        f"save every {cadence['save_steps']}",
        logger=log,
    )
    return replace(config, **cadence)


class _EpochEndEvalCallback(TrainerCallback):
    """Ensure an eval runs at epoch end when step cadence misses the boundary."""

    def on_epoch_end(self, args, state, control, **kwargs):
        if args.eval_strategy != "steps" or args.eval_steps < 1:
            return control
        if state.global_step > 0 and state.global_step % args.eval_steps != 0:
            control.should_evaluate = True
        return control


def has_eval_examples(eval_dataset) -> bool:
    """True when eval has rows (accepts a single dataset or a named mapping)."""
    if eval_dataset is None:
        return False
    if isinstance(eval_dataset, Mapping):
        return len(eval_dataset) > 0 and all(
            len(dataset) > 0 for dataset in eval_dataset.values()
        )
    return len(eval_dataset) > 0


def reporting_disabled(report_to: str | list[str] | tuple[str, ...] | None) -> bool:
    """True when HF reporting should be off (no W&B / integrations)."""
    if report_to is None:
        return True
    if isinstance(report_to, str):
        return report_to.strip().lower() in {"", "none"}
    return len(report_to) == 0 or all(
        isinstance(item, str) and item.strip().lower() == "none" for item in report_to
    )


def _transformers_major_version() -> int:
    return int(version("transformers").split(".", maxsplit=1)[0])


def _warmup_training_args(config: CausalLmTrainerConfig) -> dict[str, int | float]:
    """Map alien-ink warmup fields onto HF ``TrainingArguments``.

    Transformers v5+ accepts a float ``warmup_steps`` (ratio) and deprecated
    ``warmup_ratio``. Passing both, or passing ``warmup_ratio=0`` alongside
    explicit step counts, overwrites ``warmup_steps`` in recent releases.
    """
    if config.warmup_ratio is not None:
        if _transformers_major_version() >= 5:
            return {"warmup_steps": config.warmup_ratio}
        return {"warmup_steps": 0, "warmup_ratio": config.warmup_ratio}
    steps = config.warmup_steps if config.warmup_steps is not None else 0
    if _transformers_major_version() >= 5:
        return {"warmup_steps": steps}
    return {"warmup_steps": steps, "warmup_ratio": 0.0}


def best_model_metric(eval_dataset) -> str:
    """Metric that drives best-model selection and early stopping.

    Named eval sets (a mapping) report per-set losses (``eval_<name>_loss``);
    the first named set is the reference. A single dataset keeps ``eval_loss``.
    """
    if isinstance(eval_dataset, Mapping) and len(eval_dataset) > 0:
        first = next(iter(eval_dataset))
        return f"eval_{first}_loss"
    return "eval_loss"


def build_training_arguments(
    config: CausalLmTrainerConfig,
    *,
    has_eval: bool,
    metric_for_best_model: str = "eval_loss",
) -> TrainingArguments:
    """Map ``CausalLmTrainerConfig`` onto HF ``TrainingArguments``."""
    config.validate()
    device, use_fp16, use_bf16 = device_info(
        prefer_bf16=config.prefer_bf16,
        prefer_fp16=config.prefer_fp16,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    report_to: str | list[str] = (
        "none" if reporting_disabled(config.report_to) else config.report_to
    )

    # Epoch-based subset runs use step cadence (N evals/epoch including epoch end)
    # rather than HF's once-per-epoch strategy.
    if not has_eval:
        eval_strategy = "no"
    else:
        eval_strategy = "steps"

    num_workers = config.dataloader_num_workers if device == "cuda" else 0
    args: dict[str, Any] = dict(
        output_dir=str(config.output_dir),
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        **_warmup_training_args(config),
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        optim=config.optim,
        logging_steps=config.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=has_eval,
        metric_for_best_model=metric_for_best_model if has_eval else None,
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=config.tf32,
        torch_compile=config.torch_compile,
        report_to=report_to,
        run_name=config.run_name,
        dataloader_pin_memory=device == "cuda",
        dataloader_num_workers=num_workers,
        dataloader_persistent_workers=(
            config.dataloader_persistent_workers and num_workers > 0
        ),
        seed=config.seed,
        data_seed=config.data_seed,
        remove_unused_columns=False,
    )
    if config.dataloader_prefetch_factor is not None and num_workers > 0:
        args["dataloader_prefetch_factor"] = config.dataloader_prefetch_factor
    return TrainingArguments(**args)


def build_lm_data_collator(tokenizer):
    """Pad causal-LM features while preserving precomputed ``labels``.

    Packed train blocks set ``labels = input_ids``; completion-eval rows mask
    the prompt with ``-100``. Hugging Face's
    ``DataCollatorForLanguageModeling(mlm=False)`` always overwrites labels
    from ``input_ids``, which would destroy that mask — so pad labels with
    ``-100`` when they are already present.
    """
    import torch

    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty feature list")
        has_labels = "labels" in features[0]
        labels_list = [feature.pop("labels") for feature in features] if has_labels else None
        batch = tokenizer.pad(features, return_tensors="pt")
        if labels_list is not None:
            max_len = int(batch["input_ids"].shape[1])
            padded = [
                list(labels) + [-100] * (max_len - len(labels)) for labels in labels_list
            ]
            batch["labels"] = torch.tensor(padded, dtype=torch.long)
        else:
            labels = batch["input_ids"].clone()
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    return collate


def build_trainer_callbacks(
    config: CausalLmTrainerConfig,
    *,
    has_eval: bool,
) -> list:
    callbacks = []
    if has_eval and config.uses_epochs():
        callbacks.append(_EpochEndEvalCallback())
    if has_eval and config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        )
    return callbacks


def build_causal_lm_trainer(
    *,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: CausalLmTrainerConfig,
) -> Trainer:
    """Build a HF ``Trainer`` for causal LM with optional eval / early stopping.

    ``eval_dataset`` may be a single dataset or a mapping of named datasets
    (reported as ``eval_<name>_loss``; the first name drives best-model
    selection and early stopping).
    """
    has_eval = has_eval_examples(eval_dataset)
    # Materialized epoch runs: derive log/eval/save from dataset length so we
    # hit ~5 evals per epoch (including epoch end). Streamed / step-capped runs
    # already carry an explicit cadence from the manifest.
    if config.uses_epochs():
        try:
            num_examples = len(train_dataset)
        except TypeError:
            num_examples = 0
        if num_examples > 0:
            config = apply_epoch_cadence(config, num_train_examples=num_examples)
    args = build_training_arguments(
        config,
        has_eval=has_eval,
        metric_for_best_model=best_model_metric(eval_dataset),
    )
    if not has_eval:
        eval_arg = None
    elif isinstance(eval_dataset, Mapping):
        eval_arg = dict(eval_dataset)
    else:
        eval_arg = eval_dataset
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_arg,
        processing_class=tokenizer,
        data_collator=build_lm_data_collator(tokenizer),
        callbacks=build_trainer_callbacks(config, has_eval=has_eval),
    )


def save_model_and_tokenizer(trainer: Trainer, tokenizer, output_dir: Path) -> None:
    step("Saving model and tokenizer...", logger=log)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    detail(f"saved to {output_dir}", logger=log)


def _trainer_global_step(trainer: Trainer) -> int:
    state = getattr(trainer, "state", None)
    if state is None:
        return 0
    return int(getattr(state, "global_step", 0) or 0)


def summarize_training(
    *,
    trainer: Trainer,
    run_name: str,
    run_label: str,
    status: str,
    max_steps: int,
    tokens_per_step: int,
    model_size: ModelSize,
    accelerator: AcceleratorInfo,
    train_metrics: dict[str, Any] | None = None,
    train_loss: float | None = None,
) -> RunSummary:
    """Build, persist, log, and (if active) push an end-of-run summary."""
    summary = build_run_summary(
        run_name=run_name,
        run_label=run_label,
        status=status,
        global_step=_trainer_global_step(trainer),
        max_steps=max_steps,
        tokens_per_step=tokens_per_step,
        train_runtime_sec=None,
        train_loss=train_loss,
        model_size=model_size,
        accelerator=accelerator,
        trainer_metrics=train_metrics,
    )
    blank(logger=log)
    log_run_summary(summary)
    save_run_summary(Path(trainer.args.output_dir), summary)
    push_summary_to_wandb(summary)
    return summary


def train_and_save(
    *,
    trainer: Trainer,
    tokenizer,
    output_dir: Path,
    resume_from_checkpoint: str | Path | bool | None = None,
    run_name: str = "causal-lm",
    run_label: str = "regular",
    max_steps: int | None = None,
    tokens_per_step: int = 0,
    model_size: ModelSize | None = None,
    accelerator: AcceleratorInfo | None = None,
) -> tuple[Trainer, RunSummary | None]:
    """Run ``trainer.train()``, save artifacts, and auto-summarize metrics.

    Always attempts an end-of-run summary (completed / interrupted / failed)
    when ``model_size`` and ``accelerator`` are provided so cross-GPU speed
    comparisons have a local ``run_summary.json`` even without W&B.
    """
    step("Starting training...", logger=log)
    if resume_from_checkpoint:
        detail(f"resume_from_checkpoint: {resume_from_checkpoint}", logger=log)

    status = "completed"
    train_metrics: dict[str, Any] | None = None
    train_loss: float | None = None
    train_error: BaseException | None = None

    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        train_metrics = dict(getattr(train_result, "metrics", None) or {})
        train_loss = getattr(train_result, "training_loss", None)
        blank(logger=log)
        save_model_and_tokenizer(trainer, tokenizer, output_dir)
    except KeyboardInterrupt:
        status = "interrupted"
        step("Training interrupted; writing run summary...", logger=log)
    except Exception as exc:
        status = "failed"
        train_error = exc
        step(f"Training failed ({type(exc).__name__}); writing run summary...", logger=log)

    summary: RunSummary | None = None
    if model_size is not None and accelerator is not None:
        resolved_max = (
            max_steps if max_steps is not None else int(trainer.args.max_steps)
        )
        # Epoch mode leaves args.max_steps at -1; prefer the Trainer-computed plan.
        if resolved_max < 0:
            resolved_max = int(getattr(trainer.state, "max_steps", 0) or 0)
        summary = summarize_training(
            trainer=trainer,
            run_name=run_name,
            run_label=run_label,
            status=status,
            max_steps=resolved_max,
            tokens_per_step=tokens_per_step,
            model_size=model_size,
            accelerator=accelerator,
            train_metrics=train_metrics,
            train_loss=train_loss,
        )

    if train_error is not None:
        raise train_error
    return trainer, summary
