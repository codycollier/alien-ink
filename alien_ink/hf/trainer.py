"""Hugging Face Trainer helpers for causal language modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from alien_ink.device import device_info, distributed_world_size
from alien_ink.log import blank, detail, get_logger, step

log = get_logger("hf.trainer")


@dataclass(frozen=True)
class CausalLmTrainerConfig:
    """Hyperparameters and bookkeeping for a causal-LM ``Trainer`` run."""

    output_dir: Path
    max_steps: int = 50_000
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 6e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 2_000
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
    prefer_fp16: bool = True
    prefer_bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    run_name: str = "causal-lm"
    report_to: str = "wandb"
    resume_from_checkpoint: str | Path | bool | None = None

    def validate(self) -> None:
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
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
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")


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


def precision_label(*, use_fp16: bool, use_bf16: bool) -> str:
    if use_bf16:
        return "bf16"
    if use_fp16:
        return "fp16"
    return "fp32"


def has_eval_examples(eval_dataset) -> bool:
    return eval_dataset is not None and len(eval_dataset) > 0


def reporting_disabled(report_to: str | list[str] | tuple[str, ...] | None) -> bool:
    """True when HF reporting should be off (no W&B / integrations)."""
    if report_to is None:
        return True
    if isinstance(report_to, str):
        return report_to.strip().lower() in {"", "none"}
    return len(report_to) == 0 or all(
        isinstance(item, str) and item.strip().lower() == "none" for item in report_to
    )


def build_training_arguments(
    config: CausalLmTrainerConfig,
    *,
    has_eval: bool,
) -> TrainingArguments:
    """Map ``CausalLmTrainerConfig`` onto HF ``TrainingArguments``."""
    device, use_fp16, use_bf16 = device_info(
        prefer_bf16=config.prefer_bf16,
        prefer_fp16=config.prefer_fp16,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    report_to: str | list[str] = (
        "none" if reporting_disabled(config.report_to) else config.report_to
    )

    return TrainingArguments(
        output_dir=str(config.output_dir),
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to=report_to,
        run_name=config.run_name,
        dataloader_pin_memory=device == "cuda",
        dataloader_num_workers=(
            config.dataloader_num_workers if device == "cuda" else 0
        ),
        seed=config.seed,
        remove_unused_columns=False,
    )


def build_lm_data_collator(tokenizer) -> DataCollatorForLanguageModeling:
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


def build_trainer_callbacks(
    config: CausalLmTrainerConfig,
    *,
    has_eval: bool,
) -> list:
    callbacks = []
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
    """Build a HF ``Trainer`` for causal LM with optional eval / early stopping."""
    has_eval = has_eval_examples(eval_dataset)
    args = build_training_arguments(config, has_eval=has_eval)
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if has_eval else None,
        processing_class=tokenizer,
        data_collator=build_lm_data_collator(tokenizer),
        callbacks=build_trainer_callbacks(config, has_eval=has_eval),
    )


def save_model_and_tokenizer(trainer: Trainer, tokenizer, output_dir: Path) -> None:
    step("Saving model and tokenizer...", logger=log)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    detail(f"saved to {output_dir}", logger=log)


def train_and_save(
    *,
    trainer: Trainer,
    tokenizer,
    output_dir: Path,
    resume_from_checkpoint: str | Path | bool | None = None,
) -> Trainer:
    """Run ``trainer.train()`` then save model and tokenizer."""
    step("Starting training...", logger=log)
    if resume_from_checkpoint:
        detail(f"resume_from_checkpoint: {resume_from_checkpoint}", logger=log)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    blank(logger=log)
    save_model_and_tokenizer(trainer, tokenizer, output_dir)
    return trainer
