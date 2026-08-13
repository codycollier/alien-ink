"""From-scratch GPT-2 pretraining on tinygrad, driven by an HF ``Manifest``."""

from __future__ import annotations

import math
import shutil
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from datasets import IterableDataset
from tinygrad import Device, Tensor, dtypes
from tinygrad.nn.state import get_parameters
from transformers import set_seed

from alien_ink.com.device import (
    AcceleratorInfo,
    collect_accelerator_info,
    device_info,
    distributed_world_size,
    introspect,
)
from alien_ink.com.env import load_env
from alien_ink.com.log import banner, blank, detail, get_logger, header, step
from alien_ink.com.wb import build_run_config, require_wandb_identity, set_wandb_dir, wandb_run
from alien_ink.hf.curriculum import Curriculum
from alien_ink.hf.completion import CompletionDataConfig
from alien_ink.hf.manifest import Manifest
from alien_ink.hf.metrics import (
    ModelSize,
    RunSummary,
    build_run_config_payload,
    build_run_summary,
    collect_software_versions,
    log_run_summary,
    push_summary_to_wandb,
    save_run_config,
    save_run_summary,
)
from alien_ink.hf.model import CausalLmArchConfig, load_tokenizer
from alien_ink.hf.pretrain import (
    PretrainConfig,
    prepare_lm_datasets,
    resolve_use_wandb,
    with_trainer,
)
from alien_ink.hf.trainer import (
    apply_epoch_cadence,
    optimizer_steps_per_epoch,
    reporting_disabled,
    tokens_per_optimizer_step,
)
from alien_ink.tg.model import GPT2, build_gpt2, count_gpt2_params
from alien_ink.tg.trainer import (
    build_optimizer,
    cosine_lr,
    evaluate_loss,
    group_is_jit_ready,
    group_microbatches,
    iter_shuffled_microbatches,
    make_train_step,
    save_gpt2,
    set_optimizer_lr,
    train_step_variable,
)

log = get_logger("tg.pretrain")

_IGNORED_HARDWARE = (
    "torch_compile",
    "tf32",
    "optim",
    "gradient_checkpointing",
)


def validate_tg_manifest(manifest: Manifest) -> None:
    """Tinygrad pretrain is GPT-2, from-scratch, packed map-style data only."""
    if manifest.stage != "pre":
        raise ValueError(
            "tinygrad pretrain requires stage='pre', "
            f"got {manifest.stage!r}"
        )
    if not isinstance(manifest.model, CausalLmArchConfig):
        raise ValueError(
            "tinygrad pretrain requires a CausalLmArchConfig, got "
            f"{type(manifest.model).__name__}"
        )
    if manifest.model.family != "gpt-2":
        raise ValueError(
            "tinygrad backend only supports family='gpt-2', "
            f"got {manifest.model.family!r}"
        )
    if isinstance(manifest.data, Curriculum):
        raise ValueError("tinygrad pretrain does not support Curriculum data")
    if isinstance(manifest.data, CompletionDataConfig):
        raise ValueError("tinygrad pretrain does not support CompletionDataConfig")


def resolve_tg_dtype(*, prefer_bf16: bool, prefer_fp16: bool):
    """Same bf16-on-Ampere policy as the HF trainer; maps onto tinygrad dtypes."""
    _device, use_fp16, use_bf16 = device_info(
        prefer_bf16=prefer_bf16,
        prefer_fp16=prefer_fp16,
    )
    if use_bf16:
        return dtypes.bfloat16
    if use_fp16:
        return dtypes.float16
    return dtypes.float32


def _cast_parameters(model: GPT2, dtype) -> None:
    if dtype == dtypes.float32:
        return
    for param in get_parameters(model):
        param.replace(param.cast(dtype).contiguous()).realize()


def _log_banner(
    *,
    title: str,
    run_label: str,
    zdeck_name: str | None,
    config: PretrainConfig,
    dtype,
) -> tuple[AcceleratorInfo, int]:
    header(logger=log)
    accel = collect_accelerator_info(
        prefer_bf16=config.trainer.prefer_bf16,
        prefer_fp16=config.trainer.prefer_fp16,
    )
    for line in introspect(info=accel).splitlines():
        log.info(line)
    blank(logger=log)
    banner(title, logger=log)
    step(f"run: {run_label}", logger=log)
    if zdeck_name:
        detail(f"zdeck: {zdeck_name}", logger=log)
    detail("backend: tinygrad", logger=log)
    detail(f"tinygrad device: {Device.DEFAULT}", logger=log)
    detail(f"dtype: {dtype}", logger=log)
    world_size = accel.world_size or distributed_world_size()
    tps = tokens_per_optimizer_step(
        per_device_train_batch_size=config.trainer.per_device_train_batch_size,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        block_size=config.data.block_size,
        world_size=world_size,
    )
    detail(f"model family: {config.arch.family}", logger=log)
    if config.trainer.uses_epochs():
        step(
            f"Training epochs: {config.trainer.num_train_epochs:g} "
            f"(~{tps:,} tokens/step)",
            logger=log,
        )
    else:
        step(
            f"Training steps: {config.trainer.max_steps:,} "
            f"(~{tps:,} tokens/step, "
            f"~{config.trainer.max_steps * tps:,} tokens total)",
            logger=log,
        )
    step(
        "ignoring HardwareConfig "
        + ", ".join(_IGNORED_HARDWARE),
        logger=log,
    )
    return accel, tps


def _warmup_steps(config: PretrainConfig, max_steps: int) -> int:
    if config.trainer.warmup_ratio is not None:
        return int(config.trainer.warmup_ratio * max_steps)
    if config.trainer.warmup_steps is not None:
        return int(config.trainer.warmup_steps)
    return 0


def _rotate_checkpoints(output_dir: Path, save_total_limit: int) -> None:
    checkpoints = sorted(
        (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    extra = len(checkpoints) - save_total_limit
    if extra <= 0:
        return
    for path in checkpoints[:extra]:
        shutil.rmtree(path, ignore_errors=True)


def _save_checkpoint(
    model: GPT2,
    tokenizer,
    output_dir: Path,
    global_step: int,
    save_total_limit: int,
) -> None:
    ckpt = output_dir / f"checkpoint-{global_step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    save_gpt2(model, ckpt / "model.safetensors")
    tokenizer.save_pretrained(ckpt)
    _rotate_checkpoints(output_dir, save_total_limit)


def _materialize_eval_batches(eval_dataset, batch_size: int) -> list[np.ndarray]:
    n = len(eval_dataset)
    batches: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        rows = [
            eval_dataset[i]["input_ids"]
            for i in range(start, min(start + batch_size, n))
        ]
        batches.append(np.asarray(rows, dtype=np.int32))
    return batches


def _summarize(
    *,
    run_name: str,
    run_label: str,
    status: str,
    global_step: int,
    max_steps: int,
    tokens_per_step: int,
    train_runtime_sec: float | None,
    train_loss: float | None,
    model_size: ModelSize,
    accelerator: AcceleratorInfo,
    output_dir: Path,
    trainer_metrics: dict[str, Any] | None = None,
) -> RunSummary:
    summary = build_run_summary(
        run_name=run_name,
        run_label=run_label,
        status=status,
        global_step=global_step,
        max_steps=max_steps,
        tokens_per_step=tokens_per_step,
        train_runtime_sec=train_runtime_sec,
        train_loss=train_loss,
        model_size=model_size,
        accelerator=accelerator,
        trainer_metrics=trainer_metrics,
    )
    blank(logger=log)
    log_run_summary(summary)
    save_run_summary(output_dir, summary)
    push_summary_to_wandb(summary)
    return summary


def pretrain(
    manifest: Manifest,
    *,
    run_label: str = "zdeck",
    zdeck_name: str | None = None,
    title: str | None = None,
    env_files: tuple[Path, ...] | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    use_wandb: bool | None = None,
    extra_configs: Mapping[str, Any] | None = None,
):
    """End-to-end tinygrad GPT-2 pretrain from a zdeck ``Manifest``.

    Reuses HF tokenize/pack and writes ``run_config.json`` / ``run_summary.json``
    plus safetensors weights (not a ``GPT2LMHeadModel`` checkpoint).
    """
    validate_tg_manifest(manifest)
    env_files = env_files if env_files is not None else (Path.cwd() / ".env",)
    zdeck_name = zdeck_name or manifest.run_name
    title = title or manifest.title
    wandb_entity = wandb_entity if wandb_entity is not None else manifest.wandb.entity
    wandb_project = (
        wandb_project if wandb_project is not None else manifest.wandb.project
    )
    wandb_name = wandb_name or manifest.wandb.resolved_name(manifest.run_name)

    config = manifest.to_pretrain_config()
    want_wandb = resolve_use_wandb(config, use_wandb)
    if not want_wandb and not reporting_disabled(config.trainer.report_to):
        config = with_trainer(config, report_to="none")
    if want_wandb:
        require_wandb_identity(entity=wandb_entity, project=wandb_project)

    dtype = resolve_tg_dtype(
        prefer_bf16=config.trainer.prefer_bf16,
        prefer_fp16=config.trainer.prefer_fp16,
    )
    accel, tokens_per_step = _log_banner(
        title=title,
        run_label=run_label,
        zdeck_name=zdeck_name,
        config=config,
        dtype=dtype,
    )

    blank(logger=log)
    step("Loading credentials...", logger=log)
    env = load_env(
        *env_files,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )
    if wandb_name and wandb_name != config.trainer.run_name:
        config = with_trainer(config, run_name=wandb_name)

    set_seed(config.trainer.seed)
    Tensor.manual_seed(config.trainer.seed)
    if want_wandb:
        set_wandb_dir(config.trainer.output_dir)

    step(f"Loading tokenizer ({config.arch.tokenizer_name})...", logger=log)
    tokenizer = load_tokenizer(config.arch.tokenizer_name)
    vocab_size = (
        len(tokenizer) if hasattr(tokenizer, "__len__") else tokenizer.vocab_size
    )
    step("Initializing tinygrad GPT-2 from config with random weights...", logger=log)
    model = build_gpt2(config.arch, vocab_size)
    _cast_parameters(model, dtype)
    model_size = count_gpt2_params(model)
    detail(f"parameters: {model_size.total_params:,}", logger=log)

    train_dataset, eval_dataset = prepare_lm_datasets(config.data, tokenizer)
    if isinstance(train_dataset, IterableDataset):
        raise ValueError(
            "tinygrad pretrain requires a map-style train set; "
            "use data.mode='complete' or 'subset'"
        )

    world_size = max(1, accel.world_size or distributed_world_size())
    if config.trainer.uses_epochs():
        config = replace(
            config,
            trainer=apply_epoch_cadence(
                config.trainer,
                num_train_examples=len(train_dataset),
                world_size=world_size,
            ),
        )
        steps_per_epoch = optimizer_steps_per_epoch(
            len(train_dataset),
            per_device_train_batch_size=config.trainer.per_device_train_batch_size,
            gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
            world_size=world_size,
        )
        max_steps = max(
            1, math.ceil(config.trainer.num_train_epochs * steps_per_epoch)
        )
    else:
        max_steps = config.trainer.max_steps

    warmup = _warmup_steps(config, max_steps)
    output_dir = Path(config.trainer.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_configs: dict[str, Any] = {
        "data": config.data,
        "arch": config.arch,
        "trainer": config.trainer,
        "backend": {"name": "tinygrad", "device": str(Device.DEFAULT), "dtype": str(dtype)},
    }
    if extra_configs:
        log_configs.update(dict(extra_configs))
    else:
        log_configs.update(
            {
                "stage": {"name": manifest.stage},
                "hardware": manifest.hardware,
                "schedule": manifest.schedule,
                "wandb": manifest.wandb,
            }
        )

    config_payload = build_run_config_payload(
        run_label=run_label,
        run_name=config.trainer.run_name,
        title=title,
        configs=log_configs,
        accelerator=accel,
        model_size=model_size,
        tokens_per_step=tokens_per_step,
    )
    blank(logger=log)
    step("Writing run config...", logger=log)
    save_run_config(output_dir, config_payload)

    optimizer = build_optimizer(
        model,
        learning_rate=config.trainer.learning_rate,
        adam_beta1=config.trainer.adam_beta1,
        adam_beta2=config.trainer.adam_beta2,
        weight_decay=config.trainer.weight_decay,
    )
    accum = config.trainer.gradient_accumulation_steps
    batch_size = config.trainer.per_device_train_batch_size
    jit_step = make_train_step(
        model,
        optimizer,
        max_grad_norm=config.trainer.max_grad_norm,
        accum_steps=accum,
    )
    eval_batches = _materialize_eval_batches(
        eval_dataset, config.trainer.per_device_eval_batch_size
    )

    software = collect_software_versions()
    run_config = build_run_config(
        run_label=run_label,
        env=env,
        configs=log_configs,
        prefer_bf16=config.trainer.prefer_bf16,
        prefer_fp16=config.trainer.prefer_fp16,
        accelerator=accel,
        tokens_per_step=tokens_per_step,
        model=model_size.as_dict(),
        software=software,
    )

    status = "completed"
    train_error: BaseException | None = None
    global_step = 0
    last_loss: float | None = None
    t0 = time.perf_counter()
    train_metrics: dict[str, Any] = {}

    blank(logger=log)
    with wandb_run(
        entity=env.wandb_entity or "",
        project=env.wandb_project or "",
        name=config.trainer.run_name,
        config=run_config,
        dir=output_dir,
        enabled=want_wandb,
    ):
        try:
            step("Starting tinygrad training...", logger=log)
            epoch = 0
            while global_step < max_steps:
                micros = iter_shuffled_microbatches(
                    train_dataset,
                    batch_size=batch_size,
                    seed=config.trainer.seed,
                    epoch=epoch,
                )
                for group in group_microbatches(micros, accum):
                    if global_step >= max_steps:
                        break
                    lr = cosine_lr(
                        global_step,
                        max_steps=max_steps,
                        warmup_steps=warmup,
                        learning_rate=config.trainer.learning_rate,
                    )
                    set_optimizer_lr(optimizer, lr)
                    if group_is_jit_ready(group, accum):
                        stacked = Tensor(np.stack(group))
                        loss_t = jit_step(stacked)
                        last_loss = float(loss_t.item())
                    else:
                        last_loss = train_step_variable(
                            model,
                            optimizer,
                            group,
                            max_grad_norm=config.trainer.max_grad_norm,
                        )
                    global_step += 1
                    if global_step % config.trainer.logging_steps == 0:
                        detail(
                            f"step {global_step}/{max_steps}  "
                            f"loss={last_loss:.4f}  lr={lr:.6g}",
                            logger=log,
                        )
                        if want_wandb:
                            import wandb

                            if wandb.run is not None:
                                wandb.log(
                                    {
                                        "train/loss": last_loss,
                                        "train/lr": lr,
                                    },
                                    step=global_step,
                                )
                    if global_step % config.trainer.eval_steps == 0:
                        eval_loss = evaluate_loss(model, eval_batches)
                        detail(f"eval loss: {eval_loss:.4f}", logger=log)
                        if want_wandb:
                            import wandb

                            if wandb.run is not None:
                                wandb.log(
                                    {"eval/loss": eval_loss}, step=global_step
                                )
                    if global_step % config.trainer.save_steps == 0:
                        _save_checkpoint(
                            model,
                            tokenizer,
                            output_dir,
                            global_step,
                            config.trainer.save_total_limit,
                        )
                epoch += 1

            tokenizer.save_pretrained(output_dir)
            save_gpt2(model, output_dir / "model.safetensors")
        except KeyboardInterrupt:
            status = "interrupted"
            step("Training interrupted; writing run summary...", logger=log)
        except Exception as exc:
            status = "failed"
            train_error = exc
            step(
                f"Training failed ({type(exc).__name__}); writing run summary...",
                logger=log,
            )

        runtime = time.perf_counter() - t0
        train_metrics["train_runtime"] = runtime
        if last_loss is not None:
            train_metrics["train_loss"] = last_loss
        summary = _summarize(
            run_name=config.trainer.run_name,
            run_label=run_label,
            status=status,
            global_step=global_step,
            max_steps=max_steps,
            tokens_per_step=tokens_per_step,
            train_runtime_sec=runtime,
            train_loss=last_loss,
            model_size=model_size,
            accelerator=accel,
            output_dir=output_dir,
            trainer_metrics=train_metrics,
        )

    if train_error is not None:
        raise train_error
    return model, summary
