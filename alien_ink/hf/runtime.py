"""Shared training lifecycle; model and dataset selection stay in the stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from transformers import set_seed

from alien_ink.com.device import (
    AcceleratorInfo,
    collect_accelerator_info,
    distributed_world_size,
    introspect,
)
from alien_ink.com.env import EnvConfig, load_env
from alien_ink.com.log import banner, blank, detail, get_logger, header, step
from alien_ink.com.wb import (
    build_run_config,
    require_wandb_identity,
    set_wandb_dir,
    wandb_run,
)
from alien_ink.hf.metrics import ModelSize, build_run_config_payload, save_run_config
from alien_ink.hf.trainer import (
    CausalLmTrainerConfig,
    build_causal_lm_trainer,
    reporting_disabled,
    tokens_per_optimizer_step,
    train_and_save,
)

log = get_logger("hf.runtime")


def resolve_use_wandb(trainer: CausalLmTrainerConfig, use_wandb: bool | None) -> bool:
    """An explicit reporting flag takes precedence over trainer reporting."""
    return (
        use_wandb
        if use_wandb is not None
        else not reporting_disabled(trainer.report_to)
    )


def resolve_trainer_config(
    trainer: CausalLmTrainerConfig,
    *,
    use_wandb: bool | None,
    wandb_name: str | None,
    resume_from_checkpoint: str | Path | bool | None,
) -> tuple[CausalLmTrainerConfig, bool]:
    """Resolve call-time overrides before validating and logging a run."""
    enabled = resolve_use_wandb(trainer, use_wandb)
    changes = {}
    if not enabled:
        changes["report_to"] = "none"
    if wandb_name:
        changes["run_name"] = wandb_name
    if resume_from_checkpoint is not None:
        changes["resume_from_checkpoint"] = resume_from_checkpoint
    return replace(trainer, **changes), enabled


def initialize_run(
    trainer: CausalLmTrainerConfig,
    *,
    env_files: tuple[Path, ...] | None,
    wandb_entity: str | None,
    wandb_project: str | None,
    enabled: bool,
) -> EnvConfig:
    """Load runtime credentials, seed RNGs, and prepare optional reporting."""
    if enabled:
        require_wandb_identity(entity=wandb_entity, project=wandb_project)
    blank(logger=log)
    step("Loading credentials...", logger=log)
    env = load_env(
        *(env_files if env_files is not None else (Path.cwd() / ".env",)),
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )
    set_seed(trainer.seed)
    if enabled:
        set_wandb_dir(trainer.output_dir)
    return env


def log_training_banner(
    *,
    title: str,
    run_label: str,
    zdeck_name: str | None,
    trainer: CausalLmTrainerConfig,
    block_size: int,
    model_label: str,
    action: str,
) -> tuple[AcceleratorInfo, int]:
    header(logger=log)

    accel = collect_accelerator_info(
        prefer_bf16=trainer.prefer_bf16,
        prefer_fp16=trainer.prefer_fp16,
    )
    for line in introspect(info=accel).splitlines():
        log.info(line)
    blank(logger=log)

    banner(title, logger=log)
    step(f"run: {run_label}", logger=log)
    if zdeck_name:
        detail(f"zdeck: {zdeck_name}", logger=log)

    world_size = accel.world_size or distributed_world_size()
    tps = tokens_per_optimizer_step(
        per_device_train_batch_size=trainer.per_device_train_batch_size,
        gradient_accumulation_steps=trainer.gradient_accumulation_steps,
        block_size=block_size,
        world_size=world_size,
    )
    detail(model_label, logger=log)
    if trainer.uses_epochs():
        step(
            f"{action} epochs: {trainer.num_train_epochs:g} (~{tps:,} tokens/step)",
            logger=log,
        )
    else:
        step(
            f"{action} steps: {trainer.max_steps:,} "
            f"(~{tps:,} tokens/step, "
            f"~{trainer.max_steps * tps:,} tokens total)",
            logger=log,
        )
    if trainer.gradient_checkpointing:
        step("Gradient checkpointing enabled", logger=log)
    if trainer.torch_compile:
        step("torch.compile enabled", logger=log)
    if trainer.tf32 is True:
        step("TF32 enabled", logger=log)
    if trainer.optim != "adamw_torch":
        detail(f"optimizer: {trainer.optim}", logger=log)
    return accel, tps


def run_training(
    *,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: CausalLmTrainerConfig,
    log_configs: Mapping[str, Any],
    run_label: str,
    title: str,
    env: EnvConfig,
    accel: AcceleratorInfo,
    tokens_per_step: int,
    model_size: ModelSize,
    want_wandb: bool,
):
    """Persist run configuration, build the trainer, and train under reporting."""
    config_payload = build_run_config_payload(
        run_label=run_label,
        run_name=config.run_name,
        title=title,
        configs=log_configs,
        accelerator=accel,
        model_size=model_size,
        tokens_per_step=tokens_per_step,
    )
    blank(logger=log)
    step("Writing run config...", logger=log)
    save_run_config(config.output_dir, config_payload)

    trainer = build_causal_lm_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
    )
    software = config_payload["software"]
    run_config = build_run_config(
        run_label=run_label,
        env=env,
        configs=log_configs,
        prefer_bf16=config.prefer_bf16,
        prefer_fp16=config.prefer_fp16,
        accelerator=accel,
        tokens_per_step=tokens_per_step,
        model=model_size.as_dict(),
        software=software,
    )

    blank(logger=log)
    with wandb_run(
        entity=env.wandb_entity or "",
        project=env.wandb_project or "",
        name=config.run_name,
        config=run_config,
        dir=config.output_dir,
        enabled=want_wandb,
    ):
        return train_and_save(
            trainer=trainer,
            tokenizer=tokenizer,
            output_dir=config.output_dir,
            resume_from_checkpoint=config.resume_from_checkpoint,
            run_name=config.run_name,
            run_label=run_label,
            max_steps=config.max_steps,
            tokens_per_step=tokens_per_step,
            model_size=model_size,
            accelerator=accel,
        )
