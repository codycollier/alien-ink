"""High-level causal-LM pretraining on Hugging Face datasets.

Composes data + architecture + trainer configs. Defaults assume a local GPU
(Mist / RTX 3070 ~8 GB): microbatch 2, gradient accumulation 16, block 1024.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from datasets import IterableDataset
from transformers import set_seed

from alien_ink.device import collect_accelerator_info, distributed_world_size
from alien_ink.env import load_env
from alien_ink.hf.ds import PretrainDataConfig, load_train_eval
from alien_ink.hf.metrics import (
    build_run_config_payload,
    collect_software_versions,
    count_model_params,
    save_run_config,
)
from alien_ink.hf.model import CausalLmArchConfig, build_model_and_tokenizer, gpt2_arch
from alien_ink.hf.tok import tokenize_and_chunk
from alien_ink.hf.trainer import (
    CausalLmTrainerConfig,
    build_causal_lm_trainer,
    reporting_disabled,
    tokens_per_optimizer_step,
    train_and_save,
)
from alien_ink.log import banner, blank, detail, get_logger, header, step
from alien_ink.wb import build_run_config, require_wandb_identity, set_wandb_dir, wandb_run

log = get_logger("hf.pretrain")

# Mist RTX 3070 (~8 GB) batch recipe: effective batch = 2 * 16 = 32.
_MIST_TRAIN_BATCH = 2
_MIST_EVAL_BATCH = 2
_MIST_GRAD_ACCUM = 16


@dataclass(frozen=True)
class PretrainConfig:
    """Composable config for from-scratch causal-LM pretraining on a Hub corpus."""

    data: PretrainDataConfig
    arch: CausalLmArchConfig = field(default_factory=gpt2_arch)
    trainer: CausalLmTrainerConfig = field(
        default_factory=lambda: CausalLmTrainerConfig(
            output_dir=Path.cwd() / "output" / "pretrain",
            run_name="pretrain",
            per_device_train_batch_size=_MIST_TRAIN_BATCH,
            per_device_eval_batch_size=_MIST_EVAL_BATCH,
            gradient_accumulation_steps=_MIST_GRAD_ACCUM,
        )
    )

    def validate(self) -> None:
        self.data.validate()
        self.arch.validate()
        self.trainer.validate()
        if self.data.block_size > self.arch.n_positions:
            raise ValueError(
                f"data.block_size ({self.data.block_size}) cannot exceed "
                f"arch.n_positions ({self.arch.n_positions})"
            )


# Back-compat alias.
Gpt2PretrainConfig = PretrainConfig


def with_trainer(
    config: PretrainConfig,
    **trainer_overrides,
) -> PretrainConfig:
    """Return ``config`` with selected ``CausalLmTrainerConfig`` fields replaced."""
    return replace(config, trainer=replace(config.trainer, **trainer_overrides))


def with_data(
    config: PretrainConfig,
    **data_overrides,
) -> PretrainConfig:
    """Return ``config`` with selected ``PretrainDataConfig`` fields replaced."""
    return replace(config, data=replace(config.data, **data_overrides))


def with_arch(
    config: PretrainConfig,
    **arch_overrides,
) -> PretrainConfig:
    """Return ``config`` with selected ``CausalLmArchConfig`` fields replaced."""
    return replace(config, arch=replace(config.arch, **arch_overrides))


def resolve_use_wandb(
    config: PretrainConfig,
    use_wandb: bool | None,
) -> bool:
    """Resolve whether to start a W&B run (explicit flag wins over ``report_to``)."""
    if use_wandb is not None:
        return use_wandb
    return not reporting_disabled(config.trainer.report_to)


def prepare_lm_datasets(
    data: PretrainDataConfig,
    tokenizer,
    *,
    verbose: bool = True,
):
    """Load/tokenize a Hub corpus into train and eval LM blocks.

    Train is an iterable stream when ``mode='stream'``, otherwise a
    materialized map-style dataset. Eval is always map-style.
    """
    train_raw, eval_raw = load_train_eval(data, verbose=verbose)
    if verbose:
        blank(logger=log)
        if isinstance(train_raw, IterableDataset):
            step("Preprocessing datasets (train streams lazily)...", logger=log)
        else:
            step("Preprocessing datasets...", logger=log)
    train_text_column = data.source.text_column
    eval_text_column = (data.eval_source or data.source).text_column
    train_dataset = tokenize_and_chunk(
        train_raw,
        tokenizer,
        block_size=data.block_size,
        text_column=train_text_column,
        num_proc=data.tokenizer_num_proc,
    )
    eval_dataset = tokenize_and_chunk(
        eval_raw,
        tokenizer,
        block_size=data.block_size,
        text_column=eval_text_column,
        num_proc=data.tokenizer_num_proc,
    )
    if verbose:
        if not isinstance(train_dataset, IterableDataset):
            detail(f"train blocks: {len(train_dataset):,}", logger=log)
        detail(f"eval blocks:  {len(eval_dataset):,}", logger=log)
    if len(eval_dataset) == 0:
        raise ValueError(
            "No eval blocks produced; increase max_eval_samples or lower block_size."
        )
    return train_dataset, eval_dataset


def log_pretrain_banner(
    *,
    title: str,
    run_label: str,
    config: PretrainConfig,
):
    """Log run header / accelerator info; return ``(AcceleratorInfo, tokens_per_step)``."""
    header(logger=log)
    banner(title, logger=log)
    step(f"run: {run_label}", logger=log)

    accel = collect_accelerator_info(
        prefer_bf16=config.trainer.prefer_bf16,
        prefer_fp16=config.trainer.prefer_fp16,
    )
    world_size = accel.world_size or distributed_world_size()
    tps = tokens_per_optimizer_step(
        per_device_train_batch_size=config.trainer.per_device_train_batch_size,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        block_size=config.data.block_size,
        world_size=world_size,
    )
    gpu = accel.gpu_name or accel.device
    step(
        f"Device: {gpu} "
        f"(precision: {accel.precision}, world_size={world_size})",
        logger=log,
    )
    if accel.gpu_memory_total_gb is not None:
        detail(f"GPU memory: {accel.gpu_memory_total_gb:g} GB", logger=log)
    if accel.cuda_version:
        detail(
            f"CUDA {accel.cuda_version}"
            + (f", cuDNN {accel.cudnn_version}" if accel.cudnn_version else ""),
            logger=log,
        )
    detail(f"torch {accel.torch_version}", logger=log)
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
    if config.trainer.gradient_checkpointing:
        step("Gradient checkpointing enabled", logger=log)
    return accel, tps


def pretrain(
    config: PretrainConfig,
    *,
    run_label: str = "regular",
    title: str = "Causal LM from scratch",
    env_files: tuple[Path, ...] | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    use_wandb: bool | None = None,
    resume_from_checkpoint: str | Path | bool | None = None,
):
    """End-to-end: env → data → model → trainer → optional W&B train/save.

    Returns ``(trainer, run_summary)`` when training finishes (summary may be
    non-``None`` even on interrupt; failures still write a summary then re-raise).

    When W&B is enabled, ``wandb_entity`` and ``wandb_project`` must be passed
    explicitly (no code defaults). ``wandb_name`` overrides ``trainer.run_name``.

    Set ``use_wandb=False`` (or ``trainer.report_to="none"``) to skip Weights &
    Biases entirely. ``resume_from_checkpoint`` follows HF Trainer semantics
    (path, ``True`` for latest checkpoint, or ``None``).

    Always writes ``run_config.json`` before training and ``run_summary.json``
    when training stops (completed, interrupted, or failed) under
    ``trainer.output_dir``.
    """
    env_files = env_files if env_files is not None else (Path.cwd() / ".env",)

    if resume_from_checkpoint is not None:
        config = with_trainer(config, resume_from_checkpoint=resume_from_checkpoint)

    want_wandb = resolve_use_wandb(config, use_wandb)
    if not want_wandb and not reporting_disabled(config.trainer.report_to):
        config = with_trainer(config, report_to="none")

    if want_wandb:
        require_wandb_identity(entity=wandb_entity, project=wandb_project)

    config.validate()
    accel, tokens_per_step = log_pretrain_banner(
        title=title, run_label=run_label, config=config
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
    if want_wandb:
        set_wandb_dir(config.trainer.output_dir)
    model, tokenizer = build_model_and_tokenizer(config.arch)
    model_size = count_model_params(model, vocab_size=tokenizer.vocab_size)
    train_dataset, eval_dataset = prepare_lm_datasets(config.data, tokenizer)

    config_payload = build_run_config_payload(
        run_label=run_label,
        run_name=config.trainer.run_name,
        title=title,
        configs={"data": config.data, "arch": config.arch, "trainer": config.trainer},
        accelerator=accel,
        model_size=model_size,
        tokens_per_step=tokens_per_step,
    )
    blank(logger=log)
    step("Writing run config...", logger=log)
    save_run_config(config.trainer.output_dir, config_payload)

    trainer = build_causal_lm_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config.trainer,
    )
    software = collect_software_versions()
    run_config = build_run_config(
        run_label=run_label,
        env=env,
        configs={"data": config.data, "arch": config.arch, "trainer": config.trainer},
        prefer_bf16=config.trainer.prefer_bf16,
        prefer_fp16=config.trainer.prefer_fp16,
        accelerator=accel,
        tokens_per_step=tokens_per_step,
        model=model_size.as_dict(),
        software=software,
    )

    blank(logger=log)
    with wandb_run(
        entity=env.wandb_entity or "",
        project=env.wandb_project or "",
        name=config.trainer.run_name,
        config=run_config,
        dir=config.trainer.output_dir,
        enabled=want_wandb,
    ):
        return train_and_save(
            trainer=trainer,
            tokenizer=tokenizer,
            output_dir=config.trainer.output_dir,
            resume_from_checkpoint=config.trainer.resume_from_checkpoint,
            run_name=config.trainer.run_name,
            run_label=run_label,
            max_steps=config.trainer.max_steps,
            tokens_per_step=tokens_per_step,
            model_size=model_size,
            accelerator=accel,
        )


def pretrain_gpt2(config: PretrainConfig, **kwargs):
    """Back-compat alias for ``pretrain``."""
    return pretrain(config, **kwargs)
