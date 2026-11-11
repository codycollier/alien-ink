"""High-level causal-LM pretraining on Hugging Face datasets.

Composes data + architecture + trainer configs. Prefer defining a
:class:`~alien_ink.hf.manifest.Manifest` in zdeck programs; this module is the
runtime orchestrator that manifests materialize into.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from alien_ink.com.device import distributed_world_size
from alien_ink.com.log import get_logger
from alien_ink.hf.curriculum import Curriculum, prepare_curriculum_datasets
from alien_ink.hf.ds import PretrainDataConfig
from alien_ink.hf.ds import prepare_lm_datasets as prepare_lm_datasets
from alien_ink.hf.metrics import count_model_params
from alien_ink.hf.model import CausalLmArchConfig, build_model_and_tokenizer, gpt2_arch
from alien_ink.hf.runtime import (
    initialize_run,
    log_training_banner,
    resolve_trainer_config,
    run_training,
)
from alien_ink.hf.trainer import CausalLmTrainerConfig

log = get_logger("hf.pretrain")


@dataclass(frozen=True)
class PretrainConfig:
    """Composable config for from-scratch causal-LM pretraining.

    ``data`` is either a single Hub corpus (:class:`PretrainDataConfig`) or a
    :class:`~alien_ink.hf.curriculum.Curriculum` of sequenced corpora; both
    expose ``block_size`` and ``seed`` so the rest of the pipeline is agnostic.
    """

    data: PretrainDataConfig | Curriculum
    arch: CausalLmArchConfig = field(default_factory=gpt2_arch)
    trainer: CausalLmTrainerConfig = field(
        default_factory=lambda: CausalLmTrainerConfig(
            output_dir=Path.cwd() / "output" / "train" / "pretrain",
            run_name="pretrain",
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
    """Return ``config`` with selected ``PretrainDataConfig`` fields replaced.

    Applies to plain :class:`PretrainDataConfig` only; rebuild a
    :class:`~alien_ink.hf.curriculum.Curriculum` explicitly instead.
    """
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
    from alien_ink.hf.runtime import resolve_use_wandb as resolve

    return resolve(config.trainer, use_wandb)


def log_pretrain_banner(
    *,
    title: str,
    run_label: str,
    zdeck_name: str | None,
    config: PretrainConfig,
):
    """Log stars header, device introspection, then run summary.

    Returns ``(AcceleratorInfo, tokens_per_step)``.
    """
    return log_training_banner(
        title=title,
        run_label=run_label,
        zdeck_name=zdeck_name,
        trainer=config.trainer,
        block_size=config.data.block_size,
        model_label=f"model family: {config.arch.family}",
        action="Training",
    )


def pretrain(
    config: PretrainConfig,
    *,
    run_label: str = "regular",
    zdeck_name: str | None = None,
    title: str = "Causal LM from scratch",
    env_files: tuple[Path, ...] | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    use_wandb: bool | None = None,
    resume_from_checkpoint: str | Path | bool | None = None,
    extra_configs: Mapping[str, Any] | None = None,
):
    """End-to-end: env → data → model → trainer → optional W&B train/save.

    Returns ``(trainer, run_summary)`` when training finishes (summary may be
    non-``None`` even on interrupt; failures still write a summary then re-raise).

    When W&B is enabled, ``wandb_entity`` and ``wandb_project`` must be passed
    explicitly (no code defaults). ``wandb_name`` overrides ``trainer.run_name``.

    Set ``use_wandb=False`` (or ``trainer.report_to="none"``) to skip Weights &
    Biases entirely. ``resume_from_checkpoint`` follows HF Trainer semantics
    (path, ``True`` for latest checkpoint, or ``None``).

    ``extra_configs`` are merged into the logged run config (e.g. manifest
    ``hardware`` / ``schedule`` / ``wandb`` segments).

    Always writes ``run_config.json`` before training and ``run_summary.json``
    when training stops (completed, interrupted, or failed) under
    ``trainer.output_dir``.
    """
    trainer_config, want_wandb = resolve_trainer_config(
        config.trainer,
        use_wandb=use_wandb,
        wandb_name=wandb_name,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    config = replace(config, trainer=trainer_config)
    config.validate()
    accel, tokens_per_step = log_pretrain_banner(
        title=title, run_label=run_label, zdeck_name=zdeck_name, config=config
    )

    env = initialize_run(
        config.trainer,
        env_files=env_files,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        enabled=want_wandb,
    )
    model, tokenizer = build_model_and_tokenizer(config.arch)
    model_size = count_model_params(model, vocab_size=tokenizer.vocab_size)
    if isinstance(config.data, Curriculum):
        world_size = max(1, accel.world_size or distributed_world_size())
        samples_per_step = (
            config.trainer.per_device_train_batch_size
            * config.trainer.gradient_accumulation_steps
            * world_size
        )
        train_dataset, eval_dataset = prepare_curriculum_datasets(
            config.data,
            tokenizer,
            samples_per_step=samples_per_step,
        )
    else:
        train_dataset, eval_dataset = prepare_lm_datasets(config.data, tokenizer)

    log_configs: dict[str, Any] = {
        "data": config.data,
        "arch": config.arch,
        "trainer": config.trainer,
    }
    if extra_configs:
        log_configs.update(dict(extra_configs))

    return run_training(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config.trainer,
        log_configs=log_configs,
        run_label=run_label,
        title=title,
        env=env,
        accel=accel,
        tokens_per_step=tokens_per_step,
        model_size=model_size,
        want_wandb=want_wandb,
    )
