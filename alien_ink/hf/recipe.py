"""Composable training recipes: dataset + model + hardware + W&B + schedule.

A :class:`Recipe` is the explicit, structured source of truth for a training
program. Samples should be a recipe literal plus a thin ``main`` that calls
:meth:`Recipe.train`.

``HardwareConfig`` is the GPU-tunable subset (batch / accum / precision /
workers). Swap or ``.with_hardware(...)`` when moving machines; leave dataset,
model, W&B, and schedule alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from alien_ink.hf.ds import PretrainDataConfig
from alien_ink.hf.model import CausalLmArchConfig, gpt2_arch
from alien_ink.hf.pretrain import PretrainConfig, pretrain
from alien_ink.hf.trainer import CausalLmTrainerConfig
from alien_ink.com.wb import require_wandb_identity

__all__ = [
    "HardwareConfig",
    "Recipe",
    "ScheduleConfig",
    "WandbConfig",
    "mist_rtx_3070",
    "scaled_trainer_steps",
]

# Reference cadence for a full streamed run; shorter max_steps scale these down
# so step-capped runs keep roughly the same number of log / eval / checkpoint points.
_REF_MAX_STEPS = 50_000
_REF_LOGGING_STEPS = 50
_REF_EVAL_STEPS = 1_000
_REF_SAVE_STEPS = 1_000
_EPOCH_LOGGING_STEPS = 10


def scaled_trainer_steps(max_steps: int) -> dict[str, int]:
    """Scale log/eval/save cadence with ``max_steps``.

    At the reference ``50_000`` steps this returns the trainer defaults
    (``50`` / ``1_000`` / ``1_000``). Shorter runs keep roughly the same
    number of curve points (e.g. ``5_000`` → ``5`` / ``100`` / ``100``).
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    return {
        "logging_steps": max(1, (_REF_LOGGING_STEPS * max_steps) // _REF_MAX_STEPS),
        "eval_steps": max(1, (_REF_EVAL_STEPS * max_steps) // _REF_MAX_STEPS),
        "save_steps": max(1, (_REF_SAVE_STEPS * max_steps) // _REF_MAX_STEPS),
    }


@dataclass(frozen=True)
class HardwareConfig:
    """GPU-tunable trainer knobs — the fields that change with VRAM / host.

    Tune this (or swap a named profile like :func:`mist_rtx_3070`) when moving
    hardware. Dataset, model architecture, W&B identity, and schedule stay put.
    """

    label: str = "mist-rtx-3070"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    dataloader_num_workers: int = 2
    prefer_bf16: bool = True
    prefer_fp16: bool = True
    gradient_checkpointing: bool = True

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def validate(self) -> None:
        if not self.label.strip():
            raise ValueError("hardware.label must be a non-empty string")
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
        if self.dataloader_num_workers < 0:
            raise ValueError(
                f"dataloader_num_workers must be >= 0, got {self.dataloader_num_workers}"
            )


def mist_rtx_3070() -> HardwareConfig:
    """Mist / local RTX 3070 (~8 GB): microbatch 2 × accum 16 = effective 32."""
    return HardwareConfig(
        label="mist-rtx-3070",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        dataloader_num_workers=2,
        prefer_bf16=True,
        prefer_fp16=True,
        gradient_checkpointing=True,
    )


@dataclass(frozen=True)
class WandbConfig:
    """Explicit W&B identity — never sourced from env / package defaults."""

    entity: str | None = None
    project: str | None = None
    name: str | None = None
    enabled: bool = True

    def resolved_name(self, run_name: str) -> str:
        """W&B run name: explicit ``name`` when set, otherwise the recipe run name."""
        return self.name if self.name else run_name

    def require_identity(self) -> tuple[str, str]:
        return require_wandb_identity(entity=self.entity, project=self.project)

    def validate(self) -> None:
        if self.enabled:
            self.require_identity()


@dataclass(frozen=True)
class ScheduleConfig:
    """Length, LR, and logging cadence (host-independent training schedule)."""

    max_steps: int = 50_000
    num_train_epochs: float = 3.0
    learning_rate: float = 6e-4
    warmup_steps: int = 2_000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    seed: int = 101
    # None => derive from max_steps via scaled_trainer_steps (epoch mode uses a fixed log tick).
    logging_steps: int | None = None
    eval_steps: int | None = None
    save_steps: int | None = None
    save_total_limit: int = 2
    early_stopping_patience: int = 0

    def uses_epochs(self) -> bool:
        return self.max_steps < 0

    def cadence(self) -> dict[str, int]:
        """Resolved logging / eval / save step intervals."""
        if self.uses_epochs():
            base = {
                "logging_steps": _EPOCH_LOGGING_STEPS,
                "eval_steps": _EPOCH_LOGGING_STEPS,
                "save_steps": _EPOCH_LOGGING_STEPS,
            }
        else:
            base = scaled_trainer_steps(self.max_steps)
        if self.logging_steps is not None:
            base["logging_steps"] = self.logging_steps
        if self.eval_steps is not None:
            base["eval_steps"] = self.eval_steps
        if self.save_steps is not None:
            base["save_steps"] = self.save_steps
        return base

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
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")


@dataclass(frozen=True)
class Recipe:
    """Top-level training recipe: data ⊕ model ⊕ hardware ⊕ wandb ⊕ schedule.

    Materializes a :class:`~alien_ink.hf.pretrain.PretrainConfig` via
    :meth:`to_pretrain_config`. Compose ablations with :meth:`variant`,
    :meth:`with_hardware`, :meth:`with_data`, :meth:`with_model`,
    :meth:`with_wandb`, and :meth:`with_schedule` instead of cloning modules.
    """

    run_name: str
    title: str
    data: PretrainDataConfig
    model: CausalLmArchConfig = field(default_factory=gpt2_arch)
    hardware: HardwareConfig = field(default_factory=mist_rtx_3070)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    # Escape hatches for rarely swept CausalLmTrainerConfig fields (adam betas, …).
    trainer_overrides: Mapping[str, Any] = field(default_factory=dict)

    def variant(self, **changes: Any) -> Recipe:
        """Return a copy with selected top-level recipe fields replaced."""
        return replace(self, **changes)

    def with_hardware(self, **kw: Any) -> Recipe:
        return replace(self, hardware=replace(self.hardware, **kw))

    def with_data(self, **kw: Any) -> Recipe:
        return replace(self, data=replace(self.data, **kw))

    def with_model(self, **kw: Any) -> Recipe:
        return replace(self, model=replace(self.model, **kw))

    def with_wandb(self, **kw: Any) -> Recipe:
        return replace(self, wandb=replace(self.wandb, **kw))

    def with_schedule(self, **kw: Any) -> Recipe:
        return replace(self, schedule=replace(self.schedule, **kw))

    def with_trainer_knobs(self, **kw: Any) -> Recipe:
        """Merge extra trainer fields applied last when materializing."""
        return replace(
            self,
            trainer_overrides={**dict(self.trainer_overrides), **kw},
        )

    def output_dir(self) -> Path:
        return Path.cwd() / "output" / self.run_name

    def validate(self) -> None:
        if not self.run_name.strip():
            raise ValueError("run_name must be a non-empty string")
        if not self.title.strip():
            raise ValueError("title must be a non-empty string")
        self.data.validate()
        self.model.validate()
        self.hardware.validate()
        self.wandb.validate()
        self.schedule.validate()
        if self.data.block_size > self.model.n_positions:
            raise ValueError(
                f"data.block_size ({self.data.block_size}) cannot exceed "
                f"model.n_positions ({self.model.n_positions})"
            )

    def to_pretrain_config(self) -> PretrainConfig:
        """Materialize the runtime :class:`PretrainConfig` this recipe describes."""
        self.validate()
        cadence = self.schedule.cadence()
        trainer = CausalLmTrainerConfig(
            output_dir=self.output_dir(),
            run_name=self.wandb.resolved_name(self.run_name),
            max_steps=self.schedule.max_steps,
            num_train_epochs=self.schedule.num_train_epochs,
            learning_rate=self.schedule.learning_rate,
            warmup_steps=self.schedule.warmup_steps,
            weight_decay=self.schedule.weight_decay,
            max_grad_norm=self.schedule.max_grad_norm,
            lr_scheduler_type=self.schedule.lr_scheduler_type,
            seed=self.schedule.seed,
            logging_steps=cadence["logging_steps"],
            eval_steps=cadence["eval_steps"],
            save_steps=cadence["save_steps"],
            save_total_limit=self.schedule.save_total_limit,
            early_stopping_patience=self.schedule.early_stopping_patience,
            per_device_train_batch_size=self.hardware.per_device_train_batch_size,
            per_device_eval_batch_size=self.hardware.per_device_eval_batch_size,
            gradient_accumulation_steps=self.hardware.gradient_accumulation_steps,
            dataloader_num_workers=self.hardware.dataloader_num_workers,
            prefer_bf16=self.hardware.prefer_bf16,
            prefer_fp16=self.hardware.prefer_fp16,
            gradient_checkpointing=self.hardware.gradient_checkpointing,
            report_to="wandb" if self.wandb.enabled else "none",
            **dict(self.trainer_overrides),
        )
        cfg = PretrainConfig(data=self.data, arch=self.model, trainer=trainer)
        cfg.validate()
        return cfg

    def train(self, **pretrain_kwargs: Any):
        """Run pretraining from this recipe.

        Extra kwargs are forwarded to :func:`~alien_ink.hf.pretrain.pretrain`
        (e.g. ``resume_from_checkpoint``, ``run_label``, ``env_files``).
        """
        config = self.to_pretrain_config()
        run_label = pretrain_kwargs.pop("run_label", "sample")
        return pretrain(
            config,
            title=self.title,
            run_label=run_label,
            wandb_entity=self.wandb.entity,
            wandb_project=self.wandb.project,
            wandb_name=self.wandb.resolved_name(self.run_name),
            use_wandb=self.wandb.enabled,
            extra_configs={
                "hardware": self.hardware,
                "schedule": self.schedule,
                "wandb": self.wandb,
            },
            **pretrain_kwargs,
        )
