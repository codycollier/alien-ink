"""Composable training manifests: dataset + model + hardware + W&B + schedule.

A :class:`Manifest` is the explicit, structured source of truth for a training
program. Zdeck programs should be a manifest literal plus a thin ``main`` that
calls :meth:`Manifest.train`.

``HardwareConfig`` is the GPU-tunable subset (batch / accum / precision /
workers). Swap or ``.with_hardware(...)`` when moving machines; leave dataset,
model, W&B, and schedule alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from alien_ink.com.log import get_logger
from alien_ink.com.wb import require_wandb_identity
from alien_ink.hf.curriculum import Curriculum
from alien_ink.hf.ds import PretrainDataConfig
from alien_ink.hf.model import CausalLmArchConfig, PretrainedLmConfig, gpt2_arch
from alien_ink.hf.pretrain import PretrainConfig, pretrain
from alien_ink.hf.sft import SftConfig, finetune
from alien_ink.hf.trainer import CausalLmTrainerConfig

log = get_logger("hf.manifest")

ManifestStage = Literal["pre", "sft"]
_VALID_STAGES: frozenset[str] = frozenset({"pre", "sft"})

__all__ = [
    "HardwareConfig",
    "Manifest",
    "ScheduleConfig",
    "WandbConfig",
    "mist_rtx_3070",
    "mist_rtx_3070_gemma",
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

    # Defaults match :func:`mist_rtx_3070` (GPT-2 / NeoX class on ~8 GB).
    label: str = "mist-rtx-3070"
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    dataloader_num_workers: int = 8
    # None => HF default (2 when workers > 0). Raise on hosts with spare RAM/CPU.
    dataloader_prefetch_factor: int | None = 4
    dataloader_persistent_workers: bool = True
    prefer_bf16: bool = True
    prefer_fp16: bool = True
    gradient_checkpointing: bool = False
    # Ampere+ TF32 tensor cores; None leaves the HF / torch default alone.
    tf32: bool | None = True
    torch_compile: bool = True
    # ``adamw_torch_fused`` is faster on CUDA when the PyTorch build supports it.
    optim: str = "adamw_torch_fused"

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


def mist_rtx_3070() -> HardwareConfig:
    """Mist / RTX 3070 (~8 GB) for GPT-2 / NeoX-class models.

    Microbatch 4 × accum 8 = effective 32. Checkpointing off (VRAM headroom);
    ``torch.compile`` + TF32 + fused AdamW; host dataloader tuned for 16 cores /
    ~94 GB RAM. If this OOMs, set ``gradient_checkpointing=True`` or drop
    microbatch to 2.
    """
    return HardwareConfig()


def mist_rtx_3070_gemma() -> HardwareConfig:
    """Mist / RTX 3070 (~8 GB) for Mist-sized Gemma (large vocab / logits).

    Microbatch 1 × accum 32 = effective 32. Checkpointing stays on — batch 2
    OOMs from ~256k-vocab logits. Same compile / TF32 / fused Adam / host
    dataloader knobs as :func:`mist_rtx_3070`.
    """
    return HardwareConfig(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
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
        """W&B run name: explicit ``name`` when set, otherwise the manifest run name."""
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
    warmup_steps: int | None = 2_000
    warmup_ratio: float | None = None
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
        """Resolved logging / eval / save step intervals.

        Epoch mode starts from a fixed placeholder; ``build_causal_lm_trainer``
        replaces it with dataset-length cadence via ``apply_epoch_cadence``.
        """
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
        if self.num_train_epochs <= 0:
            raise ValueError(
                f"num_train_epochs must be > 0, got {self.num_train_epochs}"
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
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.save_total_limit < 1:
            raise ValueError(
                f"save_total_limit must be >= 1, got {self.save_total_limit}"
            )
        if self.early_stopping_patience < 0:
            raise ValueError(
                "early_stopping_patience must be >= 0, "
                f"got {self.early_stopping_patience}"
            )
        cadence = self.cadence()
        for name, value in cadence.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if cadence["save_steps"] % cadence["eval_steps"] != 0:
            raise ValueError("save_steps must be a multiple of eval_steps")


@dataclass(frozen=True)
class Manifest:
    """Top-level training manifest: data ⊕ model ⊕ hardware ⊕ wandb ⊕ schedule.

    ``stage`` distinguishes from-scratch pretraining (``pre``) from supervised
    fine-tuning (``sft``). Zdeck filenames mirror ``run_name`` (underscores vs
    hyphens), e.g. ``pre_gemma_c4_5k_mist.py`` / ``pre-gemma-c4-5k-mist``.

    ``model`` pairs with the stage: ``pre`` takes a
    :class:`~alien_ink.hf.model.CausalLmArchConfig` (random init), ``sft``
    takes a :class:`~alien_ink.hf.model.PretrainedLmConfig` (Hub id or local
    checkpoint loaded via ``AutoModelForCausalLM``).

    ``data`` is a single corpus or a
    :class:`~alien_ink.hf.curriculum.Curriculum` of sequenced corpora; with a
    curriculum, ``schedule.max_steps`` must equal ``curriculum.total_steps()``.
    Curricula are pretraining-only.

    Materializes a :class:`~alien_ink.hf.pretrain.PretrainConfig` via
    :meth:`to_pretrain_config` or an :class:`~alien_ink.hf.sft.SftConfig` via
    :meth:`to_sft_config`. Compose ablations with :meth:`variant`,
    :meth:`with_hardware`, :meth:`with_data`, :meth:`with_model`,
    :meth:`with_wandb`, and :meth:`with_schedule` instead of cloning modules.
    """

    run_name: str
    title: str
    data: PretrainDataConfig | Curriculum
    stage: ManifestStage = "pre"
    model: CausalLmArchConfig | PretrainedLmConfig = field(default_factory=gpt2_arch)
    hardware: HardwareConfig = field(default_factory=mist_rtx_3070)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    # Escape hatches for rarely swept CausalLmTrainerConfig fields (adam betas, …).
    trainer_overrides: Mapping[str, Any] = field(default_factory=dict)

    def variant(self, **changes: Any) -> Manifest:
        """Return a copy with selected top-level manifest fields replaced."""
        return replace(self, **changes)

    def with_hardware(self, **kw: Any) -> Manifest:
        return replace(self, hardware=replace(self.hardware, **kw))

    def with_data(self, **kw: Any) -> Manifest:
        """Replace fields on a plain ``PretrainDataConfig`` ``data``.

        Curricula are compositions; rebuild the ``Curriculum`` explicitly
        instead of patching it field-wise.
        """
        return replace(self, data=replace(self.data, **kw))

    def with_model(self, **kw: Any) -> Manifest:
        return replace(self, model=replace(self.model, **kw))

    def with_wandb(self, **kw: Any) -> Manifest:
        return replace(self, wandb=replace(self.wandb, **kw))

    def with_schedule(self, **kw: Any) -> Manifest:
        return replace(self, schedule=replace(self.schedule, **kw))

    def with_trainer_knobs(self, **kw: Any) -> Manifest:
        """Merge extra trainer fields applied last when materializing."""
        return replace(
            self,
            trainer_overrides={**dict(self.trainer_overrides), **kw},
        )

    def output_dir(self) -> Path:
        return Path.cwd() / "output" / "train" / self.run_name

    def gen_config(self, **overrides):
        """Family-aware generation config derived from ``self.model``.

        Pretrained hub models keep the plain-text defaults; from-scratch
        architectures dispatch on ``model.family``.
        """
        from alien_ink.hf.gen import GenConfig, gen_config_for_family

        if isinstance(self.model, PretrainedLmConfig):
            return GenConfig(**overrides)
        return gen_config_for_family(self.model.family, **overrides)

    def validate(self) -> None:
        if not self.run_name.strip():
            raise ValueError("run_name must be a non-empty string")
        if not self.title.strip():
            raise ValueError("title must be a non-empty string")
        if self.stage not in _VALID_STAGES:
            raise ValueError(
                f"stage must be one of {sorted(_VALID_STAGES)}, got {self.stage!r}"
            )
        if self.stage == "pre" and not isinstance(self.model, CausalLmArchConfig):
            raise ValueError(
                "stage='pre' requires a CausalLmArchConfig model, got "
                f"{type(self.model).__name__}"
            )
        if self.stage == "sft":
            if not isinstance(self.model, PretrainedLmConfig):
                raise ValueError(
                    "stage='sft' requires a PretrainedLmConfig model, got "
                    f"{type(self.model).__name__}"
                )
            if isinstance(self.data, Curriculum):
                raise ValueError(
                    "stage='sft' takes a single corpus, not a Curriculum"
                )
        self.data.validate()
        self.model.validate()
        self.hardware.validate()
        self.wandb.validate()
        self.schedule.validate()
        if (
            isinstance(self.model, CausalLmArchConfig)
            and self.data.block_size > self.model.n_positions
        ):
            raise ValueError(
                f"data.block_size ({self.data.block_size}) cannot exceed "
                f"model.n_positions ({self.model.n_positions})"
            )
        if isinstance(self.data, Curriculum):
            self._validate_curriculum_schedule()

    def _validate_curriculum_schedule(self) -> None:
        """Check the schedule against curriculum phase budgets.

        The schedule stays the explicit source of truth: zdeck programs set
        ``max_steps=CURRICULUM.total_steps()``. Phase boundaries that miss a
        ``save_steps`` tick only warn — a checkpoint exactly at the boundary
        lets followup phases be re-run from the same base.
        """
        total = self.data.total_steps()
        if self.schedule.uses_epochs():
            raise ValueError(
                "curriculum data requires step mode; set "
                f"schedule.max_steps={total} (curriculum.total_steps()), "
                "not epoch mode (max_steps=-1)"
            )
        if self.schedule.max_steps != total:
            raise ValueError(
                f"schedule.max_steps ({self.schedule.max_steps}) must equal "
                f"curriculum.total_steps() ({total})"
            )
        save_steps = self.schedule.cadence()["save_steps"]
        # Final boundary is training end; the model is saved there regardless.
        unaligned = [
            boundary
            for boundary in self.data.boundaries()[:-1]
            if boundary % save_steps != 0
        ]
        if unaligned:
            log.warning(
                "curriculum phase boundaries %s do not land on save_steps "
                "(%d) ticks; no checkpoint will exist exactly at those "
                "boundaries",
                unaligned,
                save_steps,
            )

    def _trainer_config(self) -> CausalLmTrainerConfig:
        """Materialize the trainer config shared by both stages."""
        cadence = self.schedule.cadence()
        return CausalLmTrainerConfig(
            output_dir=self.output_dir(),
            run_name=self.wandb.resolved_name(self.run_name),
            max_steps=self.schedule.max_steps,
            num_train_epochs=self.schedule.num_train_epochs,
            learning_rate=self.schedule.learning_rate,
            warmup_steps=self.schedule.warmup_steps,
            warmup_ratio=self.schedule.warmup_ratio,
            weight_decay=self.schedule.weight_decay,
            max_grad_norm=self.schedule.max_grad_norm,
            lr_scheduler_type=self.schedule.lr_scheduler_type,
            seed=self.schedule.seed,
            data_seed=self.data.seed,
            logging_steps=cadence["logging_steps"],
            eval_steps=cadence["eval_steps"],
            save_steps=cadence["save_steps"],
            save_total_limit=self.schedule.save_total_limit,
            early_stopping_patience=self.schedule.early_stopping_patience,
            per_device_train_batch_size=self.hardware.per_device_train_batch_size,
            per_device_eval_batch_size=self.hardware.per_device_eval_batch_size,
            gradient_accumulation_steps=self.hardware.gradient_accumulation_steps,
            dataloader_num_workers=self.hardware.dataloader_num_workers,
            dataloader_prefetch_factor=self.hardware.dataloader_prefetch_factor,
            dataloader_persistent_workers=self.hardware.dataloader_persistent_workers,
            prefer_bf16=self.hardware.prefer_bf16,
            prefer_fp16=self.hardware.prefer_fp16,
            gradient_checkpointing=self.hardware.gradient_checkpointing,
            tf32=self.hardware.tf32,
            torch_compile=self.hardware.torch_compile,
            optim=self.hardware.optim,
            report_to="wandb" if self.wandb.enabled else "none",
            **dict(self.trainer_overrides),
        )

    def to_pretrain_config(self) -> PretrainConfig:
        """Materialize the runtime :class:`PretrainConfig` this manifest describes."""
        if self.stage != "pre":
            raise ValueError(
                f"to_pretrain_config requires stage='pre', got {self.stage!r}"
            )
        self.validate()
        cfg = PretrainConfig(
            data=self.data, arch=self.model, trainer=self._trainer_config()
        )
        cfg.validate()
        return cfg

    def to_sft_config(self) -> SftConfig:
        """Materialize the runtime :class:`SftConfig` this manifest describes."""
        if self.stage != "sft":
            raise ValueError(
                f"to_sft_config requires stage='sft', got {self.stage!r}"
            )
        self.validate()
        cfg = SftConfig(
            data=self.data, model=self.model, trainer=self._trainer_config()
        )
        cfg.validate()
        return cfg

    def train(self, **stage_kwargs: Any):
        """Run training from this manifest.

        ``stage="pre"`` materializes a pretrain config and calls
        :func:`~alien_ink.hf.pretrain.pretrain`. ``stage="sft"`` materializes
        an SFT config and calls :func:`~alien_ink.hf.sft.finetune`.

        Extra kwargs are forwarded to the stage entrypoint
        (e.g. ``resume_from_checkpoint``, ``run_label``, ``env_files``).
        """
        run_label = stage_kwargs.pop("run_label", "zdeck")
        shared_kwargs: dict[str, Any] = dict(
            title=self.title,
            run_label=run_label,
            wandb_entity=self.wandb.entity,
            wandb_project=self.wandb.project,
            wandb_name=self.wandb.resolved_name(self.run_name),
            use_wandb=self.wandb.enabled,
            extra_configs={
                "stage": {"name": self.stage},
                "hardware": self.hardware,
                "schedule": self.schedule,
                "wandb": self.wandb,
            },
            **stage_kwargs,
        )
        if self.stage == "sft":
            return finetune(self.to_sft_config(), **shared_kwargs)
        return pretrain(self.to_pretrain_config(), **shared_kwargs)
