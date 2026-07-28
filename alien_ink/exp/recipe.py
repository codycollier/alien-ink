"""Shared GPT-2 pretrain experiment recipe (config + CLI + flight/spot checks).

Three layers — one job each
---------------------------
**Recipe** (:class:`Gpt2PretrainExperiment`) — *what* to train: corpus, arch,
schedule, optional pinned profile. Compose ablations here only::

    EXPERIMENT.with_arch(n_layer=6).variant(run_name="wt-sub-l6")
    EXPERIMENT.with_data(block_size=512).with_trainer(weight_decay=0.05)

**Profile** (:class:`~alien_ink.hf.hardware.AcceleratorProfile`) — *where*:
batch / accum / run-name suffix. Always via :func:`~alien_ink.hf.hardware.get_profile`::

    get_profile()            # detect
    get_profile("colab-g4")  # named
    get_profile(COLAB_G4)    # object

**Config** (:class:`~alien_ink.hf.pretrain.Gpt2PretrainConfig`) — fully resolved
snapshot from :meth:`Gpt2PretrainExperiment.config`. Ready for
:func:`~alien_ink.hf.pretrain.pretrain_gpt2`.

Runs stay thin (notebook-style)::

    from alien_ink.exp.gpt2_pretrain_wikitext_subset import EXPERIMENT

    EXPERIMENT.train(use_wandb=True, profile="colab-g4")
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from alien_ink.exp.cli import (
    add_train_override_args,
    add_wandb_args,
    train_override_kwargs,
    wandb_kwargs,
)
from alien_ink.hf.ds import PretrainDataConfig
from alien_ink.hf.gen import SpotCheckConfig, run_spot_check
from alien_ink.hf.hardware import (
    AcceleratorProfile,
    get_profile,
    trainer_overrides_for_profile,
    with_accelerator_suffix,
)
from alien_ink.hf.model import Gpt2ArchConfig
from alien_ink.hf.pretrain import Gpt2PretrainConfig, pretrain_gpt2
from alien_ink.hf.trainer import CausalLmTrainerConfig
from alien_ink.log import detail, get_logger

log = get_logger("exp.recipe")

# Reference cadence for a full streamed run; shorter max_steps scale these down
# so step-capped runs keep roughly the same number of log / eval / checkpoint points.
_REF_MAX_STEPS = 50_000
_REF_LOGGING_STEPS = 50
_REF_EVAL_STEPS = 1_000
_REF_SAVE_STEPS = 1_000
# Placeholder until build_causal_lm_trainer derives epoch cadence from dataset size
# (~5 evals/epoch including epoch end).
_EPOCH_LOGGING_STEPS = 10

# Flight-check: tiny end-to-end smoke (overrides hardware batch / length).
_FLIGHT_TRAINER = {
    "max_steps": 10,
    "warmup_steps": 0,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "logging_steps": 1,
    "eval_steps": 5,
    "save_steps": 10,
}
_FLIGHT_DATA = {
    "max_eval_samples": 10,
    "stream_shuffle_buffer": 50,
    "block_size": 128,
}
_FLIGHT_MAX_TRAIN_SAMPLES = 50

# Recipe-level fields that CLI / override bags may set via :meth:`override`.
# Everything else is treated as a trainer knob.
_RECIPE_OVERRIDE_FIELDS = frozenset(
    {
        "run_name",
        "title",
        "spot_check_title",
        "module_description",
        "max_steps",
        "num_train_epochs",
        "warmup_steps",
        "learning_rate",
        "profile",
    }
)


def scaled_trainer_steps(max_steps: int) -> dict[str, int]:
    """Scale log/eval/save cadence with ``max_steps``.

    At the reference ``50_000`` steps this returns the trainer defaults
    (``50`` / ``1_000`` / ``1_000``). Shorter runs keep ~the same number of
    curve points (e.g. ``2_000`` → ``2`` / ``40`` / ``40``).
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    return {
        "logging_steps": max(1, (_REF_LOGGING_STEPS * max_steps) // _REF_MAX_STEPS),
        "eval_steps": max(1, (_REF_EVAL_STEPS * max_steps) // _REF_MAX_STEPS),
        "save_steps": max(1, (_REF_SAVE_STEPS * max_steps) // _REF_MAX_STEPS),
    }


def trainer_length_kwargs(
    *,
    max_steps: int,
    num_train_epochs: float,
) -> dict[str, int | float]:
    """Trainer length + cadence fields for step-capped or epoch-based runs."""
    if max_steps < 0:
        return {
            "max_steps": -1,
            "num_train_epochs": num_train_epochs,
            "logging_steps": _EPOCH_LOGGING_STEPS,
        }
    return {
        "max_steps": max_steps,
        "num_train_epochs": num_train_epochs,
        **scaled_trainer_steps(max_steps),
    }


@dataclass(frozen=True)
class Gpt2PretrainExperiment:
    """Corpus recipe: labels + data factory + length/LR/arch knobs.

    Paths resolve from cwd at call time. Compose ablations with
    :meth:`variant`, :meth:`with_arch`, :meth:`with_data`, :meth:`with_trainer`,
    and :meth:`with_profile` — then call :meth:`train` (runtime kwargs only).

    Use ``max_steps >= 1`` for streamed / step-capped runs, or ``max_steps=-1``
    with ``num_train_epochs`` for finite materialized subsets.

    Pin hardware with ``profile=`` (name or :class:`AcceleratorProfile`); leave
    ``None`` to detect at config/train time via :func:`get_profile`.
    """

    run_name: str
    title: str
    spot_check_title: str
    data_factory: Callable[..., PretrainDataConfig]
    module_description: str
    max_steps: int = 50_000
    num_train_epochs: float = 3.0
    warmup_steps: int = 2_000
    learning_rate: float = 6e-4
    arch: Gpt2ArchConfig = field(default_factory=Gpt2ArchConfig)
    # Applied after data_factory() / base trainer assembly (ablation knobs).
    data_overrides: Mapping[str, Any] = field(default_factory=dict)
    trainer_overrides: Mapping[str, Any] = field(default_factory=dict)
    # Optional pinned profile; None → detect when building config / training.
    profile: AcceleratorProfile | str | None = None

    def variant(self, **changes: Any) -> Gpt2PretrainExperiment:
        """Return a copy with selected recipe fields replaced."""
        return replace(self, **changes)

    def with_arch(self, **arch_overrides: Any) -> Gpt2PretrainExperiment:
        """Return a copy with selected :class:`Gpt2ArchConfig` fields replaced."""
        return replace(self, arch=replace(self.arch, **arch_overrides))

    def with_data(self, **data_overrides: Any) -> Gpt2PretrainExperiment:
        """Return a copy merging extra :class:`PretrainDataConfig` overrides."""
        return replace(
            self,
            data_overrides={**dict(self.data_overrides), **data_overrides},
        )

    def with_trainer(self, **trainer_overrides: Any) -> Gpt2PretrainExperiment:
        """Return a copy merging extra :class:`CausalLmTrainerConfig` overrides."""
        return replace(
            self,
            trainer_overrides={**dict(self.trainer_overrides), **trainer_overrides},
        )

    def with_profile(
        self,
        profile: AcceleratorProfile | str | None,
    ) -> Gpt2PretrainExperiment:
        """Return a copy with a pinned (or cleared) hardware profile."""
        return replace(self, profile=profile)

    def override(self, **changes: Any) -> Gpt2PretrainExperiment:
        """Apply a bag of overrides — recipe fields vs trainer knobs, one path.

        Keys in the recipe dataclass (``max_steps``, ``learning_rate``,
        ``profile``, …) go through :meth:`variant`. Everything else merges via
        :meth:`with_trainer`. Prefer explicit ``variant`` / ``with_*`` in
        notebooks; this helper exists for CLI flag bags.
        """
        recipe_changes = {
            key: value
            for key, value in changes.items()
            if key in _RECIPE_OVERRIDE_FIELDS
        }
        trainer_changes = {
            key: value
            for key, value in changes.items()
            if key not in _RECIPE_OVERRIDE_FIELDS
        }
        unknown_recipe = set(recipe_changes) - {f.name for f in fields(self)}
        if unknown_recipe:
            raise TypeError(
                f"Unknown recipe override field(s): {sorted(unknown_recipe)}"
            )
        exp = self.variant(**recipe_changes) if recipe_changes else self
        return exp.with_trainer(**trainer_changes) if trainer_changes else exp

    def workdir(self) -> Path:
        return Path.cwd()

    def output_root(self) -> Path:
        return self.workdir() / "output"

    def env_file(self) -> Path:
        return self.workdir() / ".env"

    def resolve_profile(
        self,
        profile: AcceleratorProfile | str | None = None,
    ) -> AcceleratorProfile:
        """Resolve the effective profile (call arg → pinned → detect)."""
        if profile is not None:
            return get_profile(profile)
        return get_profile(self.profile)

    def resolved_run_name(
        self,
        *,
        profile: AcceleratorProfile | str | None = None,
        wandb_name: str | None = None,
        flight_check: bool = False,
    ) -> str:
        """Run / W&B name with ``-gpu`` / ``-tpu`` (or ``-cpu``) suffix."""
        resolved = self.resolve_profile(profile)
        base = wandb_name or self.run_name
        if flight_check and wandb_name is None:
            base = f"{self.run_name}-flight-check"
        elif flight_check and wandb_name is not None and "flight-check" not in wandb_name:
            base = f"{wandb_name}-flight-check"
        return with_accelerator_suffix(base, resolved.kind)

    def flight_check_run_name(
        self,
        profile: AcceleratorProfile | str | None = None,
    ) -> str:
        return self.resolved_run_name(profile=profile, flight_check=True)

    def checkpoint_output_dirs(self) -> list[Path]:
        """Candidate dirs for spot-check (device-suffixed + legacy unsuffixed)."""
        root = self.output_root()
        names: list[str] = []
        for kind in ("gpu", "tpu", "cpu"):
            names.append(with_accelerator_suffix(self.run_name, kind))
            names.append(
                with_accelerator_suffix(f"{self.run_name}-flight-check", kind)
            )
        names.extend([self.run_name, f"{self.run_name}-flight-check"])
        seen: set[str] = set()
        dirs: list[Path] = []
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            dirs.append(root / name)
        return dirs

    def _data_config(self) -> PretrainDataConfig:
        data = self.data_factory()
        if self.data_overrides:
            data = replace(data, **self.data_overrides)
        return data

    def config(
        self,
        profile: AcceleratorProfile | str | None = None,
        *,
        wandb_name: str | None = None,
        flight_check: bool = False,
    ) -> Gpt2PretrainConfig:
        """Build the fully resolved pretrain config for ``profile``.

        This is the only config-assembly entry point. Pass ``flight_check=True``
        for the tiny smoke recipe (short blocks / steps / batches).
        """
        resolved = self.resolve_profile(profile)
        run_name = self.resolved_run_name(
            profile=resolved,
            wandb_name=wandb_name,
            flight_check=flight_check,
        )
        detail(
            f"accelerator profile: {resolved.label} ({resolved.hardware}) "
            f"batch={resolved.per_device_train_batch_size} "
            f"accum={resolved.gradient_accumulation_steps} "
            f"eff={resolved.effective_batch_size} "
            f"mem×{resolved.memory_multiple:g} "
            f"compute×{resolved.vs_rtx_3070:g} "
            f"run_name={run_name}",
            logger=log,
        )
        trainer = CausalLmTrainerConfig(
            output_dir=self.output_root() / run_name,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            run_name=run_name,
            **trainer_overrides_for_profile(resolved),
            **trainer_length_kwargs(
                max_steps=self.max_steps,
                num_train_epochs=self.num_train_epochs,
            ),
        )
        if self.trainer_overrides:
            trainer = replace(trainer, **self.trainer_overrides)
        cfg = Gpt2PretrainConfig(
            data=self._data_config(),
            arch=self.arch,
            trainer=trainer,
        )
        if not flight_check:
            return cfg

        data_overrides = dict(_FLIGHT_DATA)
        if cfg.data.max_train_samples is not None:
            data_overrides["max_train_samples"] = _FLIGHT_MAX_TRAIN_SAMPLES
        return cfg.with_trainer(
            output_dir=self.output_root() / run_name,
            run_name=run_name,
            **_FLIGHT_TRAINER,
        ).with_data(**data_overrides)

    # Back-compat aliases — prefer :meth:`config`.
    def base_config(
        self,
        profile: AcceleratorProfile | str | None = None,
        *,
        wandb_name: str | None = None,
    ) -> Gpt2PretrainConfig:
        return self.config(profile, wandb_name=wandb_name)

    def flight_check_config(
        self,
        profile: AcceleratorProfile | str | None = None,
        *,
        wandb_name: str | None = None,
    ) -> Gpt2PretrainConfig:
        return self.config(profile, wandb_name=wandb_name, flight_check=True)

    def _launch(
        self,
        cfg: Gpt2PretrainConfig,
        *,
        run_label: str,
        profile: AcceleratorProfile,
        wandb_entity: str | None,
        wandb_project: str | None,
        wandb_name: str | None,
        use_wandb: bool | None,
        resume_from_checkpoint: str | Path | bool | None,
        tpu_launch: bool | None,
        tpu_num_processes: int | None,
    ):
        processes = (
            tpu_num_processes
            if tpu_num_processes is not None
            else profile.tpu_num_processes
        )
        return pretrain_gpt2(
            cfg,
            run_label=run_label,
            title=self.title,
            env_files=(self.env_file(),),
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
            use_wandb=use_wandb,
            resume_from_checkpoint=resume_from_checkpoint,
            tpu_launch=tpu_launch,
            tpu_num_processes=processes,
        )

    def train(
        self,
        *,
        profile: AcceleratorProfile | str | None = None,
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
        wandb_name: str | None = None,
        use_wandb: bool | None = None,
        resume_from_checkpoint: str | Path | bool | None = None,
        tpu_launch: bool | None = None,
        tpu_num_processes: int | None = None,
    ):
        """Full pretraining run.

        Runtime kwargs only (profile, W&B, resume, TPU launch). Ablate
        hyperparameters by composing the recipe first
        (``variant`` / ``with_arch`` / ``with_data`` / ``with_trainer``).

        Returns ``(trainer, run_summary)``. ``run_summary.json`` / ``run_config.json``
        are always written under the run ``output_dir``. On Colab TPU notebooks,
        auto-launches via ``notebook_launcher`` and returns ``(None, None)``.
        """
        resolved = self.resolve_profile(profile)
        cfg = self.config(resolved, wandb_name=wandb_name)
        return self._launch(
            cfg,
            run_label="regular",
            profile=resolved,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_name=cfg.trainer.run_name,
            use_wandb=use_wandb,
            resume_from_checkpoint=resume_from_checkpoint,
            tpu_launch=tpu_launch,
            tpu_num_processes=tpu_num_processes,
        )

    def train_flight_check(
        self,
        *,
        profile: AcceleratorProfile | str | None = None,
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
        wandb_name: str | None = None,
        use_wandb: bool | None = None,
        resume_from_checkpoint: str | Path | bool | None = None,
        tpu_launch: bool | None = None,
        tpu_num_processes: int | None = None,
    ):
        """Fast end-to-end smoke test (tiny steps / block size).

        Returns ``(trainer, run_summary)`` like :meth:`train`.
        """
        resolved = self.resolve_profile(profile)
        cfg = self.config(resolved, wandb_name=wandb_name, flight_check=True)
        return self._launch(
            cfg,
            run_label="flight_check",
            profile=resolved,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_name=cfg.trainer.run_name,
            use_wandb=use_wandb,
            resume_from_checkpoint=resume_from_checkpoint,
            tpu_launch=tpu_launch,
            tpu_num_processes=tpu_num_processes,
        )

    def spot_check(self) -> None:
        """Sample completions from the newest saved checkpoint."""
        cfg = self.config()
        run_spot_check(
            output_dirs=self.checkpoint_output_dirs(),
            spot=SpotCheckConfig(),
            text_source=cfg.data.source,
            title=self.spot_check_title,
        )

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=self.module_description)
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--train", action="store_true", help="Run full training.")
        group.add_argument(
            "--flight-check",
            action="store_true",
            help="Run a fast end-to-end smoke test.",
        )
        group.add_argument(
            "--spot-check",
            action="store_true",
            help="Sample completions from the latest saved model.",
        )
        add_wandb_args(parser)
        add_train_override_args(parser)
        return parser

    def main(self, argv: list[str] | None = None) -> None:
        args = self.build_parser().parse_args(argv)
        wb = wandb_kwargs(args)
        exp = self.override(**train_override_kwargs(args))
        if getattr(args, "profile", None) is not None:
            exp = exp.with_profile(args.profile)
        if args.train:
            exp.train(**wb)
        elif args.flight_check:
            exp.train_flight_check(**wb)
        elif args.spot_check:
            exp.spot_check()


def module_api(experiment: Gpt2PretrainExperiment) -> tuple:
    """Unpack into a thin experiment module: ``config, train, … = module_api(exp)``.

    The first value is :meth:`Gpt2PretrainExperiment.config` (also fine to bind
    as ``base_config`` for older notebook cells).
    """
    return (
        experiment.config,
        experiment.train,
        experiment.train_flight_check,
        experiment.spot_check,
        experiment.build_parser,
        experiment.main,
    )


def run_main(experiment: Gpt2PretrainExperiment, argv: list[str] | None = None) -> None:
    """CLI entry with FileNotFoundError → exit 1 (missing checkpoints, etc.)."""
    try:
        experiment.main(argv)
    except FileNotFoundError as exc:
        log.error("Error: %s", exc)
        sys.exit(1)
