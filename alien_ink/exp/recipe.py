"""Shared GPT-2 pretrain experiment recipe (config + CLI + flight/spot checks)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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
    resolve_accelerator_profile,
    trainer_overrides_for_profile,
    with_accelerator_suffix,
)
from alien_ink.hf.pretrain import Gpt2PretrainConfig, pretrain_gpt2, with_data, with_trainer
from alien_ink.hf.trainer import CausalLmTrainerConfig
from alien_ink.log import detail, get_logger

log = get_logger("exp.recipe")

# Reference cadence for a full streamed run; shorter max_steps scale these down
# so step-capped runs keep roughly the same number of log / eval / checkpoint points.
_REF_MAX_STEPS = 50_000
_REF_LOGGING_STEPS = 50
_REF_EVAL_STEPS = 1_000
_REF_SAVE_STEPS = 1_000
_CADENCE_KEYS = frozenset({"logging_steps", "eval_steps", "save_steps"})
# Placeholder until build_causal_lm_trainer derives epoch cadence from dataset size
# (~5 evals/epoch including epoch end).
_EPOCH_LOGGING_STEPS = 10


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
    """Corpus-specific labels + data factory; paths resolve from cwd at call time.

    Use ``max_steps >= 1`` for streamed / step-capped runs, or ``max_steps=-1``
    with ``num_train_epochs`` for finite materialized subsets.
    """

    run_name: str
    title: str
    spot_check_title: str
    data_factory: Callable[..., PretrainDataConfig]
    module_description: str
    max_steps: int = 50_000
    num_train_epochs: float = 3.0
    warmup_steps: int = 2_000

    def workdir(self) -> Path:
        return Path.cwd()

    def output_root(self) -> Path:
        return self.workdir() / "output"

    def env_file(self) -> Path:
        return self.workdir() / ".env"

    def resolved_run_name(
        self,
        *,
        profile: AcceleratorProfile | None = None,
        wandb_name: str | None = None,
        flight_check: bool = False,
    ) -> str:
        """Run / W&B name with ``-gpu`` / ``-tpu`` (or ``-cpu``) suffix."""
        profile = profile or resolve_accelerator_profile()
        base = wandb_name or self.run_name
        if flight_check and wandb_name is None:
            base = f"{self.run_name}-flight-check"
        elif flight_check and wandb_name is not None and "flight-check" not in wandb_name:
            base = f"{wandb_name}-flight-check"
        return with_accelerator_suffix(base, profile.kind)

    def flight_check_run_name(self, profile: AcceleratorProfile | None = None) -> str:
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
        # Preserve order, drop duplicates.
        seen: set[str] = set()
        dirs: list[Path] = []
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            dirs.append(root / name)
        return dirs

    def base_config(
        self,
        profile: AcceleratorProfile | None = None,
        *,
        wandb_name: str | None = None,
    ) -> Gpt2PretrainConfig:
        """Full pretrain defaults with accelerator-aware batch / run name."""
        profile = profile or resolve_accelerator_profile()
        run_name = self.resolved_run_name(profile=profile, wandb_name=wandb_name)
        detail(
            f"accelerator profile: {profile.label} ({profile.hardware}) "
            f"batch={profile.per_device_train_batch_size} "
            f"accum={profile.gradient_accumulation_steps} "
            f"eff={profile.effective_batch_size} "
            f"mem×{profile.memory_multiple:g} "
            f"compute×{profile.vs_rtx_3070:g} "
            f"run_name={run_name}",
            logger=log,
        )
        return Gpt2PretrainConfig(
            data=self.data_factory(),
            trainer=CausalLmTrainerConfig(
                output_dir=self.output_root() / run_name,
                learning_rate=6e-4,
                warmup_steps=self.warmup_steps,
                run_name=run_name,
                **trainer_overrides_for_profile(profile),
                **trainer_length_kwargs(
                    max_steps=self.max_steps,
                    num_train_epochs=self.num_train_epochs,
                ),
            ),
        )

    def train(
        self,
        *,
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
        wandb_name: str | None = None,
        use_wandb: bool | None = None,
        resume_from_checkpoint: str | Path | bool | None = None,
        tpu_launch: bool | None = None,
        tpu_num_processes: int | None = None,
        **trainer_overrides,
    ):
        """Full pretraining run.

        Returns ``(trainer, run_summary)``. ``run_summary.json`` / ``run_config.json``
        are always written under the run ``output_dir``. On Colab TPU notebooks,
        auto-launches via ``notebook_launcher`` and returns ``(None, None)``.
        """
        profile = resolve_accelerator_profile()
        cfg = self.base_config(profile, wandb_name=wandb_name)
        # wandb_name already folded into run_name; avoid double-apply in pretrain.
        resolved_name = cfg.trainer.run_name
        if trainer_overrides:
            cfg = with_trainer(cfg, **trainer_overrides)
            # Keep log/eval/save density aligned when max_steps is overridden
            # into step mode; leave any cadence fields the caller set alone.
            if "max_steps" in trainer_overrides and cfg.trainer.max_steps > 0:
                auto = {
                    key: value
                    for key, value in scaled_trainer_steps(cfg.trainer.max_steps).items()
                    if key not in trainer_overrides
                }
                if auto:
                    cfg = with_trainer(cfg, **auto)
        processes = (
            tpu_num_processes
            if tpu_num_processes is not None
            else profile.tpu_num_processes
        )
        return pretrain_gpt2(
            cfg,
            run_label="regular",
            title=self.title,
            env_files=(self.env_file(),),
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_name=resolved_name,
            use_wandb=use_wandb,
            resume_from_checkpoint=resume_from_checkpoint,
            tpu_launch=tpu_launch,
            tpu_num_processes=processes,
        )

    def train_flight_check(
        self,
        *,
        wandb_entity: str | None = None,
        wandb_project: str | None = None,
        wandb_name: str | None = None,
        use_wandb: bool | None = None,
        resume_from_checkpoint: str | Path | bool | None = None,
        tpu_launch: bool | None = None,
        tpu_num_processes: int | None = None,
        **trainer_overrides,
    ):
        """Fast end-to-end smoke test (tiny steps / block size).

        Returns ``(trainer, run_summary)`` like :meth:`train`.
        """
        profile = resolve_accelerator_profile()
        flight_name = self.resolved_run_name(
            profile=profile,
            wandb_name=wandb_name,
            flight_check=True,
        )
        base = self.base_config(profile)
        data_overrides: dict = {
            "max_eval_samples": 10,
            "stream_shuffle_buffer": 50,
            "block_size": 128,
        }
        # Keep flight checks cheap for materialized subset corpora.
        if base.data.max_train_samples is not None:
            data_overrides["max_train_samples"] = 50
        cfg = with_data(
            with_trainer(
                base,
                output_dir=self.output_root() / flight_name,
                max_steps=10,
                warmup_steps=0,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=1,
                eval_steps=5,
                save_steps=10,
                run_name=flight_name,
            ),
            **data_overrides,
        )
        if trainer_overrides:
            cfg = with_trainer(cfg, **trainer_overrides)
        processes = (
            tpu_num_processes
            if tpu_num_processes is not None
            else profile.tpu_num_processes
        )
        return pretrain_gpt2(
            cfg,
            run_label="flight_check",
            title=self.title,
            env_files=(self.env_file(),),
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_name=flight_name,
            use_wandb=use_wandb,
            resume_from_checkpoint=resume_from_checkpoint,
            tpu_launch=tpu_launch,
            tpu_num_processes=processes,
        )

    def spot_check(self) -> None:
        """Sample completions from the newest saved checkpoint."""
        cfg = self.base_config()
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
        overrides = train_override_kwargs(args)
        if args.train:
            self.train(**wb, **overrides)
        elif args.flight_check:
            self.train_flight_check(**wb, **overrides)
        elif args.spot_check:
            self.spot_check()


def run_main(experiment: Gpt2PretrainExperiment, argv: list[str] | None = None) -> None:
    """CLI entry with FileNotFoundError → exit 1 (missing checkpoints, etc.)."""
    try:
        experiment.main(argv)
    except FileNotFoundError as exc:
        log.error("Error: %s", exc)
        sys.exit(1)
