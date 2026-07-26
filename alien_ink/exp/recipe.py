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
from alien_ink.hf.pretrain import Gpt2PretrainConfig, pretrain_gpt2, with_data, with_trainer
from alien_ink.hf.trainer import CausalLmTrainerConfig
from alien_ink.log import get_logger

log = get_logger("exp.recipe")

# Reference cadence for a full streamed run; shorter max_steps scale these down
# so subsets keep roughly the same number of log / eval / checkpoint points.
_REF_MAX_STEPS = 50_000
_REF_LOGGING_STEPS = 50
_REF_EVAL_STEPS = 1_000
_REF_SAVE_STEPS = 1_000
_CADENCE_KEYS = frozenset({"logging_steps", "eval_steps", "save_steps"})


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


@dataclass(frozen=True)
class Gpt2PretrainExperiment:
    """Corpus-specific labels + data factory; paths resolve from cwd at call time."""

    run_name: str
    title: str
    spot_check_title: str
    data_factory: Callable[..., PretrainDataConfig]
    module_description: str
    max_steps: int = 50_000
    warmup_steps: int = 2_000

    def workdir(self) -> Path:
        return Path.cwd()

    def output_root(self) -> Path:
        return self.workdir() / "output"

    def env_file(self) -> Path:
        return self.workdir() / ".env"

    def flight_check_run_name(self) -> str:
        return f"{self.run_name}-flight-check"

    def base_config(self) -> Gpt2PretrainConfig:
        """Full pretrain defaults (~1.6B tokens at 50k steps unless overridden)."""
        return Gpt2PretrainConfig(
            data=self.data_factory(),
            trainer=CausalLmTrainerConfig(
                output_dir=self.output_root() / self.run_name,
                max_steps=self.max_steps,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=16,
                learning_rate=6e-4,
                warmup_steps=self.warmup_steps,
                run_name=self.run_name,
                **scaled_trainer_steps(self.max_steps),
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
        cfg = self.base_config()
        if trainer_overrides:
            cfg = with_trainer(cfg, **trainer_overrides)
            # Keep log/eval/save density aligned when max_steps is overridden;
            # leave any cadence fields the caller set explicitly alone.
            if "max_steps" in trainer_overrides:
                auto = {
                    key: value
                    for key, value in scaled_trainer_steps(cfg.trainer.max_steps).items()
                    if key not in trainer_overrides
                }
                if auto:
                    cfg = with_trainer(cfg, **auto)
        return pretrain_gpt2(
            cfg,
            run_label="regular",
            title=self.title,
            env_files=(self.env_file(),),
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
            use_wandb=use_wandb,
            resume_from_checkpoint=resume_from_checkpoint,
            tpu_launch=tpu_launch,
            tpu_num_processes=tpu_num_processes,
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
        flight_name = self.flight_check_run_name()
        base = self.base_config()
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
        return pretrain_gpt2(
            cfg,
            run_label="flight_check",
            title=self.title,
            env_files=(self.env_file(),),
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_name=wandb_name,
            use_wandb=use_wandb,
            resume_from_checkpoint=resume_from_checkpoint,
            tpu_launch=tpu_launch,
            tpu_num_processes=tpu_num_processes,
        )

    def spot_check(self) -> None:
        """Sample completions from the newest saved checkpoint."""
        cfg = self.base_config()
        run_spot_check(
            output_dirs=[
                self.output_root() / self.run_name,
                self.output_root() / self.flight_check_run_name(),
            ],
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
