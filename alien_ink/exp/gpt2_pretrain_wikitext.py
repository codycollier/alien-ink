#!/usr/bin/env python
"""Pretrain GPT-2 small from scratch on WikiText-103 and spot-check checkpoints.

Thin experiment entrypoints: define a base config, then small variant functions that
override only the knobs that differ (steps, LR, batch size, etc.).

Defaults are tuned to fit an 8 GB GPU (e.g. an RTX 3070): GPT-2 small (~124M
params), bf16 mixed precision, gradient checkpointing, and a small per-device
batch size with gradient accumulation. If you hit CUDA OOM, drop
``per_device_train_batch_size`` to 1 and/or lower ``block_size``.

Run from an installed environment (CLI)::

  python -m alien_ink.exp.gpt2_pretrain_wikitext --train
  python -m alien_ink.exp.gpt2_pretrain_wikitext --flight-check
  python -m alien_ink.exp.gpt2_pretrain_wikitext --spot-check

Override W&B project / run name at runtime::

  python -m alien_ink.exp.gpt2_pretrain_wikitext --train \\
    --wandb-project my-proj --wandb-name my-run

Or from a notebook / REPL::

  from alien_ink.exp.gpt2_pretrain_wikitext import train, train_flight_check
  train_flight_check(wandb_project="my-proj", wandb_name="flight-check")

Artifacts and ``.env`` resolve relative to the process working directory.
Set W&B project / run name via ``--wandb-project`` / ``--wandb-name`` (or kwargs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alien_ink.exp.cli import add_wandb_args, wandb_kwargs
from alien_ink.hf.ds import wikitext_103
from alien_ink.hf.gen import SpotCheckConfig, run_spot_check
from alien_ink.hf.pretrain import Gpt2PretrainConfig, pretrain_gpt2, with_data, with_trainer
from alien_ink.hf.trainer import CausalLmTrainerConfig


# ---------------------------------------------------------------------------
# Paths & base config
# ---------------------------------------------------------------------------

WORKDIR = Path.cwd()
OUTPUT_ROOT = WORKDIR / "output"
ENV_FILE = WORKDIR / ".env"

RUN_NAME = "gpt2-pretrain-wikitext"


def base_config() -> Gpt2PretrainConfig:
    """Full WikiText-103 pretrain defaults (~1.6B tokens at 50k steps)."""
    return Gpt2PretrainConfig(
        data=wikitext_103(),
        trainer=CausalLmTrainerConfig(
            output_dir=OUTPUT_ROOT / RUN_NAME,
            max_steps=50_000,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            learning_rate=6e-4,
            warmup_steps=2_000,
            run_name=RUN_NAME,
        ),
    )


# ---------------------------------------------------------------------------
# Training variants — override only what changes
# ---------------------------------------------------------------------------

def train(
    *,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
) -> None:
    """Full pretraining run."""
    pretrain_gpt2(
        base_config(),
        run_label="regular",
        title="GPT-2 from scratch on WikiText-103",
        env_files=(ENV_FILE,),
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )


def train_flight_check(
    *,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
) -> None:
    """Fast end-to-end smoke test (tiny steps / block size)."""
    cfg = with_data(
        with_trainer(
            base_config(),
            output_dir=OUTPUT_ROOT / f"{RUN_NAME}-flight-check",
            max_steps=10,
            warmup_steps=0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            logging_steps=1,
            eval_steps=5,
            save_steps=10,
            run_name="flight-check",
        ),
        max_eval_samples=10,
        stream_shuffle_buffer=50,
        block_size=128,
    )
    pretrain_gpt2(
        cfg,
        run_label="flight_check",
        title="GPT-2 from scratch on WikiText-103",
        env_files=(ENV_FILE,),
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )


def spot_check() -> None:
    """Sample completions from the newest saved checkpoint."""
    cfg = base_config()
    run_spot_check(
        output_dirs=[
            OUTPUT_ROOT / RUN_NAME,
            OUTPUT_ROOT / f"{RUN_NAME}-flight-check",
        ],
        spot=SpotCheckConfig(),
        text_source=cfg.data.source,
        title="GPT-2 WikiText — Spot Check",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain GPT-2 on WikiText-103 or spot-check a saved checkpoint."
        ),
    )
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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    wb = wandb_kwargs(args)
    if args.train:
        train(**wb)
    elif args.flight_check:
        train_flight_check(**wb)
    elif args.spot_check:
        spot_check()


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
