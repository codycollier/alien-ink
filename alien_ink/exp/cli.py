"""Shared CLI helpers for experiment entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from alien_ink.env import DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    """Add runtime W&B entity / project / run-name overrides (CLI / kwargs only)."""
    parser.add_argument(
        "--wandb-entity",
        default=None,
        metavar="ENTITY",
        help=f"W&B team/entity (default: {DEFAULT_WANDB_ENTITY}).",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        metavar="PROJECT",
        help=f"W&B project (default: {DEFAULT_WANDB_PROJECT}).",
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        metavar="NAME",
        help="W&B run name (default: experiment run_name).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases (sets report_to=none).",
    )


def add_train_override_args(parser: argparse.ArgumentParser) -> None:
    """Optional training hyperparameter / resume overrides."""
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="N",
        help="Override trainer max_steps (-1 for epoch-based length).",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=None,
        metavar="N",
        help="Override trainer num_train_epochs (used when max_steps=-1).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        metavar="LR",
        help="Override trainer learning_rate.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Override per-device train batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        metavar="N",
        help="Override gradient accumulation steps.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=None,
        metavar="PATH",
        help=(
            "Resume training. Pass a checkpoint path, or the flag alone to "
            "auto-pick the latest checkpoint under output_dir."
        ),
    )


def wandb_kwargs(args: argparse.Namespace) -> dict[str, str | bool | None]:
    """Keyword args to pass through to ``train`` / ``train_flight_check``."""
    return {
        "wandb_entity": args.wandb_entity,
        "wandb_project": args.wandb_project,
        "wandb_name": args.wandb_name,
        "use_wandb": False if getattr(args, "no_wandb", False) else None,
    }


def train_override_kwargs(args: argparse.Namespace) -> dict:
    """Trainer overrides + resume flag derived from CLI args."""
    overrides: dict = {}
    for name in (
        "max_steps",
        "num_train_epochs",
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
    ):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value

    resume = getattr(args, "resume_from_checkpoint", None)
    if resume is None:
        return overrides

    if resume is True:
        overrides["resume_from_checkpoint"] = True
    else:
        overrides["resume_from_checkpoint"] = Path(resume)
    return overrides
