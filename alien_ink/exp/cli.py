"""Shared CLI helpers for experiment entrypoints."""

from __future__ import annotations

import argparse


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    """Add runtime W&B project / run-name overrides (CLI / kwargs only)."""
    parser.add_argument(
        "--wandb-project",
        default=None,
        metavar="NAME",
        help="W&B project (default: alien-ink).",
    )
    parser.add_argument(
        "--wandb-name",
        default=None,
        metavar="NAME",
        help="W&B run name (default: experiment run_name).",
    )


def wandb_kwargs(args: argparse.Namespace) -> dict[str, str | None]:
    """Keyword args to pass through to ``train`` / ``train_flight_check``."""
    return {
        "wandb_project": args.wandb_project,
        "wandb_name": args.wandb_name,
    }
