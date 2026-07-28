#!/usr/bin/env python
"""Sample: GPT-2 from scratch for 50k steps on streamed WikiText-103.

Target: Mist / RTX 3070 (~8 GB). Uses the library defaults of microbatch 2
with gradient accumulation 16.

Run::

  python -m alien_ink.samples.pretrain_wikitext_50k
  python -m alien_ink.samples.pretrain_wikitext_50k --wandb
"""

from __future__ import annotations

import argparse
from pathlib import Path

from alien_ink.hf.ds import wikitext_103
from alien_ink.hf.model import ModelArchConfig
from alien_ink.hf.pretrain import PretrainConfig, pretrain
from alien_ink.hf.trainer import CausalLmTrainerConfig


RUN_NAME = "sample-gpt2-wikitext-50k"
TITLE = "Sample: GPT-2 × WikiText-103 (50k stream)"
MAX_STEPS = 50_000


def build_config(*, workdir: Path | None = None) -> PretrainConfig:
    root = workdir or Path.cwd()
    return PretrainConfig(
        data=wikitext_103(load_mode="streaming"),
        arch=ModelArchConfig(family="gpt2"),
        trainer=CausalLmTrainerConfig(
            output_dir=root / "output" / RUN_NAME,
            run_name=RUN_NAME,
            max_steps=MAX_STEPS,
            warmup_steps=2_000,
            logging_steps=50,
            eval_steps=1_000,
            save_steps=1_000,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args(argv)

    cfg = build_config()
    pretrain(
        cfg,
        title=TITLE,
        env_files=(Path.cwd() / ".env",),
        use_wandb=args.wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name or cfg.trainer.run_name,
    )


if __name__ == "__main__":
    main()
