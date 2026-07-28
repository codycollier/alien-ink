#!/usr/bin/env python
"""Pretrain GPT-2 from scratch for 5k steps on streamed English Wikipedia.

Sized for Mist (local RTX 3070, ~8 GB). W&B entity / project / name are set
explicitly below — change them before running.

  python -m alien_ink.samples.gpt2_wikipedia_5k
"""

from __future__ import annotations

from pathlib import Path

from alien_ink.hf.ds import wikipedia_english
from alien_ink.hf.model import gpt2_arch
from alien_ink.hf.pretrain import PretrainConfig, pretrain
from alien_ink.hf.trainer import CausalLmTrainerConfig

# Explicit W&B identity (required when use_wandb=True; no package defaults).
WANDB_ENTITY = "logbook"
WANDB_PROJECT = "ink-explore"
WANDB_NAME = "gpt2-wikipedia-5k-mist"

MAX_STEPS = 5_000
OUTPUT_DIR = Path.cwd() / "output" / WANDB_NAME


def main() -> None:
    config = PretrainConfig(
        data=wikipedia_english(mode="stream"),
        arch=gpt2_arch(),
        trainer=CausalLmTrainerConfig(
            output_dir=OUTPUT_DIR,
            run_name=WANDB_NAME,
            max_steps=MAX_STEPS,
            # Mist RTX 3070 (~8 GB): microbatch 2 × accum 16 = effective 32
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=16,
            warmup_steps=min(200, MAX_STEPS // 10),
            logging_steps=max(1, MAX_STEPS // 100),
            eval_steps=max(1, MAX_STEPS // 10),
            save_steps=max(1, MAX_STEPS // 10),
        ),
    )
    pretrain(
        config,
        title="GPT-2 from scratch on English Wikipedia (5k steps)",
        run_label="sample",
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        wandb_name=WANDB_NAME,
        use_wandb=True,
    )


if __name__ == "__main__":
    main()
