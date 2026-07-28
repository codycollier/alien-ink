#!/usr/bin/env python
"""Pretrain a Mist-sized Gemma from scratch for 50k steps on streamed C4.

Uses a small Gemma architecture (not full Gemma-2B) so training fits on Mist
(local RTX 3070, ~8 GB). W&B entity / project / name are set explicitly below
— change them before running.

Requires Hugging Face access to the Gemma tokenizer (`google/gemma-2b`).

  python -m alien_ink.samples.gemma_c4_50k
"""

from __future__ import annotations

from pathlib import Path

from alien_ink.hf.ds import c4_english
from alien_ink.hf.model import gemma_arch
from alien_ink.hf.pretrain import PretrainConfig, pretrain
from alien_ink.hf.trainer import CausalLmTrainerConfig

# Explicit W&B identity (required when use_wandb=True; no package defaults).
WANDB_ENTITY = "logbook"
WANDB_PROJECT = "ink-explore"
WANDB_NAME = "gemma-c4-50k-mist"

MAX_STEPS = 50_000
OUTPUT_DIR = Path.cwd() / "output" / WANDB_NAME


def main() -> None:
    config = PretrainConfig(
        data=c4_english(mode="stream"),
        arch=gemma_arch(),
        trainer=CausalLmTrainerConfig(
            output_dir=OUTPUT_DIR,
            run_name=WANDB_NAME,
            max_steps=MAX_STEPS,
            # Mist RTX 3070 (~8 GB): microbatch 2 × accum 16 = effective 32
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=16,
            warmup_steps=2_000,
            logging_steps=50,
            eval_steps=1_000,
            save_steps=1_000,
        ),
    )
    pretrain(
        config,
        title="Gemma (Mist-sized) from scratch on C4 (50k steps)",
        run_label="sample",
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        wandb_name=WANDB_NAME,
        use_wandb=True,
    )


if __name__ == "__main__":
    main()
