#!/usr/bin/env python
"""Verify the 33M TinyStories run on Mist with a ten-step CUDA smoke test."""

import argparse
import math

import torch

from alien_ink.hf.model import load_pretrained_model
from alien_ink.zdeck.pre_tinystories_33m_5k_mist import MANIFEST

MEMORY_LIMIT = 7 * 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=(2, 4), default=4)
    batch_size = parser.parse_args().batch_size
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Mist smoke test.")

    run = (
        MANIFEST.variant(run_name=f"pre-tinystories-33m-smoke-b{batch_size}-mist")
        .with_data(max_eval_samples=64)
        .with_hardware(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=16 // batch_size,
        )
        .with_schedule(max_steps=10, eval_steps=10, save_steps=10, warmup_steps=0)
        .with_trainer_knobs(logging_nan_inf_filter=False)
    )
    torch.cuda.reset_peak_memory_stats()
    trainer, summary = run.train()
    peak = torch.cuda.max_memory_reserved()
    eval_losses = [entry["eval_loss"] for entry in trainer.state.log_history if "eval_loss" in entry]
    if (
        trainer.state.global_step != 10
        or summary is None
        or summary.train_loss is None
        or not math.isfinite(summary.train_loss)
        or not eval_losses
        or not all(map(math.isfinite, eval_losses))
    ):
        raise SystemExit("Expected 10 steps with finite train and validation loss.")
    load_pretrained_model(run.output_dir() / "checkpoint-10", "cpu", family="gpt-neo")
    print(f"Peak CUDA reserved memory {peak / 1024**3:.2f} GiB")
    if peak >= MEMORY_LIMIT:
        raise SystemExit("Peak CUDA reserved memory must be below 7 GiB.")


if __name__ == "__main__":
    main()
