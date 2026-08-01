#!/usr/bin/env python
"""Pretrain a Mist-sized Gemma from scratch for 50k steps on streamed C4.

Uses a small Gemma architecture (not full Gemma-2B) so training fits on Mist
(local RTX 3070, ~8 GB). Every manifest field is spelled out below for
reproducibility — change values in place, do not rely on module defaults.

Requires Hugging Face access to the Gemma tokenizer (`google/gemma-2b`).

  python -m alien_ink.zdeck.pre_gemma_c4_50k
"""

from __future__ import annotations

from alien_ink.hf.ds import HubTextSource, PretrainDataConfig
from alien_ink.hf.model import CausalLmArchConfig
from alien_ink.hf.manifest import (
    HardwareConfig,
    Manifest,
    ScheduleConfig,
    WandbConfig,
)

MANIFEST = Manifest(
    run_name="pre-gemma-c4-50k-mist",
    title="Gemma (Mist-sized) from scratch on C4 (50k steps)",
    stage="pre",
    data=PretrainDataConfig(
        source=HubTextSource(
            dataset="allenai/c4",
            name="en",
            split="train",
            text_column="text",
        ),
        eval_source=HubTextSource(
            dataset="allenai/c4",
            name="en",
            split="validation",
            text_column="text",
        ),
        mode="stream",
        max_eval_samples=1_000,
        max_train_samples=None,
        stream_shuffle_buffer=10_000,
        block_size=1024,
        tokenizer_num_proc=4,
        seed=101,
    ),
    model=CausalLmArchConfig(
        family="gemma",
        tokenizer_name="google/gemma-2b",
        n_positions=1024,
        n_embd=512,
        n_layer=8,
        n_head=8,
        head_dim=64,
        intermediate_size=2048,
        use_cache=False,
    ),
    hardware=HardwareConfig(
        label="mist-rtx-3070",
        # batch=2 OOMs on 8 GB: Gemma vocab (~256k) materializes ~2 GiB logits.
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        dataloader_num_workers=2,
        prefer_bf16=True,
        prefer_fp16=True,
        gradient_checkpointing=True,
    ),
    wandb=WandbConfig(
        entity="logbook",
        project="ink-explore",
        name="pre-gemma-c4-50k-mist",
        enabled=True,
    ),
    schedule=ScheduleConfig(
        max_steps=50_000,
        num_train_epochs=3.0,
        learning_rate=6e-4,
        warmup_steps=2_000,
        weight_decay=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        seed=101,
        logging_steps=50,
        eval_steps=1_000,
        save_steps=1_000,
        save_total_limit=2,
        early_stopping_patience=0,
    ),
    trainer_overrides={},
)


def main() -> None:
    MANIFEST.train()


if __name__ == "__main__":
    main()
