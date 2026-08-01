#!/usr/bin/env python
"""Pretrain GPT-2 from scratch for 5k steps on streamed English Wikipedia.

Sized for Mist (local RTX 3070, ~8 GB). Every manifest field is spelled out
below for reproducibility — change values in place, do not rely on module
defaults.

  python alien_ink/zdeck/gpt-2_wikipedia_5k.py
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
    run_name="gpt-2-wikipedia-5k-mist",
    title="GPT-2 from scratch on English Wikipedia (5k steps)",
    data=PretrainDataConfig(
        source=HubTextSource(
            dataset="wikimedia/wikipedia",
            name="20231101.en",
            split="train",
            text_column="text",
        ),
        eval_source=None,
        mode="stream",
        max_eval_samples=1_000,
        max_train_samples=None,
        stream_shuffle_buffer=10_000,
        block_size=1024,
        tokenizer_num_proc=4,
        seed=101,
    ),
    model=CausalLmArchConfig(
        family="gpt-2",
        tokenizer_name="gpt2",
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        head_dim=None,
        intermediate_size=None,
        use_cache=False,
    ),
    hardware=HardwareConfig(
        label="mist-rtx-3070",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        dataloader_num_workers=2,
        prefer_bf16=True,
        prefer_fp16=True,
        gradient_checkpointing=True,
    ),
    wandb=WandbConfig(
        entity="logbook",
        project="ink-explore",
        name="gpt-2-wikipedia-5k-mist",
        enabled=True,
    ),
    schedule=ScheduleConfig(
        max_steps=5_000,
        num_train_epochs=3.0,
        learning_rate=6e-4,
        warmup_steps=200,
        weight_decay=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        seed=101,
        logging_steps=5,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        early_stopping_patience=0,
    ),
    trainer_overrides={},
)


def main() -> None:
    MANIFEST.train()


if __name__ == "__main__":
    main()
