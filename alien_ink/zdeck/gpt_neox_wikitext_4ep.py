#!/usr/bin/env python
"""Pretrain Mist-sized GPT-NeoX from scratch for 4 epochs on WikiText-103.

Sized for Mist (local RTX 3070, ~8 GB). WikiText is fully materialized
(``mode="complete"``) so epoch length is well-defined. Every manifest field is
spelled out below for reproducibility — change values in place, do not rely on
module defaults.

  python -m alien_ink.zdeck.gpt_neox_wikitext_4ep
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
    run_name="gpt-neox-wikitext-4ep-mist",
    title="GPT-NeoX from scratch on WikiText-103 (4 epochs)",
    data=PretrainDataConfig(
        source=HubTextSource(
            dataset="Salesforce/wikitext",
            name="wikitext-103-v1",
            split="train",
            text_column="text",
        ),
        eval_source=HubTextSource(
            dataset="Salesforce/wikitext",
            name="wikitext-103-v1",
            split="validation",
            text_column="text",
        ),
        mode="complete",
        max_eval_samples=1_000,
        max_train_samples=None,
        stream_shuffle_buffer=10_000,
        block_size=1024,
        tokenizer_num_proc=4,
        seed=101,
    ),
    model=CausalLmArchConfig(
        family="gpt_neox",
        tokenizer_name="EleutherAI/gpt-neox-20b",
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        head_dim=None,
        intermediate_size=3072,
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
        name="gpt-neox-wikitext-4ep-mist",
        enabled=True,
    ),
    schedule=ScheduleConfig(
        max_steps=-1,
        num_train_epochs=4.0,
        learning_rate=6e-4,
        # ~4% of ~16k planned optimizer steps (≈4 epochs × ~4k steps/epoch).
        warmup_steps=640,
        weight_decay=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        seed=101,
        # Epoch mode: cadence is derived from packed dataset length at train time.
        logging_steps=None,
        eval_steps=None,
        save_steps=None,
        save_total_limit=2,
        early_stopping_patience=0,
    ),
    trainer_overrides={},
)


def main() -> None:
    MANIFEST.train()


if __name__ == "__main__":
    main()
