#!/usr/bin/env python
"""Pretrain Mist-sized GPT-NeoX from scratch for 0.25 epochs on WikiText-103.

Performance-baseline formula: short fractional-epoch run on fully materialized
WikiText (``mode="complete"``) so wall-clock and throughput are comparable
across hardware/config tweaks. Every manifest field is spelled out below for
reproducibility — change values in place, do not rely on module defaults.

Speed notes (nanochat-inspired, tuned for Mist: 16-core / 94 GB / RTX 3070 8 GB):
  - Larger micro-batch (4) with fewer accum steps keeps tokens/step at 32,768
    while cutting Python / launch overhead (nanochat: push device-batch-size).
  - Checkpointing off: Mist NeoX has VRAM headroom; rematerialization costs
    wall-clock. If this OOMs, set ``gradient_checkpointing=True`` first.
  - ``torch.compile`` + TF32 (Ampere) + fused AdamW for kernel efficiency.
  - Host has spare CPU/RAM: more tokenize procs, dataloader workers, prefetch.

  python -m alien_ink.zdeck.baseline_perf_mist
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
    run_name="baseline-perf-mist",
    title="GPT-NeoX from scratch on WikiText-103 (0.25 epochs, baseline perf)",
    stage="pre",
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
        respect_document_boundaries=False,
        tokenizer_num_proc=8,
        seed=101,
    ),
    model=CausalLmArchConfig(
        family="gpt-neox",
        tokenizer_name="EleutherAI/gpt-neox-20b",
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        head_dim=None,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        norm_epsilon=1e-5,
        initializer_range=0.02,
        rope_theta=10_000.0,
        rotary_pct=0.25,
        tie_word_embeddings=False,
        num_key_value_heads=None,
        attention_implementation="sdpa",
        use_cache=False,
    ),
    hardware=HardwareConfig(
        label="mist-rtx-3070",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        prefer_bf16=True,
        prefer_fp16=True,
        gradient_checkpointing=False,
        tf32=True,
        torch_compile=True,
        optim="adamw_torch_fused",
    ),
    wandb=WandbConfig(
        entity="logbook",
        project="ink-explore",
        name="baseline-perf-mist",
        enabled=True,
    ),
    schedule=ScheduleConfig(
        max_steps=-1,
        num_train_epochs=0.25,
        learning_rate=6e-4,
        warmup_steps=None,
        warmup_ratio=0.04,
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
