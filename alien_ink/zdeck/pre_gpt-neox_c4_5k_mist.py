#!/usr/bin/env python
"""Pretrain Mist-sized GPT-NeoX from scratch for 5k steps on streamed C4.

Matched baseline for ``pre_gpt-neox_curriculum_geo_mist``: same architecture,
hardware, C4 data config, seeds, and 5,000-step cosine schedule — without the
geo-us-states followup phase. Train this first, run population-exact eval, then
compare against the curriculum zdeck after its extra 100 geo steps.

  python alien_ink/zdeck/pre_gpt-neox_c4_5k_mist.py
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
    run_name="pre-gpt-neox-c4-5k-mist",
    title="GPT-NeoX from scratch on C4 (5k steps)",
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
        respect_document_boundaries=True,
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
        name="pre-gpt-neox-c4-5k-mist",
        enabled=True,
    ),
    schedule=ScheduleConfig(
        max_steps=5_000,
        num_train_epochs=3.0,
        learning_rate=6e-4,
        warmup_steps=None,
        warmup_ratio=0.04,
        weight_decay=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        seed=101,
        logging_steps=10,
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
