#!/usr/bin/env python
"""Pretrain GPT-Neo from scratch on TinyStories for one full epoch."""

from __future__ import annotations

from dataclasses import replace

from alien_ink.hf.ds import tinystories
from alien_ink.hf.model import CausalLmArchConfig
from alien_ink.hf.manifest import (
    HardwareConfig,
    Manifest,
    ScheduleConfig,
    WandbConfig,
)

MANIFEST = Manifest(
    run_name="pre-tinystories-1m-1ep-mist",
    title="GPT-Neo 1M from scratch on TinyStories (1 epoch)",
    stage="pre",
    data=replace(tinystories(), mode="complete"),
    model=CausalLmArchConfig(
        family="gpt-neo",
        tokenizer_name="output/tokenizers/tinystories-bpe-6144",
        n_positions=512,
        n_embd=72,
        n_layer=8,
        n_head=4,
        head_dim=None,
        intermediate_size=288,
        hidden_act="gelu_new",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        norm_epsilon=1e-5,
        initializer_range=0.02,
        rope_theta=None,
        rotary_pct=None,
        tie_word_embeddings=True,
        num_key_value_heads=None,
        attention_implementation="eager",
        use_cache=False,
        local_attention_window=256,
    ),
    hardware=HardwareConfig(
        label="mist-rtx-3070",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=None,
        dataloader_persistent_workers=False,
        prefer_bf16=True,
        prefer_fp16=True,
        gradient_checkpointing=False,
        tf32=True,
        torch_compile=False,
        optim="adamw_torch",
    ),
    wandb=WandbConfig(
        entity="logbook",
        project="ink-explore",
        name="pre-tinystories-1m-1ep-mist",
        enabled=True,
    ),
    schedule=ScheduleConfig(
        max_steps=-1,
        num_train_epochs=1.0,
        learning_rate=6e-4,
        warmup_steps=200,
        warmup_ratio=None,
        weight_decay=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        seed=101,
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
