#!/usr/bin/env python
"""Pretrain GPT-Neo from scratch on TinyStories for 5,000 steps."""

from __future__ import annotations

from alien_ink.hf.ds import tinystories
from alien_ink.hf.model import CausalLmArchConfig
from alien_ink.hf.manifest import (
    HardwareConfig,
    Manifest,
    ScheduleConfig,
    WandbConfig,
)

MANIFEST = Manifest(
    run_name="pre-tinystories-33m-5k-mist",
    title="GPT-Neo 33M from scratch on TinyStories (5k steps)",
    stage="pre",
    data=tinystories(),
    model=CausalLmArchConfig(
        family="gpt-neo",
        tokenizer_name="output/tokenizers/tinystories-bpe-6144",
        n_positions=512,
        n_embd=768,
        n_layer=4,
        n_head=12,
        head_dim=None,
        intermediate_size=3072,
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
        name="pre-tinystories-33m-5k-mist",
        enabled=False,
    ),
    schedule=ScheduleConfig(
        max_steps=5_000,
        num_train_epochs=3.0,
        learning_rate=6e-4,
        warmup_steps=200,
        warmup_ratio=None,
        weight_decay=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        seed=101,
        logging_steps=10,
        eval_steps=250,
        save_steps=250,
        save_total_limit=2,
        early_stopping_patience=0,
    ),
    trainer_overrides={},
)


def main() -> None:
    MANIFEST.train()


if __name__ == "__main__":
    main()
