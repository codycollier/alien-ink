#!/usr/bin/env python
"""Pretrain a SmolLM2-135M-shaped Llama model from scratch on WikiText-103.

Uses the published SmolLM2-135M architecture (Llama block: RoPE, SwiGLU,
RMSNorm, GQA with 3 KV heads, tied embeddings) and the SmolLM2 tokenizer, with
random weights — this is an architecture experiment, not the pretrained
checkpoint. Same dataset, block size, effective batch, and schedule as the
GPT-2 / NeoX / Pythia WikiText baselines for direct comparison. Every manifest
field is spelled out below for reproducibility — change values in place, do
not rely on module defaults.

The 30-layer stack carries more activation memory than the 12-layer
baselines: microbatch 2 x accum 16 keeps tokens/step at 32,768. If this OOMs,
set ``gradient_checkpointing=True`` first.

Reference: https://huggingface.co/HuggingFaceTB/SmolLM2-135M

  python alien_ink/zdeck/pre_smollm2-135m_wikitext_4ep_mist.py
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
    run_name="pre-smollm2-135m-wikitext-4ep-mist",
    title="SmolLM2-135M (Llama) from scratch on WikiText-103 (4 epochs)",
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
        family="llama",
        tokenizer_name="HuggingFaceTB/SmolLM2-135M",
        # Published SmolLM2-135M shape, positions capped for Mist (ships 8192).
        n_positions=2048,
        n_embd=576,
        n_layer=30,
        n_head=9,
        head_dim=64,
        intermediate_size=1536,
        hidden_act="silu",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        norm_epsilon=1e-5,
        # SmolLM2 uses 1/sqrt(hidden) initialization.
        initializer_range=0.041666666666666664,
        rope_theta=100_000.0,
        rotary_pct=None,
        tie_word_embeddings=True,
        num_key_value_heads=3,
        attention_implementation="sdpa",
        use_cache=False,
    ),
    hardware=HardwareConfig(
        label="mist-rtx-3070",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
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
        name="pre-smollm2-135m-wikitext-4ep-mist",
        enabled=True,
    ),
    schedule=ScheduleConfig(
        max_steps=-1,
        num_train_epochs=4.0,
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
