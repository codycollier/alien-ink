#!/usr/bin/env python
"""Short GPT-2 performance baseline on fully materialized WikiText-103.

  python alien_ink/zdeck/baseline_perf_gpt-2_mist.py
"""

from alien_ink.hf.ds import HubTextSource, PretrainDataConfig
from alien_ink.hf.manifest import HardwareConfig, Manifest, ScheduleConfig, WandbConfig
from alien_ink.hf.model import CausalLmArchConfig

RUN_NAME = "baseline-perf-gpt-2-mist"

MANIFEST = Manifest(
    run_name=RUN_NAME,
    title="GPT-2 from scratch on WikiText-103 (0.25 epochs, baseline perf)",
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
        family="gpt-2",
        tokenizer_name="gpt2",
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        head_dim=None,
        intermediate_size=None,
        hidden_act="gelu_new",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        norm_epsilon=1e-5,
        initializer_range=0.02,
        rope_theta=None,
        rotary_pct=None,
        tie_word_embeddings=True,
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
        entity="logbook", project="ink-explore", name=RUN_NAME, enabled=True
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
        logging_steps=None,
        eval_steps=None,
        save_steps=None,
        save_total_limit=2,
        early_stopping_patience=0,
    ),
    trainer_overrides={},
)


if __name__ == "__main__":
    MANIFEST.train()
