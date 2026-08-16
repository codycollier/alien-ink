#!/usr/bin/env python
"""Pretrain Pythia-70M to a 20:1 token budget, ending on WikiText-103.

The 42,725-step budget is the Chinchilla planning approximation for 70M
parameters at 32,768 tokens per optimizer step. Training streams English
Wikipedia for 39,049 steps, then makes one complete packed pass through
WikiText-103 in 3,676 full optimizer steps.

  python alien_ink/zdeck/pre_pythia-70m_wikipedia_wikitext_chinchilla_mist.py
"""

from __future__ import annotations

from alien_ink.hf.curriculum import Curriculum, CurriculumPhase
from alien_ink.hf.ds import HubTextSource, PretrainDataConfig
from alien_ink.hf.manifest import HardwareConfig, Manifest, ScheduleConfig, WandbConfig
from alien_ink.hf.model import CausalLmArchConfig

WIKIPEDIA_DATA = PretrainDataConfig(
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
    respect_document_boundaries=True,
    tokenizer_num_proc=8,
    seed=101,
)

WIKITEXT_DATA = PretrainDataConfig(
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
)

CURRICULUM = Curriculum(
    phases=(
        CurriculumPhase(data=WIKIPEDIA_DATA, steps=39_049, label="wikipedia"),
        CurriculumPhase(data=WIKITEXT_DATA, steps=3_676, label="wikitext-103"),
    ),
    eval_data={"wikitext": WIKITEXT_DATA, "wikipedia": WIKIPEDIA_DATA},
)

MANIFEST = Manifest(
    run_name="pre-pythia-70m-wikipedia-wikitext-chinchilla-mist",
    title="Pythia-70M Chinchilla pretraining: Wikipedia then WikiText-103",
    stage="pre",
    data=CURRICULUM,
    model=CausalLmArchConfig(
        family="pythia",
        tokenizer_name="EleutherAI/pythia-70m",
        n_positions=2048,
        n_embd=512,
        n_layer=6,
        n_head=8,
        head_dim=None,
        intermediate_size=2048,
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
        name="pre-pythia-70m-wikipedia-wikitext-chinchilla-mist",
        enabled=True,
    ),
    schedule=ScheduleConfig(
        max_steps=CURRICULUM.total_steps(),
        num_train_epochs=3.0,
        learning_rate=1e-3,
        warmup_steps=None,
        warmup_ratio=0.04,
        weight_decay=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        seed=101,
        logging_steps=40,
        eval_steps=800,
        save_steps=800,
        save_total_limit=2,
        early_stopping_patience=0,
    ),
    trainer_overrides={},
)


def main() -> None:
    MANIFEST.train()


if __name__ == "__main__":
    main()
