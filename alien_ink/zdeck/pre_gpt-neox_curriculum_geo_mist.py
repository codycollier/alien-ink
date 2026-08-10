#!/usr/bin/env python
"""Pretrain Mist-sized GPT-NeoX on a curriculum: 5k C4 then geo-us-states.

First curriculum run: 5,000 steps of streamed C4 English followed by 100
steps over the complete ``codycollier/geo-us-states`` corpus (56 long
documents, one per US state/territory). One continuous run — single LR
schedule, single W&B run — with the phase switch landing exactly on the
step-5,000 checkpoint.

Phase math (mist hardware: batch 4 x accum 8 = 32 blocks/step): geo yields
~880 train blocks (52 rows after the 4-row eval hold-out, ~17 blocks each),
so ~27 steps/epoch — 100 steps is ~3-4 epochs via repeat-to-fill. Change the
phase ``steps`` in place to change epochs.

Eval is fixed for the whole run: a geo hold-out slice and a C4 validation
slice, reported separately as ``eval_geo_loss`` and ``eval_c4_loss`` so the
C4 curve shows any forgetting during the geo phase. ``geo`` is listed first
so best-model selection tracks the followup phase rather than reverting to a
pre-geo checkpoint.

Schedule note: cosine decays the LR to near zero by the time the geo phase
starts, which mutes what the model can absorb from it (the mid-training
literature's main caveat). Swap ``lr_scheduler_type`` to
``"warmup_stable_decay"`` to hold the LR flat until a late decay instead.

  python alien_ink/zdeck/pre_gpt-neox_curriculum_geo_mist.py
"""

from __future__ import annotations

from alien_ink.hf.curriculum import Curriculum, CurriculumPhase
from alien_ink.hf.ds import HubTextSource, PretrainDataConfig
from alien_ink.hf.model import CausalLmArchConfig
from alien_ink.hf.manifest import (
    HardwareConfig,
    Manifest,
    ScheduleConfig,
    WandbConfig,
)

C4_DATA = PretrainDataConfig(
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
)

# Reused as both the phase-2 corpus and the "geo" eval entry: the shared seed
# holds out the same 4 rows in both places, so train and eval never overlap.
GEO_DATA = PretrainDataConfig(
    source=HubTextSource(
        dataset="codycollier/geo-us-states",
        name=None,
        split="train",
        text_column="text",
    ),
    eval_source=None,
    mode="complete",
    max_eval_samples=4,
    max_train_samples=None,
    stream_shuffle_buffer=10_000,
    block_size=1024,
    respect_document_boundaries=True,
    tokenizer_num_proc=8,
    seed=101,
)

CURRICULUM = Curriculum(
    phases=(
        CurriculumPhase(data=C4_DATA, steps=5_000, label="c4"),
        CurriculumPhase(data=GEO_DATA, steps=100, label="geo"),
    ),
    eval_data={"geo": GEO_DATA, "c4": C4_DATA},
)

MANIFEST = Manifest(
    run_name="pre-gpt-neox-curriculum-geo-mist",
    title="GPT-NeoX curriculum: C4 (5k steps) then geo-us-states (100 steps)",
    stage="pre",
    data=CURRICULUM,
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
        name="pre-gpt-neox-curriculum-geo-mist",
        enabled=True,
    ),
    schedule=ScheduleConfig(
        max_steps=CURRICULUM.total_steps(),  # 5_100
        num_train_epochs=3.0,
        learning_rate=6e-4,
        warmup_steps=None,
        warmup_ratio=0.04,
        weight_decay=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        seed=101,
        # eval/save every 100 puts a checkpoint exactly at the phase boundary
        # (step 5,000), so geo-phase variants can be re-run from that base.
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
