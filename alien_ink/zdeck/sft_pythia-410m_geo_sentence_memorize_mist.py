#!/usr/bin/env python
"""Sentence memorization test for pretrained Pythia-410M on Mist.

This is the Pythia-410M scale duplicate of the 70M sentence experiment. It
continues causal-LM training on all 56 geo documents, with ordinary loss over
their retained 512-token blocks, while evaluating masked completions from the
exact-sentence JSON. The external completion eval means no corpus rows are held
out. This measures absorption of in-corpus sentences, not generalization.

Pythia-410M is the largest published Pythia base expected to support ordinary
full-parameter AdamW fine-tuning on Mist's RTX 3070 (8 GB). A 1024-token block
OOMs while allocating the full-vocabulary cross-entropy buffer. Batch one at
512 tokens plus 64 gradient-accumulation steps preserves approximately 32,768
training tokens per optimizer update while reducing that peak allocation.

  python alien_ink/zdeck/sft_pythia-410m_geo_sentence_memorize_mist.py
"""

from __future__ import annotations

from alien_ink.hf.ds import HubTextSource, PretrainDataConfig
from alien_ink.hf.manifest import HardwareConfig, Manifest, ScheduleConfig, WandbConfig
from alien_ink.hf.model import PretrainedLmConfig

RUN_NAME = "sft-pythia-410m-geo-sentence-memorize-mist"

MANIFEST = Manifest(
    run_name=RUN_NAME,
    title="Pythia-410M memorization of sentences from the full geo corpus",
    stage="sft",
    data=PretrainDataConfig(
        source=HubTextSource(
            dataset="codycollier/geo-us-states",
            name=None,
            split="train",
            text_column="text",
        ),
        eval_source=None,
        mode="complete",
        completion_eval_path="/tmp/population-exact/geo-us-states.json",
        max_eval_samples=53,
        max_train_samples=None,
        stream_shuffle_buffer=10_000,
        block_size=512,
        respect_document_boundaries=True,
        tokenizer_num_proc=8,
        seed=101,
    ),
    model=PretrainedLmConfig(
        model_name="EleutherAI/pythia-410m",
        tokenizer_name=None,
        attention_implementation="sdpa",
        use_cache=False,
        trust_remote_code=False,
    ),
    hardware=HardwareConfig(
        label="mist-rtx-3070",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=64,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        prefer_bf16=True,
        prefer_fp16=True,
        gradient_checkpointing=True,
        tf32=True,
        torch_compile=False,
        optim="adamw_torch_fused",
    ),
    wandb=WandbConfig(
        entity="logbook",
        project="ink-explore",
        name=RUN_NAME,
        enabled=True,
    ),
    schedule=ScheduleConfig(
        max_steps=-1,
        num_train_epochs=100.0,
        learning_rate=3e-5,
        warmup_steps=None,
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        seed=101,
        logging_steps=None,
        eval_steps=None,
        save_steps=None,
        save_total_limit=2,
        early_stopping_patience=0,
        stop_loss_metric=None,
        stop_loss_threshold=None,
        stop_loss_patience=0,
    ),
    trainer_overrides={
        "adam_beta2": 0.999,
        "resume_from_checkpoint": False,
    },
)


def main() -> None:
    MANIFEST.train()


if __name__ == "__main__":
    main()
