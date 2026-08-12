#!/usr/bin/env python
"""Fact-completion memorization test for pretrained Pythia-410M on Mist.

This is the Pythia-410M scale duplicate of the 70M fact experiment. It trains
on every prompt/completion pair in the population eval JSON and masks prompt
tokens from the loss. Train and eval intentionally use the same pairs, making
``eval_loss`` a direct memorization measurement rather than a generalization
measurement. Practical convergence is three distinct evaluations at or below
0.01 eval loss.

Pythia-410M is the largest published Pythia base expected to support ordinary
full-parameter AdamW fine-tuning on Mist's RTX 3070 (8 GB). Microbatches are
limited to one and gradient checkpointing is enabled to stay near that ceiling.

  python alien_ink/zdeck/sft_pythia-410m_geo_fact_memorize_mist.py
"""

from __future__ import annotations

from alien_ink.hf.completion import CompletionDataConfig
from alien_ink.hf.manifest import HardwareConfig, Manifest, ScheduleConfig, WandbConfig
from alien_ink.hf.model import PretrainedLmConfig

RUN_NAME = "sft-pythia-410m-geo-fact-memorize-mist"

MANIFEST = Manifest(
    run_name=RUN_NAME,
    title="Pythia-410M memorization of completion-trained geo facts",
    stage="sft",
    data=CompletionDataConfig(
        train_path="/tmp/population-basic/geo-us-states.json",
        eval_path="/tmp/population-basic/geo-us-states.json",
        max_eval_samples=None,
        max_train_samples=None,
        loss_on_prompt=False,
        eval_train_dataset=False,
        max_length=128,
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
        gradient_accumulation_steps=1,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=None,
        dataloader_persistent_workers=False,
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
        warmup_ratio=None,
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_scheduler_type="constant",
        seed=101,
        logging_steps=None,
        eval_steps=None,
        save_steps=None,
        save_total_limit=2,
        early_stopping_patience=0,
        stop_loss_metric="eval_loss",
        stop_loss_threshold=0.01,
        stop_loss_patience=3,
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
