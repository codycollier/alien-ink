#!/usr/bin/env python
"""Full-parameter fine-tune of pretrained Pythia-70M on geo-us-states (100 ep).

Off-the-shelf fine-tuning (learning plan Track B / F0): loads the published
``EleutherAI/pythia-70m`` base — the smallest solid Mist-fit checkpoint in the
scout table — via ``AutoModelForCausalLM`` and continues causal-LM training on
the complete geo-us-states corpus for 100 epochs. Tiny base + long epoch budget
is deliberate: enough passes to absorb the 56-document corpus while staying
cheap on an RTX 3070. Correctness before throughput: ``torch.compile`` off,
gradient checkpointing on. Every manifest field is spelled out below for
reproducibility — change values in place, do not rely on module defaults.

Swap ``model_name`` for any other Hub base or a local ``output/train/<run>``
checkpoint, and/or retarget ``data.source.dataset`` to any Hub text corpus
(same ``PretrainDataConfig`` shape). Epoch length is controlled by
``num_train_epochs`` with ``max_steps=-1``.

Reference: https://huggingface.co/EleutherAI/pythia-70m

  python alien_ink/zdeck/sft_pythia-70m_geo_100ep_mist.py
"""

from __future__ import annotations

from alien_ink.hf.ds import HubTextSource, PretrainDataConfig
from alien_ink.hf.model import PretrainedLmConfig
from alien_ink.hf.manifest import (
    HardwareConfig,
    Manifest,
    ScheduleConfig,
    WandbConfig,
)

MANIFEST = Manifest(
    run_name="sft-pythia-70m-geo-100ep-mist",
    title="Pythia-70M full fine-tune on geo-us-states (100 epochs)",
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
        # 56 long documents total; hold out just a few states for eval.
        max_eval_samples=4,
        max_train_samples=None,
        stream_shuffle_buffer=10_000,
        block_size=1024,
        respect_document_boundaries=True,
        tokenizer_num_proc=8,
        seed=101,
    ),
    model=PretrainedLmConfig(
        model_name="EleutherAI/pythia-70m",
        tokenizer_name=None,
        attention_implementation="sdpa",
        use_cache=False,
        trust_remote_code=False,
    ),
    hardware=HardwareConfig(
        label="mist-rtx-3070",
        # First fine-tune: small microbatch + checkpointing per the learning
        # plan; correctness before throughput (compile off).
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
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
        name="sft-pythia-70m-geo-100ep-mist",
        enabled=True,
    ),
    schedule=ScheduleConfig(
        max_steps=-1,
        num_train_epochs=100.0,
        # Fine-tuning LR: ~20x below the from-scratch 6e-4.
        learning_rate=3e-5,
        warmup_steps=None,
        warmup_ratio=0.03,
        weight_decay=0.01,
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
    # Fine-tuning convention: slower second-moment decay than pretraining.
    trainer_overrides={"adam_beta2": 0.999},
)


def main() -> None:
    MANIFEST.train()


if __name__ == "__main__":
    main()
