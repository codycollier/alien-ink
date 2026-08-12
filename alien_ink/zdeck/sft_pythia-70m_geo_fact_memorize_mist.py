#!/usr/bin/env python
"""Fact-completion memorization test for Pythia-70M on synthetic geo facts.

This uses Alien Ink's special completion-memorization data path. At runtime,
``CompletionDataConfig.train_path`` reads the population eval JSON and dynamically
tokenizes its ``prompt`` and ``completion`` fields instead of loading or
chunking a Hub corpus. ``max_train_samples=None`` selects every
JSON entry: all state/territory and sentence-template combinations in the
file. For example, one resulting pair is::

    prompt:     "Alabama has a population of"
    completion: "10,000."

Each resulting causal-LM row contains ``prompt + completion`` as input, but
its labels are ``-100`` over every prompt token. Cross-entropy and gradients
are therefore computed only for the completion tokens. Every JSON sentence is
visited once per epoch. With batch size one and no gradient accumulation, each
sentence produces one optimizer update.

Losses in plain English:

* Training ``loss`` measures how surprised the model is by answer tokens such
  as ``10,000.`` after seeing the prompt. Prompt tokens never count toward it.
* ``eval_completion_loss`` measures that same answer-token objective over the
  configured eval pairs.
* ``eval_train_loss`` runs the actual training pairs through the model in eval
  mode, without dropout, gradients, or parameter updates.

Because train and eval contain the same complete JSON here, the two eval
losses should be nearly identical. They are the clean, comparable measurement
of memorization. Falling toward zero means the model is assigning very high
probability to the answers it was explicitly taught. Trainer's aggregate
end-of-run ``train_loss`` may use different accumulation bookkeeping; prefer
the two eval losses when judging this experiment.

``eval_path`` independently constructs the same kind of masked
dataset. Here it points to the same JSON, and the generous eval cap includes
the complete file. Trainer evaluates it under two names:
``eval_completion_loss`` is the configured completion eval set, while
``eval_train_loss`` re-evaluates the actual training set without dropout or
parameter updates. The two should agree closely and fall toward zero;
disagreement would expose a data-path problem, while failure to fall would
expose an optimization problem. This is an intentional memorization
diagnostic of facts trained directly as completions, not a test of absorbing
sentences from a natural-text corpus or of generalization.

Set either sample limit to a small integer for a prefix-only diagnostic. Swap
``model_name`` for another Hub base or a local ``output/train/<run>``
checkpoint. Every manifest field is spelled out below for reproducibility;
epoch length is controlled by ``num_train_epochs`` with ``max_steps=-1``.

Reference: https://huggingface.co/EleutherAI/pythia-70m

  python alien_ink/zdeck/sft_pythia-70m_geo_fact_memorize_mist.py
"""

from __future__ import annotations

from alien_ink.hf.completion import CompletionDataConfig
from alien_ink.hf.model import PretrainedLmConfig
from alien_ink.hf.manifest import (
    HardwareConfig,
    Manifest,
    ScheduleConfig,
    WandbConfig,
)

MANIFEST = Manifest(
    run_name="sft-pythia-70m-geo-fact-memorize-mist",
    title="Pythia-70M memorization of completion-trained geo facts",
    stage="sft",
    data=CompletionDataConfig(
        # Dynamically tokenize every JSON pair for both train and eval.
        train_path="/tmp/population-basic/geo-us-states.json",
        eval_path="/tmp/population-basic/geo-us-states.json",
        max_eval_samples=None,
        max_train_samples=None,
        loss_on_prompt=False,
        max_length=128,
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
        # One optimizer update per sentence; keep the diagnostic simple.
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
        name="sft-pythia-70m-geo-fact-memorize-mist",
        enabled=True,
    ),
    schedule=ScheduleConfig(
        max_steps=-1,
        num_train_epochs=100.0,
        # Constant conservative LR, with no regularization fighting memorization.
        learning_rate=3e-5,
        warmup_steps=None,
        warmup_ratio=None,
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_scheduler_type="constant",
        seed=101,
        # Epoch mode derives logging/eval cadence from the JSON dataset length.
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
