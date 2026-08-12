#!/usr/bin/env python
"""Sentence memorization test for Pythia-70M on the full geo corpus.

This experiment asks whether ordinary full-document causal-LM fine-tuning can
memorize sentences that occur verbatim in its training corpus. It trains on
all 56 documents from ``codycollier/geo-us-states``. Because
``completion_eval_path`` supplies Trainer evaluation externally, no Hub rows
are held out: every geo document remains in training.

Losses in plain English:

* Training ``loss`` measures ordinary next-token prediction over every token
  retained in the 1024-token document blocks: headings, prose, punctuation,
  population facts, and all surrounding material contribute.
* ``eval_loss`` is much narrower. It supplies an exact sentence prefix as
  context, ignores that prompt in the labels, and measures only how surprised
  the model is by the sentence's remaining completion tokens.

A falling training loss means the model is learning the geo documents in
general. A falling eval loss means it is specifically becoming better at the
selected in-corpus sentence continuations. Their absolute values should not be
expected to match because they average over different tokens and objectives;
the useful question is whether both decline, especially whether eval loss
moves toward zero for memorized continuations.

Evaluation is dynamically built from all 53 entries in
``/tmp/population-exact/geo-us-states.json``. Each entry splits a sentence
from the corpus into a ``prompt`` and ``completion``. The shared completion
tokenizer restores normal boundary whitespace when the JSON fields omit it.
Evaluation is teacher-forced completion cross-entropy: prompt labels are
masked with ``-100``, while completion tokens are scored. The resulting ``eval_loss``
therefore measures whether the model can recover exact continuations from
sentences it encountered during full-document training.

This differs deliberately from ``sft_pythia-70m_geo_fact_memorize_mist.py``.
That zdeck directly trains on synthetic prompt/completion facts and optimizes
only their completions. This zdeck trains the natural-text corpus with loss on
every retained document token, then probes exact in-corpus sentences. It is a
sentence-absorption diagnostic, not a held-out generalization measurement.

Document boundaries are preserved and documents are chunked into 1024-token
blocks. As with the standard corpus path, trailing partial blocks are dropped.
Using the Pythia tokenizer, 52 of the 53 reconstructed eval strings occur
verbatim inside retained training blocks. The Minnesota prompt also includes
flattened section headings, so treat that item as a formatting probe rather
than a strictly verbatim training sentence.

Reference: https://huggingface.co/EleutherAI/pythia-70m

  python alien_ink/zdeck/sft_pythia-70m_geo_sentence_memorize_mist.py
"""

from __future__ import annotations

from alien_ink.hf.ds import HubTextSource, PretrainDataConfig
from alien_ink.hf.manifest import (
    HardwareConfig,
    Manifest,
    ScheduleConfig,
    WandbConfig,
)
from alien_ink.hf.model import PretrainedLmConfig

MANIFEST = Manifest(
    run_name="sft-pythia-70m-geo-sentence-memorize-mist",
    title="Pythia-70M memorization of sentences from the full geo corpus",
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
        # Exact prompt/completion splits from sentences in the training corpus.
        completion_eval_path="/tmp/population-exact/geo-us-states.json",
        max_eval_samples=53,
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
        name="sft-pythia-70m-geo-sentence-memorize-mist",
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
    ),
    trainer_overrides={"adam_beta2": 0.999},
)


def main() -> None:
    MANIFEST.train()


if __name__ == "__main__":
    main()
