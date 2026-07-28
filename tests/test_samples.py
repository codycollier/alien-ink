"""Tests for sample training programs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("datasets")
pytest.importorskip("transformers")

from alien_ink.samples.pretrain_wikipedia_5k import (  # noqa: E402
    MAX_STEPS as WIKI_STEPS,
)
from alien_ink.samples.pretrain_wikipedia_5k import build_config as wiki_config
from alien_ink.samples.pretrain_wikitext_50k import (  # noqa: E402
    MAX_STEPS as WT_STEPS,
)
from alien_ink.samples.pretrain_wikitext_50k import build_config as wt_config


def test_wikipedia_5k_sample_config(tmp_path: Path):
    cfg = wiki_config(workdir=tmp_path)
    assert cfg.data.load_mode == "streaming"
    assert cfg.data.source.dataset == "wikimedia/wikipedia"
    assert cfg.arch.family == "gpt2"
    assert cfg.trainer.max_steps == WIKI_STEPS == 5_000
    assert cfg.trainer.per_device_train_batch_size == 2
    assert cfg.trainer.gradient_accumulation_steps == 16
    assert cfg.trainer.output_dir == tmp_path / "output" / "sample-gpt2-wikipedia-5k"
    cfg.validate()


def test_wikitext_50k_sample_config(tmp_path: Path):
    cfg = wt_config(workdir=tmp_path)
    assert cfg.data.load_mode == "streaming"
    assert cfg.data.source.dataset == "Salesforce/wikitext"
    assert cfg.arch.family == "gpt2"
    assert cfg.trainer.max_steps == WT_STEPS == 50_000
    assert cfg.trainer.output_dir == tmp_path / "output" / "sample-gpt2-wikitext-50k"
    cfg.validate()
