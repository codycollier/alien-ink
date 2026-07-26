"""Tests for prompt extraction helpers."""

from __future__ import annotations

import pytest

datasets = pytest.importorskip("datasets")

from alien_ink.hf.ds import PretrainDataConfig, HubTextSource, text_to_prompt  # noqa: E402


def test_text_to_prompt_uses_sentence_boundary():
    text = "Hello world. More text that continues well past the soft limit here."
    assert text_to_prompt(text) == "Hello world."


def test_text_to_prompt_rejects_short_text():
    assert text_to_prompt("too short") is None


def test_text_to_prompt_hard_limit_fallback():
    text = "a" * 200
    prompt = text_to_prompt(text, soft_limit=50, hard_limit=40)
    assert prompt == "a" * 40


def test_pretrain_data_config_validate():
    cfg = PretrainDataConfig(source=HubTextSource(dataset="Salesforce/wikitext"))
    cfg.validate()
    with pytest.raises(ValueError, match="block_size"):
        PretrainDataConfig(
            source=HubTextSource(dataset="Salesforce/wikitext"),
            block_size=0,
        ).validate()
    with pytest.raises(ValueError, match="max_train_samples"):
        PretrainDataConfig(
            source=HubTextSource(dataset="Salesforce/wikitext"),
            max_train_samples=0,
        ).validate()
