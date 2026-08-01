"""Tests for prompt extraction helpers."""

from __future__ import annotations

import pytest

datasets = pytest.importorskip("datasets")

from alien_ink.hf.ds import text_to_prompt  # noqa: E402


def test_text_to_prompt_uses_sentence_boundary():
    text = "Hello world. More text that continues well past the soft limit here."
    assert text_to_prompt(text) == "Hello world."


def test_text_to_prompt_rejects_short_text():
    assert text_to_prompt("too short") is None


def test_text_to_prompt_hard_limit_fallback():
    text = "a" * 200
    prompt = text_to_prompt(text, soft_limit=50, hard_limit=40)
    assert prompt == "a" * 40
