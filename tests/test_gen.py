"""Tests for family-aware generation config."""

from __future__ import annotations

import pytest

from alien_ink.hf.gen import (
    CompletionResult,
    chat_gen_variants,
    gen_config_for_family,
)


def test_gpt2_gen_config_defaults():
    gen = gen_config_for_family("gpt-2")
    assert gen.add_special_tokens is True
    assert gen.do_sample is False
    assert gen.stop_strings == (".", "!", "?")


def test_gemma_gen_config_skips_bos():
    gen = gen_config_for_family("gemma")
    assert gen.add_special_tokens is False
    assert gen.stop_strings == (".", "!", "?")


def test_gpt_neox_gen_config_defaults():
    gen = gen_config_for_family("gpt-neox")
    assert gen.add_special_tokens is True


def test_gen_config_overrides():
    gen = gen_config_for_family("gpt-2", max_new_tokens=64, do_sample=True)
    assert gen.max_new_tokens == 64
    assert gen.do_sample is True
    assert gen.add_special_tokens is True


def test_gen_config_unknown_family():
    with pytest.raises(ValueError, match="unsupported family"):
        gen_config_for_family("llama")  # type: ignore[arg-type]


def test_chat_gen_variants_greedy_then_sampled():
    base = gen_config_for_family("gpt-2", max_new_tokens=40)
    variants = chat_gen_variants(base)
    assert len(variants) == 4
    assert variants[0].do_sample is False
    assert variants[0].temperature == 0.0
    assert [v.temperature for v in variants[1:]] == [0.5, 0.8, 1.2]
    assert all(v.do_sample for v in variants[1:])
    assert all(v.max_new_tokens == 40 for v in variants)
    assert all(v.add_special_tokens is True for v in variants)


def test_completion_result_stats_label():
    greedy = CompletionResult(
        text="Austin.",
        n_tokens=2,
        do_sample=False,
        temperature=0.0,
        top_k=50,
        top_p=0.95,
    )
    sampled = CompletionResult(
        text="Austin, Texas.",
        n_tokens=4,
        do_sample=True,
        temperature=0.5,
        top_k=50,
        top_p=0.95,
    )
    assert greedy.stats_label() == "greedy T=0, 2 tok"
    assert sampled.stats_label() == "T=0.5 top_p=0.95, 4 tok"
