"""Tests for family-aware generation config."""

from __future__ import annotations

import pytest

from alien_ink.hf.gen import gen_config_for_family


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
