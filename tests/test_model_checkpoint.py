"""Tests for checkpoint path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("transformers")

from alien_ink.hf.model import (  # noqa: E402
    CausalLmArchConfig,
    find_checkpoint_path,
    resolve_checkpoint_path,
)


def test_resolve_checkpoint_prefers_final_save(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checkpoint-100").mkdir()
    assert resolve_checkpoint_path(tmp_path) == tmp_path


def test_resolve_checkpoint_picks_latest_step(tmp_path: Path):
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-250").mkdir()
    (tmp_path / "checkpoint-50").mkdir()
    assert resolve_checkpoint_path(tmp_path) == tmp_path / "checkpoint-250"


def test_resolve_checkpoint_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No trained model"):
        resolve_checkpoint_path(tmp_path)


def test_find_checkpoint_path_tries_candidates(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    good = tmp_path / "good"
    good.mkdir()
    (good / "config.json").write_text("{}", encoding="utf-8")
    assert find_checkpoint_path(empty, good) == good


def test_arch_validate_rejects_bad_head_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        CausalLmArchConfig(n_embd=100, n_head=12).validate()


def test_arch_validate_ok():
    CausalLmArchConfig().validate()


def test_arch_families():
    from alien_ink.hf.model import gemma_arch, gpt2_arch, gpt_neox_arch

    assert gpt2_arch().family == "gpt-2"
    assert gpt_neox_arch().family == "gpt-neox"
    gemma = gemma_arch()
    assert gemma.family == "gemma"
    assert gemma.n_embd == 512
    gemma.validate()


def test_gemma_sets_num_key_value_heads(monkeypatch):
    """GemmaConfig defaults kv heads to 16; mist-sized builds must override."""
    from transformers import GemmaConfig

    from alien_ink.hf import model as model_mod
    from alien_ink.hf.model import build_model_from_scratch, gemma_arch

    # Upstream still defaults kv heads independently of attention heads.
    default = GemmaConfig(num_attention_heads=8)
    assert default.num_key_value_heads == 16

    class _FakeTok:
        vocab_size = 256

        def __len__(self) -> int:
            return 256

    class _FakeModel:
        def __init__(self, config):
            self.config = config

    # Avoid instantiating GemmaForCausalLM (needs torch>=2.4 under transformers 5).
    monkeypatch.setattr(model_mod, "GemmaForCausalLM", _FakeModel)
    model = build_model_from_scratch(_FakeTok(), gemma_arch(n_layer=1), verbose=False)
    assert model.config.num_attention_heads == 8
    assert model.config.num_key_value_heads == 8
