"""Tests for checkpoint path resolution and model arch configs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("transformers")

from alien_ink.hf.model import (  # noqa: E402
    MODEL_BUILDERS,
    ModelArchConfig,
    find_checkpoint_path,
    register_model_family,
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
        ModelArchConfig(n_embd=100, n_head=12).validate()


def test_arch_validate_ok_for_supported_families():
    for family in ("gpt2", "gpt_neox", "gemma"):
        ModelArchConfig(family=family).validate()
        assert family in MODEL_BUILDERS


def test_arch_validate_rejects_unknown_family():
    with pytest.raises(ValueError, match="unknown model family"):
        ModelArchConfig(family="nope").validate()


def test_register_model_family_extends_builders():
    def _fake_builder(tokenizer, arch):
        del tokenizer, arch
        return object()

    register_model_family(
        "toy",
        builder=_fake_builder,
        default_tokenizer="gpt2",
    )
    try:
        ModelArchConfig(family="toy").validate()
        assert "toy" in MODEL_BUILDERS
    finally:
        MODEL_BUILDERS.pop("toy", None)
