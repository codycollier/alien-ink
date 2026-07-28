"""Tests for checkpoint path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("transformers")

from alien_ink.hf.model import (  # noqa: E402
    Gpt2ArchConfig,
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
        Gpt2ArchConfig(n_embd=100, n_head=12).validate()


def test_arch_validate_ok():
    Gpt2ArchConfig().validate()


def test_arch_families():
    from alien_ink.hf.model import gemma_arch, gpt2_arch, gpt_neox_arch

    assert gpt2_arch().family == "gpt2"
    assert gpt_neox_arch().family == "gpt_neox"
    gemma = gemma_arch()
    assert gemma.family == "gemma"
    assert gemma.n_embd == 512
    gemma.validate()
