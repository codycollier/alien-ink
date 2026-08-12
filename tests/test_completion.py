"""Tests for supervised prompt/completion dataset configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alien_ink.hf.completion import CompletionDataConfig, prepare_completion_datasets


class _Tokenizer:
    def __call__(self, text, *, add_special_tokens=True, return_tensors=None):
        del return_tensors
        prefix = [1] if add_special_tokens else []
        return {"input_ids": prefix + [ord(char) for char in text]}


def _write_pairs(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {"slug": "one", "prompt": "P1", "completion": "C1"},
                {"slug": "two", "prompt": "P2", "completion": "C2"},
            ]
        ),
        encoding="utf-8",
    )


def test_completion_config_validates_and_resolves_eval_path(tmp_path: Path):
    path = tmp_path / "pairs.json"
    _write_pairs(path)
    data = CompletionDataConfig(train_path=str(path))
    data.validate()
    assert data.resolved_eval_path() == str(path)

    with pytest.raises(ValueError, match="train_path does not exist"):
        CompletionDataConfig(train_path=str(tmp_path / "missing.json")).validate()
    with pytest.raises(ValueError, match="max_train_samples"):
        CompletionDataConfig(train_path=str(path), max_train_samples=0).validate()
    with pytest.raises(ValueError, match="max_eval_samples"):
        CompletionDataConfig(train_path=str(path), max_eval_samples=0).validate()
    with pytest.raises(ValueError, match="max_length"):
        CompletionDataConfig(train_path=str(path), max_length=0).validate()


def test_prepare_completion_datasets_uses_all_rows_and_named_train_eval(tmp_path: Path):
    path = tmp_path / "pairs.json"
    _write_pairs(path)
    train, evals = prepare_completion_datasets(
        CompletionDataConfig(train_path=str(path)),
        _Tokenizer(),
        verbose=False,
    )
    assert len(train) == 2
    assert set(evals) == {"completion", "train"}
    assert len(evals["completion"]) == 2
    assert evals["train"] is train
    assert train[0]["labels"][:3] == [-100, -100, -100]


def test_prepare_completion_datasets_limits_and_prompt_loss(tmp_path: Path):
    path = tmp_path / "pairs.json"
    _write_pairs(path)
    train, evals = prepare_completion_datasets(
        CompletionDataConfig(
            train_path=str(path),
            max_train_samples=1,
            max_eval_samples=1,
            loss_on_prompt=True,
        ),
        _Tokenizer(),
        verbose=False,
    )
    assert len(train) == 1
    assert len(evals["completion"]) == 1
    assert -100 not in train[0]["labels"]

    with pytest.raises(ValueError, match="sequence length"):
        prepare_completion_datasets(
            CompletionDataConfig(train_path=str(path), max_length=1),
            _Tokenizer(),
            verbose=False,
        )
