"""Tests for tokenize / chunk packing."""

from __future__ import annotations

import pytest

datasets = pytest.importorskip("datasets")
transformers = pytest.importorskip("transformers")

from datasets import Dataset  # noqa: E402

from alien_ink.hf.tok import chunk_into_blocks, column_names, tokenize_and_chunk  # noqa: E402


class _FakeTok:
    def __call__(self, texts):
        # Variable-length token ids from character ordinals (stable, tiny).
        input_ids = [[ord(ch) % 50 for ch in text] for text in texts]
        attention_mask = [[1] * len(ids) for ids in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_column_names_from_map_dataset():
    ds = Dataset.from_list([{"text": "hello"}, {"text": "world"}])
    assert column_names(ds) == ["text"]


def test_chunk_into_blocks_packs_and_labels():
    tokenized = Dataset.from_list(
        [
            {"input_ids": list(range(10)), "attention_mask": [1] * 10},
            {"input_ids": list(range(10, 20)), "attention_mask": [1] * 10},
        ]
    )
    blocks = chunk_into_blocks(tokenized, block_size=8, num_proc=None)
    assert len(blocks) == 2
    assert blocks[0]["labels"] == blocks[0]["input_ids"]
    assert len(blocks[0]["input_ids"]) == 8


def test_tokenize_and_chunk_end_to_end():
    ds = Dataset.from_list([{"text": "abcdefghijklmnop"}] * 4)
    out = tokenize_and_chunk(
        ds,
        _FakeTok(),
        block_size=8,
        num_proc=None,
    )
    assert len(out) >= 1
    assert "labels" in out.column_names
