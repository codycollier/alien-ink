"""Tests for tokenize / chunk packing."""

from __future__ import annotations

import pytest

datasets = pytest.importorskip("datasets")
transformers = pytest.importorskip("transformers")

from datasets import Dataset  # noqa: E402

from alien_ink.hf.tok import chunk_into_blocks, column_names, tokenize_and_chunk  # noqa: E402


class _FakeTok:
    eos_token_id = 99

    def __call__(self, texts, *, add_special_tokens=True):
        assert add_special_tokens is False
        # Variable-length token ids from character ordinals (stable, tiny).
        input_ids = [[ord(ch) % 50 for ch in text] for text in texts]
        attention_mask = [[1] * len(ids) for ids in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_column_names_from_map_dataset():
    ds = Dataset.from_list([{"text": "hello"}, {"text": "world"}])
    assert column_names(ds) == ["text"]


def test_chunk_into_blocks_respects_document_boundaries():
    # Two 10-token docs → one 8-token block each; no cross-doc concat.
    tokenized = Dataset.from_list(
        [
            {"input_ids": list(range(10)), "attention_mask": [1] * 10},
            {"input_ids": list(range(10, 20)), "attention_mask": [1] * 10},
        ]
    )
    blocks = chunk_into_blocks(tokenized, block_size=8, num_proc=None)
    assert len(blocks) == 2
    assert blocks[0]["input_ids"] == list(range(8))
    assert blocks[1]["input_ids"] == list(range(10, 18))
    assert blocks[0]["labels"] == blocks[0]["input_ids"]
    assert len(blocks[0]["input_ids"]) == 8


def test_chunk_into_blocks_drops_short_documents():
    tokenized = Dataset.from_list(
        [
            {"input_ids": list(range(5)), "attention_mask": [1] * 5},
            {"input_ids": list(range(20, 40)), "attention_mask": [1] * 20},
        ]
    )
    blocks = chunk_into_blocks(tokenized, block_size=8, num_proc=None)
    # Short doc dropped; long doc yields two full blocks.
    assert len(blocks) == 2
    assert blocks[0]["input_ids"] == list(range(20, 28))
    assert blocks[1]["input_ids"] == list(range(28, 36))


def test_chunk_into_blocks_packs_across_documents_when_disabled():
    tokenized = Dataset.from_list(
        [
            {"input_ids": list(range(10)), "attention_mask": [1] * 10},
            {"input_ids": list(range(10, 20)), "attention_mask": [1] * 10},
        ]
    )
    blocks = chunk_into_blocks(
        tokenized,
        block_size=8,
        respect_document_boundaries=False,
        num_proc=None,
    )
    # 20 tokens packed → two full blocks; remainder of 4 dropped.
    assert len(blocks) == 2
    assert blocks[0]["input_ids"] == list(range(8))
    assert blocks[1]["input_ids"] == list(range(8, 16))
    assert blocks[0]["labels"] == blocks[0]["input_ids"]


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


def test_packed_documents_have_eos_and_carry_remainders_between_batches():
    ds = Dataset.from_list([{"text": "abc"}, {"text": "def"}])
    out = tokenize_and_chunk(
        ds,
        _FakeTok(),
        block_size=8,
        respect_document_boundaries=False,
        num_proc=None,
        map_batch_size=1,
    )
    assert len(out) == 1
    assert out[0]["input_ids"] == [
        ord("a") % 50,
        ord("b") % 50,
        ord("c") % 50,
        99,
        ord("d") % 50,
        ord("e") % 50,
        ord("f") % 50,
        99,
    ]


@pytest.mark.parametrize("document_bounded", [False, True])
def test_stream_packing_has_independent_iterators(document_bounded):
    from datasets import IterableDataset

    source = Dataset.from_list(
        [
            {"input_ids": list(range(6)), "attention_mask": [1] * 6},
            {"input_ids": list(range(6, 12)), "attention_mask": [1] * 6},
        ]
    ).to_iterable_dataset()
    assert isinstance(source, IterableDataset)
    blocks = chunk_into_blocks(
        source, block_size=4, respect_document_boundaries=document_bounded
    )
    expected = list(blocks)
    assert len(expected) == (2 if document_bounded else 3)
    assert list(blocks) == expected
    first, second = iter(blocks), iter(blocks)
    assert next(first) == expected[0]
    assert next(first) == expected[1]
    assert next(second) == expected[0]
    assert list(blocks) == expected
    assert list(first) == expected[2:]
    assert list(second) == expected[1:]
    for row in expected:
        assert row["labels"] == row["input_ids"]
        assert row["attention_mask"] == [1] * 4


def test_stream_packing_propagates_epochs_and_shards():
    source = (
        Dataset.from_list([{"input_ids": [i] * 4} for i in range(40)])
        .to_iterable_dataset(num_shards=2)
        .shuffle(seed=19, buffer_size=10)
    )
    blocks = chunk_into_blocks(source, block_size=4, respect_document_boundaries=False)
    assert blocks.num_shards == source.num_shards
    first = list(blocks)
    blocks.set_epoch(1)
    second = list(blocks)
    assert first != second
    assert sorted(row["input_ids"] for row in first) == sorted(
        row["input_ids"] for row in second
    )
    assert list(blocks) == second


@pytest.mark.parametrize("num_proc", [None, 2])
def test_materialized_packing_carries_across_rows_per_shard(num_proc):
    source = Dataset.from_list(
        [{"input_ids": [i] * 2, "attention_mask": [1] * 2} for i in range(8)]
    )
    blocks = chunk_into_blocks(
        source,
        block_size=4,
        respect_document_boundaries=False,
        num_proc=num_proc,
        map_batch_size=1,
    )
    assert blocks["input_ids"] == [
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [4, 4, 5, 5],
        [6, 6, 7, 7],
    ]
    assert blocks["labels"] == blocks["input_ids"]


def test_stream_packing_resume_and_wrapped_epochs():
    source = (
        Dataset.from_list([{"input_ids": [i, i + 1, i + 2]} for i in range(40)])
        .to_iterable_dataset()
        .shuffle(seed=7, buffer_size=10)
    )
    blocks = chunk_into_blocks(source, block_size=4, respect_document_boundaries=False)
    expected = list(blocks)
    iterator = iter(blocks)
    assert next(iterator) == expected[0]
    state = blocks.state_dict()
    blocks.load_state_dict(state)
    assert list(blocks) == expected[1:]
    wrapped = chunk_into_blocks(
        source, block_size=4, respect_document_boundaries=False
    ).map(lambda row: row)
    first = list(wrapped)
    wrapped.set_epoch(1)
    second = list(wrapped)
    assert first != second
    assert list(wrapped) == second


def test_stream_packing_keeps_native_shard_partition():
    source = Dataset.from_list(
        [{"input_ids": [i] * 4} for i in range(8)]
    ).to_iterable_dataset(num_shards=2)
    blocks = chunk_into_blocks(source, block_size=4, respect_document_boundaries=False)
    assert blocks.num_shards == 2
    left = list(blocks.shard(num_shards=2, index=0))
    right = list(blocks.shard(num_shards=2, index=1))
    assert left + right == list(blocks)


@pytest.mark.parametrize("rows", [[], [[1, 2]]])
def test_materialized_packing_with_no_full_blocks(rows):
    blocks = chunk_into_blocks(
        Dataset.from_dict({"input_ids": rows}), block_size=4, num_proc=None
    )
    assert len(blocks) == 0
    assert blocks.column_names == ["input_ids", "labels"]
