"""Tokenize and pack Hugging Face text datasets into fixed-length LM blocks."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, Features, IterableDataset
from datasets.iterable_dataset import ExamplesIterable


def column_names(dataset) -> list[str]:
    """Return dataset column names, peeking the first row if needed.

    Streaming datasets don't always expose ``column_names`` (e.g. after shuffle),
    so fall back to peeking the first row. Peeking is safe: iterating an
    IterableDataset returns a fresh iterator and doesn't consume the dataset.
    """
    names = dataset.column_names
    if names:
        return list(names)
    return list(next(iter(dataset)).keys())


def tokenize_text(
    dataset,
    tokenizer,
    *,
    text_column: str = "text",
    num_proc: int | None = 4,
):
    """Tokenize documents with one explicit EOS separator; drop source columns."""
    is_streaming = isinstance(dataset, IterableDataset)
    cols = column_names(dataset)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("causal-LM packing requires tokenizer.eos_token_id")

    def tokenize_function(examples):
        encoded = tokenizer(examples[text_column], add_special_tokens=False)
        for index, input_ids in enumerate(encoded["input_ids"]):
            input_ids.append(eos_token_id)
            if "attention_mask" in encoded:
                encoded["attention_mask"][index].append(1)
            for key, values in encoded.items():
                if key in {"input_ids", "attention_mask"}:
                    continue
                # Keep any tokenizer-specific per-token fields aligned.
                values[index].append(values[index][-1] if values[index] else 0)
        return encoded

    kwargs: dict[str, Any] = {"batched": True, "remove_columns": cols}
    # Streaming datasets are processed lazily and cannot use multiprocessing
    # (num_proc) or a progress ``desc``.
    if not is_streaming and num_proc is not None:
        kwargs.update(num_proc=num_proc, desc="Tokenizing")
    return dataset.map(tokenize_function, **kwargs)


def _pack_rows(rows, *, block_size: int, respect_document_boundaries: bool):
    """Pack one iterator; its incomplete final block is discarded."""
    remainder: dict[str, list[int]] = {}
    for row in rows:
        if respect_document_boundaries:
            remainder = {}
        for key, values in row.items():
            if key != "labels":
                remainder.setdefault(key, []).extend(values)
        packed_length = len(remainder["input_ids"]) // block_size * block_size
        for offset in range(0, packed_length, block_size):
            block = {
                key: values[offset : offset + block_size]
                for key, values in remainder.items()
            }
            block["labels"] = block["input_ids"][:]
            yield block
        remainder = {key: values[packed_length:] for key, values in remainder.items()}


def _pack_sources(sources, *, block_size, respect_document_boundaries):
    for source in sources:
        yield from _pack_rows(
            iter(source),
            block_size=block_size,
            respect_document_boundaries=respect_document_boundaries,
        )


def _pack_examples(source, *, block_size, respect_document_boundaries):
    source._init_state_dict()
    rows = (row for _, row in source)
    yield from enumerate(
        _pack_rows(
            rows,
            block_size=block_size,
            respect_document_boundaries=respect_document_boundaries,
        )
    )


def _fixed_examples(source):
    source._init_state_dict()
    yield from source


class _FixedExamplesIterable(ExamplesIterable):
    """Keep partition RNGs fixed while retaining native shard assignment."""

    def __init__(self, source):
        super().__init__(_fixed_examples, {"source": source})

    @property
    def num_shards(self):
        return self.kwargs["source"].num_shards

    def shuffle_data_sources(self, generator):
        return self

    def shard_data_sources(self, num_shards, index, contiguous=True):
        return type(self)(self.kwargs["source"].shard_data_sources(
            num_shards, index, contiguous=contiguous
        ))


def _has_fixed_partition(source):
    if getattr(source, "block_sources_order_when_shuffling", False):
        return True
    child = getattr(source, "ex_iterable", None)
    return child is not None and _has_fixed_partition(child)


class _PackedExamplesIterable(ExamplesIterable):
    """HF adapter that preserves source shuffling, sharding, and epoch handling.

    ExamplesIterable supplies checkpoint replay; packing state lives solely in
    the generator. Keep the dependency on HF iterable internals in this adapter.
    """

    def __init__(self, source, *, block_size, respect_document_boundaries):
        if _has_fixed_partition(source):
            source = _FixedExamplesIterable(source)
        super().__init__(
            _pack_examples,
            dict(
                source=source,
                block_size=block_size,
                respect_document_boundaries=respect_document_boundaries,
            ),
        )

    @property
    def ex_iterable(self):
        return self.kwargs["source"]

    @ex_iterable.setter
    def ex_iterable(self, source):
        self.kwargs["source"] = source

    def _with_source(self, source):
        return type(self)(
            source, **{k: v for k, v in self.kwargs.items() if k != "source"}
        )

    def shuffle_data_sources(self, generator):
        return self._with_source(self.kwargs["source"].shuffle_data_sources(generator))

    def shard_data_sources(self, num_shards, index, contiguous=True):
        return self._with_source(
            self.kwargs["source"].shard_data_sources(
                num_shards,
                index,
                contiguous=contiguous,
            )
        )

    def reshard_data_sources(self):
        return self._with_source(self.ex_iterable.reshard_data_sources())

    @property
    def num_shards(self):
        return self.kwargs["source"].num_shards


def chunk_into_blocks(
    tokenized,
    *,
    block_size: int,
    respect_document_boundaries: bool = True,
    num_proc: int | None = 4,
    map_batch_size: int = 1000,
):
    """Pack tokenized rows into causal-LM blocks with iterator-local state.

    Document-bounded packing drops each document's incomplete tail. Otherwise
    tails carry across rows until the iterator (or processing shard) ends.
    ``map_batch_size`` is retained for compatibility; packing reads rows lazily.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if num_proc is not None and num_proc < 1:
        raise ValueError("num_proc must be >= 1 when set")
    is_streaming = isinstance(tokenized, IterableDataset)
    features = tokenized.features
    if features is not None:
        features = Features({**features, "labels": features["input_ids"]})
    options = dict(
        block_size=block_size,
        respect_document_boundaries=respect_document_boundaries,
    )
    if is_streaming:
        info = tokenized.info.copy()
        info.features = features
        return IterableDataset(
            _PackedExamplesIterable(tokenized._ex_iterable, **options),
            info=info,
            split=tokenized.split,
            distributed=tokenized._distributed,
            token_per_repo_id=tokenized._token_per_repo_id,
        )
    count = min(num_proc or 1, max(1, len(tokenized)))
    sources = [tokenized.shard(num_shards=count, index=i) for i in range(count)]
    kwargs = dict(sources=sources, **options)
    # HF's generator reader rejects an empty split, even with a schema.
    if next(_pack_sources(**kwargs), None) is None:
        return Dataset.from_dict(
            {key: [] for key in [*tokenized.column_names, "labels"]},
            features=features,
        )
    return Dataset.from_generator(
        _pack_sources,
        gen_kwargs=kwargs,
        features=features,
        num_proc=num_proc,
    )


def tokenize_and_chunk(
    dataset,
    tokenizer,
    *,
    block_size: int,
    text_column: str = "text",
    respect_document_boundaries: bool = True,
    num_proc: int | None = 4,
    map_batch_size: int = 1000,
):
    """Tokenize text then pack into fixed-length causal-LM blocks."""
    tokenized = tokenize_text(
        dataset,
        tokenizer,
        text_column=text_column,
        num_proc=num_proc,
    )
    return chunk_into_blocks(
        tokenized,
        block_size=block_size,
        respect_document_boundaries=respect_document_boundaries,
        num_proc=num_proc,
        map_batch_size=map_batch_size,
    )
