"""Tokenize and pack Hugging Face text datasets into fixed-length LM blocks."""

from __future__ import annotations

from typing import Any

from datasets import IterableDataset


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


def _chunk_document(values: list, block_size: int) -> list:
    """Slice one document's token list into full ``block_size`` blocks."""
    total = (len(values) // block_size) * block_size
    if total < block_size:
        return []
    return [values[i : i + block_size] for i in range(0, total, block_size)]


def chunk_into_blocks(
    tokenized,
    *,
    block_size: int,
    respect_document_boundaries: bool = True,
    num_proc: int | None = 4,
    map_batch_size: int = 1000,
):
    """Split token streams into ``block_size`` chunks with labels.

    When ``respect_document_boundaries`` is True (default), each Hub row is
    chunked independently — a training block never spans two documents.
    Per-document remainders shorter than ``block_size`` are dropped.

    When False, token streams are concatenated across rows in each map batch
    before slicing (classic packing). Prefer this for short-row corpora such
    as WikiText where most rows are shorter than ``block_size``.
    """
    is_streaming = isinstance(tokenized, IterableDataset)

    remainder: dict[str, list] = {}

    def group_texts(examples):
        if respect_document_boundaries:
            chunks = {key: [] for key in examples}
            for i in range(len(examples["input_ids"])):
                for key, values_list in examples.items():
                    chunks[key].extend(_chunk_document(values_list[i], block_size))
            if not chunks["input_ids"]:
                return {key: [] for key in examples}
            chunks["labels"] = [ids[:] for ids in chunks["input_ids"]]
            return chunks

        concatenated = {
            key: remainder.get(key, []) + sum(examples[key], [])
            for key in examples
        }
        total_length = len(concatenated["input_ids"])
        if total_length < block_size:
            remainder.clear()
            remainder.update(concatenated)
            return {key: [] for key in concatenated}
        packed_length = (total_length // block_size) * block_size
        chunks = {
            key: [
                values[i : i + block_size]
                for i in range(0, packed_length, block_size)
            ]
            for key, values in concatenated.items()
        }
        remainder.clear()
        remainder.update(
            {key: values[packed_length:] for key, values in concatenated.items()}
        )
        chunks["labels"] = chunks["input_ids"].copy()
        return chunks

    kwargs: dict[str, Any] = {"batched": True, "batch_size": map_batch_size}
    if not is_streaming and num_proc is not None:
        boundary = "doc-bounded" if respect_document_boundaries else "packed"
        kwargs.update(
            num_proc=num_proc,
            desc=f"Chunking into {block_size}-token blocks ({boundary})",
        )
    return tokenized.map(group_texts, **kwargs)


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
