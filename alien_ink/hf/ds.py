"""Hugging Face dataset helpers — streaming text corpora and prompt extraction."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from datasets import Dataset, IterableDataset, load_dataset


@dataclass(frozen=True)
class HubTextSource:
    """Identity of a text corpus on the Hugging Face Hub."""

    dataset: str
    name: str | None = None
    split: str = "train"
    text_column: str = "text"


@dataclass(frozen=True)
class PretrainDataConfig:
    """How to stream a Hub text corpus and pack it for causal LM pretraining.

    If ``eval_source`` is set, eval rows are taken from that split (capped by
    ``max_eval_samples``). Otherwise the first ``max_eval_samples`` rows of the
    train stream are held out so train/eval do not overlap.
    """

    source: HubTextSource
    eval_source: HubTextSource | None = None
    max_eval_samples: int = 1_000
    stream_shuffle_buffer: int = 10_000
    block_size: int = 1024
    tokenizer_num_proc: int = 4
    seed: int = 101

    def validate(self) -> None:
        if self.max_eval_samples < 1:
            raise ValueError(
                f"max_eval_samples must be >= 1, got {self.max_eval_samples}"
            )
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.stream_shuffle_buffer < 1:
            raise ValueError(
                "stream_shuffle_buffer must be >= 1, "
                f"got {self.stream_shuffle_buffer}"
            )
        if self.tokenizer_num_proc < 1:
            raise ValueError(
                f"tokenizer_num_proc must be >= 1, got {self.tokenizer_num_proc}"
            )


def wikipedia_english(
    *,
    name: str = "20231101.en",
    max_eval_samples: int = 1_000,
) -> PretrainDataConfig:
    """English Wikipedia dump (single train split; eval via hold-out prefix)."""
    return PretrainDataConfig(
        source=HubTextSource(dataset="wikimedia/wikipedia", name=name),
        max_eval_samples=max_eval_samples,
    )


def wikitext_103(
    *,
    name: str = "wikitext-103-v1",
    max_eval_samples: int = 1_000,
) -> PretrainDataConfig:
    """WikiText-103 (train + validation splits)."""
    return PretrainDataConfig(
        source=HubTextSource(dataset="Salesforce/wikitext", name=name, split="train"),
        eval_source=HubTextSource(
            dataset="Salesforce/wikitext", name=name, split="validation"
        ),
        max_eval_samples=max_eval_samples,
    )


def c4_english(
    *,
    name: str = "en",
    max_eval_samples: int = 1_000,
) -> PretrainDataConfig:
    """Colossal Clean Crawled Corpus, English (train + validation splits)."""
    return PretrainDataConfig(
        source=HubTextSource(dataset="allenai/c4", name=name, split="train"),
        eval_source=HubTextSource(dataset="allenai/c4", name=name, split="validation"),
        max_eval_samples=max_eval_samples,
    )


def stream_hub_text(
    source: HubTextSource,
    *,
    trust_remote_code: bool = False,
) -> IterableDataset:
    """Open a streaming view of a Hub text split (no materialization)."""
    kwargs: dict = {
        "split": source.split,
        "streaming": True,
        "trust_remote_code": trust_remote_code,
    }
    if source.name is not None:
        return load_dataset(source.dataset, source.name, **kwargs)
    return load_dataset(source.dataset, **kwargs)


def materialize_prefix(stream: IterableDataset, n: int) -> Dataset:
    """Take the first ``n`` rows from a stream into a map-style Dataset."""
    rows = list(itertools.islice(stream, n))
    return Dataset.from_list(rows)


def skip_and_shuffle(
    stream: IterableDataset,
    *,
    skip: int,
    seed: int,
    buffer_size: int,
) -> IterableDataset:
    """Skip held-out rows, then shuffle with a bounded buffer."""
    return stream.skip(skip).shuffle(seed=seed, buffer_size=buffer_size)


def _source_label(source: HubTextSource) -> str:
    if source.name:
        return f"{source.dataset} ({source.name})"
    return source.dataset


def load_streaming_train_eval(
    data: PretrainDataConfig,
    *,
    trust_remote_code: bool = False,
    verbose: bool = True,
) -> tuple[IterableDataset, Dataset]:
    """Stream training text; materialize a bounded eval set.

    Large corpora stay streamed for training. Eval is always a map-style
    Dataset of at most ``max_eval_samples`` rows, either from ``eval_source``
    or from a held-out prefix of the train stream.
    """
    source = data.source
    if verbose:
        print(f">> Streaming {_source_label(source)} [{source.split}]...")

    train_stream = stream_hub_text(source, trust_remote_code=trust_remote_code)

    if data.eval_source is not None:
        eval_source = data.eval_source
        if verbose:
            print(
                f">> Materializing up to {data.max_eval_samples} eval rows "
                f"from {_source_label(eval_source)} [{eval_source.split}]..."
            )
        eval_stream = stream_hub_text(eval_source, trust_remote_code=trust_remote_code)
        eval_dataset = materialize_prefix(eval_stream, data.max_eval_samples)
        train_stream = train_stream.shuffle(
            seed=data.seed,
            buffer_size=data.stream_shuffle_buffer,
        )
        if verbose:
            print(f"   eval examples:  {len(eval_dataset):,} ({eval_source.split})")
            print(f"   train examples: {_source_label(source)} (streaming)")
        return train_stream, eval_dataset

    if verbose:
        print(f">> Materializing {data.max_eval_samples} held-out eval rows...")
    eval_dataset = materialize_prefix(train_stream, data.max_eval_samples)
    train_stream = skip_and_shuffle(
        train_stream,
        skip=data.max_eval_samples,
        seed=data.seed,
        buffer_size=data.stream_shuffle_buffer,
    )
    if verbose:
        print(f"   eval examples:  {len(eval_dataset):,} (held out)")
        print(f"   train examples: {_source_label(source)} (streaming)")
    return train_stream, eval_dataset


def text_to_prompt(text: str, *, soft_limit: int = 200, hard_limit: int = 120) -> str | None:
    """Turn a document into a short prompt-like prefix, or None if too short."""
    text = text.strip().replace("\n", " ")
    if len(text) < 40:
        return None

    for sep in (". ", ".\n", "! ", "? "):
        if sep in text[:soft_limit]:
            return text.split(sep, 1)[0] + sep.strip()
    return text[:hard_limit]


def load_text_prompts(
    source: HubTextSource,
    *,
    count: int,
    seed: int,
    trust_remote_code: bool = False,
) -> list[str]:
    """Sample short prompt strings from a streamed Hub text corpus."""
    start = seed * 17
    stream = stream_hub_text(source, trust_remote_code=trust_remote_code)
    rows = itertools.islice(stream, start, start + count * 3)

    prompts: list[str] = []
    for row in rows:
        prompt = text_to_prompt(row[source.text_column])
        if prompt and prompt not in prompts:
            prompts.append(prompt)
        if len(prompts) >= count:
            break
    return prompts
