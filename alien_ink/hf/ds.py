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


# Default size for non-streamed "subset" corpora (train prefix + eval cap).
DEFAULT_SUBSET_TRAIN_SAMPLES = 20_000
DEFAULT_SUBSET_EVAL_SAMPLES = 1_000


@dataclass(frozen=True)
class PretrainDataConfig:
    """How to load a Hub text corpus and pack it for causal LM pretraining.

    If ``eval_source`` is set, eval rows are taken from that split (capped by
    ``max_eval_samples``). Otherwise the first ``max_eval_samples`` rows of the
    train stream are held out so train/eval do not overlap.

    When ``max_train_samples`` is ``None``, training stays streamed. When set,
    the train prefix is materialized into a map-style Dataset (after any
    hold-out), which is useful for small local subsets.
    """

    source: HubTextSource
    eval_source: HubTextSource | None = None
    max_eval_samples: int = 1_000
    max_train_samples: int | None = None
    stream_shuffle_buffer: int = 10_000
    block_size: int = 1024
    tokenizer_num_proc: int = 4
    seed: int = 101

    def validate(self) -> None:
        if self.max_eval_samples < 1:
            raise ValueError(
                f"max_eval_samples must be >= 1, got {self.max_eval_samples}"
            )
        if self.max_train_samples is not None and self.max_train_samples < 1:
            raise ValueError(
                f"max_train_samples must be >= 1 when set, "
                f"got {self.max_train_samples}"
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
    max_train_samples: int | None = None,
) -> PretrainDataConfig:
    """English Wikipedia dump (single train split; eval via hold-out prefix)."""
    return PretrainDataConfig(
        source=HubTextSource(dataset="wikimedia/wikipedia", name=name),
        max_eval_samples=max_eval_samples,
        max_train_samples=max_train_samples,
    )


def wikipedia_english_subset(
    *,
    name: str = "20231101.en",
    max_train_samples: int = DEFAULT_SUBSET_TRAIN_SAMPLES,
    max_eval_samples: int = DEFAULT_SUBSET_EVAL_SAMPLES,
) -> PretrainDataConfig:
    """Materialized English Wikipedia prefix (default 20k train + 1k hold-out)."""
    return wikipedia_english(
        name=name,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )


def wikitext_103(
    *,
    name: str = "wikitext-103-v1",
    max_eval_samples: int = 1_000,
    max_train_samples: int | None = None,
) -> PretrainDataConfig:
    """WikiText-103 (train + validation splits)."""
    return PretrainDataConfig(
        source=HubTextSource(dataset="Salesforce/wikitext", name=name, split="train"),
        eval_source=HubTextSource(
            dataset="Salesforce/wikitext", name=name, split="validation"
        ),
        max_eval_samples=max_eval_samples,
        max_train_samples=max_train_samples,
    )


def wikitext_103_subset(
    *,
    name: str = "wikitext-103-v1",
    max_train_samples: int = DEFAULT_SUBSET_TRAIN_SAMPLES,
    max_eval_samples: int = DEFAULT_SUBSET_EVAL_SAMPLES,
) -> PretrainDataConfig:
    """Materialized WikiText-103 prefix (default 20k train + 1k validation)."""
    return wikitext_103(
        name=name,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )


def c4_english(
    *,
    name: str = "en",
    max_eval_samples: int = 1_000,
    max_train_samples: int | None = None,
) -> PretrainDataConfig:
    """Colossal Clean Crawled Corpus, English (train + validation splits)."""
    return PretrainDataConfig(
        source=HubTextSource(dataset="allenai/c4", name=name, split="train"),
        eval_source=HubTextSource(dataset="allenai/c4", name=name, split="validation"),
        max_eval_samples=max_eval_samples,
        max_train_samples=max_train_samples,
    )


def c4_english_subset(
    *,
    name: str = "en",
    max_train_samples: int = DEFAULT_SUBSET_TRAIN_SAMPLES,
    max_eval_samples: int = DEFAULT_SUBSET_EVAL_SAMPLES,
) -> PretrainDataConfig:
    """Materialized C4 English prefix (default 20k train + 1k validation)."""
    return c4_english(
        name=name,
        max_train_samples=max_train_samples,
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

    Prefer ``load_train_eval`` unless you specifically need the streaming path.
    """
    if data.max_train_samples is not None:
        raise ValueError(
            "load_streaming_train_eval requires max_train_samples=None; "
            "use load_materialized_train_eval or load_train_eval instead"
        )

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


def load_materialized_train_eval(
    data: PretrainDataConfig,
    *,
    trust_remote_code: bool = False,
    verbose: bool = True,
) -> tuple[Dataset, Dataset]:
    """Materialize bounded train and eval map-style Datasets from Hub streams.

    Requires ``max_train_samples``. Eval follows the same rules as the streaming
    loader (dedicated ``eval_source`` or hold-out prefix). Train is the next
    ``max_train_samples`` rows after any hold-out, then shuffled in memory.
    """
    if data.max_train_samples is None:
        raise ValueError(
            "load_materialized_train_eval requires max_train_samples; "
            "use load_streaming_train_eval or load_train_eval instead"
        )

    source = data.source
    n_train = data.max_train_samples
    if verbose:
        print(
            f">> Materializing up to {n_train:,} train rows from "
            f"{_source_label(source)} [{source.split}]..."
        )

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
        train_dataset = materialize_prefix(train_stream, n_train)
        eval_label = eval_source.split
    else:
        if verbose:
            print(f">> Materializing {data.max_eval_samples} held-out eval rows...")
        eval_dataset = materialize_prefix(train_stream, data.max_eval_samples)
        train_stream = train_stream.skip(data.max_eval_samples)
        train_dataset = materialize_prefix(train_stream, n_train)
        eval_label = "held out"

    train_dataset = train_dataset.shuffle(seed=data.seed)
    if verbose:
        print(f"   eval examples:  {len(eval_dataset):,} ({eval_label})")
        print(f"   train examples: {len(train_dataset):,} (materialized)")
    return train_dataset, eval_dataset


def load_train_eval(
    data: PretrainDataConfig,
    *,
    trust_remote_code: bool = False,
    verbose: bool = True,
) -> tuple[Dataset | IterableDataset, Dataset]:
    """Load train/eval for pretraining — streamed or materialized.

    Dispatches on ``max_train_samples``: ``None`` keeps train as an
    ``IterableDataset``; an integer materializes a finite map-style train set.
    Eval is always a bounded map-style Dataset.
    """
    data.validate()
    if data.max_train_samples is not None:
        return load_materialized_train_eval(
            data,
            trust_remote_code=trust_remote_code,
            verbose=verbose,
        )
    return load_streaming_train_eval(
        data,
        trust_remote_code=trust_remote_code,
        verbose=verbose,
    )


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
