"""Hugging Face dataset helpers — streaming / materialized text corpora.

Shipped corpora (extend via :func:`register_dataset`):

- ``wikipedia_english`` — wikimedia/wikipedia
- ``wikitext_103`` — Salesforce/wikitext
- ``c4_english`` — allenai/c4

Each supports three load modes:

- ``streaming`` — train stays an IterableDataset
- ``subset`` — materialize a bounded train prefix (``max_train_samples``)
- ``complete`` — materialize the full Hub split (map-style, no streaming)
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from datasets import Dataset, IterableDataset, load_dataset

from alien_ink.log import detail, get_logger, step

log = get_logger("hf.ds")

LoadMode = Literal["streaming", "subset", "complete"]


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
    train stream/split are held out so train/eval do not overlap.

    Load modes:

    - ``streaming``: train stays streamed (``max_train_samples`` must be None)
    - ``subset``: materialize ``max_train_samples`` train rows (required)
    - ``complete``: materialize the full Hub split (non-streaming load)
    """

    source: HubTextSource
    eval_source: HubTextSource | None = None
    load_mode: LoadMode = "streaming"
    max_eval_samples: int = 1_000
    max_train_samples: int | None = None
    stream_shuffle_buffer: int = 10_000
    block_size: int = 1024
    tokenizer_num_proc: int = 4
    seed: int = 101

    def validate(self) -> None:
        if self.load_mode not in ("streaming", "subset", "complete"):
            raise ValueError(
                f"load_mode must be streaming|subset|complete, got {self.load_mode!r}"
            )
        if self.max_eval_samples < 1:
            raise ValueError(
                f"max_eval_samples must be >= 1, got {self.max_eval_samples}"
            )
        if self.load_mode == "streaming" and self.max_train_samples is not None:
            raise ValueError(
                "load_mode='streaming' requires max_train_samples=None; "
                "use load_mode='subset' for a bounded materialization"
            )
        if self.load_mode == "subset" and (
            self.max_train_samples is None or self.max_train_samples < 1
        ):
            raise ValueError(
                "load_mode='subset' requires max_train_samples >= 1, "
                f"got {self.max_train_samples}"
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
    load_mode: LoadMode = "streaming",
    max_eval_samples: int = 1_000,
    max_train_samples: int | None = None,
) -> PretrainDataConfig:
    """English Wikipedia dump (single train split; eval via hold-out prefix)."""
    return PretrainDataConfig(
        source=HubTextSource(dataset="wikimedia/wikipedia", name=name),
        load_mode=load_mode,
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
        load_mode="subset",
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )


def wikipedia_english_complete(
    *,
    name: str = "20231101.en",
    max_eval_samples: int = 1_000,
) -> PretrainDataConfig:
    """Fully materialized English Wikipedia train split (non-streaming)."""
    return wikipedia_english(
        name=name,
        load_mode="complete",
        max_eval_samples=max_eval_samples,
    )


def wikitext_103(
    *,
    name: str = "wikitext-103-v1",
    load_mode: LoadMode = "streaming",
    max_eval_samples: int = 1_000,
    max_train_samples: int | None = None,
) -> PretrainDataConfig:
    """WikiText-103 (train + validation splits)."""
    return PretrainDataConfig(
        source=HubTextSource(dataset="Salesforce/wikitext", name=name, split="train"),
        eval_source=HubTextSource(
            dataset="Salesforce/wikitext", name=name, split="validation"
        ),
        load_mode=load_mode,
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
        load_mode="subset",
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )


def wikitext_103_complete(
    *,
    name: str = "wikitext-103-v1",
    max_eval_samples: int = 1_000,
) -> PretrainDataConfig:
    """Fully materialized WikiText-103 train + validation (non-streaming)."""
    return wikitext_103(
        name=name,
        load_mode="complete",
        max_eval_samples=max_eval_samples,
    )


def c4_english(
    *,
    name: str = "en",
    load_mode: LoadMode = "streaming",
    max_eval_samples: int = 1_000,
    max_train_samples: int | None = None,
) -> PretrainDataConfig:
    """Colossal Clean Crawled Corpus, English (train + validation splits)."""
    return PretrainDataConfig(
        source=HubTextSource(dataset="allenai/c4", name=name, split="train"),
        eval_source=HubTextSource(dataset="allenai/c4", name=name, split="validation"),
        load_mode=load_mode,
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
        load_mode="subset",
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )


def c4_english_complete(
    *,
    name: str = "en",
    max_eval_samples: int = 1_000,
) -> PretrainDataConfig:
    """Fully materialized C4 English train + validation (non-streaming)."""
    return c4_english(
        name=name,
        load_mode="complete",
        max_eval_samples=max_eval_samples,
    )


DatasetFactory = Callable[..., PretrainDataConfig]

DATASET_REGISTRY: dict[str, DatasetFactory] = {
    "wikipedia_english": wikipedia_english,
    "wikipedia_english_subset": wikipedia_english_subset,
    "wikipedia_english_complete": wikipedia_english_complete,
    "wikitext_103": wikitext_103,
    "wikitext_103_subset": wikitext_103_subset,
    "wikitext_103_complete": wikitext_103_complete,
    "c4_english": c4_english,
    "c4_english_subset": c4_english_subset,
    "c4_english_complete": c4_english_complete,
}


def register_dataset(name: str, factory: DatasetFactory) -> None:
    """Register an additional dataset factory under ``name``."""
    DATASET_REGISTRY[name] = factory


def get_dataset(name: str, **kwargs) -> PretrainDataConfig:
    """Build a :class:`PretrainDataConfig` from the registry."""
    try:
        factory = DATASET_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"unknown dataset {name!r}; known: {known}") from exc
    return factory(**kwargs)


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


def load_hub_text(
    source: HubTextSource,
    *,
    trust_remote_code: bool = False,
) -> Dataset:
    """Load a full Hub text split as a map-style Dataset (non-streaming)."""
    kwargs: dict = {
        "split": source.split,
        "streaming": False,
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


def _cap_dataset(dataset: Dataset, n: int) -> Dataset:
    if len(dataset) <= n:
        return dataset
    return dataset.select(range(n))


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
    if data.load_mode != "streaming":
        raise ValueError(
            "load_streaming_train_eval requires load_mode='streaming'; "
            "use load_train_eval instead"
        )
    if data.max_train_samples is not None:
        raise ValueError(
            "load_streaming_train_eval requires max_train_samples=None; "
            "use load_materialized_train_eval or load_train_eval instead"
        )

    source = data.source
    if verbose:
        step(f"Streaming {_source_label(source)} [{source.split}]...", logger=log)

    train_stream = stream_hub_text(source, trust_remote_code=trust_remote_code)

    if data.eval_source is not None:
        eval_source = data.eval_source
        if verbose:
            step(
                f"Materializing up to {data.max_eval_samples} eval rows "
                f"from {_source_label(eval_source)} [{eval_source.split}]...",
                logger=log,
            )
        eval_stream = stream_hub_text(eval_source, trust_remote_code=trust_remote_code)
        eval_dataset = materialize_prefix(eval_stream, data.max_eval_samples)
        train_stream = train_stream.shuffle(
            seed=data.seed,
            buffer_size=data.stream_shuffle_buffer,
        )
        if verbose:
            detail(
                f"eval examples:  {len(eval_dataset):,} ({eval_source.split})",
                logger=log,
            )
            detail(
                f"train examples: {_source_label(source)} (streaming)",
                logger=log,
            )
        return train_stream, eval_dataset

    if verbose:
        step(
            f"Materializing {data.max_eval_samples} held-out eval rows...",
            logger=log,
        )
    eval_dataset = materialize_prefix(train_stream, data.max_eval_samples)
    train_stream = skip_and_shuffle(
        train_stream,
        skip=data.max_eval_samples,
        seed=data.seed,
        buffer_size=data.stream_shuffle_buffer,
    )
    if verbose:
        detail(f"eval examples:  {len(eval_dataset):,} (held out)", logger=log)
        detail(
            f"train examples: {_source_label(source)} (streaming)",
            logger=log,
        )
    return train_stream, eval_dataset


def load_materialized_train_eval(
    data: PretrainDataConfig,
    *,
    trust_remote_code: bool = False,
    verbose: bool = True,
) -> tuple[Dataset, Dataset]:
    """Materialize bounded train and eval map-style Datasets from Hub streams.

    Requires ``load_mode='subset'`` and ``max_train_samples``. Eval follows the
    same rules as the streaming loader (dedicated ``eval_source`` or hold-out
    prefix). Train is the next ``max_train_samples`` rows after any hold-out,
    then shuffled in memory.
    """
    if data.load_mode != "subset":
        raise ValueError(
            "load_materialized_train_eval requires load_mode='subset'; "
            "use load_train_eval instead"
        )
    if data.max_train_samples is None:
        raise ValueError(
            "load_materialized_train_eval requires max_train_samples; "
            "use load_streaming_train_eval or load_train_eval instead"
        )

    source = data.source
    n_train = data.max_train_samples
    if verbose:
        step(
            f"Materializing up to {n_train:,} train rows from "
            f"{_source_label(source)} [{source.split}]...",
            logger=log,
        )

    train_stream = stream_hub_text(source, trust_remote_code=trust_remote_code)

    if data.eval_source is not None:
        eval_source = data.eval_source
        if verbose:
            step(
                f"Materializing up to {data.max_eval_samples} eval rows "
                f"from {_source_label(eval_source)} [{eval_source.split}]...",
                logger=log,
            )
        eval_stream = stream_hub_text(eval_source, trust_remote_code=trust_remote_code)
        eval_dataset = materialize_prefix(eval_stream, data.max_eval_samples)
        train_dataset = materialize_prefix(train_stream, n_train)
        eval_label = eval_source.split
    else:
        if verbose:
            step(
                f"Materializing {data.max_eval_samples} held-out eval rows...",
                logger=log,
            )
        eval_dataset = materialize_prefix(train_stream, data.max_eval_samples)
        train_stream = train_stream.skip(data.max_eval_samples)
        train_dataset = materialize_prefix(train_stream, n_train)
        eval_label = "held out"

    train_dataset = train_dataset.shuffle(seed=data.seed)
    if verbose:
        detail(f"eval examples:  {len(eval_dataset):,} ({eval_label})", logger=log)
        detail(
            f"train examples: {len(train_dataset):,} (materialized subset)",
            logger=log,
        )
    return train_dataset, eval_dataset


def load_complete_train_eval(
    data: PretrainDataConfig,
    *,
    trust_remote_code: bool = False,
    verbose: bool = True,
) -> tuple[Dataset, Dataset]:
    """Materialize full Hub train/eval splits (non-streaming).

    Eval is capped to ``max_eval_samples``. When there is no dedicated
    ``eval_source``, the first ``max_eval_samples`` train rows are held out.
    """
    if data.load_mode != "complete":
        raise ValueError(
            "load_complete_train_eval requires load_mode='complete'; "
            "use load_train_eval instead"
        )

    source = data.source
    if verbose:
        step(
            f"Loading complete {_source_label(source)} [{source.split}] "
            f"(non-streaming)...",
            logger=log,
        )
    train_full = load_hub_text(source, trust_remote_code=trust_remote_code)

    if data.eval_source is not None:
        eval_source = data.eval_source
        if verbose:
            step(
                f"Loading complete {_source_label(eval_source)} "
                f"[{eval_source.split}] (non-streaming)...",
                logger=log,
            )
        eval_full = load_hub_text(eval_source, trust_remote_code=trust_remote_code)
        eval_dataset = _cap_dataset(eval_full, data.max_eval_samples)
        train_dataset = train_full.shuffle(seed=data.seed)
        eval_label = eval_source.split
    else:
        if verbose:
            step(
                f"Holding out {data.max_eval_samples} eval rows from train...",
                logger=log,
            )
        n_eval = min(data.max_eval_samples, len(train_full))
        eval_dataset = train_full.select(range(n_eval))
        train_dataset = train_full.select(range(n_eval, len(train_full))).shuffle(
            seed=data.seed
        )
        eval_label = "held out"

    if verbose:
        detail(f"eval examples:  {len(eval_dataset):,} ({eval_label})", logger=log)
        detail(
            f"train examples: {len(train_dataset):,} (materialized complete)",
            logger=log,
        )
    return train_dataset, eval_dataset


def load_train_eval(
    data: PretrainDataConfig,
    *,
    trust_remote_code: bool = False,
    verbose: bool = True,
) -> tuple[Dataset | IterableDataset, Dataset]:
    """Load train/eval for pretraining — streamed, subset, or complete.

    Dispatches on ``load_mode``. Eval is always a bounded map-style Dataset.
    """
    data.validate()
    if data.load_mode == "subset":
        return load_materialized_train_eval(
            data,
            trust_remote_code=trust_remote_code,
            verbose=verbose,
        )
    if data.load_mode == "complete":
        return load_complete_train_eval(
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
