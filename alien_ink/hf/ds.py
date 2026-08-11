"""Hugging Face dataset helpers — Hub text corpora for causal-LM pretraining.

Supports three load modes:
  - ``stream``     — train stays an ``IterableDataset``
  - ``subset``     — materialize a bounded train prefix (``max_train_samples``)
  - ``complete``   — materialize the full Hub split (non-streaming download)

Built-in corpora: English Wikipedia, WikiText-103, C4 English, and
geo-us-states. Use :func:`hub_text` for any other Hub text corpus, or add a
factory that returns ``PretrainDataConfig`` to extend.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from datasets import Dataset, IterableDataset, load_dataset

from alien_ink.com.log import detail, get_logger, step

log = get_logger("hf.ds")

LoadMode = Literal["stream", "subset", "complete"]

# Default size for non-streamed "subset" corpora (train prefix + eval cap).
DEFAULT_SUBSET_TRAIN_SAMPLES = 20_000
DEFAULT_SUBSET_EVAL_SAMPLES = 1_000


@dataclass(frozen=True)
class HubTextSource:
    """Identity of a text corpus on the Hugging Face Hub."""

    dataset: str
    name: str | None = None
    split: str = "train"
    text_column: str = "text"


@dataclass(frozen=True)
class PretrainDataConfig:
    """How to load a Hub text corpus and pack it for causal LM pretraining.

    If ``completion_eval_path`` is set, Trainer eval comes from that external
    prompt/completion JSON (loaded at runtime) and the full train split is
    used — no Hub hold-out. ``eval_source`` / ``max_eval_samples`` are ignored
    for building the Trainer eval set.

    Otherwise: if ``eval_source`` is set, eval rows are taken from that split
    (capped by ``max_eval_samples``). Else the first ``max_eval_samples`` rows
    of the train stream/split are held out so train/eval do not overlap.

    ``mode``:
      - ``stream``   — train is streamed; ``max_train_samples`` must be ``None``
      - ``subset``   — materialize ``max_train_samples`` train rows
      - ``complete`` — download/materialize the full train split

    ``respect_document_boundaries``:
      - ``True``  (default) — chunk each Hub row independently; blocks never
        span two documents. Short-row corpora (e.g. WikiText) should set
        ``False`` or most rows produce zero blocks.
      - ``False`` — concatenate tokens across rows before slicing (classic packing)
    """

    source: HubTextSource
    eval_source: HubTextSource | None = None
    completion_eval_path: str | None = None
    mode: LoadMode = "stream"
    max_eval_samples: int = 1_000
    max_train_samples: int | None = None
    stream_shuffle_buffer: int = 10_000
    block_size: int = 1024
    respect_document_boundaries: bool = True
    tokenizer_num_proc: int = 4
    seed: int = 101

    def validate(self) -> None:
        if self.mode not in {"stream", "subset", "complete"}:
            raise ValueError(
                f"mode must be one of stream, subset, complete; got {self.mode!r}"
            )
        if self.completion_eval_path is not None:
            path = Path(self.completion_eval_path)
            if not str(self.completion_eval_path).strip():
                raise ValueError("completion_eval_path must be a non-empty path")
            if not path.is_file():
                raise ValueError(f"completion_eval_path does not exist: {path}")
        elif self.max_eval_samples < 1:
            raise ValueError(
                f"max_eval_samples must be >= 1, got {self.max_eval_samples}"
            )
        if self.mode == "subset":
            if self.max_train_samples is None or self.max_train_samples < 1:
                raise ValueError(
                    "mode='subset' requires max_train_samples >= 1, "
                    f"got {self.max_train_samples}"
                )
        elif self.max_train_samples is not None and self.max_train_samples < 1:
            raise ValueError(
                f"max_train_samples must be >= 1 when set, "
                f"got {self.max_train_samples}"
            )
        if self.mode == "stream" and self.max_train_samples is not None:
            raise ValueError(
                "mode='stream' requires max_train_samples=None; "
                "use mode='subset' for a materialized prefix"
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


def _data_config(
    source: HubTextSource,
    *,
    eval_source: HubTextSource | None = None,
    mode: LoadMode = "stream",
    max_eval_samples: int = 1_000,
    max_train_samples: int | None = None,
    respect_document_boundaries: bool = True,
) -> PretrainDataConfig:
    return PretrainDataConfig(
        source=source,
        eval_source=eval_source,
        mode=mode,
        max_eval_samples=max_eval_samples,
        max_train_samples=max_train_samples,
        respect_document_boundaries=respect_document_boundaries,
    )


def wikipedia_english(
    *,
    name: str = "20231101.en",
    mode: LoadMode = "stream",
    max_eval_samples: int = 1_000,
    max_train_samples: int | None = None,
) -> PretrainDataConfig:
    """English Wikipedia dump (single train split; eval via hold-out prefix)."""
    if mode == "subset" and max_train_samples is None:
        max_train_samples = DEFAULT_SUBSET_TRAIN_SAMPLES
    return _data_config(
        HubTextSource(dataset="wikimedia/wikipedia", name=name),
        mode=mode,
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
        mode="subset",
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )


def wikipedia_english_complete(
    *,
    name: str = "20231101.en",
    max_eval_samples: int = 1_000,
) -> PretrainDataConfig:
    """Fully materialized English Wikipedia train split."""
    return wikipedia_english(
        name=name,
        mode="complete",
        max_eval_samples=max_eval_samples,
    )


def wikitext_103(
    *,
    name: str = "wikitext-103-v1",
    mode: LoadMode = "stream",
    max_eval_samples: int = 1_000,
    max_train_samples: int | None = None,
) -> PretrainDataConfig:
    """WikiText-103 (train + validation splits).

    Uses cross-document packing because Hub rows are line-oriented and usually
    shorter than ``block_size``.
    """
    if mode == "subset" and max_train_samples is None:
        max_train_samples = DEFAULT_SUBSET_TRAIN_SAMPLES
    return _data_config(
        HubTextSource(dataset="Salesforce/wikitext", name=name, split="train"),
        eval_source=HubTextSource(
            dataset="Salesforce/wikitext", name=name, split="validation"
        ),
        mode=mode,
        max_eval_samples=max_eval_samples,
        max_train_samples=max_train_samples,
        respect_document_boundaries=False,
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
        mode="subset",
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )


def wikitext_103_complete(
    *,
    name: str = "wikitext-103-v1",
    max_eval_samples: int = 1_000,
) -> PretrainDataConfig:
    """Fully materialized WikiText-103 train + validation splits."""
    return wikitext_103(
        name=name,
        mode="complete",
        max_eval_samples=max_eval_samples,
    )


def c4_english(
    *,
    name: str = "en",
    mode: LoadMode = "stream",
    max_eval_samples: int = 1_000,
    max_train_samples: int | None = None,
) -> PretrainDataConfig:
    """Colossal Clean Crawled Corpus, English (train + validation splits)."""
    if mode == "subset" and max_train_samples is None:
        max_train_samples = DEFAULT_SUBSET_TRAIN_SAMPLES
    return _data_config(
        HubTextSource(dataset="allenai/c4", name=name, split="train"),
        eval_source=HubTextSource(dataset="allenai/c4", name=name, split="validation"),
        mode=mode,
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
        mode="subset",
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )


def c4_english_complete(
    *,
    name: str = "en",
    max_eval_samples: int = 1_000,
) -> PretrainDataConfig:
    """Fully materialized C4 English train + validation splits."""
    return c4_english(
        name=name,
        mode="complete",
        max_eval_samples=max_eval_samples,
    )


def hub_text(
    dataset: str,
    *,
    name: str | None = None,
    split: str = "train",
    text_column: str = "text",
    eval_source: HubTextSource | None = None,
    mode: LoadMode = "complete",
    max_eval_samples: int = 1_000,
    max_train_samples: int | None = None,
    respect_document_boundaries: bool = True,
) -> PretrainDataConfig:
    """Generic factory for any Hub text corpus (e.g. personal datasets).

    Defaults to ``mode="complete"`` with a hold-out eval, which suits small
    custom corpora. For corpora with fewer rows than ``max_eval_samples``,
    lower it or the hold-out will swallow the whole train split.
    """
    return _data_config(
        HubTextSource(
            dataset=dataset,
            name=name,
            split=split,
            text_column=text_column,
        ),
        eval_source=eval_source,
        mode=mode,
        max_eval_samples=max_eval_samples,
        max_train_samples=max_train_samples,
        respect_document_boundaries=respect_document_boundaries,
    )


def geo_us_states(*, max_eval_samples: int = 4) -> PretrainDataConfig:
    """``codycollier/geo-us-states`` — 56 long documents, one per US state/territory.

    Rows are ~17k tokens each, so document-bounded chunking still yields many
    blocks per row. The corpus is tiny; eval holds out just a few rows so the
    train split keeps nearly all states.
    """
    return hub_text(
        "codycollier/geo-us-states",
        mode="complete",
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


def load_hub_text(
    source: HubTextSource,
    *,
    trust_remote_code: bool = False,
) -> Dataset:
    """Download/materialize a full Hub text split (non-streaming)."""
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


def shuffled_stream(
    stream: IterableDataset,
    *,
    seed: int,
    buffer_size: int,
) -> IterableDataset:
    """Return a deterministic bounded-buffer shuffle for sampling or training."""
    return stream.shuffle(seed=seed, buffer_size=buffer_size)


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


def uses_completion_eval(data: PretrainDataConfig) -> bool:
    """True when Trainer eval comes from an external prompt/completion JSON."""
    return data.completion_eval_path is not None


def _empty_hub_eval(text_column: str) -> Dataset:
    """Placeholder Hub eval when completion JSON supplies Trainer eval instead."""
    return Dataset.from_dict({text_column: []})


def load_streaming_train_eval(
    data: PretrainDataConfig,
    *,
    trust_remote_code: bool = False,
    verbose: bool = True,
) -> tuple[IterableDataset, Dataset]:
    """Stream training text; materialize a bounded eval set."""
    if data.mode != "stream":
        raise ValueError(
            f"load_streaming_train_eval requires mode='stream', got {data.mode!r}"
        )
    if data.max_train_samples is not None:
        raise ValueError(
            "load_streaming_train_eval requires max_train_samples=None; "
            "use mode='subset' or load_train_eval instead"
        )

    source = data.source
    if verbose:
        step(f"Streaming {_source_label(source)} [{source.split}]...", logger=log)

    train_stream = stream_hub_text(source, trust_remote_code=trust_remote_code)

    if uses_completion_eval(data):
        train_stream = shuffled_stream(
            train_stream,
            seed=data.seed,
            buffer_size=data.stream_shuffle_buffer,
        )
        if verbose:
            detail(
                f"eval: deferred to completion JSON ({data.completion_eval_path})",
                logger=log,
            )
            detail(
                f"train examples: {_source_label(source)} (streaming, no hold-out)",
                logger=log,
            )
        return train_stream, _empty_hub_eval(source.text_column)

    if data.eval_source is not None:
        eval_source = data.eval_source
        if verbose:
            step(
                f"Materializing up to {data.max_eval_samples} eval rows "
                f"from {_source_label(eval_source)} [{eval_source.split}]...",
                logger=log,
            )
        eval_stream = shuffled_stream(
            stream_hub_text(eval_source, trust_remote_code=trust_remote_code),
            seed=data.seed,
            buffer_size=data.stream_shuffle_buffer,
        )
        eval_dataset = materialize_prefix(eval_stream, data.max_eval_samples)
        train_stream = shuffled_stream(
            train_stream,
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
    sampled_stream = shuffled_stream(
        train_stream,
        seed=data.seed,
        buffer_size=data.stream_shuffle_buffer,
    )
    eval_dataset = materialize_prefix(sampled_stream, data.max_eval_samples)
    train_stream = sampled_stream.skip(data.max_eval_samples)
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

    Requires ``mode='subset'`` and ``max_train_samples``. Eval follows the same
    rules as the streaming loader (dedicated ``eval_source`` or hold-out prefix).
    """
    if data.mode != "subset":
        raise ValueError(
            f"load_materialized_train_eval requires mode='subset', got {data.mode!r}"
        )
    if data.max_train_samples is None:
        raise ValueError(
            "load_materialized_train_eval requires max_train_samples; "
            "use mode='stream'/'complete' or load_train_eval instead"
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

    if uses_completion_eval(data):
        train_dataset = materialize_prefix(train_stream, n_train)
        train_dataset = train_dataset.shuffle(seed=data.seed)
        if verbose:
            detail(
                f"eval: deferred to completion JSON ({data.completion_eval_path})",
                logger=log,
            )
            detail(
                f"train examples: {len(train_dataset):,} (materialized subset, no hold-out)",
                logger=log,
            )
        return train_dataset, _empty_hub_eval(source.text_column)

    if data.eval_source is not None:
        eval_source = data.eval_source
        if verbose:
            step(
                f"Materializing up to {data.max_eval_samples} eval rows "
                f"from {_source_label(eval_source)} [{eval_source.split}]...",
                logger=log,
            )
        eval_stream = shuffled_stream(
            stream_hub_text(eval_source, trust_remote_code=trust_remote_code),
            seed=data.seed,
            buffer_size=data.stream_shuffle_buffer,
        )
        eval_dataset = materialize_prefix(eval_stream, data.max_eval_samples)
        train_dataset = materialize_prefix(train_stream, n_train)
        eval_label = eval_source.split
    else:
        if verbose:
            step(
                f"Materializing {data.max_eval_samples} held-out eval rows...",
                logger=log,
            )
        sampled_stream = shuffled_stream(
            train_stream,
            seed=data.seed,
            buffer_size=data.stream_shuffle_buffer,
        )
        eval_dataset = materialize_prefix(sampled_stream, data.max_eval_samples)
        train_dataset = materialize_prefix(
            sampled_stream.skip(data.max_eval_samples),
            n_train,
        )
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
    """Download full Hub splits into map-style Datasets (non-streaming)."""
    if data.mode != "complete":
        raise ValueError(
            f"load_complete_train_eval requires mode='complete', got {data.mode!r}"
        )

    source = data.source
    if verbose:
        step(
            f"Downloading complete {_source_label(source)} [{source.split}]...",
            logger=log,
        )
    train_full = load_hub_text(source, trust_remote_code=trust_remote_code)

    if uses_completion_eval(data):
        train_dataset = train_full.shuffle(seed=data.seed)
        if verbose:
            detail(
                f"eval: deferred to completion JSON ({data.completion_eval_path})",
                logger=log,
            )
            detail(
                f"train examples: {len(train_dataset):,} (complete, no hold-out)",
                logger=log,
            )
        return train_dataset, _empty_hub_eval(source.text_column)

    if data.eval_source is not None:
        eval_source = data.eval_source
        if verbose:
            step(
                f"Downloading complete {_source_label(eval_source)} "
                f"[{eval_source.split}] (cap {data.max_eval_samples})...",
                logger=log,
            )
        eval_full = load_hub_text(eval_source, trust_remote_code=trust_remote_code)
        eval_dataset = eval_full.shuffle(seed=data.seed).select(
            range(min(data.max_eval_samples, len(eval_full)))
        )
        train_dataset = train_full.shuffle(seed=data.seed)
        eval_label = eval_source.split
    else:
        if verbose:
            step(
                f"Holding out {data.max_eval_samples} eval rows from complete split...",
                logger=log,
            )
        shuffled = train_full.shuffle(seed=data.seed)
        n_eval = min(data.max_eval_samples, len(shuffled))
        eval_dataset = shuffled.select(range(n_eval))
        train_dataset = shuffled.select(range(n_eval, len(shuffled)))
        eval_label = "held out"

    if verbose:
        detail(f"eval examples:  {len(eval_dataset):,} ({eval_label})", logger=log)
        detail(
            f"train examples: {len(train_dataset):,} (complete)",
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

    Dispatches on ``mode``. Eval is always a bounded map-style Dataset.
    """
    data.validate()
    if data.mode == "subset":
        return load_materialized_train_eval(
            data,
            trust_remote_code=trust_remote_code,
            verbose=verbose,
        )
    if data.mode == "complete":
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
