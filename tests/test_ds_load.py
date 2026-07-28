"""Tests for streamed / subset / complete train/eval loading."""

from __future__ import annotations

from unittest.mock import patch

import pytest

datasets = pytest.importorskip("datasets")
from datasets import Dataset, IterableDataset  # noqa: E402

from alien_ink.hf.ds import (  # noqa: E402
    DEFAULT_SUBSET_EVAL_SAMPLES,
    DEFAULT_SUBSET_TRAIN_SAMPLES,
    HubTextSource,
    PretrainDataConfig,
    c4_english_complete,
    c4_english_subset,
    load_complete_train_eval,
    load_materialized_train_eval,
    load_streaming_train_eval,
    load_train_eval,
    wikipedia_english,
    wikipedia_english_complete,
    wikipedia_english_subset,
    wikitext_103_subset,
)


def _rows(prefix: str, n: int) -> list[dict[str, str]]:
    return [{"text": f"{prefix}-{i}"} for i in range(n)]


def _iterable(rows: list[dict[str, str]]) -> IterableDataset:
    return Dataset.from_list(rows).to_iterable_dataset()


def test_subset_factories_default_sizes():
    wiki = wikipedia_english_subset()
    assert wiki.mode == "subset"
    assert wiki.max_train_samples == DEFAULT_SUBSET_TRAIN_SAMPLES
    assert wiki.max_eval_samples == DEFAULT_SUBSET_EVAL_SAMPLES
    assert wiki.eval_source is None

    wt = wikitext_103_subset()
    assert wt.mode == "subset"
    assert wt.max_train_samples == DEFAULT_SUBSET_TRAIN_SAMPLES
    assert wt.max_eval_samples == DEFAULT_SUBSET_EVAL_SAMPLES
    assert wt.eval_source is not None
    assert wt.eval_source.split == "validation"

    c4 = c4_english_subset()
    assert c4.mode == "subset"
    assert c4.max_train_samples == DEFAULT_SUBSET_TRAIN_SAMPLES
    assert c4.max_eval_samples == DEFAULT_SUBSET_EVAL_SAMPLES
    assert c4.eval_source is not None


def test_complete_factories():
    wiki = wikipedia_english_complete()
    assert wiki.mode == "complete"
    assert wiki.max_train_samples is None
    c4 = c4_english_complete()
    assert c4.mode == "complete"


def test_stream_factory_default():
    cfg = wikipedia_english()
    assert cfg.mode == "stream"
    assert cfg.max_train_samples is None


def test_load_train_eval_dispatches_on_mode():
    streamed = PretrainDataConfig(
        source=HubTextSource(dataset="dummy", split="train"),
        mode="stream",
        max_eval_samples=2,
    )
    subset = PretrainDataConfig(
        source=HubTextSource(dataset="dummy", split="train"),
        mode="subset",
        max_eval_samples=2,
        max_train_samples=3,
    )
    complete = PretrainDataConfig(
        source=HubTextSource(dataset="dummy", split="train"),
        mode="complete",
        max_eval_samples=2,
    )

    with patch(
        "alien_ink.hf.ds.load_streaming_train_eval",
        return_value=("stream", "eval"),
    ) as stream_mock, patch(
        "alien_ink.hf.ds.load_materialized_train_eval",
        return_value=("mat", "eval"),
    ) as mat_mock, patch(
        "alien_ink.hf.ds.load_complete_train_eval",
        return_value=("full", "eval"),
    ) as complete_mock:
        assert load_train_eval(streamed, verbose=False) == ("stream", "eval")
        stream_mock.assert_called_once()
        mat_mock.assert_not_called()
        complete_mock.assert_not_called()

        stream_mock.reset_mock()
        mat_mock.reset_mock()
        complete_mock.reset_mock()
        assert load_train_eval(subset, verbose=False) == ("mat", "eval")
        mat_mock.assert_called_once()
        stream_mock.assert_not_called()
        complete_mock.assert_not_called()

        stream_mock.reset_mock()
        mat_mock.reset_mock()
        complete_mock.reset_mock()
        assert load_train_eval(complete, verbose=False) == ("full", "eval")
        complete_mock.assert_called_once()
        stream_mock.assert_not_called()
        mat_mock.assert_not_called()


def test_load_materialized_hold_out_skips_eval_prefix():
    train_rows = _rows("t", 10)
    cfg = PretrainDataConfig(
        source=HubTextSource(dataset="dummy", split="train"),
        mode="subset",
        max_eval_samples=2,
        max_train_samples=3,
        seed=0,
    )

    with patch(
        "alien_ink.hf.ds.stream_hub_text",
        return_value=_iterable(train_rows),
    ):
        train_ds, eval_ds = load_materialized_train_eval(cfg, verbose=False)

    assert isinstance(train_ds, Dataset)
    assert isinstance(eval_ds, Dataset)
    assert len(eval_ds) == 2
    assert len(train_ds) == 3
    assert set(eval_ds["text"]) == {"t-0", "t-1"}
    assert set(train_ds["text"]).isdisjoint(set(eval_ds["text"]))
    assert set(train_ds["text"]) == {"t-2", "t-3", "t-4"}


def test_load_materialized_uses_eval_source():
    train_rows = _rows("train", 5)
    eval_rows = _rows("eval", 4)
    cfg = PretrainDataConfig(
        source=HubTextSource(dataset="dummy", split="train"),
        eval_source=HubTextSource(dataset="dummy", split="validation"),
        mode="subset",
        max_eval_samples=2,
        max_train_samples=3,
        seed=0,
    )

    def fake_stream(source, *, trust_remote_code=False):
        del trust_remote_code
        if source.split == "validation":
            return _iterable(eval_rows)
        return _iterable(train_rows)

    with patch("alien_ink.hf.ds.stream_hub_text", side_effect=fake_stream):
        train_ds, eval_ds = load_materialized_train_eval(cfg, verbose=False)

    assert len(train_ds) == 3
    assert len(eval_ds) == 2
    assert set(train_ds["text"]) == {"train-0", "train-1", "train-2"}
    assert set(eval_ds["text"]) == {"eval-0", "eval-1"}


def test_load_complete_hold_out():
    train_rows = _rows("t", 6)
    cfg = PretrainDataConfig(
        source=HubTextSource(dataset="dummy", split="train"),
        mode="complete",
        max_eval_samples=2,
        seed=0,
    )

    with patch(
        "alien_ink.hf.ds.load_hub_text",
        return_value=Dataset.from_list(train_rows),
    ):
        train_ds, eval_ds = load_complete_train_eval(cfg, verbose=False)

    assert len(eval_ds) == 2
    assert len(train_ds) == 4
    assert set(eval_ds["text"]) == {"t-0", "t-1"}
    assert set(train_ds["text"]).isdisjoint(set(eval_ds["text"]))


def test_load_streaming_rejects_wrong_mode():
    cfg = PretrainDataConfig(
        source=HubTextSource(dataset="dummy"),
        mode="subset",
        max_train_samples=10,
    )
    with pytest.raises(ValueError, match="mode='stream'"):
        load_streaming_train_eval(cfg, verbose=False)


def test_load_materialized_requires_subset_mode():
    cfg = PretrainDataConfig(source=HubTextSource(dataset="dummy"), mode="stream")
    with pytest.raises(ValueError, match="mode='subset'"):
        load_materialized_train_eval(cfg, verbose=False)


def test_stream_mode_rejects_max_train_samples():
    cfg = PretrainDataConfig(
        source=HubTextSource(dataset="dummy"),
        mode="stream",
        max_train_samples=10,
    )
    with pytest.raises(ValueError, match="max_train_samples=None"):
        cfg.validate()
