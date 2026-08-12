"""Tests for post-training completion eval helpers (CPU-only)."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from alien_ink.hf.eval import (
    EvalItem,
    ItemResult,
    build_completion_eval_dataset,
    build_report,
    char_similarity,
    load_eval_items,
    normalize_text,
    relative_output_path,
    results_path,
    rouge_l_f1,
    save_eval_report,
    score_completion,
    suggest_max_new_tokens,
    token_f1,
    tokenize_prompt_completion,
)
from alien_ink.hf.manifest import Manifest
from alien_ink.hf.model import gpt2_arch
from alien_ink.hf.ds import HubTextSource, PretrainDataConfig


def _tiny_manifest(run_name: str = "eval-test-run") -> Manifest:
    return Manifest(
        run_name=run_name,
        title="eval test",
        data=PretrainDataConfig(
            source=HubTextSource(dataset="dummy", split="train", text_column="text"),
            max_train_samples=1,
            max_eval_samples=1,
        ),
        model=gpt2_arch(),
    )


def _item(
    *,
    slug: str,
    predicted: str,
    exact: bool,
    prefix: bool,
    n_tokens: int,
    loss: float | None = None,
    ppl: float | None = None,
    char_sim: float = 0.0,
    token_f1_score: float = 0.0,
    rouge_l: float = 0.0,
) -> ItemResult:
    return ItemResult(
        slug=slug,
        predicted=predicted,
        exact=exact,
        prefix=prefix,
        n_tokens=n_tokens,
        loss=loss,
        ppl=ppl,
        char_sim=char_sim,
        token_f1=token_f1_score,
        rouge_l=rouge_l,
    )


def test_normalize_text_strips_and_collapses_whitespace():
    assert normalize_text("  hello   world\n\t!") == "hello world !"
    assert normalize_text("") == ""


def test_score_completion_exact_and_prefix():
    scores = score_completion("Austin, Texas.", "Austin, Texas.")
    assert scores.exact and scores.prefix
    assert scores.char_sim == pytest.approx(1.0)
    assert scores.token_f1 == pytest.approx(1.0)
    assert scores.rouge_l == pytest.approx(1.0)

    scores = score_completion("  Austin,  Texas.  ", "Austin, Texas.")
    assert scores.exact and scores.prefix

    scores = score_completion("Austin, Texas. Population grew.", "Austin, Texas.")
    assert not scores.exact and scores.prefix
    assert 0.0 < scores.char_sim < 1.0

    scores = score_completion("Dallas.", "Austin, Texas.")
    assert not scores.exact and not scores.prefix
    assert scores.char_sim < 1.0


def test_score_completion_empty_expected():
    scores = score_completion("", "")
    assert scores.exact and scores.prefix
    assert scores.char_sim == 1.0

    scores = score_completion("x", "")
    assert not scores.exact and not scores.prefix
    assert scores.char_sim == 0.0


def test_char_similarity_and_token_metrics():
    assert char_similarity("abc", "abc") == pytest.approx(1.0)
    assert char_similarity("abc", "abd") == pytest.approx(2 / 3)
    assert char_similarity("", "abc") == pytest.approx(0.0)

    assert token_f1("the capital of texas", "the capital of texas") == pytest.approx(1.0)
    assert token_f1("austin texas", "dallas texas") == pytest.approx(0.5)
    assert token_f1("a b c", "x y z") == pytest.approx(0.0)

    # Shared ordered subsequence "the population" → ROUGE-L > bag F1 alone.
    assert rouge_l_f1(
        "the population grew quickly",
        "the population of texas",
    ) == pytest.approx(0.5)


def test_load_eval_items_ok(tmp_path: Path):
    path = tmp_path / "geo.json"
    path.write_text(
        json.dumps(
            [
                {
                    "slug": "texas",
                    "prompt": "The capital of Texas is",
                    "completion": "Austin.",
                },
                {
                    "slug": "alaska",
                    "prompt": "With a population of",
                    "completion": "740,133.",
                },
            ]
        ),
        encoding="utf-8",
    )
    items = load_eval_items(path)
    assert len(items) == 2
    assert items[0] == EvalItem(
        slug="texas",
        prompt="The capital of Texas is",
        completion="Austin.",
    )


class _FakeEvalTok:
    """Minimal tokenizer: prompt chars → ids 1..n, completion chars → 100+."""

    def __call__(self, text, *, add_special_tokens=True, return_tensors=None):
        del return_tensors
        if add_special_tokens:
            ids = [1] + [ord(ch) % 40 + 2 for ch in text]
        else:
            ids = [100 + (ord(ch) % 40) for ch in text]
        return {"input_ids": ids}


def test_tokenize_prompt_completion_masks_prompt():
    tok = _FakeEvalTok()
    encoded = tokenize_prompt_completion(
        tok,
        "Hi",
        "Yo",
        add_special_tokens=True,
    )
    assert encoded is not None
    prompt_len = len(tok("Hi", add_special_tokens=True)["input_ids"])
    completion_ids = tok(" Yo", add_special_tokens=False)["input_ids"]
    assert encoded["labels"][:prompt_len] == [-100] * prompt_len
    assert encoded["labels"][prompt_len:] == completion_ids
    assert encoded["input_ids"] == (
        tok("Hi", add_special_tokens=True)["input_ids"] + completion_ids
    )
    assert encoded["attention_mask"] == [1] * len(encoded["input_ids"])


def test_tokenize_prompt_completion_preserves_existing_text_boundaries():
    tok = _FakeEvalTok()
    after_open_paren = tokenize_prompt_completion(
        tok,
        "population (",
        "90%)",
        add_special_tokens=True,
    )
    assert after_open_paren is not None
    assert after_open_paren["labels"][-len(tok("90%)", add_special_tokens=False)["input_ids"]):] == tok(
        "90%)", add_special_tokens=False
    )["input_ids"]

    explicit_space = tokenize_prompt_completion(
        tok,
        "population of ",
        "10,000",
        add_special_tokens=True,
    )
    assert explicit_space is not None
    assert explicit_space["labels"][-len(tok("10,000", add_special_tokens=False)["input_ids"]):] == tok(
        "10,000", add_special_tokens=False
    )["input_ids"]


def test_tokenize_prompt_completion_empty_completion():
    tok = _FakeEvalTok()
    assert tokenize_prompt_completion(tok, "Hi", "", add_special_tokens=True) is None


def test_build_completion_eval_dataset(tmp_path: Path):
    path = tmp_path / "pop.json"
    path.write_text(
        json.dumps(
            [
                {
                    "slug": "alabama-has-population",
                    "sentence": "Alabama has a population of 10,000.",
                    "prompt": "Alabama has a population of",
                    "completion": "10,000.",
                },
                {
                    "slug": "alaska-has-population",
                    "prompt": "Alaska has a population of",
                    "completion": "740,133.",
                },
            ]
        ),
        encoding="utf-8",
    )
    ds = build_completion_eval_dataset(path, _FakeEvalTok(), add_special_tokens=True)
    assert len(ds) == 2
    assert "input_ids" in ds.column_names and "labels" in ds.column_names
    assert ds[0]["labels"].count(-100) > 0
    assert any(label != -100 for label in ds[0]["labels"])

    limited = build_completion_eval_dataset(
        path,
        _FakeEvalTok(),
        add_special_tokens=True,
        max_samples=1,
    )
    assert len(limited) == 1

    with pytest.raises(ValueError, match="max_samples"):
        build_completion_eval_dataset(
            path,
            _FakeEvalTok(),
            add_special_tokens=True,
            max_samples=0,
        )


def test_load_eval_items_rejects_bad_shape(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"slug": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON list"):
        load_eval_items(bad)

    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_eval_items(empty)

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps([{"slug": "a", "prompt": "p"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_eval_items(missing)

    dup = tmp_path / "dup.json"
    dup.write_text(
        json.dumps(
            [
                {"slug": "a", "prompt": "p", "completion": "c"},
                {"slug": "a", "prompt": "p2", "completion": "c2"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate slug"):
        load_eval_items(dup)


def test_build_report_rates_and_score_rollups():
    items = [
        _item(
            slug="a",
            predicted="yes",
            exact=True,
            prefix=True,
            n_tokens=1,
            loss=1.0,
            ppl=2.718281828,
            char_sim=1.0,
            token_f1_score=1.0,
            rouge_l=1.0,
        ),
        _item(
            slug="b",
            predicted="yes please",
            exact=False,
            prefix=True,
            n_tokens=2,
            loss=3.0,
            ppl=20.08553692,
            char_sim=0.5,
            token_f1_score=0.4,
            rouge_l=0.6,
        ),
        _item(
            slug="c",
            predicted="no",
            exact=False,
            prefix=False,
            n_tokens=1,
            loss=None,
            ppl=None,
            char_sim=0.0,
            token_f1_score=0.0,
            rouge_l=0.0,
        ),
    ]
    report = build_report(
        run_name="run",
        zdeck="zdeck",
        checkpoint="/ckpt",
        evals_path="/evals.json",
        max_new_tokens=32,
        do_sample=False,
        items=items,
        timestamp="2026-01-01T00:00:00Z",
    )
    assert report.n == 3
    assert report.exact_count == 1
    assert report.prefix_count == 2
    assert report.exact_rate == pytest.approx(1 / 3)
    assert report.prefix_rate == pytest.approx(2 / 3)
    assert report.mean_loss == pytest.approx(2.0)
    assert report.mean_ppl == pytest.approx(math.exp(2.0))
    assert report.mean_char_sim == pytest.approx(0.5)
    assert report.mean_token_f1 == pytest.approx(1.4 / 3)
    assert report.mean_rouge_l == pytest.approx(1.6 / 3)

    payload = report.as_dict()
    assert "prompt" not in payload["items"][0]
    assert "completion" not in payload["items"][0]
    assert set(payload["items"][0]) == {
        "slug",
        "predicted",
        "exact",
        "prefix",
        "n_tokens",
        "loss",
        "ppl",
        "char_sim",
        "token_f1",
        "rouge_l",
    }


def test_relative_output_path_uses_dot_slash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    nested = tmp_path / "output" / "train" / "run" / "evals" / "geo.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("{}", encoding="utf-8")
    assert relative_output_path(nested) == "./output/train/run/evals/geo.json"
    assert relative_output_path(nested.resolve()) == "./output/train/run/evals/geo.json"


def test_save_eval_report_writes_under_evals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    manifest = _tiny_manifest("save-run")
    evals = Path("/tmp/population-exact/geo-us-states.json")
    when = datetime(2026, 8, 10, 19, 0, 0, tzinfo=timezone.utc)
    out = results_path(manifest, evals, when=when)
    assert out == tmp_path / "output" / "train" / "save-run" / "evals" / "geo-us-states-20260810-190000.json"

    report = build_report(
        run_name=manifest.run_name,
        zdeck="label",
        checkpoint="/ckpt",
        evals_path=evals,
        max_new_tokens=16,
        do_sample=False,
        items=[
            _item(
                slug="texas",
                predicted="Austin.",
                exact=True,
                prefix=True,
                n_tokens=2,
                loss=0.5,
                ppl=1.648721,
                char_sim=1.0,
                token_f1_score=1.0,
                rouge_l=1.0,
            )
        ],
        timestamp="2026-08-10T19:00:00Z",
    )
    saved = save_eval_report(out, report)
    assert saved.is_file()
    data = json.loads(saved.read_text(encoding="utf-8"))
    assert data["exact_count"] == 1
    assert data["mean_loss"] == pytest.approx(0.5)
    assert data["mean_char_sim"] == pytest.approx(1.0)
    assert data["items"][0]["slug"] == "texas"
    assert data["items"][0]["token_f1"] == pytest.approx(1.0)
    assert "prompt" not in data["items"][0]
    # Eval file contents must not be embedded.
    assert "The capital" not in saved.read_text(encoding="utf-8")
    assert relative_output_path(saved) == (
        "./output/train/save-run/evals/geo-us-states-20260810-190000.json"
    )


class _FakeTok:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # One fake token per whitespace-separated word.
        return list(range(max(1, len(text.split()))))


def test_suggest_max_new_tokens_uses_longest_plus_cushion():
    items = [
        EvalItem(slug="a", prompt="p", completion="one"),
        EvalItem(slug="b", prompt="p", completion="one two three four"),
    ]
    assert suggest_max_new_tokens(items, _FakeTok(), cushion=8) == 12
    assert suggest_max_new_tokens([], _FakeTok()) == 128
