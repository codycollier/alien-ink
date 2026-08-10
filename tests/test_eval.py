"""Tests for post-training completion eval helpers (CPU-only)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from alien_ink.hf.eval import (
    EvalItem,
    ItemResult,
    build_report,
    load_eval_items,
    normalize_text,
    results_path,
    save_eval_report,
    score_completion,
    suggest_max_new_tokens,
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


def test_normalize_text_strips_and_collapses_whitespace():
    assert normalize_text("  hello   world\n\t!") == "hello world !"
    assert normalize_text("") == ""


def test_score_completion_exact_and_prefix():
    exact, prefix = score_completion("Austin, Texas.", "Austin, Texas.")
    assert exact and prefix

    exact, prefix = score_completion("  Austin,  Texas.  ", "Austin, Texas.")
    assert exact and prefix

    exact, prefix = score_completion("Austin, Texas. Population grew.", "Austin, Texas.")
    assert not exact and prefix

    exact, prefix = score_completion("Dallas.", "Austin, Texas.")
    assert not exact and not prefix


def test_score_completion_empty_expected():
    assert score_completion("", "") == (True, True)
    assert score_completion("x", "") == (False, False)


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


def test_build_report_rates_and_no_prompt_fields():
    items = [
        ItemResult(slug="a", predicted="yes", exact=True, prefix=True, n_tokens=1),
        ItemResult(slug="b", predicted="yes please", exact=False, prefix=True, n_tokens=2),
        ItemResult(slug="c", predicted="no", exact=False, prefix=False, n_tokens=1),
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

    payload = report.as_dict()
    assert "prompt" not in payload["items"][0]
    assert "completion" not in payload["items"][0]
    assert set(payload["items"][0]) == {
        "slug",
        "predicted",
        "exact",
        "prefix",
        "n_tokens",
    }


def test_save_eval_report_writes_under_evals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    manifest = _tiny_manifest("save-run")
    evals = Path("/tmp/population-exact/geo-us-states.json")
    when = datetime(2026, 8, 10, 19, 0, 0, tzinfo=timezone.utc)
    out = results_path(manifest, evals, when=when)
    assert out == tmp_path / "output" / "save-run" / "evals" / "geo-us-states-20260810-190000.json"

    report = build_report(
        run_name=manifest.run_name,
        zdeck="label",
        checkpoint="/ckpt",
        evals_path=evals,
        max_new_tokens=16,
        do_sample=False,
        items=[
            ItemResult(
                slug="texas",
                predicted="Austin.",
                exact=True,
                prefix=True,
                n_tokens=2,
            )
        ],
        timestamp="2026-08-10T19:00:00Z",
    )
    saved = save_eval_report(out, report)
    assert saved.is_file()
    data = json.loads(saved.read_text(encoding="utf-8"))
    assert data["exact_count"] == 1
    assert data["items"][0]["slug"] == "texas"
    assert "prompt" not in data["items"][0]
    # Eval file contents must not be embedded.
    assert "The capital" not in saved.read_text(encoding="utf-8")


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
