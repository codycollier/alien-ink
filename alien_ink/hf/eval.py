"""Post-training completion eval against an external JSON item file.

Loads a trained checkpoint, generates greedy continuations for each
``prompt``, and scores against ``completion`` with normalized exact match
(primary) and prefix match (secondary). Eval file contents are never copied
into the run output — only results are written under ``evals/``.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from alien_ink.com.device import move_module_to_device
from alien_ink.com.log import banner, blank, detail, get_logger, header, step
from alien_ink.hf.gen import generate_completion_result
from alien_ink.hf.manifest import Manifest
from alien_ink.hf.metrics import write_json
from alien_ink.hf.model import (
    CausalLmArchConfig,
    PretrainedLmConfig,
    find_checkpoint_path,
    load_pretrained_model,
    load_tokenizer,
)

log = get_logger("hf.eval")

_WS = re.compile(r"\s+")
_TOKEN_CUSHION = 8
_DEFAULT_MAX_NEW_TOKENS = 128


@dataclass(frozen=True)
class EvalItem:
    """One prompt / expected-completion pair from an external eval file."""

    slug: str
    prompt: str
    completion: str


@dataclass(frozen=True)
class ItemResult:
    """Per-item prediction and scores (no prompt / expected text)."""

    slug: str
    predicted: str
    exact: bool
    prefix: bool
    n_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalReport:
    """Aggregate + per-item results for one eval run."""

    run_name: str
    zdeck: str
    checkpoint: str
    evals_path: str
    timestamp: str
    max_new_tokens: int
    do_sample: bool
    n: int
    exact_count: int
    prefix_count: int
    exact_rate: float
    prefix_rate: float
    items: tuple[ItemResult, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["items"] = [item.as_dict() for item in self.items]
        return payload


def normalize_text(text: str) -> str:
    """Strip and collapse whitespace for stable string comparison."""
    return _WS.sub(" ", text.strip())


def score_completion(predicted: str, expected: str) -> tuple[bool, bool]:
    """Return ``(exact_match, prefix_match)`` after normalization."""
    pred = normalize_text(predicted)
    exp = normalize_text(expected)
    if not exp:
        exact = pred == ""
        return exact, exact
    exact = pred == exp
    prefix = pred.startswith(exp)
    return exact, prefix


def load_eval_items(path: Path | str) -> list[EvalItem]:
    """Load and validate an eval JSON file (list of slug/prompt/completion)."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"eval file must be a JSON list, got {type(raw).__name__}")
    if not raw:
        raise ValueError("eval file is empty")

    items: list[EvalItem] = []
    seen: set[str] = set()
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"eval item {index} must be an object")
        missing = [key for key in ("slug", "prompt", "completion") if key not in row]
        if missing:
            raise ValueError(f"eval item {index} missing fields: {missing}")
        slug = str(row["slug"]).strip()
        prompt = str(row["prompt"])
        completion = str(row["completion"])
        if not slug:
            raise ValueError(f"eval item {index} has an empty slug")
        if not prompt:
            raise ValueError(f"eval item {index} ({slug!r}) has an empty prompt")
        if slug in seen:
            raise ValueError(f"duplicate slug {slug!r} in eval file")
        seen.add(slug)
        items.append(EvalItem(slug=slug, prompt=prompt, completion=completion))
    return items


def suggest_max_new_tokens(
    items: list[EvalItem],
    tokenizer: PreTrainedTokenizerBase,
    *,
    cushion: int = _TOKEN_CUSHION,
) -> int:
    """Token length of the longest expected completion, plus a small cushion."""
    if not items:
        return _DEFAULT_MAX_NEW_TOKENS
    longest = 0
    for item in items:
        n = len(tokenizer.encode(item.completion, add_special_tokens=False))
        if n > longest:
            longest = n
    return max(1, longest + max(0, cushion))


def load_model_for_eval(
    manifest: Manifest,
    device: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, Path]:
    """Resolve the run checkpoint and load it for greedy evaluation."""
    model_path = find_checkpoint_path(manifest.output_dir())
    if isinstance(manifest.model, PretrainedLmConfig):
        step(f"Loading model from {model_path}...", logger=log)
        tokenizer = load_tokenizer(model_path)
        model = AutoModelForCausalLM.from_pretrained(str(model_path))
        move_module_to_device(model, device)
        model.eval()
    elif isinstance(manifest.model, CausalLmArchConfig):
        model, tokenizer = load_pretrained_model(
            model_path,
            device,
            family=manifest.model.family,
        )
    else:
        raise TypeError(
            f"unsupported model config type: {type(manifest.model).__name__}"
        )
    model.config.use_cache = True
    return model, tokenizer, model_path


def evals_output_dir(manifest: Manifest) -> Path:
    """``output/train/<run_name>/evals`` under the current working directory."""
    return manifest.output_dir() / "evals"


def results_path(manifest: Manifest, evals_path: Path, *, when: datetime | None = None) -> Path:
    """Timestamped results file path under the run's ``evals/`` directory."""
    when = when or datetime.now(timezone.utc)
    stamp = when.strftime("%Y%m%d-%H%M%S")
    stem = evals_path.stem or "eval"
    return evals_output_dir(manifest) / f"{stem}-{stamp}.json"


def build_report(
    *,
    run_name: str,
    zdeck: str,
    checkpoint: Path | str,
    evals_path: Path | str,
    max_new_tokens: int,
    do_sample: bool,
    items: list[ItemResult],
    timestamp: str | None = None,
) -> EvalReport:
    """Assemble aggregate rates from per-item results."""
    n = len(items)
    exact_count = sum(1 for item in items if item.exact)
    prefix_count = sum(1 for item in items if item.prefix)
    return EvalReport(
        run_name=run_name,
        zdeck=zdeck,
        checkpoint=str(checkpoint),
        evals_path=str(evals_path),
        timestamp=timestamp
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        n=n,
        exact_count=exact_count,
        prefix_count=prefix_count,
        exact_rate=(exact_count / n) if n else 0.0,
        prefix_rate=(prefix_count / n) if n else 0.0,
        items=tuple(items),
    )


def save_eval_report(path: Path, report: EvalReport) -> Path:
    """Write the eval report JSON (results only; no eval prompts/completions)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_json(path, report.as_dict())


def relative_output_path(path: Path | str) -> str:
    """Cwd-relative path with a ``./`` prefix when the file is under cwd."""
    resolved = Path(path).resolve()
    try:
        rel = resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return str(resolved)
    return f"./{rel.as_posix()}"


def log_eval_summary(report: EvalReport, *, results_file: Path | str | None = None) -> None:
    """Print a compact exact/prefix summary to the package logger."""
    blank(logger=log)
    step("Eval summary", logger=log)
    detail(f"items:        {report.n}", logger=log)
    detail(
        f"exact match:  {report.exact_count}/{report.n} "
        f"({100.0 * report.exact_rate:.1f}%)",
        logger=log,
    )
    detail(
        f"prefix match: {report.prefix_count}/{report.n} "
        f"({100.0 * report.prefix_rate:.1f}%)",
        logger=log,
    )
    if results_file is not None:
        detail(f"results:      {relative_output_path(results_file)}", logger=log)


def run_completion_eval(
    *,
    manifest: Manifest,
    evals_path: Path | str,
    device: str,
    zdeck_label: str = "",
    max_new_tokens: int | None = None,
    title: str | None = None,
) -> EvalReport:
    """Load checkpoint + eval file, score greedy completions, write results."""
    evals_path = Path(evals_path)
    items = load_eval_items(evals_path)

    header(logger=log)
    banner(title or f"completion eval — {manifest.run_name}", logger=log)
    step(f"Device: {device}", logger=log)
    detail(f"zdeck:    {zdeck_label or manifest.run_name}", logger=log)
    detail(f"run_name: {manifest.run_name}", logger=log)
    detail(f"evals:    {evals_path}", logger=log)
    detail(f"items:    {len(items)}", logger=log)

    model, tokenizer, checkpoint = load_model_for_eval(manifest, device)
    detail(f"checkpoint: {checkpoint}", logger=log)

    if max_new_tokens is None:
        max_new_tokens = suggest_max_new_tokens(items, tokenizer)
    gen = manifest.gen_config(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        stop_strings=(),
    )
    detail(
        f"gen: greedy max_new_tokens={gen.max_new_tokens} "
        f"add_special_tokens={gen.add_special_tokens}",
        logger=log,
    )
    blank(logger=log)

    results: list[ItemResult] = []
    for index, item in enumerate(items, start=1):
        completion = generate_completion_result(
            model, tokenizer, item.prompt, device, gen
        )
        exact, prefix = score_completion(completion.text, item.completion)
        results.append(
            ItemResult(
                slug=item.slug,
                predicted=completion.text,
                exact=exact,
                prefix=prefix,
                n_tokens=completion.n_tokens,
            )
        )
        mark = "exact" if exact else ("prefix" if prefix else "miss")
        log.info("[%s/%s] %s — %s", index, len(items), item.slug, mark)

    when = datetime.now(timezone.utc)
    report = build_report(
        run_name=manifest.run_name,
        zdeck=zdeck_label or manifest.run_name,
        checkpoint=checkpoint,
        evals_path=evals_path,
        max_new_tokens=gen.max_new_tokens,
        do_sample=gen.do_sample,
        items=results,
        timestamp=when.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    out = results_path(manifest, evals_path, when=when)
    save_eval_report(out, report)
    log_eval_summary(report, results_file=out)
    log.info("Done.")
    return report
