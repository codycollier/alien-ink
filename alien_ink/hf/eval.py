"""Post-training completion eval against an external JSON item file.

Loads a trained checkpoint, generates greedy continuations for each
``prompt``, and scores against ``completion`` with:

* normalized exact / prefix match (discrete hit rates)
* teacher-forced mean CE loss + perplexity on the expected completion
  (same signal as training ``eval_loss``)
* text similarity of predicted vs expected: char edit similarity, token F1,
  and ROUGE-L F1

Eval file contents are never copied into the run output — only results are
written under ``evals/``.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from alien_ink.com.device import move_module_to_device, torch_device
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
class TextScores:
    """Discrete hits plus continuous similarity of predicted vs expected."""

    exact: bool
    prefix: bool
    char_sim: float
    token_f1: float
    rouge_l: float


@dataclass(frozen=True)
class ItemResult:
    """Per-item prediction and scores (no prompt / expected text)."""

    slug: str
    predicted: str
    exact: bool
    prefix: bool
    n_tokens: int
    loss: float | None
    ppl: float | None
    char_sim: float
    token_f1: float
    rouge_l: float

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
    mean_loss: float | None
    mean_ppl: float | None
    mean_char_sim: float
    mean_token_f1: float
    mean_rouge_l: float
    items: tuple[ItemResult, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["items"] = [item.as_dict() for item in self.items]
        return payload


def normalize_text(text: str) -> str:
    """Strip and collapse whitespace for stable string comparison."""
    return _WS.sub(" ", text.strip())


def _words(text: str) -> list[str]:
    """Whitespace tokens after normalization (empty string → empty list)."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    return normalized.split(" ")


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert / delete / substitute)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Keep only the previous row to save memory on longer completions.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def char_similarity(predicted: str, expected: str) -> float:
    """``1 - lev / max(len)`` on normalized strings (1.0 = identical)."""
    pred = normalize_text(predicted)
    exp = normalize_text(expected)
    if not pred and not exp:
        return 1.0
    denom = max(len(pred), len(exp))
    if denom == 0:
        return 1.0
    return 1.0 - (levenshtein(pred, exp) / denom)


def token_f1(predicted: str, expected: str) -> float:
    """Bag-of-words F1 over whitespace tokens after normalization."""
    pred_toks = _words(predicted)
    exp_toks = _words(expected)
    if not pred_toks and not exp_toks:
        return 1.0
    if not pred_toks or not exp_toks:
        return 0.0
    pred_counts = Counter(pred_toks)
    exp_counts = Counter(exp_toks)
    overlap = sum((pred_counts & exp_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(exp_toks)
    return 2.0 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists."""
    if not a or not b:
        return 0
    # DP row over b; O(len(a) * len(b)) time, O(len(b)) memory.
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        cur = [0]
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(predicted: str, expected: str) -> float:
    """ROUGE-L F1 over whitespace tokens (LCS-based)."""
    pred_toks = _words(predicted)
    exp_toks = _words(expected)
    if not pred_toks and not exp_toks:
        return 1.0
    if not pred_toks or not exp_toks:
        return 0.0
    lcs = _lcs_length(pred_toks, exp_toks)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_toks)
    recall = lcs / len(exp_toks)
    return 2.0 * precision * recall / (precision + recall)


def score_completion(predicted: str, expected: str) -> TextScores:
    """Exact/prefix hits plus continuous text-similarity scores."""
    pred = normalize_text(predicted)
    exp = normalize_text(expected)
    if not exp:
        exact = pred == ""
        sim = 1.0 if exact else 0.0
        return TextScores(
            exact=exact,
            prefix=exact,
            char_sim=sim,
            token_f1=sim,
            rouge_l=sim,
        )
    exact = pred == exp
    prefix = pred.startswith(exp)
    return TextScores(
        exact=exact,
        prefix=prefix,
        char_sim=char_similarity(pred, exp),
        token_f1=token_f1(pred, exp),
        rouge_l=rouge_l_f1(pred, exp),
    )


@torch.inference_mode()
def expected_completion_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    expected: str,
    device: str,
    *,
    add_special_tokens: bool,
) -> float | None:
    """Teacher-forced mean CE loss of ``expected`` given ``prompt``.

    Matches the training eval signal (``eval_loss``): cross-entropy averaged
    over expected-completion tokens only. Returns ``None`` when there are no
    completion tokens to score.
    """
    if not expected:
        return None
    target = torch_device(device)
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )["input_ids"]
    completion_ids = tokenizer(
        expected,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    if completion_ids.shape[1] == 0:
        return None
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(target)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    if loss is None:
        return None
    return float(loss.item())


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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


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
    """Assemble aggregate rates and mean scores from per-item results."""
    n = len(items)
    exact_count = sum(1 for item in items if item.exact)
    prefix_count = sum(1 for item in items if item.prefix)
    losses = [item.loss for item in items if item.loss is not None]
    mean_loss = _mean(losses)
    mean_ppl = math.exp(mean_loss) if mean_loss is not None else None
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
        mean_loss=mean_loss,
        mean_ppl=mean_ppl,
        mean_char_sim=_mean([item.char_sim for item in items]) or 0.0,
        mean_token_f1=_mean([item.token_f1 for item in items]) or 0.0,
        mean_rouge_l=_mean([item.rouge_l for item in items]) or 0.0,
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


def _hit_label(exact: bool, prefix: bool) -> str:
    if exact:
        return "exact"
    if prefix:
        return "prefix"
    return "miss"


def log_eval_summary(report: EvalReport, *, results_file: Path | str | None = None) -> None:
    """Print hit rates plus mean loss / similarity rollups."""
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
    if report.mean_loss is not None and report.mean_ppl is not None:
        detail(
            f"mean loss:    {report.mean_loss:.4f}  "
            f"(ppl {report.mean_ppl:.2f})",
            logger=log,
        )
    else:
        detail("mean loss:    n/a", logger=log)
    detail(f"mean char_sim: {report.mean_char_sim:.4f}", logger=log)
    detail(f"mean token_f1: {report.mean_token_f1:.4f}", logger=log)
    detail(f"mean rouge_l:  {report.mean_rouge_l:.4f}", logger=log)
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
        scores = score_completion(completion.text, item.completion)
        loss = expected_completion_loss(
            model,
            tokenizer,
            item.prompt,
            item.completion,
            device,
            add_special_tokens=gen.add_special_tokens,
        )
        ppl = math.exp(loss) if loss is not None else None
        results.append(
            ItemResult(
                slug=item.slug,
                predicted=completion.text,
                exact=scores.exact,
                prefix=scores.prefix,
                n_tokens=completion.n_tokens,
                loss=loss,
                ppl=ppl,
                char_sim=scores.char_sim,
                token_f1=scores.token_f1,
                rouge_l=scores.rouge_l,
            )
        )
        hit = _hit_label(scores.exact, scores.prefix)
        loss_part = f" loss={loss:.3f}" if loss is not None else ""
        log.info(
            "[%s/%s] %s — %s  char=%.3f tok_f1=%.3f rouge_l=%.3f%s",
            index,
            len(items),
            item.slug,
            hit,
            scores.char_sim,
            scores.token_f1,
            scores.rouge_l,
            loss_part,
        )

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
