"""Prompt/completion datasets for supervised causal-LM fine-tuning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from transformers import PreTrainedTokenizerBase

from alien_ink.com.log import blank, detail, get_logger, step

log = get_logger("hf.completion")

__all__ = [
    "CompletionDataConfig",
    "EvalItem",
    "load_eval_items",
    "tokenize_prompt_completion",
    "build_completion_eval_dataset",
    "prepare_completion_datasets",
]


@dataclass(frozen=True)
class EvalItem:
    """One prompt / expected-completion pair from an external eval file."""

    slug: str
    prompt: str
    completion: str


def tokenize_prompt_completion(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    completion: str,
    *,
    add_special_tokens: bool,
    loss_on_prompt: bool = False,
) -> dict[str, list[int]] | None:
    """Tokenize one prompt/completion pair with prompt tokens masked in labels.

    Same masking contract as :func:`expected_completion_loss`: ``labels`` are
    ``-100`` over the prompt span and real ids over the completion. Returns
    ``None`` when the completion encodes to zero tokens.
    """
    if not completion:
        return None
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=add_special_tokens,
    )["input_ids"]
    # Eval JSON stores human-readable fields without requiring callers to
    # encode boundary whitespace into either value. Supply the ordinary text
    # boundary unless whitespace/punctuation already makes it unnecessary.
    no_space_after = "([{/'\""
    no_space_before = ".,;:!?)]}"
    needs_space = (
        not prompt[-1].isspace()
        and not completion[0].isspace()
        and prompt[-1] not in no_space_after
        and completion[0] not in no_space_before
    )
    completion_text = f" {completion}" if needs_space else completion
    completion_ids = tokenizer(
        completion_text,
        add_special_tokens=False,
    )["input_ids"]
    if not completion_ids:
        return None
    input_ids = list(prompt_ids) + list(completion_ids)
    labels = (
        list(prompt_ids) + list(completion_ids)
        if loss_on_prompt
        else [-100] * len(prompt_ids) + list(completion_ids)
    )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def build_completion_eval_dataset(
    path: Path | str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    add_special_tokens: bool = True,
    max_samples: int | None = None,
    loss_on_prompt: bool = False,
):
    """Map-style Dataset of masked prompt/completion rows from an eval JSON.

    Eval file contents are read at runtime and never copied into the run
    output. Rows with empty completion encodings are skipped.
    """
    from datasets import Dataset

    if max_samples is not None and max_samples < 1:
        raise ValueError(f"max_samples must be >= 1 when set, got {max_samples}")
    items = load_eval_items(path)
    if max_samples is not None:
        items = items[:max_samples]
    rows: list[dict[str, list[int]]] = []
    for item in items:
        encoded = tokenize_prompt_completion(
            tokenizer,
            item.prompt,
            item.completion,
            add_special_tokens=add_special_tokens,
            loss_on_prompt=loss_on_prompt,
        )
        if encoded is not None:
            rows.append(encoded)
    if not rows:
        raise ValueError(f"no scorable completion tokens in eval file: {path}")
    return Dataset.from_list(rows)


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


@dataclass(frozen=True)
class CompletionDataConfig:
    """JSON prompt/completion data with an explicit causal-LM loss mask.

    Each JSON file is a list of objects containing ``slug``, ``prompt``, and
    ``completion``. ``None`` sample limits mean the complete file. When no
    ``eval_path`` is supplied, evaluation reuses ``train_path``.
    """

    train_path: str
    eval_path: str | None = None
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    loss_on_prompt: bool = False
    eval_train_dataset: bool = False
    max_length: int = 128
    seed: int = 101

    def resolved_eval_path(self) -> str:
        return self.eval_path or self.train_path

    def validate(self) -> None:
        for field_name, value in (
            ("train_path", self.train_path),
            ("eval_path", self.eval_path),
        ):
            if value is None:
                continue
            if not str(value).strip():
                raise ValueError(f"{field_name} must be a non-empty path")
            if not Path(value).is_file():
                raise ValueError(f"{field_name} does not exist: {value}")
        for field_name, value in (
            ("max_train_samples", self.max_train_samples),
            ("max_eval_samples", self.max_eval_samples),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{field_name} must be >= 1 when set, got {value}")
        if self.max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {self.max_length}")


def prepare_completion_datasets(
    data: CompletionDataConfig,
    tokenizer,
    *,
    verbose: bool = True,
    add_special_tokens: bool = True,
):
    """Build completion-only train/eval datasets directly from JSON files."""
    data.validate()
    if verbose:
        blank(logger=log)
        step(f"Building completion train from {data.train_path}...", logger=log)
    train_dataset = build_completion_eval_dataset(
        data.train_path,
        tokenizer,
        add_special_tokens=add_special_tokens,
        max_samples=data.max_train_samples,
        loss_on_prompt=data.loss_on_prompt,
    )
    eval_dataset = build_completion_eval_dataset(
        data.resolved_eval_path(),
        tokenizer,
        add_special_tokens=add_special_tokens,
        max_samples=data.max_eval_samples,
        loss_on_prompt=data.loss_on_prompt,
    )
    longest = max(
        max(len(row) for row in train_dataset["input_ids"]),
        max(len(row) for row in eval_dataset["input_ids"]),
    )
    if longest > data.max_length:
        raise ValueError(
            f"completion sequence length ({longest}) exceeds "
            f"max_length ({data.max_length})"
        )
    if verbose:
        detail(f"train examples: {len(train_dataset):,} (completion JSON)", logger=log)
        detail(f"eval examples:  {len(eval_dataset):,} (completion JSON)", logger=log)
    if data.eval_train_dataset:
        return train_dataset, {"completion": eval_dataset, "train": train_dataset}
    return train_dataset, eval_dataset
