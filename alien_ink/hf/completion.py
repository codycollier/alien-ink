"""Prompt/completion datasets for supervised causal-LM fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alien_ink.com.log import blank, detail, get_logger, step
from alien_ink.hf.eval import build_completion_eval_dataset

log = get_logger("hf.completion")

__all__ = ["CompletionDataConfig", "prepare_completion_datasets"]


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
