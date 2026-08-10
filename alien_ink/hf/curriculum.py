"""Curriculum: sequence Hub corpora (or subsets) into one pretraining stream.

A :class:`Curriculum` is an ordered tuple of phases, each pairing a
:class:`~alien_ink.hf.ds.PretrainDataConfig` with an optimizer-step budget.
It materializes as a single ``IterableDataset`` of LM blocks whose phase
boundaries fall exactly on optimizer steps, so the HF ``Trainer`` sees a
normal streaming dataset — one run, one LR schedule, one W&B run.

One optimizer step consumes ``per_device_train_batch_size *
gradient_accumulation_steps * world_size`` blocks (``samples_per_step``), so a
phase of ``steps`` contributes exactly ``steps * samples_per_step`` blocks.
Small materialized phases repeat (multiple epochs) to fill their budget;
streamed phases must be large enough to supply theirs.

Eval is fixed for the whole run so curves stay comparable across phase
transitions: a single set (first phase's config by default) or a mapping of
named sets (e.g. ``{"geo": ..., "c4": ...}``) for per-domain loss curves —
the standard way to watch a late phase for catastrophic forgetting.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from datasets import Dataset, IterableDataset, concatenate_datasets

from alien_ink.com.log import blank, detail, get_logger, step
from alien_ink.hf.ds import PretrainDataConfig, load_train_eval
from alien_ink.hf.tok import tokenize_and_chunk

log = get_logger("hf.curriculum")

__all__ = [
    "Curriculum",
    "CurriculumPhase",
    "bounded_phase_blocks",
    "chain_phase_blocks",
    "prepare_curriculum_datasets",
]


@dataclass(frozen=True)
class CurriculumPhase:
    """One curriculum phase: a corpus config plus an optimizer-step budget."""

    data: PretrainDataConfig
    steps: int
    label: str | None = None

    def name(self) -> str:
        if self.label:
            return self.label
        return self.data.source.dataset

    def validate(self) -> None:
        if self.steps < 1:
            raise ValueError(f"phase steps must be >= 1, got {self.steps}")
        self.data.validate()


@dataclass(frozen=True)
class Curriculum:
    """Ordered phases trained back-to-back in a single run.

    ``eval_data`` selects the fixed eval set(s) for the whole run:
      - ``None`` — eval from the first phase's config (hold-out or eval split)
      - a ``PretrainDataConfig`` — one explicit eval set
      - a mapping of name → config — named eval sets; the trainer reports
        ``eval_<name>_loss`` per entry and the *first* entry drives best-model
        selection and early stopping

    Reusing a phase's exact config as an eval entry is safe for hold-out
    corpora: the deterministic seed yields the same held-out rows in both
    places, so train and eval slices never overlap.
    """

    phases: tuple[CurriculumPhase, ...]
    eval_data: PretrainDataConfig | Mapping[str, PretrainDataConfig] | None = None

    @property
    def block_size(self) -> int:
        """Shared LM block size (uniform across phases)."""
        return self.phases[0].data.block_size

    @property
    def seed(self) -> int:
        """Data seed (first phase), mirroring ``PretrainDataConfig.seed``."""
        return self.phases[0].data.seed

    def total_steps(self) -> int:
        return sum(phase.steps for phase in self.phases)

    def boundaries(self) -> tuple[int, ...]:
        """Cumulative optimizer step at the end of each phase."""
        marks: list[int] = []
        total = 0
        for phase in self.phases:
            total += phase.steps
            marks.append(total)
        return tuple(marks)

    def eval_configs(self) -> dict[str, PretrainDataConfig]:
        """Resolved name → eval config mapping (default: first phase's data)."""
        if self.eval_data is None:
            return {"eval": self.phases[0].data}
        if isinstance(self.eval_data, PretrainDataConfig):
            return {"eval": self.eval_data}
        return dict(self.eval_data)

    def validate(self) -> None:
        if not self.phases:
            raise ValueError("curriculum requires at least one phase")
        for phase in self.phases:
            phase.validate()
        block_sizes = {phase.data.block_size for phase in self.phases}
        if len(block_sizes) > 1:
            raise ValueError(
                "all curriculum phases must share the same block_size, "
                f"got {sorted(block_sizes)}"
            )
        evals = self.eval_configs()
        if not evals:
            raise ValueError("eval_data mapping must have at least one entry")
        for name, config in evals.items():
            if not name.strip():
                raise ValueError("eval_data names must be non-empty strings")
            config.validate()


def bounded_phase_blocks(blocks, *, num_samples: int) -> IterableDataset:
    """Bound one phase's block dataset to exactly ``num_samples`` rows.

    Materialized phases repeat (cycle) so a small corpus can fill its full
    step budget; streamed phases are taken as-is and must be large enough.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if isinstance(blocks, IterableDataset):
        return blocks.take(num_samples)
    if len(blocks) == 0:
        raise ValueError("phase produced no training blocks")
    return blocks.to_iterable_dataset().repeat(None).take(num_samples)


def chain_phase_blocks(parts: list[IterableDataset]) -> IterableDataset:
    """Concatenate bounded phase streams into one train stream.

    Lazily-mapped streams carry no feature schema and would re-infer types at
    concat time (e.g. int64 vs a materialized phase's int32), so every part is
    cast to the first concrete schema before concatenation.
    """
    if not parts:
        raise ValueError("chain_phase_blocks requires at least one part")
    if len(parts) == 1:
        return parts[0]
    features = next(
        (part.features for part in parts if part.features is not None), None
    )
    if features is not None:
        parts = [
            part if part.features == features else part.cast(features)
            for part in parts
        ]
    return concatenate_datasets(parts)


def _phase_train_config(data: PretrainDataConfig) -> PretrainDataConfig:
    """Config for loading a phase's train stream (its own eval is discarded).

    With a dedicated ``eval_source`` the train stream is independent of
    ``max_eval_samples``, so shrink it to 1 to avoid materializing eval rows
    we throw away. Hold-out configs keep their value — it determines which
    rows are excluded from train, and must match any eval reuse of the config.
    """
    if data.eval_source is not None and data.max_eval_samples > 1:
        return replace(data, max_eval_samples=1)
    return data


def _phase_blocks(
    phase: CurriculumPhase,
    tokenizer,
    *,
    verbose: bool,
):
    train_raw, _ = load_train_eval(_phase_train_config(phase.data), verbose=verbose)
    return tokenize_and_chunk(
        train_raw,
        tokenizer,
        block_size=phase.data.block_size,
        text_column=phase.data.source.text_column,
        respect_document_boundaries=phase.data.respect_document_boundaries,
        num_proc=phase.data.tokenizer_num_proc,
    )


def _eval_blocks(
    name: str,
    data: PretrainDataConfig,
    tokenizer,
    *,
    verbose: bool,
) -> Dataset:
    _, eval_raw = load_train_eval(data, verbose=verbose)
    eval_dataset = tokenize_and_chunk(
        eval_raw,
        tokenizer,
        block_size=data.block_size,
        text_column=(data.eval_source or data.source).text_column,
        respect_document_boundaries=data.respect_document_boundaries,
        num_proc=data.tokenizer_num_proc,
    )
    if len(eval_dataset) == 0:
        raise ValueError(
            f"eval set {name!r} produced no blocks; "
            "increase max_eval_samples or lower block_size."
        )
    return eval_dataset


def prepare_curriculum_datasets(
    curriculum: Curriculum,
    tokenizer,
    *,
    samples_per_step: int,
    verbose: bool = True,
) -> tuple[IterableDataset, Dataset | dict[str, Dataset]]:
    """Load/tokenize a curriculum into one train stream plus fixed eval set(s).

    Train is a single ``IterableDataset`` of exactly
    ``total_steps() * samples_per_step`` blocks, phases in order. Eval is a
    map-style ``Dataset`` (single set) or a name → ``Dataset`` dict.
    """
    curriculum.validate()
    if samples_per_step < 1:
        raise ValueError(f"samples_per_step must be >= 1, got {samples_per_step}")

    if verbose:
        step(
            f"Preparing curriculum: {len(curriculum.phases)} phases, "
            f"{curriculum.total_steps():,} total steps "
            f"({samples_per_step} blocks/step)...",
            logger=log,
        )

    parts: list[IterableDataset] = []
    for index, phase in enumerate(curriculum.phases, start=1):
        num_samples = phase.steps * samples_per_step
        if verbose:
            detail(
                f"phase {index}/{len(curriculum.phases)}: {phase.name()} — "
                f"{phase.steps:,} steps ({num_samples:,} blocks)",
                logger=log,
            )
        blocks = _phase_blocks(phase, tokenizer, verbose=verbose)
        parts.append(bounded_phase_blocks(blocks, num_samples=num_samples))
    train_dataset = chain_phase_blocks(parts)

    if verbose:
        blank(logger=log)
        step("Preparing curriculum eval set(s)...", logger=log)
    eval_configs = curriculum.eval_configs()
    eval_sets = {
        name: _eval_blocks(name, data, tokenizer, verbose=verbose)
        for name, data in eval_configs.items()
    }
    if verbose:
        for name, dataset in eval_sets.items():
            detail(f"eval blocks [{name}]: {len(dataset):,}", logger=log)

    if curriculum.eval_data is None or isinstance(
        curriculum.eval_data, PretrainDataConfig
    ):
        return train_dataset, eval_sets["eval"]
    return train_dataset, eval_sets
