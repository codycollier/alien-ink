"""Weights & Biases run helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Iterator

from alien_ink.device import AcceleratorInfo, collect_accelerator_info
from alien_ink.env import DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT, EnvConfig
from alien_ink.log import detail, get_logger, step

log = get_logger("wb")

__all__ = [
    "DEFAULT_WANDB_ENTITY",
    "DEFAULT_WANDB_PROJECT",
    "build_run_config",
    "resolve_wandb_root",
    "serialize_config",
    "set_wandb_dir",
    "wandb_run",
]


def serialize_config(obj: Any) -> dict[str, Any]:
    """Flatten a dataclass (or mapping) for logging; stringify Paths recursively."""

    def _convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        return value

    if hasattr(obj, "__dataclass_fields__"):
        raw = asdict(obj)
    elif isinstance(obj, dict):
        raw = dict(obj)
    else:
        raw = {f.name: getattr(obj, f.name) for f in fields(obj)}
    return {k: _convert(v) for k, v in raw.items()}


def build_run_config(
    *,
    run_label: str,
    env: EnvConfig,
    configs: dict[str, Any],
    prefer_bf16: bool = True,
    prefer_fp16: bool = True,
    accelerator: AcceleratorInfo | None = None,
    tokens_per_step: int | None = None,
    model: dict[str, Any] | None = None,
    software: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Flat config dict suitable for ``wandb.init(config=...)``.

    When ``configs`` has multiple entries, keys are namespaced as ``{name}.{field}``.
    A single entry is flattened without a prefix. Accelerator / software fields
    are prefixed with ``accel.`` / ``sw.`` for easy W&B filtering across GPUs.
    """
    accel = accelerator or collect_accelerator_info(
        prefer_bf16=prefer_bf16,
        prefer_fp16=prefer_fp16,
    )
    flat: dict[str, Any] = {
        "run_label": run_label,
        "wandb_entity": env.wandb_entity,
        "wandb_project": env.wandb_project,
        "device": accel.device,
        "use_fp16": accel.use_fp16,
        "use_bf16": accel.use_bf16,
        "precision": accel.precision,
        "world_size": accel.world_size,
        "tokens_per_optimizer_step": tokens_per_step,
    }
    for k, v in accel.as_dict().items():
        flat[f"accel.{k}"] = v
    if model:
        for k, v in model.items():
            flat[f"model.{k}"] = v
    if software:
        for k, v in software.items():
            flat[f"sw.{k}"] = v
    if len(configs) == 1:
        flat.update(serialize_config(next(iter(configs.values()))))
    else:
        for name, cfg in configs.items():
            for k, v in serialize_config(cfg).items():
                flat[f"{name}.{k}"] = v
    return flat


def resolve_wandb_root(output_dir: Path | str) -> Path:
    """Absolute run root; W&B local files go under ``<root>/wandb``."""
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def set_wandb_dir(output_dir: Path | str) -> Path:
    """Point W&B local run + artifact dirs under ``output_dir`` (absolute)."""
    root = resolve_wandb_root(output_dir)
    os.environ["WANDB_DIR"] = str(root)
    # Default is ``./artifacts`` in cwd; keep downloads with the run.
    os.environ["WANDB_ARTIFACT_DIR"] = str(root / "artifacts")
    return root


@contextmanager
def wandb_run(
    *,
    entity: str,
    project: str,
    name: str,
    config: dict[str, Any],
    dir: Path | str,
    enabled: bool = True,
) -> Iterator[Any]:
    """Init a W&B run under ``dir``, yield it, then finish (even on error).

    When ``enabled`` is False, yields ``None`` without importing or calling wandb.
    """
    if not enabled:
        yield None
        return

    import wandb  # lazy: keep wandb optional until a run is actually started

    root = set_wandb_dir(dir)
    step("Starting Weights & Biases run...", logger=log)
    detail(f"entity:    {entity}", logger=log)
    detail(f"project:   {project}", logger=log)
    detail(f"name:      {name}", logger=log)
    detail(f"wandb dir: {root / 'wandb'}", logger=log)
    run = wandb.init(
        entity=entity,
        project=project,
        name=name,
        config=config,
        dir=str(root),
    )
    try:
        yield run
    finally:
        wandb.finish()
