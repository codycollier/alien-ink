"""Training run metrics: FLOPs estimates, throughput, and end-of-run summaries."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from alien_ink.com.device import AcceleratorInfo
from alien_ink.com.log import banner, detail, get_logger, step

log = get_logger("hf.metrics")

# Kaplan et al.: ~6N FLOPs per training token for Transformer LM (fwd+bwd).
FLOPS_PER_PARAM_PER_TOKEN = 6

RUN_CONFIG_FILENAME = "run_config.json"
RUN_SUMMARY_FILENAME = "run_summary.json"


@dataclass(frozen=True)
class ModelSize:
    """Parameter counts used for FLOP accounting."""

    total_params: int
    trainable_params: int
    non_embedding_params: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunSummary:
    """Derived metrics written when training stops (success or interrupt)."""

    # Identity
    run_name: str
    run_label: str
    status: str  # completed | interrupted | failed

    # Steps / tokens
    global_step: int
    max_steps: int
    tokens_per_step: int
    tokens_trained: int

    # Time / throughput
    train_runtime_sec: float | None
    steps_per_sec: float | None
    tokens_per_sec: float | None
    samples_per_sec: float | None

    # FLOPs (Kaplan 6N approximation on non-embedding params)
    flops_total: float | None
    flops_per_sec: float | None
    tflops_per_sec: float | None
    mfu: float | None

    # Loss
    train_loss: float | None

    # Context
    model: dict[str, Any] = field(default_factory=dict)
    accelerator: dict[str, Any] = field(default_factory=dict)
    trainer_metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def count_model_params(model, *, vocab_size: int | None = None) -> ModelSize:
    """Count total / trainable / non-embedding parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    embed = 0
    seen: set[int] = set()
    # GPT-2 / HF naming (dedupe by storage id for tied weights).
    for name in ("wte", "wpe", "embed_tokens", "embeddings"):
        module = getattr(getattr(model, "transformer", model), name, None)
        if module is None:
            module = getattr(model, name, None)
        weight = getattr(module, "weight", None) if module is not None else None
        if weight is None:
            continue
        key = id(weight)
        if key in seen:
            continue
        seen.add(key)
        embed += weight.numel()

    # Fallback: vocab * hidden if embeddings not found by name.
    if embed == 0 and vocab_size is not None:
        cfg = getattr(model, "config", None)
        hidden = getattr(cfg, "n_embd", None) or getattr(cfg, "hidden_size", None)
        n_pos = getattr(cfg, "n_positions", None) or getattr(
            cfg, "max_position_embeddings", 0
        )
        if hidden:
            embed = int(vocab_size) * int(hidden) + int(n_pos or 0) * int(hidden)

    non_embed = max(0, total - embed)
    return ModelSize(
        total_params=total,
        trainable_params=trainable,
        non_embedding_params=non_embed,
    )


def estimate_train_flops(*, non_embedding_params: int, tokens: int) -> float:
    """Approximate training FLOPs: ``6 * N_non_embed * tokens`` (Kaplan)."""
    return float(FLOPS_PER_PARAM_PER_TOKEN * non_embedding_params * tokens)


def collect_software_versions() -> dict[str, str | None]:
    """Versions of key packages for reproducibility notes."""

    def _ver(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {
        "alien_ink": _ver("alien-ink"),
        "torch": _ver("torch"),
        "transformers": _ver("transformers"),
        "datasets": _ver("datasets"),
        "accelerate": _ver("accelerate"),
        "wandb": _ver("wandb"),
    }


def build_run_config_payload(
    *,
    run_label: str,
    run_name: str,
    title: str,
    configs: dict[str, Any],
    accelerator: AcceleratorInfo,
    model_size: ModelSize | None = None,
    tokens_per_step: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """JSON-serializable resolved experiment config for offline reproduction."""
    from alien_ink.com.wb import serialize_config

    payload: dict[str, Any] = {
        "run_label": run_label,
        "run_name": run_name,
        "title": title,
        "created_at_unix": time.time(),
        "accelerator": accelerator.as_dict(),
        "software": collect_software_versions(),
        "tokens_per_optimizer_step": tokens_per_step,
    }
    if model_size is not None:
        payload["model"] = model_size.as_dict()
    for name, cfg in configs.items():
        payload[name] = serialize_config(cfg)
    if extra:
        payload.update(extra)
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def save_run_config(output_dir: Path, payload: dict[str, Any]) -> Path:
    """Write ``run_config.json`` under the run output directory."""
    path = write_json(output_dir / RUN_CONFIG_FILENAME, payload)
    detail(f"run config: {path}", logger=log)
    return path


def _first_float(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                continue
    return None


def build_run_summary(
    *,
    run_name: str,
    run_label: str,
    status: str,
    global_step: int,
    max_steps: int,
    tokens_per_step: int,
    train_runtime_sec: float | None,
    train_loss: float | None,
    model_size: ModelSize,
    accelerator: AcceleratorInfo,
    trainer_metrics: dict[str, Any] | None = None,
    samples_per_sec: float | None = None,
    steps_per_sec: float | None = None,
) -> RunSummary:
    """Derive tokens/FLOPs/throughput/MFU from step counts and wall time."""
    metrics = dict(trainer_metrics or {})
    runtime = train_runtime_sec
    if runtime is None:
        runtime = _first_float(metrics, "train_runtime")
    if samples_per_sec is None:
        samples_per_sec = _first_float(metrics, "train_samples_per_second")
    if steps_per_sec is None:
        steps_per_sec = _first_float(metrics, "train_steps_per_second")
    if train_loss is None:
        train_loss = _first_float(metrics, "train_loss")

    tokens_trained = int(global_step) * int(tokens_per_step)
    tokens_per_sec: float | None = None
    if runtime is not None and runtime > 0:
        tokens_per_sec = tokens_trained / runtime
        if steps_per_sec is None:
            steps_per_sec = global_step / runtime

    flops_total = estimate_train_flops(
        non_embedding_params=model_size.non_embedding_params,
        tokens=tokens_trained,
    )
    flops_per_sec: float | None = None
    tflops_per_sec: float | None = None
    mfu: float | None = None
    if runtime is not None and runtime > 0:
        flops_per_sec = flops_total / runtime
        tflops_per_sec = flops_per_sec / 1e12
        if accelerator.peak_tflops and accelerator.peak_tflops > 0:
            mfu = tflops_per_sec / accelerator.peak_tflops

    return RunSummary(
        run_name=run_name,
        run_label=run_label,
        status=status,
        global_step=int(global_step),
        max_steps=int(max_steps),
        tokens_per_step=int(tokens_per_step),
        tokens_trained=tokens_trained,
        train_runtime_sec=runtime,
        steps_per_sec=steps_per_sec,
        tokens_per_sec=tokens_per_sec,
        samples_per_sec=samples_per_sec,
        flops_total=flops_total,
        flops_per_sec=flops_per_sec,
        tflops_per_sec=tflops_per_sec,
        mfu=mfu,
        train_loss=train_loss,
        model=model_size.as_dict(),
        accelerator=accelerator.as_dict(),
        trainer_metrics=metrics,
    )


def save_run_summary(output_dir: Path, summary: RunSummary) -> Path:
    """Write ``run_summary.json`` under the run output directory."""
    path = write_json(output_dir / RUN_SUMMARY_FILENAME, summary.as_dict())
    detail(f"run summary: {path}", logger=log)
    return path


def log_run_summary(summary: RunSummary) -> None:
    """Pretty-print the end-of-run metrics block."""
    banner("Run summary", logger=log)
    step(f"status: {summary.status}", logger=log)
    detail(
        f"steps: {summary.global_step:,} / {summary.max_steps:,}",
        logger=log,
    )
    detail(f"tokens/step: {summary.tokens_per_step:,}", logger=log)
    detail(f"tokens trained: {summary.tokens_trained:,}", logger=log)
    if summary.train_loss is not None:
        detail(f"train loss: {summary.train_loss:.6f}", logger=log)
    if summary.train_runtime_sec is not None:
        detail(f"runtime: {summary.train_runtime_sec:,.2f}s", logger=log)
    if summary.steps_per_sec is not None:
        detail(f"steps/sec: {summary.steps_per_sec:.4f}", logger=log)
    if summary.tokens_per_sec is not None:
        detail(f"tokens/sec: {summary.tokens_per_sec:,.1f}", logger=log)
    if summary.tflops_per_sec is not None:
        detail(f"TFLOP/s (est.): {summary.tflops_per_sec:.3f}", logger=log)
    if summary.mfu is not None:
        detail(f"MFU (est.): {100.0 * summary.mfu:.1f}%", logger=log)
    elif summary.accelerator.get("gpu_name"):
        detail(
            "MFU: n/a (no peak TFLOPS entry for "
            f"{summary.accelerator.get('gpu_name')})",
            logger=log,
        )
    gpu = summary.accelerator.get("gpu_name") or summary.accelerator.get("device")
    detail(
        f"accelerator: {gpu} "
        f"({summary.accelerator.get('precision')}, "
        f"world_size={summary.accelerator.get('world_size')})",
        logger=log,
    )


def push_summary_to_wandb(summary: RunSummary) -> None:
    """Update the active W&B run summary when a run is in progress."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    payload = {
        k: v
        for k, v in summary.as_dict().items()
        if k not in {"accelerator", "model", "trainer_metrics"} and v is not None
    }
    # Flatten a few high-value accelerator fields for easy W&B filtering.
    payload["gpu_name"] = summary.accelerator.get("gpu_name")
    payload["precision"] = summary.accelerator.get("precision")
    payload["world_size"] = summary.accelerator.get("world_size")
    payload["torch_version"] = summary.accelerator.get("torch_version")
    payload["total_params"] = summary.model.get("total_params")
    payload["non_embedding_params"] = summary.model.get("non_embedding_params")
    wandb.run.summary.update(payload)
