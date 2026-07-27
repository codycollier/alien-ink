"""Accelerator profiles: batch/run-name defaults for local RTX, Colab G4, TPU v6e-1."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from alien_ink.device import resolve_device


@dataclass(frozen=True)
class AcceleratorProfile:
    """Training defaults + naming for the active accelerator class."""

    kind: str  # "gpu" | "tpu" | "cpu" — appended to run names
    label: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    tpu_num_processes: int | None = None


# Local consumer RTX (~8 GB, e.g. 3070): keep the proven small-batch + accum recipe.
LOCAL_RTX = AcceleratorProfile(
    kind="gpu",
    label="local-rtx",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
)

# Colab GPU G4 = NVIDIA L4 (~24 GB): large microbatches, no grad accum.
COLAB_G4 = AcceleratorProfile(
    kind="gpu",
    label="colab-g4",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
)

# Colab TPU v6e-1: single chip, 32 GB HBM — large batch, no accum, 1 process.
COLAB_TPU_V6E1 = AcceleratorProfile(
    kind="tpu",
    label="colab-tpu-v6e1",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
    tpu_num_processes=1,
)

CPU_PROFILE = AcceleratorProfile(
    kind="cpu",
    label="cpu",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
)


def is_colab() -> bool:
    """True when running under Google Colab."""
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"):
        return True
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def with_accelerator_suffix(run_name: str, kind: str) -> str:
    """Append ``-{kind}`` once (e.g. ``...-gpu``, ``...-tpu``)."""
    suffix = f"-{kind}"
    if run_name.endswith(suffix):
        return run_name
    return f"{run_name}{suffix}"


def _cuda_memory_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024**3)
    except Exception:
        return None


def _cuda_name() -> str:
    if not torch.cuda.is_available():
        return ""
    try:
        return torch.cuda.get_device_name(0) or ""
    except Exception:
        return ""


def resolve_accelerator_profile() -> AcceleratorProfile:
    """Pick batch/run defaults for the current machine.

    Assumptions for this project:
    - Colab GPU notebook → G4 (L4, ~24 GB)
    - Colab TPU notebook → v6e-1 (1 chip, 32 GB HBM)
    - Local CUDA with ≤12 GB → consumer RTX recipe
    """
    device = resolve_device()
    if device == "xla":
        return COLAB_TPU_V6E1
    if device == "cuda":
        name = _cuda_name()
        mem = _cuda_memory_gb()
        if is_colab() or "L4" in name or (mem is not None and mem >= 20):
            return COLAB_G4
        return LOCAL_RTX
    if device == "mps":
        # Apple GPU: keep the conservative local recipe; still tag as -gpu.
        return LOCAL_RTX
    return CPU_PROFILE


def trainer_overrides_for_profile(profile: AcceleratorProfile) -> dict[str, int]:
    """Fields applied onto ``CausalLmTrainerConfig`` for ``profile``."""
    return {
        "per_device_train_batch_size": profile.per_device_train_batch_size,
        "per_device_eval_batch_size": profile.per_device_eval_batch_size,
        "gradient_accumulation_steps": profile.gradient_accumulation_steps,
    }
