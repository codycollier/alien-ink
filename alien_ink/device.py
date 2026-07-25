"""Torch device and mixed-precision selection."""

from __future__ import annotations

import os

import torch


def distributed_world_size() -> int:
    """Process group size from the environment (default 1 for single-process)."""
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        return 1


def resolve_device() -> str:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_precision(
    device: str,
    *,
    prefer_bf16: bool = True,
    prefer_fp16: bool = True,
) -> tuple[bool, bool]:
    """Return ``(use_fp16, use_bf16)`` for the given device.

    Prefers bf16 on Ampere+ GPUs: same dynamic range as fp32, so it avoids the
    loss-scaling / overflow instabilities that fp16 can hit. MPS stays on fp32;
    mixed precision there is still uneven across PyTorch builds.
    """
    if device == "cuda":
        use_bf16 = prefer_bf16 and torch.cuda.is_bf16_supported()
        use_fp16 = prefer_fp16 and not use_bf16
        return use_fp16, use_bf16
    return False, False


def device_info(
    *,
    prefer_bf16: bool = True,
    prefer_fp16: bool = True,
) -> tuple[str, bool, bool]:
    """Return ``(device, use_fp16, use_bf16)``."""
    device = resolve_device()
    use_fp16, use_bf16 = resolve_precision(
        device,
        prefer_bf16=prefer_bf16,
        prefer_fp16=prefer_fp16,
    )
    return device, use_fp16, use_bf16
