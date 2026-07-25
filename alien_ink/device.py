"""Torch device and mixed-precision selection."""

from __future__ import annotations

import torch


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_precision(
    device: str,
    *,
    prefer_bf16: bool = True,
    prefer_fp16: bool = True,
) -> tuple[bool, bool]:
    """Return ``(use_fp16, use_bf16)`` for the given device.

    Prefers bf16 on Ampere+ GPUs: same dynamic range as fp32, so it avoids the
    loss-scaling / overflow instabilities that fp16 can hit.
    """
    use_bf16 = (
        device == "cuda"
        and prefer_bf16
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = device == "cuda" and prefer_fp16 and not use_bf16
    return use_fp16, use_bf16


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
