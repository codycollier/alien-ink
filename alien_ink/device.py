"""Torch device, mixed-precision selection, and accelerator fingerprinting."""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass
from typing import Any

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


@dataclass(frozen=True)
class AcceleratorInfo:
    """Hardware / runtime fingerprint for cross-machine experiment comparison."""

    device: str
    use_fp16: bool
    use_bf16: bool
    precision: str
    world_size: int
    gpu_count: int
    gpu_name: str | None
    gpu_memory_total_gb: float | None
    cuda_available: bool
    cuda_version: str | None
    cudnn_version: str | None
    torch_version: str
    platform: str
    python_version: str
    # Optional peak TFLOPS for the active precision (lookup; None if unknown).
    peak_tflops: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def precision_label(*, use_fp16: bool, use_bf16: bool) -> str:
    if use_bf16:
        return "bf16"
    if use_fp16:
        return "fp16"
    return "fp32"


# Approximate dense Tensor-core peaks for common cards (TFLOPS).
# Used only for MFU; throughput / FLOPs totals do not depend on this table.
_PEAK_TFLOPS: dict[str, dict[str, float]] = {
    # Consumer
    "NVIDIA GeForce RTX 3070": {"fp16": 20.31, "bf16": 20.31, "fp32": 20.31},
    "NVIDIA GeForce RTX 3080": {"fp16": 29.77, "bf16": 29.77, "fp32": 29.77},
    "NVIDIA GeForce RTX 3090": {"fp16": 35.58, "bf16": 35.58, "fp32": 35.58},
    "NVIDIA GeForce RTX 4070": {"fp16": 29.15, "bf16": 29.15, "fp32": 29.15},
    "NVIDIA GeForce RTX 4080": {"fp16": 48.74, "bf16": 48.74, "fp32": 48.74},
    "NVIDIA GeForce RTX 4090": {"fp16": 82.58, "bf16": 82.58, "fp32": 82.58},
    # Data center / cloud
    "NVIDIA A10": {"fp16": 125.0, "bf16": 125.0, "fp32": 31.2},
    "NVIDIA A100-SXM4-40GB": {"fp16": 312.0, "bf16": 312.0, "fp32": 19.5},
    "NVIDIA A100-SXM4-80GB": {"fp16": 312.0, "bf16": 312.0, "fp32": 19.5},
    "NVIDIA A100 40GB PCIe": {"fp16": 312.0, "bf16": 312.0, "fp32": 19.5},
    "NVIDIA A100 80GB PCIe": {"fp16": 312.0, "bf16": 312.0, "fp32": 19.5},
    "NVIDIA L4": {"fp16": 120.0, "bf16": 120.0, "fp32": 30.3},
    "NVIDIA L40S": {"fp16": 362.0, "bf16": 362.0, "fp32": 91.6},
    "NVIDIA H100 PCIe": {"fp16": 756.5, "bf16": 756.5, "fp32": 51.0},
    "NVIDIA H100 80GB HBM3": {"fp16": 989.0, "bf16": 989.0, "fp32": 67.0},
    "Tesla T4": {"fp16": 65.0, "bf16": 0.0, "fp32": 8.1},
    "Tesla V100-SXM2-16GB": {"fp16": 125.0, "bf16": 0.0, "fp32": 15.7},
}


def lookup_peak_tflops(gpu_name: str | None, precision: str) -> float | None:
    """Return approximate peak TFLOPS for ``gpu_name`` at ``precision``, if known."""
    if not gpu_name:
        return None
    # Exact match first, then substring (driver names vary slightly).
    entry = _PEAK_TFLOPS.get(gpu_name)
    if entry is None:
        for key, value in _PEAK_TFLOPS.items():
            if key in gpu_name or gpu_name in key:
                entry = value
                break
    if entry is None:
        return None
    peak = entry.get(precision)
    if peak is None or peak <= 0:
        return None
    return peak


def collect_accelerator_info(
    *,
    prefer_bf16: bool = True,
    prefer_fp16: bool = True,
) -> AcceleratorInfo:
    """Snapshot device, precision, GPU identity, and runtime versions."""
    device, use_fp16, use_bf16 = device_info(
        prefer_bf16=prefer_bf16,
        prefer_fp16=prefer_fp16,
    )
    prec = precision_label(use_fp16=use_fp16, use_bf16=use_bf16)

    gpu_count = 0
    gpu_name: str | None = None
    gpu_memory_total_gb: float | None = None
    cuda_version: str | None = None
    cudnn_version: str | None = None
    cuda_available = bool(torch.cuda.is_available())

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_memory_total_gb = round(props.total_memory / (1024**3), 2)
        cuda_version = getattr(torch.version, "cuda", None)
        try:
            cudnn_version = str(torch.backends.cudnn.version())
        except Exception:
            cudnn_version = None
    elif device == "mps":
        gpu_name = "Apple MPS"
        gpu_count = 1

    return AcceleratorInfo(
        device=device,
        use_fp16=use_fp16,
        use_bf16=use_bf16,
        precision=prec,
        world_size=distributed_world_size(),
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        gpu_memory_total_gb=gpu_memory_total_gb,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        torch_version=torch.__version__,
        platform=platform.platform(),
        python_version=platform.python_version(),
        peak_tflops=lookup_peak_tflops(gpu_name, prec),
    )
