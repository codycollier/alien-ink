"""Torch device, mixed-precision selection, and accelerator fingerprinting."""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass
from typing import Any

import torch


def distributed_world_size() -> int:
    """Process group size (env ``WORLD_SIZE``, else XLA runtime, else 1)."""
    try:
        env = os.environ.get("WORLD_SIZE")
        if env is not None:
            return max(1, int(env))
    except ValueError:
        pass
    xla_ws = _xla_world_size()
    if xla_ws is not None:
        return xla_ws
    return 1


def is_torch_xla_available() -> bool:
    """True when the ``torch_xla`` package imports successfully."""
    try:
        import torch_xla  # noqa: F401
    except ImportError:
        return False
    return True


def is_xla_tpu_available() -> bool:
    """True when PyTorch/XLA reports a TPU backend."""
    if not is_torch_xla_available():
        return False
    try:
        import torch_xla.runtime as xr

        return str(xr.device_type()).upper() == "TPU"
    except Exception:
        return False


def _xla_world_size() -> int | None:
    if not is_torch_xla_available():
        return None
    try:
        import torch_xla.runtime as xr

        return max(1, int(xr.world_size()))
    except Exception:
        return None


def _xla_device_count() -> int:
    if not is_torch_xla_available():
        return 0
    try:
        import torch_xla.runtime as xr

        return max(0, int(xr.global_device_count()))
    except Exception:
        try:
            import torch_xla.runtime as xr

            return max(0, int(xr.world_size()))
        except Exception:
            return 0


def _tpu_chip_name() -> str | None:
    for key in (
        "TPU_ACCELERATOR_TYPE",
        "ACCELERATOR_TYPE",
        "TPU_TYPE",
        "TPU_NAME",
    ):
        value = os.environ.get(key)
        if value:
            return value
    # Colab TPU notebooks in this project are assumed to be v6e-1.
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"):
        return "TPU v6e-1"
    return None


def resolve_device() -> str:
    """Prefer CUDA, then XLA/TPU, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if is_xla_tpu_available():
        return "xla"
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
    loss-scaling / overflow instabilities that fp16 can hit. TPUs also prefer
    bf16. MPS stays on fp32; mixed precision there is still uneven across
    PyTorch builds.
    """
    if device == "cuda":
        use_bf16 = prefer_bf16 and torch.cuda.is_bf16_supported()
        use_fp16 = prefer_fp16 and not use_bf16
        return use_fp16, use_bf16
    if device == "xla":
        # Cloud TPUs are efficient in bf16; avoid fp16 loss-scaling paths.
        use_bf16 = prefer_bf16
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


def torch_device(device: str | None = None) -> torch.device | str:
    """Resolve a library device string to a ``torch.device`` (XLA-aware)."""
    resolved = resolve_device() if device is None else device
    if resolved == "xla":
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    return resolved


def move_module_to_device(module: torch.nn.Module, device: str | None = None):
    """Move ``module`` onto the resolved device (handles XLA/TPU)."""
    return module.to(torch_device(device))


def xla_mark_step() -> None:
    """Flush the pending XLA graph when running on TPU; no-op otherwise."""
    if resolve_device() != "xla":
        return
    try:
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    except Exception:
        return


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
    xla_available: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def precision_label(*, use_fp16: bool, use_bf16: bool) -> str:
    if use_bf16:
        return "bf16"
    if use_fp16:
        return "fp16"
    return "fp32"


# Approximate peak TFLOPS for MFU (throughput / FLOPs totals do not use this).
# GPU FP16/BF16 entries are dense Tensor-core peaks (no sparsity). Aligns with
# alien_ink.hf.hardware comparison baseline (RTX 3070 = 40.6 FP16).
_PEAK_TFLOPS: dict[str, dict[str, float]] = {
    # Consumer
    "NVIDIA GeForce RTX 3070": {"fp16": 40.6, "bf16": 40.6, "fp32": 20.31},
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
    "NVIDIA L4": {"fp16": 121.0, "bf16": 121.0, "fp32": 30.3},
    "NVIDIA L40S": {"fp16": 362.0, "bf16": 362.0, "fp32": 91.6},
    # Colab G4 = RTX PRO 6000 Blackwell Server Edition (~96 GB).
    # Table peak 500 FP16; Colab marketing also cites ~960 BF16.
    "NVIDIA RTX PRO 6000 Blackwell": {"fp16": 500.0, "bf16": 960.0, "fp32": 0.0},
    "RTX PRO 6000": {"fp16": 500.0, "bf16": 960.0, "fp32": 0.0},
    "NVIDIA H100 PCIe": {"fp16": 756.5, "bf16": 756.5, "fp32": 51.0},
    "NVIDIA H100 80GB HBM3": {"fp16": 989.0, "bf16": 989.0, "fp32": 67.0},
    "Tesla T4": {"fp16": 65.0, "bf16": 0.0, "fp32": 8.1},
    "Tesla V100-SXM2-16GB": {"fp16": 125.0, "bf16": 0.0, "fp32": 15.7},
    # Cloud TPU (per-chip peaks; Colab TPU notebook assumed v6e-1 Trillium)
    "TPU v6e": {"fp16": 918.0, "bf16": 918.0, "fp32": 0.0},
    "TPU v6e-1": {"fp16": 918.0, "bf16": 918.0, "fp32": 0.0},
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
    """Snapshot device, precision, GPU/TPU identity, and runtime versions."""
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
    xla_available = is_torch_xla_available()

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
    elif device == "xla":
        gpu_count = _xla_device_count() or distributed_world_size()
        gpu_name = _tpu_chip_name() or "TPU"
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
        xla_available=xla_available,
    )


_INTROSPECT_BORDER = 80 * "-"
_INTROSPECT_LABEL_WIDTH = 12  # includes trailing colon
_INTROSPECT_TITLE = ":: introspection complete — echoes from the substrate"


def _torch_xla_version() -> str | None:
    if not is_torch_xla_available():
        return None
    try:
        import torch_xla

        return getattr(torch_xla, "__version__", None)
    except Exception:
        return None


def _xla_device_type() -> str | None:
    if not is_torch_xla_available():
        return None
    try:
        import torch_xla.runtime as xr

        return str(xr.device_type())
    except Exception:
        return None


def _xla_device_str() -> str | None:
    if not is_torch_xla_available():
        return None
    try:
        return str(torch_device("xla"))
    except Exception:
        return None


def _kv(label: str, value: Any) -> str:
    """Format ``label: value`` with a right-aligned label column."""
    return f"{label + ':':>{_INTROSPECT_LABEL_WIDTH}} {value}"


def introspect(
    *,
    prefer_bf16: bool = True,
    prefer_fp16: bool = True,
) -> str:
    """Return a bordered runtime summary for notebooks / verification cells.

    Safe in any environment (CPU, CUDA, MPS, XLA/TPU). Intended usage::

        import alien_ink
        print(alien_ink.stars)
        print(alien_ink.device.introspect())
    """
    info = collect_accelerator_info(
        prefer_bf16=prefer_bf16,
        prefer_fp16=prefer_fp16,
    )
    rows: list[tuple[str, Any]] = [("torch", info.torch_version)]

    if info.device == "xla":
        rows.append(("xla", _torch_xla_version() or "?"))
        xla_dev = _xla_device_str()
        if xla_dev:
            rows.append(("using", xla_dev))
        xla_type = _xla_device_type()
        if xla_type:
            rows.append(("device_type", f"{xla_type} → {info.device}"))
        rows.append(("tpu", info.gpu_name or "TPU"))
        rows.append(("cores", info.gpu_count))
    else:
        cuda_value: Any = info.cuda_available
        if info.cuda_available and info.cuda_version:
            cuda_value = f"True  ({info.cuda_version}"
            if info.cudnn_version:
                cuda_value += f", cudnn {info.cudnn_version}"
            cuda_value += ")"
        rows.append(("cuda", cuda_value))
        if info.cuda_available and info.gpu_name:
            gpu = info.gpu_name
            if info.gpu_memory_total_gb is not None:
                gpu = f"{gpu} ({info.gpu_memory_total_gb:g} GB)"
            rows.append(("gpu", gpu))
        elif info.device == "mps" and info.gpu_name:
            rows.append(("gpu", info.gpu_name))

    rows.append(("device", info.device))
    rows.append(("precision", info.precision))
    rows.append(("world_size", info.world_size))
    if info.peak_tflops is not None:
        rows.append(("peak_tflops", f"{info.peak_tflops:g}"))

    try:
        from alien_ink.hf.hardware import get_profile

        profile = get_profile()
        rows.append(
            (
                "profile",
                f"{profile.label}  batch: {profile.per_device_train_batch_size}  "
                f"accum: {profile.gradient_accumulation_steps}  "
                f"mem×{profile.memory_multiple:g}  "
                f"compute×{profile.vs_rtx_3070:g}",
            )
        )
    except Exception:
        pass

    body = "\n".join(_kv(label, value) for label, value in rows)
    return (
        f"{_INTROSPECT_BORDER}\n"
        f"{_INTROSPECT_TITLE}\n"
        f"\n"
        f"{body}\n"
        f"{_INTROSPECT_BORDER}"
    )
