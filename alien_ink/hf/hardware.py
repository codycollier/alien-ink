"""Training machine profiles: batch/run-name defaults scaled by accelerator capacity.

Primary targets (see README-exp.md):
  - Mist / local RTX 3070 (~8 GB)
  - Colab G4 = RTX PRO 6000 Blackwell (~96 GB)
  - Colab TPU v6e-1 Trillium (1×32 GB HBM)

``vs_rtx_3070`` and peak TFLOPS follow the project comparison table
(FP16 for GPUs, BF16 for TPU). Batch knobs prioritize filling device memory
with microbatches on cloud (accum=1); local keeps small microbatch + accum.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from alien_ink.device import resolve_device

# Comparison-table baseline (RTX 3070 local Mist machine).
_BASELINE_MEMORY_GB = 8.0
_BASELINE_PEAK_TFLOPS_FP16 = 40.6


@dataclass(frozen=True)
class AcceleratorProfile:
    """Training defaults + capacity metrics for one machine class."""

    kind: str  # "gpu" | "tpu" | "cpu" — appended to run names
    label: str
    hardware: str
    memory_gb: float
    peak_tflops: float
    precision: str  # "fp16" | "bf16" | "fp32"
    vs_rtx_3070: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    tpu_num_processes: int | None = None

    @property
    def memory_multiple(self) -> float:
        """Device HBM / VRAM relative to the 8 GB RTX 3070 baseline."""
        return self.memory_gb / _BASELINE_MEMORY_GB

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


# Local Mist: consumer RTX (~8 GB, e.g. 3070). Small microbatch + grad accum.
# Proven stable for GPT-2 @ block_size=1024; effective batch = 32.
LOCAL_RTX = AcceleratorProfile(
    kind="gpu",
    label="local-rtx",
    hardware="RTX 3070 (local Mist)",
    memory_gb=8.0,
    peak_tflops=_BASELINE_PEAK_TFLOPS_FP16,
    precision="fp16",
    vs_rtx_3070=1.0,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
)

# Colab L4-class mid-tier (~22–24 GB usable). Batch 32 @ block_size=1024
# OOMs (~25G activation headroom needed; TPU's 32G HBM is the floor that
# fits 32). Microbatch 16 + accum 2 keeps effective batch 32.
COLAB_L4 = AcceleratorProfile(
    kind="gpu",
    label="colab-l4",
    hardware="NVIDIA L4 (Colab / ~24 GB)",
    memory_gb=24.0,
    peak_tflops=121.0,
    precision="fp16",
    vs_rtx_3070=3.0,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
)

# Colab A100 40 GB (Ampere). memory_multiple=5, compute×7.7. Batch 32 /
# accum=1 — 40 GB is under the ~51G needed for GPT-2 batch 64 @ 1024.
COLAB_A100_40GB = AcceleratorProfile(
    kind="gpu",
    label="colab-a100-40gb",
    hardware="NVIDIA A100 40GB (Colab / Ampere)",
    memory_gb=40.0,
    peak_tflops=312.0,
    precision="fp16",
    vs_rtx_3070=7.7,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
)

# Colab G4 = RTX PRO 6000 Blackwell (~96 GB). memory_multiple=12;
# microbatch 64 fills HBM far better than L4's 16 (TPU OOM at
# batch 64 needed ~51G — G4's 96G has comfortable headroom).
COLAB_G4 = AcceleratorProfile(
    kind="gpu",
    label="colab-g4",
    hardware="RTX PRO 6000 Blackwell (Colab G4)",
    memory_gb=96.0,
    peak_tflops=500.0,
    precision="fp16",
    vs_rtx_3070=12.3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
)

# Colab TPU v6e-1 Trillium: 1 chip, 32 GB HBM. Batch 64 OOMs GPT-2 @
# block_size=1024 (~51G HBM for attention/logits temps); 32 fits.
COLAB_TPU_V6E1 = AcceleratorProfile(
    kind="tpu",
    label="colab-tpu-v6e1",
    hardware="TPU v6e-1 Trillium (Colab, 1-chip)",
    memory_gb=32.0,
    peak_tflops=918.0,
    precision="bf16",
    vs_rtx_3070=22.6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    tpu_num_processes=1,
)

CPU_PROFILE = AcceleratorProfile(
    kind="cpu",
    label="cpu",
    hardware="CPU",
    memory_gb=0.0,
    peak_tflops=0.0,
    precision="fp32",
    vs_rtx_3070=0.0,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
)

# Public aliases for known training machines.
TRAINING_MACHINES: dict[str, AcceleratorProfile] = {
    "mist-rtx-3070": LOCAL_RTX,
    "colab-g4": COLAB_G4,
    "colab-a100-40gb": COLAB_A100_40GB,
    "colab-tpu-v6e1": COLAB_TPU_V6E1,
    "colab-l4": COLAB_L4,
}


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


def _is_g4_gpu(name: str, mem: float | None) -> bool:
    """Colab G4 / RTX PRO 6000 Blackwell (~96 GB), not L4."""
    upper = name.upper()
    if "BLACKWELL" in upper:
        return True
    if "6000" in upper and ("PRO" in upper or "RTX" in upper):
        return True
    # Memory band: G4 is 96 GB; leave headroom above A100-80 / H100-80.
    if mem is not None and mem >= 90:
        return True
    return False


def _is_a100_40gb(name: str, mem: float | None) -> bool:
    """A100 40 GB (not the 80 GB SKU)."""
    upper = name.upper()
    if "A100" not in upper:
        return False
    if mem is not None:
        return 30.0 <= mem < 60.0
    return "40" in upper


def resolve_accelerator_profile() -> AcceleratorProfile:
    """Pick batch/run defaults for the current machine.

    Assumptions for this project:
    - Colab GPU G4 → RTX PRO 6000 Blackwell (~96 GB)
    - Colab GPU A100 40 GB → Ampere 40 GB recipe
    - Colab GPU L4 / other ≥20 GB mid-tier → L4 recipe
    - Colab TPU notebook → v6e-1 (1 chip, 32 GB HBM)
    - Local CUDA with ≤12 GB → consumer RTX / Mist recipe
    """
    device = resolve_device()
    if device == "xla":
        return COLAB_TPU_V6E1
    if device == "cuda":
        name = _cuda_name()
        mem = _cuda_memory_gb()
        if _is_g4_gpu(name, mem):
            return COLAB_G4
        if _is_a100_40gb(name, mem):
            return COLAB_A100_40GB
        if is_colab() or "L4" in name or (mem is not None and mem >= 20):
            return COLAB_L4
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
