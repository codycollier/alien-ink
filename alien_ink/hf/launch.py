"""TPU / notebook launch helpers for Hugging Face training entrypoints."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from alien_ink.device import distributed_world_size, is_xla_tpu_available
from alien_ink.log import detail, get_logger, step

log = get_logger("hf.launch")

__all__ = [
    "in_notebook",
    "launch_tpu",
    "should_auto_launch_tpu",
    "tpu_num_processes",
    "warn_if_tpu_single_process",
]


def in_notebook() -> bool:
    """True when running under an IPython/Jupyter kernel (e.g. Colab)."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    ipython = get_ipython()
    if ipython is None:
        return False
    return ipython.__class__.__name__ == "ZMQInteractiveShell"


def tpu_num_processes(default: int = 1) -> int:
    """Best-effort TPU process count for ``notebook_launcher`` / xla_spawn.

    Defaults to ``1`` (Colab TPU v6e-1 is a single chip). Multi-chip VMs should
    set ``TPU_NUM_DEVICES`` or rely on ``torch_xla.runtime.global_device_count``.
    """
    for key in ("TPU_NUM_DEVICES", "TPU_PROCESS_COUNT", "COLAB_TPU_NUM_DEVICES"):
        raw = os.environ.get(key)
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
    if is_xla_tpu_available():
        try:
            import torch_xla.runtime as xr

            count = int(xr.global_device_count())
            if count > 0:
                return count
        except Exception:
            pass
    return default


def should_auto_launch_tpu(*, force: bool | None = None) -> bool:
    """Whether to wrap training in Accelerate's ``notebook_launcher``.

    Defaults to True on a TPU notebook when not already inside a multi-process
    XLA job. Pass ``force=True/False`` to override detection.
    """
    if force is False:
        return False
    if not is_xla_tpu_available():
        return False
    if force is True:
        return True
    if distributed_world_size() > 1:
        return False
    return in_notebook()


def launch_tpu(
    function: Callable[..., Any],
    args: tuple[Any, ...] = (),
    *,
    num_processes: int | None = None,
    mixed_precision: str = "bf16",
) -> None:
    """Spawn TPU workers via Accelerate ``notebook_launcher``."""
    from accelerate import notebook_launcher

    processes = tpu_num_processes() if num_processes is None else max(1, num_processes)
    step(
        f"Launching on TPU via notebook_launcher (num_processes={processes}, "
        f"mixed_precision={mixed_precision})...",
        logger=log,
    )
    notebook_launcher(
        function,
        args=args,
        num_processes=processes,
        mixed_precision=mixed_precision,
    )


def warn_if_tpu_single_process() -> None:
    """Log a hint when a TPU is visible but training is single-process."""
    if not is_xla_tpu_available():
        return
    if distributed_world_size() > 1:
        return
    if in_notebook():
        return
    detail(
        "TPU detected with world_size=1. For multi-core training launch with "
        "`python -m torch_xla.distributed.xla_spawn --num_cores=N -m ...` "
        "or `accelerate launch` (notebooks auto-launch via notebook_launcher).",
        logger=log,
    )
