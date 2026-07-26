"""XLA / TPU training compatibility helpers."""

from __future__ import annotations


def native_gradient_checkpointing_supported(device: str) -> bool:
    """False on XLA: ``torch.utils.checkpoint`` does ``getattr(torch, device)``.

    That path raises ``AttributeError: module 'torch' has no attribute 'xla'``.
    PyTorch/XLA provides ``torch_xla.utils.checkpoint`` instead; until HF Trainer
    wires that in automatically, keep native checkpointing off on TPU.
    """
    return device != "xla"
