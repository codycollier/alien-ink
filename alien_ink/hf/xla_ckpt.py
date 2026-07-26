"""XLA / TPU training compatibility helpers."""

from __future__ import annotations


def native_gradient_checkpointing_supported(device: str) -> bool:
    """False on XLA: ``torch.utils.checkpoint`` does ``getattr(torch, device)``.

    That path raises ``AttributeError: module 'torch' has no attribute 'xla'``.
    PyTorch/XLA provides ``torch_xla.utils.checkpoint`` instead; until HF Trainer
    wires that in automatically, keep native checkpointing off on TPU.
    """
    return device != "xla"


def resolve_trainer_optim(device: str, *, default: str | None = None) -> str:
    """Pick an HF ``TrainingArguments.optim`` value safe for ``device``.

    Transformers defaults to ``adamw_torch_fused`` on recent torch, but fused
    AdamW rejects XLA params (``fused=True`` only allows cuda/mps/cpu/...).
    """
    if device == "xla":
        return "adamw_torch"
    return default if default is not None else "adamw_torch"