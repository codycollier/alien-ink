"""Tests for XLA/TPU training compatibility helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from alien_ink.hf.xla_ckpt import (
    native_gradient_checkpointing_supported,
    resolve_trainer_optim,
)

transformers = pytest.importorskip("transformers")

from alien_ink.hf.trainer import (  # noqa: E402
    CausalLmTrainerConfig,
    build_training_arguments,
)


def test_native_gradient_checkpointing_supported():
    assert native_gradient_checkpointing_supported("cuda") is True
    assert native_gradient_checkpointing_supported("cpu") is True
    assert native_gradient_checkpointing_supported("mps") is True
    assert native_gradient_checkpointing_supported("xla") is False


def test_resolve_trainer_optim_xla_avoids_fused():
    assert resolve_trainer_optim("xla") == "adamw_torch"
    assert resolve_trainer_optim("cuda") == "adamw_torch"
    assert resolve_trainer_optim("cuda", default="adamw_torch_fused") == (
        "adamw_torch_fused"
    )


def test_build_training_arguments_disables_grad_ckpt_on_xla(tmp_path: Path):
    cfg = CausalLmTrainerConfig(
        output_dir=tmp_path / "out",
        gradient_checkpointing=True,
        max_steps=10,
        report_to="none",
    )
    with patch(
        "alien_ink.hf.trainer.device_info",
        return_value=("xla", False, True),
    ):
        args = build_training_arguments(cfg, has_eval=False)
    assert args.gradient_checkpointing is False
    assert args.bf16 is True
    assert args.fp16 is False
    assert getattr(args.optim, "value", args.optim) == "adamw_torch"


def test_build_training_arguments_keeps_grad_ckpt_on_cuda(tmp_path: Path):
    cfg = CausalLmTrainerConfig(
        output_dir=tmp_path / "out",
        gradient_checkpointing=True,
        max_steps=10,
        report_to="none",
    )
    with patch(
        "alien_ink.hf.trainer.device_info",
        return_value=("cuda", False, True),
    ):
        args = build_training_arguments(cfg, has_eval=False)
    assert args.gradient_checkpointing is True
