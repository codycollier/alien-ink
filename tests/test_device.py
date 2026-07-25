"""Tests for device / precision helpers."""

from __future__ import annotations

import alien_ink.device as device_mod


def test_distributed_world_size_defaults_to_one(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert device_mod.distributed_world_size() == 1


def test_distributed_world_size_reads_env(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert device_mod.distributed_world_size() == 4


def test_distributed_world_size_invalid_env(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "nope")
    assert device_mod.distributed_world_size() == 1


def test_resolve_precision_cpu_is_fp32():
    use_fp16, use_bf16 = device_mod.resolve_precision("cpu")
    assert (use_fp16, use_bf16) == (False, False)


def test_resolve_precision_mps_is_fp32():
    use_fp16, use_bf16 = device_mod.resolve_precision("mps")
    assert (use_fp16, use_bf16) == (False, False)


def test_resolve_precision_cuda_prefers_bf16_when_supported(monkeypatch):
    monkeypatch.setattr(
        device_mod.torch.cuda,
        "is_bf16_supported",
        lambda: True,
    )
    use_fp16, use_bf16 = device_mod.resolve_precision(
        "cuda", prefer_bf16=True, prefer_fp16=True
    )
    assert (use_fp16, use_bf16) == (False, True)


def test_resolve_precision_cuda_falls_back_to_fp16(monkeypatch):
    monkeypatch.setattr(
        device_mod.torch.cuda,
        "is_bf16_supported",
        lambda: False,
    )
    use_fp16, use_bf16 = device_mod.resolve_precision(
        "cuda", prefer_bf16=True, prefer_fp16=True
    )
    assert (use_fp16, use_bf16) == (True, False)
