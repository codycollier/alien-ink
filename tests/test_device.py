"""Tests for device / precision helpers."""

from __future__ import annotations

import alien_ink.device as device_mod


def test_distributed_world_size_defaults_to_one(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(device_mod, "_xla_world_size", lambda: None)
    assert device_mod.distributed_world_size() == 1


def test_distributed_world_size_reads_env(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert device_mod.distributed_world_size() == 4


def test_distributed_world_size_invalid_env(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "nope")
    monkeypatch.setattr(device_mod, "_xla_world_size", lambda: None)
    assert device_mod.distributed_world_size() == 1


def test_distributed_world_size_falls_back_to_xla(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(device_mod, "_xla_world_size", lambda: 8)
    assert device_mod.distributed_world_size() == 8


def test_resolve_precision_cpu_is_fp32():
    use_fp16, use_bf16 = device_mod.resolve_precision("cpu")
    assert (use_fp16, use_bf16) == (False, False)


def test_resolve_precision_mps_is_fp32():
    use_fp16, use_bf16 = device_mod.resolve_precision("mps")
    assert (use_fp16, use_bf16) == (False, False)


def test_resolve_precision_xla_prefers_bf16():
    use_fp16, use_bf16 = device_mod.resolve_precision(
        "xla", prefer_bf16=True, prefer_fp16=True
    )
    assert (use_fp16, use_bf16) == (False, True)


def test_resolve_precision_xla_fp16_when_bf16_disabled():
    use_fp16, use_bf16 = device_mod.resolve_precision(
        "xla", prefer_bf16=False, prefer_fp16=True
    )
    assert (use_fp16, use_bf16) == (True, False)


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


def test_resolve_device_prefers_xla_over_mps(monkeypatch):
    monkeypatch.setattr(device_mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_mod, "is_xla_tpu_available", lambda: True)
    assert device_mod.resolve_device() == "xla"


def test_lookup_peak_tflops_known_and_unknown():
    assert device_mod.lookup_peak_tflops("NVIDIA GeForce RTX 3070", "bf16") == 20.31
    assert device_mod.lookup_peak_tflops("TPU v6e-1", "bf16") == 918.0
    assert device_mod.lookup_peak_tflops("Totally Fake GPU", "bf16") is None
    assert device_mod.lookup_peak_tflops(None, "bf16") is None


def test_collect_accelerator_info_cpu(monkeypatch):
    monkeypatch.setattr(device_mod, "resolve_device", lambda: "cpu")
    monkeypatch.setattr(device_mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_mod, "is_torch_xla_available", lambda: False)
    monkeypatch.setattr(device_mod, "_xla_world_size", lambda: None)
    info = device_mod.collect_accelerator_info()
    assert info.device == "cpu"
    assert info.precision == "fp32"
    assert info.world_size >= 1
    assert info.torch_version
    assert info.gpu_name is None
    assert info.xla_available is False


def test_collect_accelerator_info_xla(monkeypatch):
    monkeypatch.setattr(device_mod, "resolve_device", lambda: "xla")
    monkeypatch.setattr(device_mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_mod, "is_torch_xla_available", lambda: True)
    monkeypatch.setattr(device_mod, "_xla_device_count", lambda: 8)
    monkeypatch.setattr(device_mod, "_tpu_chip_name", lambda: "v5e-8")
    monkeypatch.setattr(device_mod, "_xla_world_size", lambda: 8)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    info = device_mod.collect_accelerator_info(prefer_bf16=True, prefer_fp16=True)
    assert info.device == "xla"
    assert info.precision == "bf16"
    assert info.gpu_count == 8
    assert info.gpu_name == "v5e-8"
    assert info.xla_available is True
    assert info.world_size == 8
