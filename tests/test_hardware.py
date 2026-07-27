"""Tests for accelerator / training-machine profiles."""

from __future__ import annotations

import alien_ink.hf.hardware as hw


def test_with_accelerator_suffix_appends_once():
    assert hw.with_accelerator_suffix("run-a", "gpu") == "run-a-gpu"
    assert hw.with_accelerator_suffix("run-a-gpu", "gpu") == "run-a-gpu"
    assert hw.with_accelerator_suffix("run-a", "tpu") == "run-a-tpu"


def test_profile_metrics_local_rtx():
    p = hw.LOCAL_RTX
    assert p.memory_gb == 8.0
    assert p.memory_multiple == 1.0
    assert p.vs_rtx_3070 == 1.0
    assert p.effective_batch_size == 32
    assert p.per_device_train_batch_size == 2
    assert p.gradient_accumulation_steps == 16


def test_profile_metrics_colab_g4():
    p = hw.COLAB_G4
    assert p.hardware.startswith("RTX PRO 6000")
    assert p.memory_gb == 96.0
    assert p.memory_multiple == 12.0
    assert p.vs_rtx_3070 == 12.3
    assert p.peak_tflops == 500.0
    assert p.per_device_train_batch_size == 64
    assert p.gradient_accumulation_steps == 1
    assert p.effective_batch_size == 64


def test_profile_metrics_a100_40gb():
    p = hw.COLAB_A100_40GB
    assert p.memory_gb == 40.0
    assert p.memory_multiple == 5.0
    assert p.vs_rtx_3070 == 7.7
    assert p.peak_tflops == 312.0
    assert p.per_device_train_batch_size == 32
    assert p.gradient_accumulation_steps == 1
    assert p.effective_batch_size == 32


def test_profile_metrics_tpu_v6e1():
    p = hw.COLAB_TPU_V6E1
    assert p.memory_gb == 32.0
    assert p.memory_multiple == 4.0
    assert p.vs_rtx_3070 == 22.6
    assert p.peak_tflops == 918.0
    assert p.precision == "bf16"
    assert p.per_device_train_batch_size == 32
    assert p.gradient_accumulation_steps == 1
    assert p.tpu_num_processes == 1


def test_training_machines_registry():
    assert hw.TRAINING_MACHINES["mist-rtx-3070"] is hw.LOCAL_RTX
    assert hw.TRAINING_MACHINES["colab-g4"] is hw.COLAB_G4
    assert hw.TRAINING_MACHINES["colab-a100-40gb"] is hw.COLAB_A100_40GB
    assert hw.TRAINING_MACHINES["colab-tpu-v6e1"] is hw.COLAB_TPU_V6E1


def test_resolve_profile_tpu(monkeypatch):
    monkeypatch.setattr(hw, "resolve_device", lambda: "xla")
    profile = hw.resolve_accelerator_profile()
    assert profile == hw.COLAB_TPU_V6E1
    assert profile.kind == "tpu"
    assert profile.per_device_train_batch_size == 32
    assert profile.gradient_accumulation_steps == 1
    assert profile.tpu_num_processes == 1


def test_resolve_profile_colab_g4_by_name(monkeypatch):
    monkeypatch.setattr(hw, "resolve_device", lambda: "cuda")
    monkeypatch.setattr(hw, "is_colab", lambda: True)
    monkeypatch.setattr(hw, "_cuda_name", lambda: "NVIDIA RTX PRO 6000 Blackwell")
    monkeypatch.setattr(hw, "_cuda_memory_gb", lambda: 95.5)
    assert hw.resolve_accelerator_profile() == hw.COLAB_G4


def test_resolve_profile_colab_g4_by_memory(monkeypatch):
    monkeypatch.setattr(hw, "resolve_device", lambda: "cuda")
    monkeypatch.setattr(hw, "is_colab", lambda: True)
    monkeypatch.setattr(hw, "_cuda_name", lambda: "NVIDIA Graphics Device")
    monkeypatch.setattr(hw, "_cuda_memory_gb", lambda: 96.0)
    assert hw.resolve_accelerator_profile() == hw.COLAB_G4


def test_resolve_profile_colab_a100_40gb_by_name(monkeypatch):
    monkeypatch.setattr(hw, "resolve_device", lambda: "cuda")
    monkeypatch.setattr(hw, "is_colab", lambda: True)
    monkeypatch.setattr(hw, "_cuda_name", lambda: "NVIDIA A100-SXM4-40GB")
    monkeypatch.setattr(hw, "_cuda_memory_gb", lambda: 39.4)
    assert hw.resolve_accelerator_profile() == hw.COLAB_A100_40GB


def test_resolve_profile_a100_80gb_not_40gb_profile(monkeypatch):
    """80 GB A100 falls through to mid-tier (not the 40 GB recipe)."""
    monkeypatch.setattr(hw, "resolve_device", lambda: "cuda")
    monkeypatch.setattr(hw, "is_colab", lambda: True)
    monkeypatch.setattr(hw, "_cuda_name", lambda: "NVIDIA A100-SXM4-80GB")
    monkeypatch.setattr(hw, "_cuda_memory_gb", lambda: 79.2)
    assert hw.resolve_accelerator_profile() == hw.COLAB_L4


def test_resolve_profile_colab_l4(monkeypatch):
    monkeypatch.setattr(hw, "resolve_device", lambda: "cuda")
    monkeypatch.setattr(hw, "is_colab", lambda: True)
    monkeypatch.setattr(hw, "_cuda_name", lambda: "NVIDIA L4")
    monkeypatch.setattr(hw, "_cuda_memory_gb", lambda: 22.5)
    assert hw.resolve_accelerator_profile() == hw.COLAB_L4


def test_resolve_profile_local_rtx(monkeypatch):
    monkeypatch.setattr(hw, "resolve_device", lambda: "cuda")
    monkeypatch.setattr(hw, "is_colab", lambda: False)
    monkeypatch.setattr(hw, "_cuda_name", lambda: "NVIDIA GeForce RTX 3070")
    monkeypatch.setattr(hw, "_cuda_memory_gb", lambda: 8.0)
    profile = hw.resolve_accelerator_profile()
    assert profile == hw.LOCAL_RTX
    assert profile.per_device_train_batch_size == 2
    assert profile.gradient_accumulation_steps == 16


def test_resolve_profile_large_vram_non_colab_l4(monkeypatch):
    monkeypatch.setattr(hw, "resolve_device", lambda: "cuda")
    monkeypatch.setattr(hw, "is_colab", lambda: False)
    monkeypatch.setattr(hw, "_cuda_name", lambda: "NVIDIA L4")
    monkeypatch.setattr(hw, "_cuda_memory_gb", lambda: 23.0)
    assert hw.resolve_accelerator_profile() == hw.COLAB_L4


def test_resolve_profile_cpu(monkeypatch):
    monkeypatch.setattr(hw, "resolve_device", lambda: "cpu")
    assert hw.resolve_accelerator_profile() == hw.CPU_PROFILE
