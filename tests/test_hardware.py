"""Tests for accelerator profiles (local RTX / Colab G4 / TPU v6e-1)."""

from __future__ import annotations

import alien_ink.hf.hardware as hw


def test_with_accelerator_suffix_appends_once():
    assert hw.with_accelerator_suffix("run-a", "gpu") == "run-a-gpu"
    assert hw.with_accelerator_suffix("run-a-gpu", "gpu") == "run-a-gpu"
    assert hw.with_accelerator_suffix("run-a", "tpu") == "run-a-tpu"


def test_resolve_profile_tpu(monkeypatch):
    monkeypatch.setattr(hw, "resolve_device", lambda: "xla")
    profile = hw.resolve_accelerator_profile()
    assert profile == hw.COLAB_TPU_V6E1
    assert profile.kind == "tpu"
    assert profile.per_device_train_batch_size == 64
    assert profile.gradient_accumulation_steps == 1
    assert profile.tpu_num_processes == 1


def test_resolve_profile_colab_g4(monkeypatch):
    monkeypatch.setattr(hw, "resolve_device", lambda: "cuda")
    monkeypatch.setattr(hw, "is_colab", lambda: True)
    monkeypatch.setattr(hw, "_cuda_name", lambda: "NVIDIA L4")
    monkeypatch.setattr(hw, "_cuda_memory_gb", lambda: 22.5)
    assert hw.resolve_accelerator_profile() == hw.COLAB_G4


def test_resolve_profile_local_rtx(monkeypatch):
    monkeypatch.setattr(hw, "resolve_device", lambda: "cuda")
    monkeypatch.setattr(hw, "is_colab", lambda: False)
    monkeypatch.setattr(hw, "_cuda_name", lambda: "NVIDIA GeForce RTX 3070")
    monkeypatch.setattr(hw, "_cuda_memory_gb", lambda: 8.0)
    profile = hw.resolve_accelerator_profile()
    assert profile == hw.LOCAL_RTX
    assert profile.per_device_train_batch_size == 2
    assert profile.gradient_accumulation_steps == 16


def test_resolve_profile_large_vram_non_colab(monkeypatch):
    monkeypatch.setattr(hw, "resolve_device", lambda: "cuda")
    monkeypatch.setattr(hw, "is_colab", lambda: False)
    monkeypatch.setattr(hw, "_cuda_name", lambda: "NVIDIA L4")
    monkeypatch.setattr(hw, "_cuda_memory_gb", lambda: 23.0)
    assert hw.resolve_accelerator_profile() == hw.COLAB_G4


def test_resolve_profile_cpu(monkeypatch):
    monkeypatch.setattr(hw, "resolve_device", lambda: "cpu")
    assert hw.resolve_accelerator_profile() == hw.CPU_PROFILE
