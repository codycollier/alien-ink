"""Tests for device / precision helpers."""

from __future__ import annotations

import alien_ink.device as device_mod
import alien_ink.hf.hardware as hw


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
    assert device_mod.lookup_peak_tflops("NVIDIA GeForce RTX 3070", "bf16") == 40.6
    assert device_mod.lookup_peak_tflops("NVIDIA GeForce RTX 3070", "fp32") == 20.31
    assert device_mod.lookup_peak_tflops("TPU v6e-1", "bf16") == 918.0
    assert device_mod.lookup_peak_tflops("NVIDIA RTX PRO 6000 Blackwell", "fp16") == 500.0
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


def test_introspect_cpu(monkeypatch):
    monkeypatch.setattr(device_mod, "resolve_device", lambda: "cpu")
    monkeypatch.setattr(hw, "resolve_device", lambda: "cpu")
    monkeypatch.setattr(device_mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_mod, "is_torch_xla_available", lambda: False)
    monkeypatch.setattr(device_mod, "_xla_world_size", lambda: None)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    text = device_mod.introspect()
    border = 80 * "-"
    assert text.startswith(border)
    assert text.endswith(border)
    assert device_mod._INTROSPECT_TITLE in text
    assert device_mod._kv("torch", device_mod.torch.__version__) in text
    assert device_mod._kv("cuda", False) in text
    assert device_mod._kv("device", "cpu") in text
    assert device_mod._kv("precision", "fp32") in text
    assert (
        device_mod._kv("profile", "cpu  batch: 1  accum: 8  mem×0  compute×0")
        in text
    )


def test_introspect_cuda(monkeypatch):
    fake = device_mod.AcceleratorInfo(
        device="cuda",
        use_fp16=False,
        use_bf16=True,
        precision="bf16",
        world_size=1,
        gpu_count=1,
        gpu_name="NVIDIA RTX PRO 6000 Blackwell",
        gpu_memory_total_gb=95.5,
        cuda_available=True,
        cuda_version="12.8",
        cudnn_version="90100",
        torch_version="2.6.0+cu128",
        platform="Linux",
        python_version="3.11.0",
        peak_tflops=500.0,
        xla_available=False,
    )
    monkeypatch.setattr(device_mod, "collect_accelerator_info", lambda **_: fake)
    monkeypatch.setattr(hw, "get_profile", lambda profile=None: hw.COLAB_G4)

    text = device_mod.introspect()
    assert device_mod._kv("torch", "2.6.0+cu128") in text
    assert device_mod._kv("cuda", "True  (12.8, cudnn 90100)") in text
    assert device_mod._kv("gpu", "NVIDIA RTX PRO 6000 Blackwell (95.5 GB)") in text
    assert device_mod._kv("device", "cuda") in text
    assert device_mod._kv("precision", "bf16") in text
    assert device_mod._kv("world_size", 1) in text
    assert device_mod._kv("peak_tflops", "500") in text
    assert (
        device_mod._kv(
            "profile",
            "colab-g4  batch: 64  accum: 1  mem×12  compute×12.3",
        )
        in text
    )
    assert "using:" not in text
    assert "device_type:" not in text


def test_introspect_xla(monkeypatch):
    fake = device_mod.AcceleratorInfo(
        device="xla",
        use_fp16=False,
        use_bf16=True,
        precision="bf16",
        world_size=1,
        gpu_count=1,
        gpu_name="TPU v6e-1",
        gpu_memory_total_gb=None,
        cuda_available=False,
        cuda_version=None,
        cudnn_version=None,
        torch_version="2.6.0",
        platform="Linux",
        python_version="3.11.0",
        peak_tflops=918.0,
        xla_available=True,
    )
    monkeypatch.setattr(device_mod, "collect_accelerator_info", lambda **_: fake)
    monkeypatch.setattr(device_mod, "_torch_xla_version", lambda: "2.6.0")
    monkeypatch.setattr(device_mod, "_xla_device_str", lambda: "xla:0")
    monkeypatch.setattr(device_mod, "_xla_device_type", lambda: "TPU")
    monkeypatch.setattr(hw, "get_profile", lambda profile=None: hw.COLAB_TPU_V6E1)

    text = device_mod.introspect()
    assert device_mod._kv("torch", "2.6.0") in text
    assert device_mod._kv("xla", "2.6.0") in text
    assert device_mod._kv("using", "xla:0") in text
    assert device_mod._kv("device_type", "TPU → xla") in text
    assert device_mod._kv("tpu", "TPU v6e-1") in text
    assert device_mod._kv("cores", 1) in text
    assert device_mod._kv("device", "xla") in text
    assert device_mod._kv("precision", "bf16") in text
    assert device_mod._kv("peak_tflops", "918") in text
    assert (
        device_mod._kv(
            "profile",
            "colab-tpu-v6e1  batch: 32  accum: 1  mem×4  compute×22.6",
        )
        in text
    )
    assert text.startswith(80 * "-")
    assert text.endswith(80 * "-")
