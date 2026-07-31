"""Tests for W&B config helpers (no live wandb runs)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from alien_ink.com.env import EnvConfig
from alien_ink.com.wb import (
    build_run_config,
    require_wandb_identity,
    serialize_config,
    wandb_run,
)


@dataclass
class _Tiny:
    name: str
    path: Path
    values: tuple[int, ...]


def test_require_wandb_identity_ok():
    assert require_wandb_identity(entity="logbook", project="ink-explore") == (
        "logbook",
        "ink-explore",
    )


def test_require_wandb_identity_missing():
    with pytest.raises(ValueError, match="wandb_entity"):
        require_wandb_identity(entity=None, project="p")
    with pytest.raises(ValueError, match="wandb_project"):
        require_wandb_identity(entity="e", project="  ")


def test_serialize_config_stringifies_paths():
    raw = serialize_config(_Tiny(name="x", path=Path("/tmp/out"), values=(1, 2)))
    assert raw == {"name": "x", "path": "/tmp/out", "values": [1, 2]}


def test_serialize_config_dict():
    assert serialize_config({"a": Path("p"), "b": 1}) == {"a": "p", "b": 1}


def test_build_run_config_namespaces_multiple(monkeypatch):
    from alien_ink.com.device import AcceleratorInfo

    fake = AcceleratorInfo(
        device="cpu",
        use_fp16=False,
        use_bf16=False,
        precision="fp32",
        world_size=1,
        gpu_count=0,
        gpu_name=None,
        gpu_memory_total_gb=None,
        cuda_available=False,
        cuda_version=None,
        cudnn_version=None,
        torch_version="2.0.0",
        platform="test",
        python_version="3.11.0",
        peak_tflops=None,
    )
    monkeypatch.setattr("alien_ink.com.wb.collect_accelerator_info", lambda **_: fake)
    env = EnvConfig(
        hf_token=None,
        wandb_api_key=None,
        wandb_entity="logbook",
        wandb_project="proj",
    )
    flat = build_run_config(
        run_label="r",
        env=env,
        configs={"data": {"block_size": 128}, "arch": {"n_layer": 2}},
        tokens_per_step=1024,
        model={"total_params": 100},
        software={"torch": "2.0.0"},
    )
    assert flat["run_label"] == "r"
    assert flat["wandb_entity"] == "logbook"
    assert flat["wandb_project"] == "proj"
    assert flat["data.block_size"] == 128
    assert flat["arch.n_layer"] == 2
    assert flat["tokens_per_optimizer_step"] == 1024
    assert flat["accel.device"] == "cpu"
    assert flat["model.total_params"] == 100
    assert flat["sw.torch"] == "2.0.0"


def test_wandb_run_disabled_does_not_import_wandb():
    with wandb_run(
        entity="e",
        project="p",
        name="n",
        config={},
        dir=Path("."),
        enabled=False,
    ) as run:
        assert run is None


def test_wandb_run_enabled_requires_identity(tmp_path):
    with pytest.raises(ValueError, match="wandb_entity"):
        with wandb_run(
            entity="",
            project="p",
            name="n",
            config={},
            dir=tmp_path,
            enabled=True,
        ):
            pass
