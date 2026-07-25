"""Tests for W&B config helpers (no live wandb runs)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alien_ink.env import DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT, EnvConfig
from alien_ink.wb import build_run_config, serialize_config, wandb_run


@dataclass
class _Tiny:
    name: str
    path: Path
    values: tuple[int, ...]


def test_default_wandb_entity_and_project():
    assert DEFAULT_WANDB_ENTITY == "logbook"
    assert DEFAULT_WANDB_PROJECT == "ink-explore"


def test_serialize_config_stringifies_paths():
    raw = serialize_config(_Tiny(name="x", path=Path("/tmp/out"), values=(1, 2)))
    assert raw == {"name": "x", "path": "/tmp/out", "values": [1, 2]}


def test_serialize_config_dict():
    assert serialize_config({"a": Path("p"), "b": 1}) == {"a": "p", "b": 1}


def test_build_run_config_namespaces_multiple(monkeypatch):
    monkeypatch.setattr(
        "alien_ink.wb.device_info",
        lambda **_: ("cpu", False, False),
    )
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
    )
    assert flat["run_label"] == "r"
    assert flat["wandb_entity"] == "logbook"
    assert flat["wandb_project"] == "proj"
    assert flat["data.block_size"] == 128
    assert flat["arch.n_layer"] == 2


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
