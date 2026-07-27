"""Tests for TPU / notebook launch helpers."""

from __future__ import annotations

import alien_ink.hf.launch as launch_mod


def test_should_auto_launch_tpu_force_false(monkeypatch):
    monkeypatch.setattr(launch_mod, "is_xla_tpu_available", lambda: True)
    assert launch_mod.should_auto_launch_tpu(force=False) is False


def test_should_auto_launch_tpu_requires_tpu(monkeypatch):
    monkeypatch.setattr(launch_mod, "is_xla_tpu_available", lambda: False)
    assert launch_mod.should_auto_launch_tpu(force=True) is False
    assert launch_mod.should_auto_launch_tpu() is False


def test_should_auto_launch_tpu_force_true(monkeypatch):
    monkeypatch.setattr(launch_mod, "is_xla_tpu_available", lambda: True)
    assert launch_mod.should_auto_launch_tpu(force=True) is True


def test_should_auto_launch_tpu_notebook_default(monkeypatch):
    monkeypatch.setattr(launch_mod, "is_xla_tpu_available", lambda: True)
    monkeypatch.setattr(launch_mod, "distributed_world_size", lambda: 1)
    monkeypatch.setattr(launch_mod, "in_notebook", lambda: True)
    assert launch_mod.should_auto_launch_tpu() is True


def test_should_auto_launch_tpu_skips_when_already_multiprocess(monkeypatch):
    monkeypatch.setattr(launch_mod, "is_xla_tpu_available", lambda: True)
    monkeypatch.setattr(launch_mod, "distributed_world_size", lambda: 8)
    monkeypatch.setattr(launch_mod, "in_notebook", lambda: True)
    assert launch_mod.should_auto_launch_tpu() is False


def test_tpu_num_processes_reads_env(monkeypatch):
    monkeypatch.setenv("TPU_NUM_DEVICES", "4")
    assert launch_mod.tpu_num_processes() == 4


def test_launch_tpu_calls_notebook_launcher(monkeypatch):
    calls: list[dict] = []

    def fake_notebook_launcher(fn, args=(), num_processes=None, mixed_precision="no"):
        calls.append(
            {
                "fn": fn,
                "args": args,
                "num_processes": num_processes,
                "mixed_precision": mixed_precision,
            }
        )

    class FakeAccelerate:
        notebook_launcher = staticmethod(fake_notebook_launcher)

    import sys

    monkeypatch.setitem(sys.modules, "accelerate", FakeAccelerate)
    monkeypatch.setattr(launch_mod, "tpu_num_processes", lambda default=1: 1)

    def _fn():
        return None

    launch_mod.launch_tpu(_fn, mixed_precision="bf16")
    assert len(calls) == 1
    assert calls[0]["num_processes"] == 1
    assert calls[0]["mixed_precision"] == "bf16"
    assert calls[0]["fn"] is _fn
