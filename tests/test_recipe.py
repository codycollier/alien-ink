"""Tests for shared experiment recipe helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("datasets")
pytest.importorskip("transformers")

from alien_ink.exp.cli import train_override_kwargs, wandb_kwargs  # noqa: E402
from alien_ink.exp.recipe import Gpt2PretrainExperiment  # noqa: E402
from alien_ink.hf.ds import HubTextSource, PretrainDataConfig  # noqa: E402


def _fake_data(**_kwargs) -> PretrainDataConfig:
    return PretrainDataConfig(source=HubTextSource(dataset="Salesforce/wikitext", name="x"))


def test_flight_check_run_name_appends_suffix():
    exp = Gpt2PretrainExperiment(
        run_name="gpt2-pretrain-wikitext",
        title="t",
        spot_check_title="s",
        data_factory=_fake_data,
        module_description="d",
    )
    assert exp.flight_check_run_name() == "gpt2-pretrain-wikitext-flight-check"


def test_paths_resolve_from_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp = Gpt2PretrainExperiment(
        run_name="run-a",
        title="t",
        spot_check_title="s",
        data_factory=_fake_data,
        module_description="d",
    )
    assert exp.output_root() == tmp_path / "output"
    assert exp.env_file() == tmp_path / ".env"
    cfg = exp.base_config()
    assert cfg.trainer.output_dir == tmp_path / "output" / "run-a"
    assert cfg.trainer.run_name == "run-a"


def test_wandb_kwargs_no_wandb():
    ns = type(
        "N",
        (),
        {
            "wandb_entity": "e",
            "wandb_project": "p",
            "wandb_name": "n",
            "no_wandb": True,
        },
    )()
    assert wandb_kwargs(ns) == {
        "wandb_entity": "e",
        "wandb_project": "p",
        "wandb_name": "n",
        "use_wandb": False,
    }


def test_train_override_kwargs_resume_path():
    ns = type(
        "N",
        (),
        {
            "max_steps": 100,
            "learning_rate": None,
            "per_device_train_batch_size": None,
            "gradient_accumulation_steps": None,
            "resume_from_checkpoint": "/ckpt",
        },
    )()
    out = train_override_kwargs(ns)
    assert out["max_steps"] == 100
    assert out["resume_from_checkpoint"] == Path("/ckpt")


def test_train_override_kwargs_resume_true():
    ns = type(
        "N",
        (),
        {
            "max_steps": None,
            "learning_rate": None,
            "per_device_train_batch_size": None,
            "gradient_accumulation_steps": None,
            "resume_from_checkpoint": True,
        },
    )()
    assert train_override_kwargs(ns)["resume_from_checkpoint"] is True


def test_parser_includes_new_flags():
    exp = Gpt2PretrainExperiment(
        run_name="run-a",
        title="t",
        spot_check_title="s",
        data_factory=_fake_data,
        module_description="d",
    )
    args = exp.build_parser().parse_args(
        ["--train", "--no-wandb", "--max-steps", "5", "--resume-from-checkpoint"]
    )
    assert args.no_wandb is True
    assert args.max_steps == 5
    assert args.resume_from_checkpoint is True
