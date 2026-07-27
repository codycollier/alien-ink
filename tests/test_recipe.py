"""Tests for shared experiment recipe helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("datasets")
pytest.importorskip("transformers")

from alien_ink.exp.cli import train_override_kwargs, wandb_kwargs  # noqa: E402
from alien_ink.exp.recipe import (  # noqa: E402
    Gpt2PretrainExperiment,
    scaled_trainer_steps,
    trainer_length_kwargs,
)
from alien_ink.hf.ds import HubTextSource, PretrainDataConfig  # noqa: E402
from alien_ink.hf.hardware import COLAB_G4, COLAB_TPU_V6E1, LOCAL_RTX  # noqa: E402


def _fake_data(**_kwargs) -> PretrainDataConfig:
    return PretrainDataConfig(source=HubTextSource(dataset="Salesforce/wikitext", name="x"))


def test_scaled_trainer_steps_matches_reference_defaults():
    assert scaled_trainer_steps(50_000) == {
        "logging_steps": 50,
        "eval_steps": 1_000,
        "save_steps": 1_000,
    }


def test_scaled_trainer_steps_short_run():
    assert scaled_trainer_steps(2_000) == {
        "logging_steps": 2,
        "eval_steps": 40,
        "save_steps": 40,
    }


def test_scaled_trainer_steps_rejects_non_positive():
    with pytest.raises(ValueError, match="max_steps"):
        scaled_trainer_steps(0)


def test_trainer_length_kwargs_epoch_mode():
    assert trainer_length_kwargs(max_steps=-1, num_train_epochs=3) == {
        "max_steps": -1,
        "num_train_epochs": 3,
        "logging_steps": 10,
    }


def test_flight_check_run_name_appends_suffix(monkeypatch):
    monkeypatch.setattr(
        "alien_ink.exp.recipe.resolve_accelerator_profile",
        lambda: LOCAL_RTX,
    )
    exp = Gpt2PretrainExperiment(
        run_name="gpt2-pretrain-wikitext",
        title="t",
        spot_check_title="s",
        data_factory=_fake_data,
        module_description="d",
    )
    assert exp.flight_check_run_name() == "gpt2-pretrain-wikitext-flight-check-gpu"


def test_resolved_run_name_tpu_suffix(monkeypatch):
    monkeypatch.setattr(
        "alien_ink.exp.recipe.resolve_accelerator_profile",
        lambda: COLAB_TPU_V6E1,
    )
    exp = Gpt2PretrainExperiment(
        run_name="gpt2-pretrain-wpe-subset",
        title="t",
        spot_check_title="s",
        data_factory=_fake_data,
        module_description="d",
    )
    assert exp.resolved_run_name() == "gpt2-pretrain-wpe-subset-tpu"
    assert exp.resolved_run_name(wandb_name="custom") == "custom-tpu"


def test_subset_experiment_defaults(monkeypatch):
    monkeypatch.setattr(
        "alien_ink.exp.recipe.resolve_accelerator_profile",
        lambda: LOCAL_RTX,
    )
    from alien_ink.exp.gpt2_pretrain_wikipedia_english_subset import EXPERIMENT

    cfg = EXPERIMENT.base_config()
    assert cfg.data.max_train_samples == 20_000
    assert cfg.data.max_eval_samples == 1_000
    assert cfg.trainer.max_steps == -1
    assert cfg.trainer.num_train_epochs == 3
    assert cfg.trainer.uses_epochs()
    assert cfg.trainer.warmup_steps == 200
    assert cfg.trainer.logging_steps == 10
    assert EXPERIMENT.run_name == "gpt2-pretrain-wpe-subset"
    assert cfg.trainer.run_name == "gpt2-pretrain-wpe-subset-gpu"
    assert cfg.trainer.per_device_train_batch_size == 2
    assert cfg.trainer.gradient_accumulation_steps == 16


def test_base_config_colab_g4_batch(monkeypatch):
    monkeypatch.setattr(
        "alien_ink.exp.recipe.resolve_accelerator_profile",
        lambda: COLAB_G4,
    )
    exp = Gpt2PretrainExperiment(
        run_name="run-a",
        title="t",
        spot_check_title="s",
        data_factory=_fake_data,
        module_description="d",
    )
    cfg = exp.base_config()
    assert cfg.trainer.per_device_train_batch_size == 64
    assert cfg.trainer.gradient_accumulation_steps == 1
    assert cfg.trainer.run_name == "run-a-gpu"


def test_base_config_tpu_v6e1_batch(monkeypatch):
    monkeypatch.setattr(
        "alien_ink.exp.recipe.resolve_accelerator_profile",
        lambda: COLAB_TPU_V6E1,
    )
    exp = Gpt2PretrainExperiment(
        run_name="run-a",
        title="t",
        spot_check_title="s",
        data_factory=_fake_data,
        module_description="d",
    )
    cfg = exp.base_config()
    assert cfg.trainer.per_device_train_batch_size == 32
    assert cfg.trainer.gradient_accumulation_steps == 1
    assert cfg.trainer.run_name == "run-a-tpu"


def test_streamed_experiment_keeps_reference_cadence(monkeypatch):
    monkeypatch.setattr(
        "alien_ink.exp.recipe.resolve_accelerator_profile",
        lambda: LOCAL_RTX,
    )
    from alien_ink.exp.gpt2_pretrain_wikitext import EXPERIMENT

    cfg = EXPERIMENT.base_config()
    assert cfg.trainer.max_steps == 50_000
    assert cfg.trainer.logging_steps == 50
    assert cfg.trainer.eval_steps == 1_000
    assert cfg.trainer.save_steps == 1_000


def test_train_rescales_cadence_when_max_steps_overridden(monkeypatch):
    captured: dict = {}

    def fake_pretrain(cfg, **kwargs):
        captured["trainer"] = cfg.trainer
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "alien_ink.exp.recipe.resolve_accelerator_profile",
        lambda: LOCAL_RTX,
    )
    monkeypatch.setattr("alien_ink.exp.recipe.pretrain_gpt2", fake_pretrain)
    exp = Gpt2PretrainExperiment(
        run_name="run-a",
        title="t",
        spot_check_title="s",
        data_factory=_fake_data,
        module_description="d",
        max_steps=2_000,
        warmup_steps=200,
    )
    exp.train(use_wandb=False, max_steps=5_000)
    assert captured["trainer"].max_steps == 5_000
    assert captured["trainer"].logging_steps == 5
    assert captured["trainer"].eval_steps == 100
    assert captured["trainer"].save_steps == 100
    assert captured["kwargs"]["wandb_name"] == "run-a-gpu"


def test_train_passes_tpu_num_processes_from_profile(monkeypatch):
    captured: dict = {}

    def fake_pretrain(cfg, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "alien_ink.exp.recipe.resolve_accelerator_profile",
        lambda: COLAB_TPU_V6E1,
    )
    monkeypatch.setattr("alien_ink.exp.recipe.pretrain_gpt2", fake_pretrain)
    exp = Gpt2PretrainExperiment(
        run_name="run-a",
        title="t",
        spot_check_title="s",
        data_factory=_fake_data,
        module_description="d",
    )
    exp.train(use_wandb=False)
    assert captured["kwargs"]["tpu_num_processes"] == 1


def test_train_keeps_explicit_cadence_overrides(monkeypatch):
    captured: dict = {}

    def fake_pretrain(cfg, **_kwargs):
        captured["trainer"] = cfg.trainer

    monkeypatch.setattr(
        "alien_ink.exp.recipe.resolve_accelerator_profile",
        lambda: LOCAL_RTX,
    )
    monkeypatch.setattr("alien_ink.exp.recipe.pretrain_gpt2", fake_pretrain)
    exp = Gpt2PretrainExperiment(
        run_name="run-a",
        title="t",
        spot_check_title="s",
        data_factory=_fake_data,
        module_description="d",
        max_steps=2_000,
        warmup_steps=200,
    )
    exp.train(use_wandb=False, max_steps=5_000, eval_steps=250)
    assert captured["trainer"].max_steps == 5_000
    assert captured["trainer"].eval_steps == 250
    # Unspecified cadence fields still follow the new max_steps.
    assert captured["trainer"].logging_steps == 5
    assert captured["trainer"].save_steps == 100


def test_flight_check_shrinks_materialized_train(monkeypatch):
    def _subset_data(**_kwargs) -> PretrainDataConfig:
        return PretrainDataConfig(
            source=HubTextSource(dataset="dummy"),
            max_train_samples=20_000,
            max_eval_samples=1_000,
        )

    captured: dict = {}

    def fake_pretrain(cfg, **_kwargs):
        captured["data"] = cfg.data
        captured["trainer"] = cfg.trainer

    monkeypatch.setattr(
        "alien_ink.exp.recipe.resolve_accelerator_profile",
        lambda: LOCAL_RTX,
    )
    monkeypatch.setattr("alien_ink.exp.recipe.pretrain_gpt2", fake_pretrain)
    exp = Gpt2PretrainExperiment(
        run_name="subset-run",
        title="t",
        spot_check_title="s",
        data_factory=_subset_data,
        module_description="d",
        max_steps=2_000,
        warmup_steps=200,
    )
    exp.train_flight_check(use_wandb=False)
    assert captured["data"].max_train_samples == 50
    assert captured["data"].max_eval_samples == 10
    assert captured["trainer"].max_steps == 10
    assert captured["trainer"].run_name == "subset-run-flight-check-gpu"


def test_paths_resolve_from_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "alien_ink.exp.recipe.resolve_accelerator_profile",
        lambda: LOCAL_RTX,
    )
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
    assert cfg.trainer.output_dir == tmp_path / "output" / "run-a-gpu"
    assert cfg.trainer.run_name == "run-a-gpu"


def test_checkpoint_output_dirs_include_device_suffixes(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp = Gpt2PretrainExperiment(
        run_name="run-a",
        title="t",
        spot_check_title="s",
        data_factory=_fake_data,
        module_description="d",
    )
    dirs = {p.name for p in exp.checkpoint_output_dirs()}
    assert "run-a-gpu" in dirs
    assert "run-a-tpu" in dirs
    assert "run-a-flight-check-gpu" in dirs
    assert "run-a" in dirs  # legacy


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
            "num_train_epochs": None,
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
            "num_train_epochs": None,
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
        [
            "--train",
            "--no-wandb",
            "--max-steps",
            "-1",
            "--num-train-epochs",
            "3",
            "--resume-from-checkpoint",
        ]
    )
    assert args.no_wandb is True
    assert args.max_steps == -1
    assert args.num_train_epochs == 3
    assert args.resume_from_checkpoint is True
