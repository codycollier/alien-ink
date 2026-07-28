"""Tests for Recipe composition and materialization."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("transformers")

from alien_ink.hf.ds import HubTextSource, PretrainDataConfig  # noqa: E402
from alien_ink.hf.model import gpt2_arch  # noqa: E402
from alien_ink.hf.recipe import (  # noqa: E402
    HardwareConfig,
    Recipe,
    ScheduleConfig,
    WandbConfig,
    mist_rtx_3070,
    scaled_trainer_steps,
)


def _minimal_data(**overrides) -> PretrainDataConfig:
    return PretrainDataConfig(
        source=HubTextSource(dataset="Salesforce/wikitext"),
        **overrides,
    )


def _recipe(**overrides) -> Recipe:
    base = dict(
        run_name="test-run",
        title="Test run",
        data=_minimal_data(),
        model=gpt2_arch(),
        hardware=mist_rtx_3070(),
        wandb=WandbConfig(entity="logbook", project="ink-explore", enabled=False),
        schedule=ScheduleConfig(max_steps=5_000, warmup_steps=200),
    )
    base.update(overrides)
    return Recipe(**base)


def test_scaled_trainer_steps_at_reference():
    assert scaled_trainer_steps(50_000) == {
        "logging_steps": 50,
        "eval_steps": 1_000,
        "save_steps": 1_000,
    }


def test_scaled_trainer_steps_short_run():
    assert scaled_trainer_steps(5_000) == {
        "logging_steps": 5,
        "eval_steps": 100,
        "save_steps": 100,
    }


def test_mist_rtx_3070_effective_batch():
    hw = mist_rtx_3070()
    assert hw.label == "mist-rtx-3070"
    assert hw.effective_batch_size == 32


def test_recipe_to_pretrain_config_merges_segments(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recipe = _recipe()
    cfg = recipe.to_pretrain_config()

    assert cfg.data is recipe.data
    assert cfg.arch is recipe.model
    assert cfg.trainer.output_dir == tmp_path / "output" / "test-run"
    assert cfg.trainer.run_name == "test-run"
    assert cfg.trainer.max_steps == 5_000
    assert cfg.trainer.warmup_steps == 200
    assert cfg.trainer.per_device_train_batch_size == 2
    assert cfg.trainer.gradient_accumulation_steps == 16
    assert cfg.trainer.logging_steps == 5
    assert cfg.trainer.eval_steps == 100
    assert cfg.trainer.save_steps == 100
    assert cfg.trainer.report_to == "none"


def test_recipe_wandb_name_override(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recipe = _recipe(
        wandb=WandbConfig(
            entity="logbook",
            project="ink-explore",
            name="custom-wandb-name",
            enabled=True,
        ),
    )
    cfg = recipe.to_pretrain_config()
    assert cfg.trainer.run_name == "custom-wandb-name"
    assert cfg.trainer.report_to == "wandb"


def test_recipe_explicit_cadence_overrides_scale(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recipe = _recipe(
        schedule=ScheduleConfig(
            max_steps=5_000,
            warmup_steps=200,
            logging_steps=50,
            eval_steps=500,
            save_steps=500,
        ),
    )
    cfg = recipe.to_pretrain_config()
    assert cfg.trainer.logging_steps == 50
    assert cfg.trainer.eval_steps == 500
    assert cfg.trainer.save_steps == 500


def test_recipe_with_hardware_composition(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recipe = _recipe().with_hardware(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        label="bigger-gpu",
    )
    assert recipe.hardware.label == "bigger-gpu"
    assert recipe.hardware.effective_batch_size == 32
    cfg = recipe.to_pretrain_config()
    assert cfg.trainer.per_device_train_batch_size == 4
    assert cfg.trainer.gradient_accumulation_steps == 8


def test_recipe_with_model_and_schedule(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recipe = (
        _recipe()
        .with_model(n_layer=6)
        .with_schedule(learning_rate=3e-4)
        .variant(run_name="ablation-l6")
    )
    assert recipe.run_name == "ablation-l6"
    assert recipe.model.n_layer == 6
    assert recipe.schedule.learning_rate == 3e-4
    cfg = recipe.to_pretrain_config()
    assert cfg.arch.n_layer == 6
    assert cfg.trainer.learning_rate == 3e-4
    assert cfg.trainer.output_dir == tmp_path / "output" / "ablation-l6"


def test_recipe_validate_requires_wandb_identity():
    recipe = _recipe(wandb=WandbConfig(enabled=True))
    with pytest.raises(ValueError, match="wandb_entity"):
        recipe.validate()


def test_recipe_validate_block_vs_positions():
    recipe = _recipe(
        data=_minimal_data(block_size=2048),
        model=gpt2_arch(n_positions=1024),
    )
    with pytest.raises(ValueError, match="block_size"):
        recipe.validate()


def test_hardware_config_validate():
    with pytest.raises(ValueError, match="per_device_train_batch_size"):
        HardwareConfig(per_device_train_batch_size=0).validate()
