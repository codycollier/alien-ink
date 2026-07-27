"""Tests for trainer config helpers and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")

from alien_ink.hf.pretrain import (  # noqa: E402
    Gpt2PretrainConfig,
    resolve_use_wandb,
    with_trainer,
)
from alien_ink.hf.ds import HubTextSource, PretrainDataConfig  # noqa: E402
from alien_ink.hf.model import Gpt2ArchConfig  # noqa: E402
from alien_ink.hf.trainer import (  # noqa: E402
    CausalLmTrainerConfig,
    reporting_disabled,
    tokens_per_optimizer_step,
)


def test_tokens_per_optimizer_step_includes_world_size():
    assert (
        tokens_per_optimizer_step(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            block_size=128,
            world_size=2,
        )
        == 2 * 4 * 128 * 2
    )


def test_reporting_disabled():
    assert reporting_disabled("none")
    assert reporting_disabled("None")
    assert reporting_disabled("")
    assert reporting_disabled(None)
    assert reporting_disabled([])
    assert not reporting_disabled("wandb")


def test_resolve_use_wandb_explicit_and_report_to():
    cfg = Gpt2PretrainConfig(
        data=PretrainDataConfig(source=HubTextSource(dataset="Salesforce/wikitext")),
        trainer=CausalLmTrainerConfig(
            output_dir=Path("output/x"),
            report_to="wandb",
        ),
    )
    assert resolve_use_wandb(cfg, True) is True
    assert resolve_use_wandb(cfg, False) is False
    assert resolve_use_wandb(cfg, None) is True
    cfg_off = with_trainer(cfg, report_to="none")
    assert resolve_use_wandb(cfg_off, None) is False


def test_pretrain_config_validate_block_vs_positions():
    cfg = Gpt2PretrainConfig(
        data=PretrainDataConfig(
            source=HubTextSource(dataset="Salesforce/wikitext"),
            block_size=2048,
        ),
        arch=Gpt2ArchConfig(n_positions=1024),
        trainer=CausalLmTrainerConfig(output_dir=Path("output/x")),
    )
    with pytest.raises(ValueError, match="block_size"):
        cfg.validate()


def test_trainer_config_validate_learning_rate():
    with pytest.raises(ValueError, match="learning_rate"):
        CausalLmTrainerConfig(
            output_dir=Path("output/x"),
            learning_rate=0,
        ).validate()


def test_trainer_config_epoch_mode_validation():
    CausalLmTrainerConfig(
        output_dir=Path("output/x"),
        max_steps=-1,
        num_train_epochs=3,
    ).validate()
    with pytest.raises(ValueError, match="max_steps"):
        CausalLmTrainerConfig(output_dir=Path("output/x"), max_steps=0).validate()
    with pytest.raises(ValueError, match="num_train_epochs"):
        CausalLmTrainerConfig(
            output_dir=Path("output/x"),
            max_steps=-1,
            num_train_epochs=0,
        ).validate()


def test_build_training_arguments_epoch_strategy(monkeypatch, tmp_path: Path):
    from alien_ink.hf import trainer as trainer_mod

    monkeypatch.setattr(
        trainer_mod,
        "device_info",
        lambda **_kwargs: ("cpu", False, False),
    )
    cfg = CausalLmTrainerConfig(
        output_dir=tmp_path / "out",
        max_steps=-1,
        num_train_epochs=3,
        logging_steps=10,
    )
    args = trainer_mod.build_training_arguments(cfg, has_eval=True)
    assert args.max_steps == -1
    assert args.num_train_epochs == 3
    assert args.eval_strategy == "epoch"
    assert args.save_strategy == "epoch"
