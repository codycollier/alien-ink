"""Tests for trainer config helpers and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")

from alien_ink.hf.pretrain import (  # noqa: E402
    PretrainConfig,
    resolve_use_wandb,
    with_trainer,
)
from alien_ink.hf.ds import HubTextSource, PretrainDataConfig  # noqa: E402
from alien_ink.hf.model import CausalLmArchConfig  # noqa: E402
from alien_ink.hf.trainer import (  # noqa: E402
    CausalLmTrainerConfig,
    ConsecutiveLossThresholdCallback,
    apply_epoch_cadence,
    build_lm_data_collator,
    epoch_cadence_steps,
    optimizer_steps_per_epoch,
    reporting_disabled,
    tokens_per_optimizer_step,
)


class _State:
    global_step = 0


class _Control:
    should_training_stop = False


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


def test_consecutive_loss_threshold_stops_after_distinct_eval_steps():
    callback = ConsecutiveLossThresholdCallback(
        metric="eval_completion_loss",
        threshold=1e-4,
        patience=3,
    )
    state = _State()
    control = _Control()

    for step, loss in ((10, 9e-5), (10, 8e-5), (20, 2e-4), (30, 9e-5), (40, 8e-5)):
        state.global_step = step
        callback.on_evaluate(
            None,
            state,
            control,
            metrics={"eval_completion_loss": loss},
        )
        assert not control.should_training_stop

    state.global_step = 50
    callback.on_evaluate(
        None,
        state,
        control,
        metrics={"eval_completion_loss": 7e-5},
    )
    assert control.should_training_stop


def test_trainer_config_requires_complete_loss_stop_condition(tmp_path: Path):
    with pytest.raises(ValueError, match="must be set together"):
        CausalLmTrainerConfig(
            output_dir=tmp_path / "out",
            stop_loss_metric="eval_loss",
        ).validate()
    assert reporting_disabled([])
    assert not reporting_disabled("wandb")


def test_resolve_use_wandb_explicit_and_report_to():
    cfg = PretrainConfig(
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
    cfg = PretrainConfig(
        data=PretrainDataConfig(
            source=HubTextSource(dataset="Salesforce/wikitext"),
            block_size=2048,
        ),
        arch=CausalLmArchConfig(n_positions=1024),
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
        eval_steps=20,
        save_steps=100,
    )
    args = trainer_mod.build_training_arguments(cfg, has_eval=True)
    assert args.max_steps == -1
    assert args.num_train_epochs == 3
    # Epoch runs use step cadence (applied earlier) so mid-epoch evals work.
    assert args.eval_strategy == "steps"
    assert args.save_strategy == "steps"
    assert args.eval_steps == 20
    assert args.save_steps == 100


def test_build_training_arguments_speed_knobs(monkeypatch, tmp_path: Path):
    from alien_ink.hf import trainer as trainer_mod
    from transformers import training_args as training_args_mod

    monkeypatch.setattr(
        trainer_mod,
        "device_info",
        lambda **_kwargs: ("cuda", False, True),
    )
    monkeypatch.setattr(training_args_mod, "is_torch_tf32_available", lambda: True)
    cfg = CausalLmTrainerConfig(
        output_dir=tmp_path / "out",
        max_steps=100,
        warmup_steps=None,
        warmup_ratio=0.04,
        data_seed=303,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        tf32=True,
        torch_compile=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=False,
    )
    args = trainer_mod.build_training_arguments(cfg, has_eval=False)
    assert args.dataloader_num_workers == 8
    assert args.dataloader_prefetch_factor == 4
    assert args.dataloader_persistent_workers is True
    assert args.tf32 is True
    assert args.torch_compile is True
    assert args.optim == "adamw_torch_fused"
    assert args.gradient_checkpointing is False
    if int(transformers.__version__.split(".", maxsplit=1)[0]) >= 5:
        assert args.warmup_steps == 0.04
    else:
        assert args.warmup_steps == 0
        assert args.warmup_ratio == 0.04
    assert args.data_seed == 303


def test_build_training_arguments_warmup_steps_not_mangled(monkeypatch, tmp_path: Path):
    from alien_ink.hf import trainer as trainer_mod

    monkeypatch.setattr(
        trainer_mod,
        "device_info",
        lambda **_kwargs: ("cpu", False, False),
    )
    cfg = CausalLmTrainerConfig(
        output_dir=tmp_path / "out",
        max_steps=10_000,
        warmup_steps=200,
        eval_steps=100,
        save_steps=100,
    )
    args = trainer_mod.build_training_arguments(cfg, has_eval=False)
    assert args.warmup_steps == 200


def test_trainer_config_validates_warmup_and_cadence(tmp_path: Path):
    with pytest.raises(ValueError, match="only one"):
        CausalLmTrainerConfig(
            output_dir=tmp_path / "out",
            warmup_steps=10,
            warmup_ratio=0.1,
        ).validate()
    with pytest.raises(ValueError, match="cannot exceed"):
        CausalLmTrainerConfig(
            output_dir=tmp_path / "out",
            max_steps=10,
            warmup_steps=11,
        ).validate()
    with pytest.raises(ValueError, match="multiple"):
        CausalLmTrainerConfig(
            output_dir=tmp_path / "out",
            eval_steps=30,
            save_steps=100,
        ).validate()


def test_optimizer_steps_per_epoch_matches_hf_ceil_math():
    # 100 examples, batch 8, world 1 → 13 dataloader batches; accum 4 → 4 steps.
    assert (
        optimizer_steps_per_epoch(
            100,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            world_size=1,
        )
        == 4
    )
    # Exact division: 128 / (8*1) = 16 batches; /4 accum = 4 steps.
    assert (
        optimizer_steps_per_epoch(
            128,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            world_size=1,
        )
        == 4
    )


def test_epoch_cadence_steps_five_evals_including_epoch_end():
    # 100 steps → eval every 20 → ticks at 20,40,60,80,100 (epoch end).
    assert epoch_cadence_steps(100) == {
        "logging_steps": 1,
        "eval_steps": 20,
        "save_steps": 100,
    }
    # Non-divisible: ~5 step evals, save ≈ once/epoch; epoch-end callback covers the remainder.
    assert epoch_cadence_steps(103) == {
        "logging_steps": 1,
        "eval_steps": 20,
        "save_steps": 100,
    }
    assert epoch_cadence_steps(7) == {
        "logging_steps": 1,
        "eval_steps": 1,
        "save_steps": 5,
    }


def test_apply_epoch_cadence_updates_config(tmp_path: Path):
    cfg = CausalLmTrainerConfig(
        output_dir=tmp_path / "out",
        max_steps=-1,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        logging_steps=10,
    )
    # 640 examples / (2*16) = 20 steps/epoch → 5 evals of 4 steps.
    out = apply_epoch_cadence(cfg, num_train_examples=640, world_size=1)
    assert out.eval_steps == 4
    assert out.save_steps == 20
    assert out.logging_steps == 1
    # Step-capped configs are left alone.
    stepped = CausalLmTrainerConfig(output_dir=tmp_path / "out", max_steps=1_000)
    assert apply_epoch_cadence(stepped, num_train_examples=640) is stepped


class _PadTok:
    pad_token_id = 0

    def pad(self, features, *, return_tensors="pt"):
        import torch

        assert return_tensors == "pt"
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        for feature in features:
            ids = list(feature["input_ids"])
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            mask = list(feature.get("attention_mask", [1] * len(ids)))
            attention_mask.append(mask + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def test_lm_collator_preserves_prompt_mask():
    collate = build_lm_data_collator(_PadTok())
    batch = collate(
        [
            {
                "input_ids": [10, 11, 12, 13],
                "attention_mask": [1, 1, 1, 1],
                "labels": [-100, -100, 12, 13],
            },
            {
                "input_ids": [20, 21],
                "attention_mask": [1, 1],
                "labels": [-100, 21],
            },
        ]
    )
    assert batch["labels"].tolist() == [
        [-100, -100, 12, 13],
        [-100, 21, -100, -100],
    ]
    assert batch["input_ids"].tolist() == [
        [10, 11, 12, 13],
        [20, 21, 0, 0],
    ]


def test_lm_collator_clones_input_ids_without_labels():
    collate = build_lm_data_collator(_PadTok())
    batch = collate(
        [
            {"input_ids": [10, 11, 12], "attention_mask": [1, 1, 1]},
            {"input_ids": [20], "attention_mask": [1]},
        ]
    )
    assert batch["labels"].tolist() == [
        [10, 11, 12],
        [20, -100, -100],
    ]
