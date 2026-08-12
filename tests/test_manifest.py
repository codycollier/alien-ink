"""Tests for Manifest composition and materialization."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from alien_ink.hf.ds import HubTextSource, PretrainDataConfig  # noqa: E402
from alien_ink.hf.completion import CompletionDataConfig  # noqa: E402
from alien_ink.hf.model import CausalLmArchConfig, PretrainedLmConfig, gpt2_arch  # noqa: E402
from alien_ink.hf.manifest import (  # noqa: E402
    HardwareConfig,
    Manifest,
    ScheduleConfig,
    WandbConfig,
    mist_rtx_3070,
    mist_rtx_3070_gemma,
    scaled_trainer_steps,
)


def _minimal_data(**overrides) -> PretrainDataConfig:
    return PretrainDataConfig(
        source=HubTextSource(dataset="Salesforce/wikitext"),
        **overrides,
    )


def _manifest(**overrides) -> Manifest:
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
    return Manifest(**base)


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
    assert hw.per_device_train_batch_size == 4
    assert hw.gradient_accumulation_steps == 8
    assert hw.effective_batch_size == 32
    assert hw.torch_compile is True
    assert hw.gradient_checkpointing is False
    assert hw.optim == "adamw_torch_fused"


def test_mist_rtx_3070_gemma_keeps_vram_safe_microbatch():
    hw = mist_rtx_3070_gemma()
    assert hw.label == "mist-rtx-3070"
    assert hw.per_device_train_batch_size == 1
    assert hw.gradient_accumulation_steps == 32
    assert hw.effective_batch_size == 32
    assert hw.gradient_checkpointing is True
    assert hw.torch_compile is True
    assert hw.optim == "adamw_torch_fused"


def test_manifest_to_pretrain_config_merges_segments(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = _manifest()
    cfg = manifest.to_pretrain_config()

    assert cfg.data is manifest.data
    assert cfg.arch is manifest.model
    assert cfg.trainer.output_dir == tmp_path / "output" / "train" / "test-run"
    assert cfg.trainer.run_name == "test-run"
    assert cfg.trainer.max_steps == 5_000
    assert cfg.trainer.warmup_steps == 200
    assert cfg.trainer.warmup_ratio is None
    assert cfg.trainer.data_seed == manifest.data.seed
    assert cfg.trainer.per_device_train_batch_size == 4
    assert cfg.trainer.gradient_accumulation_steps == 8
    assert cfg.trainer.torch_compile is True
    assert cfg.trainer.logging_steps == 5
    assert cfg.trainer.eval_steps == 100
    assert cfg.trainer.save_steps == 100
    assert cfg.trainer.report_to == "none"


def test_manifest_routes_completion_data_to_sft_only(tmp_path: Path):
    pairs = tmp_path / "pairs.json"
    pairs.write_text(
        '[{"slug":"x","prompt":"p","completion":"c"}]',
        encoding="utf-8",
    )
    data = CompletionDataConfig(train_path=str(pairs))
    sft = _manifest(
        stage="sft",
        data=data,
        model=PretrainedLmConfig(model_name="EleutherAI/pythia-70m"),
    )
    assert sft.to_sft_config().data is data

    pre = _manifest(data=data)
    with pytest.raises(ValueError, match="CompletionDataConfig"):
        pre.validate()


def test_manifest_wandb_name_override(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = _manifest(
        wandb=WandbConfig(
            entity="logbook",
            project="ink-explore",
            name="custom-wandb-name",
            enabled=True,
        ),
    )
    cfg = manifest.to_pretrain_config()
    assert cfg.trainer.run_name == "custom-wandb-name"
    assert cfg.trainer.report_to == "wandb"


def test_manifest_explicit_cadence_overrides_scale(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = _manifest(
        schedule=ScheduleConfig(
            max_steps=5_000,
            warmup_steps=200,
            logging_steps=50,
            eval_steps=500,
            save_steps=500,
        ),
    )
    cfg = manifest.to_pretrain_config()
    assert cfg.trainer.logging_steps == 50
    assert cfg.trainer.eval_steps == 500
    assert cfg.trainer.save_steps == 500


def test_manifest_with_hardware_composition(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = _manifest().with_hardware(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        label="bigger-gpu",
    )
    assert manifest.hardware.label == "bigger-gpu"
    assert manifest.hardware.effective_batch_size == 32
    cfg = manifest.to_pretrain_config()
    assert cfg.trainer.per_device_train_batch_size == 4
    assert cfg.trainer.gradient_accumulation_steps == 8


def test_manifest_with_model_and_schedule(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = (
        _manifest()
        .with_model(n_layer=6)
        .with_schedule(learning_rate=3e-4)
        .variant(run_name="ablation-l6")
    )
    assert manifest.run_name == "ablation-l6"
    assert manifest.model.n_layer == 6
    assert manifest.schedule.learning_rate == 3e-4
    cfg = manifest.to_pretrain_config()
    assert cfg.arch.n_layer == 6
    assert cfg.trainer.learning_rate == 3e-4
    assert cfg.trainer.output_dir == tmp_path / "output" / "train" / "ablation-l6"


def test_manifest_validate_requires_wandb_identity():
    manifest = _manifest(wandb=WandbConfig(enabled=True))
    with pytest.raises(ValueError, match="wandb_entity"):
        manifest.validate()


def test_manifest_validate_block_vs_positions():
    manifest = _manifest(
        data=_minimal_data(block_size=2048),
        model=gpt2_arch(n_positions=1024),
    )
    with pytest.raises(ValueError, match="block_size"):
        manifest.validate()


def test_hardware_config_validate():
    with pytest.raises(ValueError, match="per_device_train_batch_size"):
        HardwareConfig(per_device_train_batch_size=0).validate()
    with pytest.raises(ValueError, match="dataloader_prefetch_factor"):
        HardwareConfig(dataloader_prefetch_factor=0).validate()
    with pytest.raises(ValueError, match="dataloader_persistent_workers"):
        HardwareConfig(
            dataloader_num_workers=0,
            dataloader_persistent_workers=True,
        ).validate()


def test_manifest_speed_hardware_knobs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = _manifest().with_hardware(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        gradient_checkpointing=False,
        tf32=True,
        torch_compile=True,
        optim="adamw_torch_fused",
    )
    cfg = manifest.to_pretrain_config()
    assert cfg.trainer.per_device_train_batch_size == 4
    assert cfg.trainer.gradient_accumulation_steps == 8
    assert cfg.trainer.dataloader_num_workers == 8
    assert cfg.trainer.dataloader_prefetch_factor == 4
    assert cfg.trainer.dataloader_persistent_workers is True
    assert cfg.trainer.gradient_checkpointing is False
    assert cfg.trainer.tf32 is True
    assert cfg.trainer.torch_compile is True
    assert cfg.trainer.optim == "adamw_torch_fused"


def test_manifest_gen_config_follows_model_family():
    from alien_ink.hf.gen import GenConfig
    from alien_ink.hf.model import gemma_arch

    gpt2_manifest = _manifest()
    assert gpt2_manifest.gen_config().add_special_tokens is True

    gemma_manifest = _manifest(model=gemma_arch())
    assert gemma_manifest.gen_config(max_new_tokens=120) == GenConfig(
        max_new_tokens=120,
        do_sample=False,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        stop_strings=(".", "!", "?"),
        add_special_tokens=False,
    )


def test_manifest_stage_defaults_to_pre():
    assert _manifest().stage == "pre"


def test_manifest_stage_variant_and_validate():
    assert _manifest().variant(stage="sft").stage == "sft"
    with pytest.raises(ValueError, match="stage"):
        _manifest().variant(stage="rlhf").validate()  # type: ignore[arg-type]


def _sft_manifest(**overrides) -> Manifest:
    base = dict(
        stage="sft",
        model=PretrainedLmConfig(model_name="EleutherAI/pythia-160m"),
        schedule=ScheduleConfig(max_steps=200, warmup_steps=20, learning_rate=3e-5),
    )
    base.update(overrides)
    return _manifest(**base)


def test_manifest_sft_requires_pretrained_model():
    with pytest.raises(ValueError, match="PretrainedLmConfig"):
        _manifest(stage="sft").validate()


def test_manifest_pre_rejects_pretrained_model():
    with pytest.raises(ValueError, match="CausalLmArchConfig"):
        _manifest(
            model=PretrainedLmConfig(model_name="EleutherAI/pythia-160m")
        ).validate()


def test_manifest_to_sft_config(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = _sft_manifest()
    cfg = manifest.to_sft_config()

    assert cfg.data is manifest.data
    assert cfg.model is manifest.model
    assert cfg.model.model_name == "EleutherAI/pythia-160m"
    assert cfg.trainer.output_dir == tmp_path / "output" / "train" / "test-run"
    assert cfg.trainer.max_steps == 200
    assert cfg.trainer.learning_rate == 3e-5
    assert cfg.trainer.report_to == "none"


def test_manifest_stage_config_dispatch(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="requires stage='pre'"):
        _sft_manifest().to_pretrain_config()
    with pytest.raises(ValueError, match="requires stage='sft'"):
        _manifest().to_sft_config()


def test_manifest_sft_gen_config_uses_plain_defaults():
    from alien_ink.hf.gen import GenConfig

    manifest = _sft_manifest()
    assert manifest.gen_config() == GenConfig()
    assert manifest.gen_config(max_new_tokens=120).max_new_tokens == 120


def test_schedule_validates_ratio_and_cadence():
    ScheduleConfig(max_steps=100, warmup_steps=None, warmup_ratio=0.04).validate()
    with pytest.raises(ValueError, match="only one"):
        ScheduleConfig(warmup_steps=10, warmup_ratio=0.1).validate()
    with pytest.raises(ValueError, match="multiple"):
        ScheduleConfig(eval_steps=30, save_steps=100).validate()


def test_all_zdeck_manifests_are_explicit_and_valid():
    import alien_ink.zdeck as zdeck

    required_arch_fields = {
        "hidden_act",
        "hidden_dropout",
        "attention_dropout",
        "norm_epsilon",
        "initializer_range",
        "rope_theta",
        "rotary_pct",
        "tie_word_embeddings",
        "num_key_value_heads",
        "attention_implementation",
    }
    required_pretrained_fields = {
        "model_name",
        "tokenizer_name",
        "attention_implementation",
        "use_cache",
        "trust_remote_code",
    }
    modules = [
        info.name
        for info in pkgutil.iter_modules(zdeck.__path__)
        if not info.name.startswith("_")
    ]
    assert modules
    for name in modules:
        module = importlib.import_module(f"alien_ink.zdeck.{name}")
        manifest = module.MANIFEST
        manifest.validate()
        source = inspect.getsource(module)
        if isinstance(manifest.model, CausalLmArchConfig):
            required_model_fields = required_arch_fields
        else:
            assert isinstance(manifest.model, PretrainedLmConfig)
            required_model_fields = required_pretrained_fields
        for field_name in required_model_fields:
            assert f"{field_name}=" in source, f"{name} omits {field_name}"
        assert "warmup_ratio=" in source, f"{name} omits warmup_ratio"
