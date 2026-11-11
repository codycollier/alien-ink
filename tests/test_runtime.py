"""Exercise both training entrypoints without Hub, GPU, or reporting services."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from alien_ink.com.device import AcceleratorInfo
from alien_ink.com.env import EnvConfig
from alien_ink.hf import pretrain, runtime, sft, trainer
from alien_ink.hf.ds import HubTextSource, PretrainDataConfig
from alien_ink.hf.metrics import ModelSize
from alien_ink.hf.model import PretrainedLmConfig
from alien_ink.hf.trainer import CausalLmTrainerConfig


@pytest.mark.parametrize("stage", ["pre", "sft"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_training_entrypoint_artifacts_and_reporting(
    tmp_path, monkeypatch, stage, enabled, fails
):
    accel = AcceleratorInfo(
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
        torch_version="test",
        platform="test",
        python_version="test",
    )
    monkeypatch.setattr(runtime, "collect_accelerator_info", lambda **kw: accel)
    monkeypatch.setattr(runtime, "introspect", lambda **kw: "CPU")
    monkeypatch.setattr(
        runtime,
        "load_env",
        lambda *args, **kw: EnvConfig(
            None,
            None,
            kw["wandb_entity"],
            kw["wandb_project"],
        ),
    )
    seeds, reporting_dirs, events = [], [], []
    monkeypatch.setattr(runtime, "set_seed", seeds.append)
    monkeypatch.setattr(runtime, "set_wandb_dir", reporting_dirs.append)
    monkeypatch.setattr(trainer, "push_summary_to_wandb", lambda summary: None)

    @contextmanager
    def reporting(**kw):
        events.append(("enter", kw))
        try:
            yield
        finally:
            events.append(("exit", None))

    monkeypatch.setattr(runtime, "wandb_run", reporting)
    error = RuntimeError("training failed")

    class FakeTrainer:
        def __init__(self, **kw):
            self.args = kw["config"]
            self.state = SimpleNamespace(global_step=0)

        def train(self, *, resume_from_checkpoint):
            assert resume_from_checkpoint == "checkpoint-2"
            self.state.global_step = 3
            if fails:
                raise error
            return SimpleNamespace(metrics={"train_runtime": 1.0}, training_loss=0.5)

        def save_model(self, output_dir):
            (tmp_path / "model.saved").touch()

    tokenizer = SimpleNamespace(
        vocab_size=16, save_pretrained=lambda path: (path / "tokenizer.saved").touch()
    )
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32))
    stage_module = pretrain if stage == "pre" else sft
    loader = (
        "build_model_and_tokenizer"
        if stage == "pre"
        else "load_hub_model_and_tokenizer"
    )
    monkeypatch.setattr(stage_module, loader, lambda config: (model, tokenizer))
    monkeypatch.setattr(
        stage_module, "count_model_params", lambda *args, **kw: ModelSize(10, 10, 8)
    )
    monkeypatch.setattr(stage_module, "prepare_lm_datasets", lambda *args: ([{}], [{}]))
    monkeypatch.setattr(
        runtime, "build_causal_lm_trainer", lambda **kw: FakeTrainer(**kw)
    )
    data = PretrainDataConfig(source=HubTextSource(dataset="fake/text"), block_size=8)
    trainer_config = CausalLmTrainerConfig(
        output_dir=tmp_path, max_steps=10, warmup_steps=0
    )
    if stage == "pre":
        config = pretrain.PretrainConfig(data=data, trainer=trainer_config)
        entrypoint = pretrain.pretrain
    else:
        config = sft.SftConfig(
            data=data,
            model=PretrainedLmConfig(model_name="fake/model"),
            trainer=trainer_config,
        )
        entrypoint = sft.finetune
    kwargs = dict(
        use_wandb=enabled,
        wandb_entity="test",
        wandb_project="test",
        wandb_name="resolved-name",
        resume_from_checkpoint="checkpoint-2",
        env_files=(),
    )
    if fails:
        with pytest.raises(RuntimeError) as raised:
            entrypoint(config, **kwargs)
        assert raised.value is error
    else:
        result, summary = entrypoint(config, **kwargs)
        assert result.state.global_step == 3
        assert summary.status == "completed"
    payload = json.loads((tmp_path / "run_config.json").read_text())
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert payload["model_size"] == {
        "total_params": 10,
        "trainable_params": 10,
        "non_embedding_params": 8,
    }
    assert payload["trainer"]["run_name"] == "resolved-name"
    assert payload["trainer"]["report_to"] == ("wandb" if enabled else "none")
    assert summary["status"] == ("failed" if fails else "completed")
    if stage == "sft":
        assert payload["model"]["model_name"] == "fake/model"
    else:
        assert "arch" in payload
    assert (tmp_path / "model.saved").exists() is not fails
    assert (tmp_path / "tokenizer.saved").exists() is not fails
    assert seeds == [trainer_config.seed]
    assert reporting_dirs == ([tmp_path] if enabled else [])
    assert events[0][1]["enabled"] is enabled
    assert events[-1][0] == "exit"


def test_moved_helpers_keep_compatibility_imports():
    from alien_ink.com import serialization, wb
    from alien_ink.hf import completion, ds, eval

    assert wb.serialize_config is serialization.serialize_config
    assert pretrain.prepare_lm_datasets is ds.prepare_lm_datasets
    for name in (
        "EvalItem",
        "load_eval_items",
        "tokenize_prompt_completion",
        "build_completion_eval_dataset",
    ):
        assert getattr(eval, name) is getattr(completion, name)
