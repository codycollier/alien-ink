"""Tests for FLOPs estimates and end-of-run summaries."""

from __future__ import annotations

from pathlib import Path

from alien_ink.com.device import AcceleratorInfo
from alien_ink.hf.metrics import (
    ModelSize,
    build_run_summary,
    count_model_params,
    estimate_train_flops,
    save_run_config,
    save_run_summary,
)


def _accel(**overrides) -> AcceleratorInfo:
    base = dict(
        device="cuda",
        use_fp16=False,
        use_bf16=True,
        precision="bf16",
        world_size=1,
        gpu_count=1,
        gpu_name="NVIDIA GeForce RTX 3070",
        gpu_memory_total_gb=8.0,
        cuda_available=True,
        cuda_version="12.1",
        cudnn_version="8902",
        torch_version="2.4.0",
        platform="test",
        python_version="3.11.0",
        peak_tflops=20.31,
    )
    base.update(overrides)
    return AcceleratorInfo(**base)


def test_estimate_train_flops_kaplan_6n():
    assert estimate_train_flops(non_embedding_params=100, tokens=10) == 6 * 100 * 10


def _true_non_embedding_params(model) -> int:
    """Ground truth: total minus deduped input/output/positional tables."""
    total = sum(p.numel() for p in model.parameters())
    seen: set[int] = set()
    embed = 0
    wpe = getattr(getattr(model, "transformer", None), "wpe", None)
    for module in (model.get_input_embeddings(), model.get_output_embeddings(), wpe):
        weight = getattr(module, "weight", None)
        if weight is not None and id(weight) not in seen:
            seen.add(id(weight))
            embed += weight.numel()
    return total - embed


def test_count_model_params_per_family_embedding_accounting():
    """Tied (GPT-2/Gemma), untied (NeoX), and rotary-only families all count right."""
    from transformers import (
        GemmaConfig,
        GemmaForCausalLM,
        GPT2Config,
        GPT2LMHeadModel,
        GPTNeoXConfig,
        GPTNeoXForCausalLM,
    )

    tiny = dict(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    models = [
        GPT2LMHeadModel(
            GPT2Config(vocab_size=128, n_embd=32, n_layer=2, n_head=4, n_positions=64)
        ),
        GPTNeoXForCausalLM(GPTNeoXConfig(**tiny)),
        GemmaForCausalLM(GemmaConfig(**tiny, head_dim=8)),
    ]
    for model in models:
        size = count_model_params(model, vocab_size=128)
        assert size.non_embedding_params == _true_non_embedding_params(model)
        assert size.total_params == sum(p.numel() for p in model.parameters())
        assert size.trainable_params == size.total_params


def test_build_run_summary_throughput_and_mfu():
    summary = build_run_summary(
        run_name="r",
        run_label="regular",
        status="completed",
        global_step=100,
        max_steps=100,
        tokens_per_step=1_024,
        train_runtime_sec=50.0,
        train_loss=2.5,
        model_size=ModelSize(
            total_params=124_000_000,
            trainable_params=124_000_000,
            non_embedding_params=100_000_000,
        ),
        accelerator=_accel(),
    )
    assert summary.tokens_trained == 100 * 1_024
    assert summary.tokens_per_sec == summary.tokens_trained / 50.0
    assert summary.steps_per_sec == 2.0
    assert summary.flops_total == 6 * 100_000_000 * summary.tokens_trained
    assert summary.tflops_per_sec is not None
    assert summary.mfu is not None
    assert 0 < summary.mfu < 2  # sanity; synthetic numbers can exceed 1


def test_build_run_summary_uses_trainer_metrics_fallback():
    summary = build_run_summary(
        run_name="r",
        run_label="flight_check",
        status="completed",
        global_step=10,
        max_steps=10,
        tokens_per_step=128,
        train_runtime_sec=None,
        train_loss=None,
        model_size=ModelSize(10, 10, 8),
        accelerator=_accel(peak_tflops=None),
        trainer_metrics={
            "train_runtime": 5.0,
            "train_loss": 3.1,
            "train_steps_per_second": 2.0,
        },
    )
    assert summary.train_runtime_sec == 5.0
    assert summary.train_loss == 3.1
    assert summary.steps_per_sec == 2.0
    assert summary.mfu is None


def test_save_run_config_and_summary(tmp_path: Path):
    payload = {"run_name": "x", "tokens_per_optimizer_step": 128}
    cfg_path = save_run_config(tmp_path, payload)
    assert cfg_path.name == "run_config.json"
    assert cfg_path.is_file()

    summary = build_run_summary(
        run_name="x",
        run_label="regular",
        status="interrupted",
        global_step=3,
        max_steps=10,
        tokens_per_step=128,
        train_runtime_sec=1.5,
        train_loss=None,
        model_size=ModelSize(10, 10, 8),
        accelerator=_accel(),
    )
    path = save_run_summary(tmp_path, summary)
    assert path.name == "run_summary.json"
    text = path.read_text(encoding="utf-8")
    assert '"status": "interrupted"' in text
    assert '"tokens_trained": 384' in text
