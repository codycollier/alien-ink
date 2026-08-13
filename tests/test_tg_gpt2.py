"""Tinygrad GPT-2: architecture match, train step, and manifest guards."""

from __future__ import annotations

import math

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import get_parameters

from alien_ink.hf.ds import HubTextSource, PretrainDataConfig
from alien_ink.hf.manifest import HardwareConfig, Manifest, ScheduleConfig, WandbConfig
from alien_ink.hf.metrics import collect_software_versions
from alien_ink.hf.model import CausalLmArchConfig, gpt2_arch, gpt_neox_arch, PretrainedLmConfig
from alien_ink.tg.model import build_gpt2, count_gpt2_params, gelu_new
from alien_ink.tg.pretrain import validate_tg_manifest
from alien_ink.tg.trainer import build_optimizer, cosine_lr, make_train_step


def _tiny_arch(**overrides) -> CausalLmArchConfig:
    return gpt2_arch(
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        **overrides,
    )


def _pre_manifest(**overrides) -> Manifest:
    base = dict(
        run_name="tg-test",
        title="tg test",
        data=PretrainDataConfig(source=HubTextSource(dataset="Salesforce/wikitext")),
        model=gpt2_arch(),
        hardware=HardwareConfig(),
        wandb=WandbConfig(entity="logbook", project="ink-explore", enabled=False),
        schedule=ScheduleConfig(max_steps=8, warmup_steps=1),
    )
    base.update(overrides)
    return Manifest(**base)


def test_gelu_new_matches_tanh_formula():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    expected = 0.5 * x * (
        1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3))
    )
    got = gelu_new(Tensor(x)).numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_dropout_inactive_when_not_training():
    Tensor.training = False
    ones = Tensor.ones(8, 8)
    out = ones.dropout(0.9)
    np.testing.assert_array_equal(out.numpy(), ones.numpy())


def test_build_gpt2_rejects_non_gpt2_family():
    import pytest

    with pytest.raises(ValueError, match="family='gpt-2'"):
        build_gpt2(gpt_neox_arch(n_embd=32, n_layer=2, n_head=4), vocab_size=128)


def test_validate_tg_manifest_rejects_sft_and_other_families():
    import pytest

    validate_tg_manifest(_pre_manifest())
    with pytest.raises(ValueError, match="stage='pre'"):
        validate_tg_manifest(
            _pre_manifest(
                stage="sft",
                model=PretrainedLmConfig(model_name="gpt2"),
            )
        )
    with pytest.raises(ValueError, match="family='gpt-2'"):
        validate_tg_manifest(_pre_manifest(model=gpt_neox_arch()))


def test_tiny_gpt2_forward_shape_and_tying():
    model = build_gpt2(_tiny_arch(), vocab_size=128)
    assert id(model.wte.weight) == id(model.lm_head.weight)
    tokens = Tensor(np.random.default_rng(0).integers(0, 128, size=(2, 16), dtype=np.int32))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 16, 128)
    assert math.isfinite(float(loss.item()))

    sizes = count_gpt2_params(model)
    assert sizes.total_params == sizes.trainable_params
    # Tied head: embedding tables are wte + wpe only.
    embed = int(model.wte.weight.numel()) + int(model.wpe.weight.numel())
    assert sizes.non_embedding_params == sizes.total_params - embed


def test_tiny_gpt2_train_step_changes_weights():
    Tensor.manual_seed(0)
    model = build_gpt2(_tiny_arch(), vocab_size=128)
    before = model.wte.weight.numpy().copy()
    optimizer = build_optimizer(
        model,
        learning_rate=1e-2,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
    )
    step = make_train_step(model, optimizer, max_grad_norm=1.0, accum_steps=1)
    tokens = Tensor(np.random.default_rng(1).integers(0, 128, size=(2, 16), dtype=np.int32))
    loss, grad_norm = step(tokens)
    assert math.isfinite(float(loss.item()))
    assert grad_norm is not None and math.isfinite(grad_norm)
    after = model.wte.weight.numpy()
    assert not np.allclose(before, after)


def test_cosine_lr_warmup_then_decay():
    assert cosine_lr(0, max_steps=100, warmup_steps=10, learning_rate=1.0) == 0.0
    assert cosine_lr(10, max_steps=100, warmup_steps=10, learning_rate=1.0) == 1.0
    mid = cosine_lr(55, max_steps=100, warmup_steps=10, learning_rate=1.0)
    end = cosine_lr(100, max_steps=100, warmup_steps=10, learning_rate=1.0)
    assert 0.0 <= end <= 1e-6
    assert 0.0 < mid < 1.0


def test_clip_grad_norm_preserves_bf16_dtype():
    Tensor.manual_seed(0)
    model = build_gpt2(_tiny_arch(), vocab_size=128)
    for param in get_parameters(model):
        param.replace(param.cast(dtypes.bfloat16).contiguous()).realize()
    optimizer = build_optimizer(
        model,
        learning_rate=1e-2,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
    )
    step = make_train_step(model, optimizer, max_grad_norm=1.0, accum_steps=2)
    tokens = Tensor(
        np.random.default_rng(2).integers(0, 128, size=(2, 2, 16), dtype=np.int32)
    )
    loss, grad_norm = step(tokens)
    assert math.isfinite(float(loss.item()))
    assert grad_norm is not None and math.isfinite(grad_norm)
    for param in optimizer.params:
        if param.grad is not None:
            assert param.grad.dtype == dtypes.bfloat16


def test_software_versions_include_tinygrad():
    versions = collect_software_versions()
    assert versions["tinygrad"]
