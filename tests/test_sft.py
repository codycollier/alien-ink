"""Tests for pretrained-model loading and the SFT config path."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("transformers")

from alien_ink.hf.ds import HubTextSource, PretrainDataConfig  # noqa: E402
from alien_ink.hf.model import (  # noqa: E402
    PretrainedLmConfig,
    load_hub_model_and_tokenizer,
    model_max_positions,
)
from alien_ink.hf.sft import SftConfig  # noqa: E402
from alien_ink.hf.trainer import CausalLmTrainerConfig  # noqa: E402


def test_pretrained_config_resolves_tokenizer_to_model_name():
    config = PretrainedLmConfig(model_name="EleutherAI/pythia-160m")
    assert config.resolved_tokenizer_name() == "EleutherAI/pythia-160m"
    config = PretrainedLmConfig(model_name="a/b", tokenizer_name="c/d")
    assert config.resolved_tokenizer_name() == "c/d"


def test_pretrained_config_validate():
    PretrainedLmConfig(model_name="EleutherAI/pythia-160m").validate()
    with pytest.raises(ValueError, match="model_name"):
        PretrainedLmConfig(model_name=" ").validate()
    with pytest.raises(ValueError, match="tokenizer_name"):
        PretrainedLmConfig(model_name="a/b", tokenizer_name=" ").validate()
    with pytest.raises(ValueError, match="attention_implementation"):
        PretrainedLmConfig(
            model_name="a/b",
            attention_implementation="flash_attention_2",  # type: ignore[arg-type]
        ).validate()


class _FakeConfig:
    def __init__(self, **kw):
        self.pad_token_id = kw.get("pad_token_id")
        self.use_cache = True
        self.max_position_embeddings = kw.get("max_position_embeddings", 2048)


class _FakeEmbedding:
    def __init__(self, rows: int):
        import torch

        self.weight = torch.zeros(rows, 4)


class _FakeModel:
    def __init__(self, rows: int = 300, pad_token_id=None):
        self.config = _FakeConfig(pad_token_id=pad_token_id)
        self._embedding = _FakeEmbedding(rows)
        self.resized_to: int | None = None

    def get_input_embeddings(self):
        return self._embedding

    def resize_token_embeddings(self, n: int):
        self.resized_to = n
        self._embedding = _FakeEmbedding(n)

    def parameters(self):
        return iter([self._embedding.weight])


class _FakeTok:
    vocab_size = 256
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2

    def __len__(self) -> int:
        return 256


def _patch_hub_loading(monkeypatch, model: _FakeModel):
    from alien_ink.hf import model as model_mod

    class _FakeAuto:
        @staticmethod
        def from_pretrained(name, **kw):
            return model

    monkeypatch.setattr(model_mod, "AutoModelForCausalLM", _FakeAuto)
    monkeypatch.setattr(model_mod, "load_tokenizer", lambda name: _FakeTok())


def test_load_hub_model_sets_cache_and_pad(monkeypatch):
    fake = _FakeModel(rows=300, pad_token_id=None)
    _patch_hub_loading(monkeypatch, fake)
    model, tokenizer = load_hub_model_and_tokenizer(
        PretrainedLmConfig(model_name="fake/model"), verbose=False
    )
    assert model is fake
    assert model.config.use_cache is False
    assert model.config.pad_token_id == tokenizer.pad_token_id
    # Embedding table (300) already covers the tokenizer (256): no resize.
    assert fake.resized_to is None


def test_load_hub_model_resizes_small_embeddings(monkeypatch):
    fake = _FakeModel(rows=100)
    _patch_hub_loading(monkeypatch, fake)
    load_hub_model_and_tokenizer(
        PretrainedLmConfig(model_name="fake/model"), verbose=False
    )
    assert fake.resized_to == 256


def test_model_max_positions():
    assert model_max_positions(_FakeModel()) == 2048


def test_sft_config_validates_segments(tmp_path: Path):
    data = PretrainDataConfig(
        source=HubTextSource(dataset="codycollier/geo-us-states"),
        mode="complete",
        max_eval_samples=4,
    )
    config = SftConfig(
        data=data,
        model=PretrainedLmConfig(model_name="EleutherAI/pythia-160m"),
        trainer=CausalLmTrainerConfig(
            output_dir=tmp_path / "out",
            run_name="sft-test",
            max_steps=100,
            warmup_steps=10,
        ),
    )
    config.validate()

    bad = SftConfig(
        data=data,
        model=PretrainedLmConfig(model_name=" "),
        trainer=config.trainer,
    )
    with pytest.raises(ValueError, match="model_name"):
        bad.validate()
