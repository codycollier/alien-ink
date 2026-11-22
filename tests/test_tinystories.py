import importlib
import math
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from datasets import Dataset
from transformers import GPTNeoForCausalLM

from alien_ink.hf import ds
from alien_ink.hf.gen import generate_completion_result
from alien_ink.hf.model import build_model_from_scratch, load_pretrained_model, load_tokenizer
from alien_ink.hf.tok import tokenize_and_chunk
from alien_ink.zdeck.pre_tinystories_1m_1ep_mist import MANIFEST as EPOCH_MANIFEST
from alien_ink.zdeck.pre_tinystories_1m_5k_mist import MANIFEST

STORIES = ["A cat sat.", "A dog ran.", "They met.", "Café ☕."]


@pytest.fixture
def tokenizer(tmp_path):
    script = Path(__file__).resolve().parents[1] / "bin/prepare-tinystories-tokenizer.py"
    train = runpy.run_path(str(script))["train_tokenizer"]
    tokenizer = train(STORIES, vocab_size=300)
    path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(path)
    loaded = load_tokenizer(path)
    assert len(loaded) == len(tokenizer)
    assert loaded.get_vocab() == tokenizer.get_vocab()
    assert len({loaded.bos_token_id, loaded.eos_token_id, loaded.pad_token_id}) == 3
    assert loaded.decode(loaded(STORIES[-1]).input_ids) == STORIES[-1]
    return path, loaded


def test_pinned_splits_and_shuffling(monkeypatch):
    calls = []
    stories = {split: [f"{split} story {i}" for i in range(20)] for split in ("train", "validation")}

    def load(dataset, name, *, split, streaming, **kwargs):
        calls.append((dataset, name, split, streaming, kwargs["revision"]))
        result = Dataset.from_dict({"text": stories[split]})
        return result.to_iterable_dataset() if streaming else result

    monkeypatch.setattr(ds, "load_dataset", load)
    config = ds.tinystories(max_eval_samples=5)
    train, validation = ds.load_train_eval(config, verbose=False)
    expected = []
    for split, streaming in (("train", True), ("validation", True), ("validation", False)):
        expected.append((
            "roneneldan/TinyStories", "default", split, streaming,
            "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        ))
    assert list(ds.load_hub_text(config.eval_source)["text"]) == stories["validation"]
    assert calls == expected
    for split, actual in (("train", [row["text"] for row in train]), ("validation", list(validation["text"]))):
        raw = Dataset.from_dict({"text": stories[split]}).to_iterable_dataset()
        shuffled = list(raw.shuffle(seed=101, buffer_size=10_000))
        assert actual == [row["text"] for row in shuffled[:len(actual)]]
    assert len(validation) == 5


def test_packs_short_stories_with_eos(tokenizer):
    _, tok = tokenizer
    ids = [tok(text, add_special_tokens=False).input_ids for text in STORIES[:2]]
    expected = [*ids[0], tok.eos_token_id, *ids[1], tok.eos_token_id]
    raw = Dataset.from_dict({"text": STORIES[:2]})
    for dataset in (raw, raw.to_iterable_dataset()):
        blocks = tokenize_and_chunk(
            dataset, tok, block_size=len(expected), respect_document_boundaries=False, num_proc=1,
        )
        block, = list(blocks)
        assert block["input_ids"] == block["labels"] == expected
        assert block["attention_mask"] == [1] * len(expected)


@pytest.mark.parametrize("size,count", [("1m", 982_800), ("3m", 2_960_928), ("33m", 33_455_616)])
def test_parameter_counts(size, count, tokenizer):
    manifest = importlib.import_module(f"alien_ink.zdeck.pre_tinystories_{size}_5k_mist").MANIFEST

    _, tok = tokenizer
    shape = SimpleNamespace(
        vocab_size=6_144,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    model = build_model_from_scratch(shape, manifest.model, verbose=False)
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        assert getattr(model.config, field) == getattr(tok, field)
    assert isinstance(model, GPTNeoForCausalLM)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == count
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert model.config.attention_layers == ["global", "local"] * (manifest.model.n_layer // 2)
    assert model.config.window_size == 256
    assert model.config.max_position_embeddings == 512
    assert model.config.use_cache is False


@pytest.mark.parametrize("mode", ["stream", "complete"])
def test_offline_training_checkpoint_and_generation(tmp_path, tokenizer, mode):
    path, _ = tokenizer
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(path), mode],
        cwd=tmp_path,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"},
        check=True,
    )


def run_cpu_fixture(path, mode):
    tok = load_tokenizer(path)
    raw = Dataset.from_dict({"text": STORIES * 8})
    manifest = EPOCH_MANIFEST if mode == "complete" else MANIFEST
    run = (
        manifest.variant(run_name="tinystories-cpu-fixture")
        .with_data(block_size=8, max_eval_samples=8)
        .with_model(tokenizer_name=str(path), n_positions=16, local_attention_window=4,
                    n_embd=16, n_layer=2, n_head=2, intermediate_size=32)
        .with_hardware(per_device_train_batch_size=2, per_device_eval_batch_size=2,
                       gradient_accumulation_steps=1, tf32=False, prefer_bf16=False,
                       prefer_fp16=False)
        .with_schedule(warmup_steps=0)
        .with_trainer_knobs(logging_nan_inf_filter=False)
    )
    if mode == "stream":
        run = run.with_schedule(max_steps=1, eval_steps=1, save_steps=1, logging_steps=1)
    with (
        patch.object(ds, "stream_hub_text", side_effect=lambda *a, **kw: raw.to_iterable_dataset()),
        patch.object(ds, "load_hub_text", return_value=raw),
    ):
        trainer, summary = run.train()
    if mode == "complete":
        token_count = sum(len(ids) + 1 for ids in tok(list(raw["text"])).input_ids)
        blocks = token_count // run.data.block_size
        assert summary.global_step == math.ceil(blocks / 2)
        assert trainer.state.epoch == pytest.approx(1.0)
    else:
        assert summary.global_step == 1
    assert math.isfinite(summary.train_loss)
    assert any(math.isfinite(log.get("eval_loss", math.nan)) for log in trainer.state.log_history)
    model, loaded = load_pretrained_model(run.output_dir() / f"checkpoint-{summary.global_step}", "cpu", family="gpt-neo")
    assert loaded.get_vocab() == tok.get_vocab()
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    result = generate_completion_result(
        model, loaded, "A cat", "cpu", run.gen_config(max_new_tokens=2, do_sample=False),
    )
    assert 1 <= result.n_tokens <= 2


if __name__ == "__main__":
    run_cpu_fixture(Path(sys.argv[1]), sys.argv[2])
