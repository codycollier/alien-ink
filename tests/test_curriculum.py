"""Tests for curriculum sequencing of pretraining datasets."""

from __future__ import annotations

from unittest.mock import patch

import pytest

datasets = pytest.importorskip("datasets")
transformers = pytest.importorskip("transformers")

from datasets import Dataset, IterableDataset  # noqa: E402

from alien_ink.hf.curriculum import (  # noqa: E402
    Curriculum,
    CurriculumPhase,
    bounded_phase_blocks,
    chain_phase_blocks,
    prepare_curriculum_datasets,
)
from alien_ink.hf.ds import (  # noqa: E402
    HubTextSource,
    PretrainDataConfig,
    geo_us_states,
    hub_text,
)
from alien_ink.hf.manifest import (  # noqa: E402
    Manifest,
    ScheduleConfig,
    WandbConfig,
    mist_rtx_3070,
)
from alien_ink.hf.model import gpt2_arch  # noqa: E402
from alien_ink.hf.trainer import best_model_metric, has_eval_examples  # noqa: E402


class _FakeTok:
    eos_token_id = 99

    def __call__(self, texts, *, add_special_tokens=True):
        assert add_special_tokens is False
        input_ids = [[ord(ch) % 50 for ch in text] for text in texts]
        attention_mask = [[1] * len(ids) for ids in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _config(prefix: str, **overrides) -> PretrainDataConfig:
    base = dict(
        source=HubTextSource(dataset=f"dummy/{prefix}", split="train"),
        mode="stream",
        max_eval_samples=1,
        block_size=4,
        stream_shuffle_buffer=2,
        tokenizer_num_proc=1,
        seed=0,
    )
    base.update(overrides)
    return PretrainDataConfig(**base)


def _phase(prefix: str, steps: int, **overrides) -> CurriculumPhase:
    return CurriculumPhase(data=_config(prefix, **overrides), steps=steps)


def _blocks(value: int, n: int) -> Dataset:
    return Dataset.from_list(
        [{"input_ids": [value] * 4, "labels": [value] * 4} for _ in range(n)]
    )


def test_curriculum_totals_and_boundaries():
    cur = Curriculum(phases=(_phase("a", 5), _phase("b", 2), _phase("c", 3)))
    assert cur.total_steps() == 10
    assert cur.boundaries() == (5, 7, 10)
    assert cur.block_size == 4
    assert cur.seed == 0


def test_curriculum_validate_rejects_bad_shapes():
    with pytest.raises(ValueError, match="at least one phase"):
        Curriculum(phases=()).validate()
    with pytest.raises(ValueError, match="steps"):
        Curriculum(phases=(_phase("a", 0),)).validate()
    with pytest.raises(ValueError, match="block_size"):
        Curriculum(
            phases=(_phase("a", 1), _phase("b", 1, block_size=8)),
        ).validate()
    with pytest.raises(ValueError, match="names"):
        Curriculum(
            phases=(_phase("a", 1),),
            eval_data={" ": _config("a")},
        ).validate()


def test_curriculum_eval_configs_default_to_first_phase():
    first = _config("a")
    cur = Curriculum(phases=(CurriculumPhase(data=first, steps=1),))
    assert cur.eval_configs() == {"eval": first}

    explicit = _config("b")
    assert Curriculum(
        phases=(CurriculumPhase(data=first, steps=1),),
        eval_data=explicit,
    ).eval_configs() == {"eval": explicit}

    named = Curriculum(
        phases=(CurriculumPhase(data=first, steps=1),),
        eval_data={"geo": explicit, "c4": first},
    )
    assert list(named.eval_configs()) == ["geo", "c4"]


def test_bounded_phase_blocks_repeats_materialized_to_fill():
    rows = [r["input_ids"][0] for r in bounded_phase_blocks(_blocks(7, 3), num_samples=7)]
    assert rows == [7] * 7


def test_bounded_phase_blocks_takes_from_stream():
    stream = _blocks(5, 10).to_iterable_dataset()
    bounded = bounded_phase_blocks(stream, num_samples=4)
    assert isinstance(bounded, IterableDataset)
    assert len(list(bounded)) == 4


def test_bounded_phase_blocks_rejects_empty_materialized():
    empty = Dataset.from_dict({"input_ids": [], "labels": []})
    with pytest.raises(ValueError, match="no training blocks"):
        bounded_phase_blocks(empty, num_samples=2)


def test_chain_phase_blocks_preserves_order_and_aligns_features():
    # Lazy-mapped stream (features=None) followed by a concrete repeated phase.
    lazy = _blocks(1, 5).to_iterable_dataset().map(lambda ex: ex).take(3)
    assert lazy.features is None
    concrete = _blocks(2, 2).to_iterable_dataset().repeat(None).take(5)
    chained = chain_phase_blocks([lazy, concrete])
    values = [r["input_ids"][0] for r in chained]
    assert values == [1, 1, 1, 2, 2, 2, 2, 2]


def _fake_stream_factory(rows_by_dataset):
    def fake_stream(source, *, trust_remote_code=False):
        del trust_remote_code
        rows = rows_by_dataset[source.dataset]
        return Dataset.from_list(rows).to_iterable_dataset()

    return fake_stream


def test_prepare_curriculum_datasets_end_to_end():
    # block_size=4; each 8-char row tokenizes to 9 ids (incl. EOS) → 2 blocks.
    rows_by_dataset = {
        "dummy/a": [{"text": "a" * 8} for _ in range(20)],
        "dummy/b": [{"text": "b" * 8} for _ in range(4)],
    }
    cur = Curriculum(
        phases=(
            _phase("a", 2),  # stream: 2 steps * 2 blocks/step = 4 blocks
            _phase("b", 3, mode="complete"),  # repeats to fill 6 blocks
        ),
        eval_data={"b": _config("b", mode="complete"), "a": _config("a")},
    )

    with patch(
        "alien_ink.hf.ds.stream_hub_text",
        side_effect=_fake_stream_factory(rows_by_dataset),
    ), patch(
        "alien_ink.hf.ds.load_hub_text",
        side_effect=lambda source, **kw: Dataset.from_list(
            rows_by_dataset[source.dataset]
        ),
    ):
        train, evals = prepare_curriculum_datasets(
            cur,
            _FakeTok(),
            samples_per_step=2,
            verbose=False,
        )

    blocks = list(train)
    assert len(blocks) == 10
    a_id, b_id = ord("a") % 50, ord("b") % 50
    assert all(block["input_ids"][0] == a_id for block in blocks[:4])
    assert all(block["input_ids"][0] == b_id for block in blocks[4:])

    assert set(evals) == {"b", "a"}
    for eval_set in evals.values():
        assert len(eval_set) >= 1


def test_prepare_curriculum_single_eval_returns_dataset():
    rows_by_dataset = {"dummy/a": [{"text": "a" * 8} for _ in range(10)]}
    cur = Curriculum(phases=(_phase("a", 1),))

    with patch(
        "alien_ink.hf.ds.stream_hub_text",
        side_effect=_fake_stream_factory(rows_by_dataset),
    ):
        train, eval_set = prepare_curriculum_datasets(
            cur,
            _FakeTok(),
            samples_per_step=2,
            verbose=False,
        )

    assert isinstance(eval_set, Dataset)
    assert len(list(train)) == 2


def test_has_eval_examples_accepts_mapping():
    filled = _blocks(1, 2)
    empty = Dataset.from_dict({"input_ids": [], "labels": []})
    assert has_eval_examples({"a": filled, "b": filled}) is True
    assert has_eval_examples({"a": filled, "b": empty}) is False
    assert has_eval_examples({}) is False
    assert has_eval_examples(filled) is True
    assert has_eval_examples(None) is False


def test_best_model_metric_follows_first_named_eval():
    filled = _blocks(1, 2)
    assert best_model_metric({"geo": filled, "c4": filled}) == "eval_geo_loss"
    assert best_model_metric(filled) == "eval_loss"
    assert best_model_metric(None) == "eval_loss"


def _curriculum_manifest(schedule: ScheduleConfig, phases) -> Manifest:
    return Manifest(
        run_name="test-curriculum",
        title="Test curriculum",
        data=Curriculum(phases=phases),
        model=gpt2_arch(),
        hardware=mist_rtx_3070(),
        wandb=WandbConfig(entity="logbook", project="ink-explore", enabled=False),
        schedule=schedule,
    )


def test_manifest_accepts_matching_curriculum():
    phases = (_phase("a", 400, block_size=1024), _phase("b", 100, block_size=1024))
    schedule = ScheduleConfig(
        max_steps=500, warmup_steps=50, eval_steps=100, save_steps=100
    )
    manifest = _curriculum_manifest(schedule, phases)
    manifest.validate()
    cfg = manifest.to_pretrain_config()
    assert cfg.trainer.max_steps == 500
    assert cfg.trainer.data_seed == 0


def test_manifest_rejects_curriculum_step_mismatch():
    phases = (_phase("a", 400, block_size=1024),)
    schedule = ScheduleConfig(max_steps=500, warmup_steps=50)
    with pytest.raises(ValueError, match="total_steps"):
        _curriculum_manifest(schedule, phases).validate()


def test_manifest_rejects_curriculum_epoch_mode():
    phases = (_phase("a", 400, block_size=1024),)
    schedule = ScheduleConfig(max_steps=-1, warmup_steps=None, warmup_ratio=0.04)
    with pytest.raises(ValueError, match="step mode"):
        _curriculum_manifest(schedule, phases).validate()


def test_manifest_warns_on_unaligned_phase_boundary():
    phases = (_phase("a", 450, block_size=1024), _phase("b", 50, block_size=1024))
    schedule = ScheduleConfig(
        max_steps=500, warmup_steps=50, eval_steps=100, save_steps=100
    )
    manifest = _curriculum_manifest(schedule, phases)
    with patch("alien_ink.hf.manifest.log") as mock_log:
        manifest.validate()
    mock_log.warning.assert_called_once()
    assert 450 in mock_log.warning.call_args.args[1]


def test_hub_text_factory_defaults():
    cfg = hub_text("codycollier/example")
    assert cfg.mode == "complete"
    assert cfg.source.dataset == "codycollier/example"
    assert cfg.source.text_column == "text"
    assert cfg.eval_source is None
    cfg.validate()


def test_geo_us_states_factory_small_holdout():
    cfg = geo_us_states()
    assert cfg.source.dataset == "codycollier/geo-us-states"
    assert cfg.mode == "complete"
    # 56-row corpus: the default 1,000-row hold-out would swallow it.
    assert cfg.max_eval_samples == 4
    assert cfg.respect_document_boundaries is True
    cfg.validate()
