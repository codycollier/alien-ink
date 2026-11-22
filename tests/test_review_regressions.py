from types import SimpleNamespace

import pytest
from datasets import Dataset

from alien_ink.hf.curriculum import bounded_phase_blocks
from alien_ink.hf.tok import chunk_into_blocks
from alien_ink.hf.trainer import SaveImprovementCallback, build_lm_data_collator


def test_improvement_requests_checkpoint():
    control = SimpleNamespace(should_save=False)
    SaveImprovementCallback().on_evaluate(
        SimpleNamespace(metric_for_best_model='eval_loss', greater_is_better=False),
        SimpleNamespace(best_metric=2.0), control, {'eval_loss': 1.0},
    )
    assert control.should_save


def test_short_stream_fails():
    stream = Dataset.from_dict({'input_ids': [[1], [2]]}).to_iterable_dataset()
    with pytest.raises(ValueError, match='expected 3 blocks, got 2'):
        list(bounded_phase_blocks(stream, num_samples=3))


def test_fixed_partition_survives_epochs():
    source = Dataset.from_dict({'input_ids': [[i] * 2 for i in range(20)]})
    stream = source.to_iterable_dataset().shuffle(seed=7, buffer_size=10).skip(4)
    blocks = chunk_into_blocks(stream, block_size=2)
    expected = list(blocks)
    blocks.set_epoch(1)
    assert list(blocks) == expected


def test_left_padding_preserves_labels_and_features():
    import torch

    class Tokenizer:
        padding_side = 'left'

        def pad(self, features, **kwargs):
            return {'input_ids': torch.tensor([[1, 2], [0, 1]])}

    features = [{'input_ids': [1, 2], 'labels': [-100, 2]},
                {'input_ids': [1], 'labels': [1]}]
    result = build_lm_data_collator(Tokenizer())(features)
    assert result['labels'].tolist() == [[-100, 2], [-100, 1]]
    assert features[1]['labels'] == [1]


def test_manifest_overrides_without_wandb_identity(monkeypatch):
    from alien_ink.hf import manifest
    from alien_ink.hf.ds import HubTextSource, PretrainDataConfig
    m = manifest.Manifest(run_name='test', title='test',
                          data=PretrainDataConfig(source=HubTextSource('fake')))
    monkeypatch.setattr(manifest, 'pretrain', lambda config, **kwargs: kwargs)
    result = m.train(use_wandb=False, title='override', extra_configs={'custom': {}})
    assert result['title'] == 'override'
    assert not result['use_wandb']
    assert 'custom' in result['extra_configs']
    assert 'hardware' in result['extra_configs']


def test_curriculum_completion_eval_dispatch(monkeypatch):
    from alien_ink.hf import completion, curriculum
    from alien_ink.hf.ds import HubTextSource, PretrainDataConfig
    sentinel = object()
    monkeypatch.setattr(completion, 'build_completion_eval_dataset',
                        lambda *args, **kwargs: sentinel)
    data = PretrainDataConfig(source=HubTextSource('fake'), completion_eval_path='eval.json')
    assert curriculum._eval_blocks('eval', data, object(), verbose=False) is sentinel


def test_resumed_summary_uses_session_work():
    from alien_ink.hf.metrics import ModelSize, build_run_summary
    accelerator = SimpleNamespace(peak_tflops=1, as_dict=lambda: {})
    result = build_run_summary(
        run_name='test', run_label='test', status='completed', global_step=1000,
        start_global_step=900, max_steps=1000, tokens_per_step=10,
        train_runtime_sec=10, train_loss=1, model_size=ModelSize(100, 100, 80),
        accelerator=accelerator,
    )
    assert result.tokens_trained == 10000
    assert result.session_tokens == 1000
    assert result.steps_per_sec == 10
    assert result.tokens_per_sec == 100
    assert result.flops_per_sec == result.session_flops / 10
