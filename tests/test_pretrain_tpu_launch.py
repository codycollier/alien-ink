"""Tests for TPU auto-launch wrapping in pretrain_gpt2."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("transformers")
pytest.importorskip("datasets")

from alien_ink.hf.ds import HubTextSource, PretrainDataConfig  # noqa: E402
from alien_ink.hf.pretrain import Gpt2PretrainConfig, pretrain_gpt2  # noqa: E402
from alien_ink.hf.trainer import CausalLmTrainerConfig  # noqa: E402


def test_pretrain_gpt2_auto_launches_on_tpu_notebook():
    cfg = Gpt2PretrainConfig(
        data=PretrainDataConfig(
            source=HubTextSource(dataset="Salesforce/wikitext"),
        ),
        trainer=CausalLmTrainerConfig(output_dir=Path("output/tpu-test")),
    )
    launched: list[object] = []

    def fake_launch(fn, args=(), *, num_processes=None, mixed_precision="bf16"):
        launched.append(
            {
                "fn": fn,
                "num_processes": num_processes,
                "mixed_precision": mixed_precision,
            }
        )

    with (
        patch("alien_ink.hf.pretrain.should_auto_launch_tpu", return_value=True),
        patch("alien_ink.hf.pretrain.launch_tpu", side_effect=fake_launch),
    ):
        result = pretrain_gpt2(cfg, use_wandb=False)

    assert result == (None, None)
    assert len(launched) == 1
    assert launched[0]["mixed_precision"] == "bf16"
