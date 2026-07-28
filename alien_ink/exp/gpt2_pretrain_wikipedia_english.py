#!/usr/bin/env python
"""Pretrain GPT-2 small from scratch on English Wikipedia and spot-check checkpoints.

Thin experiment entrypoint over :class:`alien_ink.exp.recipe.Gpt2PretrainExperiment`.

Batch / accum defaults come from :mod:`alien_ink.hf.hardware` (Mist RTX 3070,
Colab G4 Blackwell 96 GB, L4 mid-tier, or TPU v6e-1). Run names gain a
``-gpu`` / ``-tpu`` suffix. If you hit CUDA OOM, drop
``per_device_train_batch_size`` and/or ``block_size``.

Run from an installed environment (CLI)::

  python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --train
  python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --flight-check
  python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --spot-check

Compose ablations without new modules::

  from alien_ink.exp.gpt2_pretrain_wikipedia_english import EXPERIMENT
  EXPERIMENT.with_data(block_size=512).variant(run_name="wpe-b512").train()
"""

from __future__ import annotations

from alien_ink.exp.recipe import Gpt2PretrainExperiment, module_api, run_main
from alien_ink.hf.ds import wikipedia_english

EXPERIMENT = Gpt2PretrainExperiment(
    run_name="gpt2-pretrain-wpe",
    title="GPT-2 from scratch on English Wikipedia",
    spot_check_title="GPT-2 Wikipedia — Spot Check",
    data_factory=wikipedia_english,
    module_description=(
        "Pretrain GPT-2 on English Wikipedia or spot-check a saved checkpoint."
    ),
)

config, train, train_flight_check, spot_check, build_parser, main = module_api(
    EXPERIMENT
)
base_config = config  # alias


if __name__ == "__main__":
    run_main(EXPERIMENT)
