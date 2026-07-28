#!/usr/bin/env python
"""Pretrain GPT-2 small from scratch on C4 (English) and spot-check checkpoints.

Thin experiment entrypoint over :class:`alien_ink.exp.recipe.Gpt2PretrainExperiment`.

Batch / accum defaults come from :mod:`alien_ink.hf.hardware` (Mist RTX 3070,
Colab G4 Blackwell 96 GB, L4 mid-tier, or TPU v6e-1). Run names gain a
``-gpu`` / ``-tpu`` suffix. If you hit CUDA OOM, drop
``per_device_train_batch_size`` and/or ``block_size``.

Run from an installed environment (CLI)::

  python -m alien_ink.exp.gpt2_pretrain_c4 --train
  python -m alien_ink.exp.gpt2_pretrain_c4 --flight-check
  python -m alien_ink.exp.gpt2_pretrain_c4 --spot-check

Compose ablations without new modules::

  from alien_ink.exp.gpt2_pretrain_c4 import EXPERIMENT
  EXPERIMENT.with_arch(n_layer=6).variant(run_name="c4-l6").train()
"""

from __future__ import annotations

from alien_ink.exp.recipe import Gpt2PretrainExperiment, module_api, run_main
from alien_ink.hf.ds import c4_english

EXPERIMENT = Gpt2PretrainExperiment(
    run_name="gpt2-pretrain-c4",
    title="GPT-2 from scratch on C4 (English)",
    spot_check_title="GPT-2 C4 — Spot Check",
    data_factory=c4_english,
    module_description=(
        "Pretrain GPT-2 on C4 (English) or spot-check a saved checkpoint."
    ),
)

config, train, train_flight_check, spot_check, build_parser, main = module_api(
    EXPERIMENT
)
base_config = config  # alias


if __name__ == "__main__":
    run_main(EXPERIMENT)
