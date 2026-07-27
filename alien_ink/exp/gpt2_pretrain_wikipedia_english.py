#!/usr/bin/env python
"""Pretrain GPT-2 small from scratch on English Wikipedia and spot-check checkpoints.

Thin experiment entrypoint over :class:`alien_ink.exp.recipe.Gpt2PretrainExperiment`.

Batch / accum defaults come from :mod:`alien_ink.hf.hardware` (local RTX ~8 GB,
Colab G4/L4, or Colab TPU v6e-1). Run names gain a ``-gpu`` / ``-tpu`` suffix.
If you hit CUDA OOM, drop ``per_device_train_batch_size`` and/or ``block_size``.

Run from an installed environment (CLI)::

  python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --train
  python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --flight-check
  python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --spot-check

Override W&B entity / project / run name at runtime::

  python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --train \\
    --wandb-entity logbook --wandb-project ink-explore --wandb-name gpt2-pretrain-wpe

Or from a notebook / REPL::

  from alien_ink.exp.gpt2_pretrain_wikipedia_english import train, train_flight_check
  train_flight_check(wandb_entity="logbook", wandb_project="ink-explore")

Artifacts and ``.env`` resolve relative to the process working directory at call
time. Set W&B entity / project / run name via ``--wandb-entity`` /
``--wandb-project`` / ``--wandb-name`` (or kwargs). Use ``--no-wandb`` to skip
Weights & Biases.
"""

from __future__ import annotations

from alien_ink.exp.recipe import Gpt2PretrainExperiment, run_main
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

base_config = EXPERIMENT.base_config
train = EXPERIMENT.train
train_flight_check = EXPERIMENT.train_flight_check
spot_check = EXPERIMENT.spot_check
build_parser = EXPERIMENT.build_parser
main = EXPERIMENT.main


if __name__ == "__main__":
    run_main(EXPERIMENT)
