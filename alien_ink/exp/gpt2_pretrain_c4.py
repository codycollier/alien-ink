#!/usr/bin/env python
"""Pretrain GPT-2 small from scratch on C4 (English) and spot-check checkpoints.

Thin experiment entrypoint over :class:`alien_ink.exp.recipe.Gpt2PretrainExperiment`.

Defaults are tuned to fit an 8 GB GPU (e.g. an RTX 3070): GPT-2 small (~124M
params), bf16 mixed precision, gradient checkpointing, and a small per-device
batch size with gradient accumulation. If you hit CUDA OOM, drop
``per_device_train_batch_size`` to 1 and/or lower ``block_size``.

Run from an installed environment (CLI)::

  python -m alien_ink.exp.gpt2_pretrain_c4 --train
  python -m alien_ink.exp.gpt2_pretrain_c4 --flight-check
  python -m alien_ink.exp.gpt2_pretrain_c4 --spot-check

Override W&B project / run name at runtime::

  python -m alien_ink.exp.gpt2_pretrain_c4 --train \\
    --wandb-project my-proj --wandb-name my-run

Or from a notebook / REPL::

  from alien_ink.exp.gpt2_pretrain_c4 import train, train_flight_check
  train_flight_check(wandb_project="my-proj", wandb_name="flight")

Artifacts and ``.env`` resolve relative to the process working directory at call
time. Set W&B project / run name via ``--wandb-project`` / ``--wandb-name``
(or kwargs). Use ``--no-wandb`` to skip Weights & Biases.
"""

from __future__ import annotations

from alien_ink.exp.recipe import Gpt2PretrainExperiment, run_main
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

base_config = EXPERIMENT.base_config
train = EXPERIMENT.train
train_flight_check = EXPERIMENT.train_flight_check
spot_check = EXPERIMENT.spot_check
build_parser = EXPERIMENT.build_parser
main = EXPERIMENT.main


if __name__ == "__main__":
    run_main(EXPERIMENT)
