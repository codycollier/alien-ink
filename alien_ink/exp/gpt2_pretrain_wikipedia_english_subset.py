#!/usr/bin/env python
"""Pretrain GPT-2 on a small materialized English Wikipedia subset.

Uses the first ~20k train docs (plus 1k hold-out eval) instead of the full
streamed Wikipedia dump. Thin entrypoint over
:class:`alien_ink.exp.recipe.Gpt2PretrainExperiment`.

Run from an installed environment (CLI)::

  python -m alien_ink.exp.gpt2_pretrain_wikipedia_english_subset --train
  python -m alien_ink.exp.gpt2_pretrain_wikipedia_english_subset --flight-check
  python -m alien_ink.exp.gpt2_pretrain_wikipedia_english_subset --spot-check

Override W&B entity / project / run name at runtime::

  python -m alien_ink.exp.gpt2_pretrain_wikipedia_english_subset --train \\
    --wandb-entity logbook --wandb-project ink-explore \\
    --wandb-name gpt2-pretrain-wpe-subset

Or from a notebook / REPL::

  from alien_ink.exp.gpt2_pretrain_wikipedia_english_subset import (
      train,
      train_flight_check,
  )
  train_flight_check(wandb_entity="logbook", wandb_project="ink-explore")
"""

from __future__ import annotations

from alien_ink.exp.recipe import Gpt2PretrainExperiment, run_main
from alien_ink.hf.ds import wikipedia_english_subset

EXPERIMENT = Gpt2PretrainExperiment(
    run_name="gpt2-pretrain-wpe-subset",
    title="GPT-2 from scratch on English Wikipedia (20k subset)",
    spot_check_title="GPT-2 Wikipedia subset — Spot Check",
    data_factory=wikipedia_english_subset,
    module_description=(
        "Pretrain GPT-2 on a materialized English Wikipedia subset "
        "or spot-check a saved checkpoint."
    ),
    max_steps=2_000,
    warmup_steps=200,
)

base_config = EXPERIMENT.base_config
train = EXPERIMENT.train
train_flight_check = EXPERIMENT.train_flight_check
spot_check = EXPERIMENT.spot_check
build_parser = EXPERIMENT.build_parser
main = EXPERIMENT.main


if __name__ == "__main__":
    run_main(EXPERIMENT)
