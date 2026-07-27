#!/usr/bin/env python
"""Pretrain GPT-2 on a small materialized WikiText-103 subset.

Uses the first ~20k train docs (plus 1k validation) instead of the full
streamed corpus. Thin entrypoint over
:class:`alien_ink.exp.recipe.Gpt2PretrainExperiment`.

Run from an installed environment (CLI)::

  python -m alien_ink.exp.gpt2_pretrain_wikitext_subset --train
  python -m alien_ink.exp.gpt2_pretrain_wikitext_subset --flight-check
  python -m alien_ink.exp.gpt2_pretrain_wikitext_subset --spot-check

Override W&B project / run name at runtime::

  python -m alien_ink.exp.gpt2_pretrain_wikitext_subset --train \\
    --wandb-project my-proj --wandb-name my-run

Or from a notebook / REPL::

  from alien_ink.exp.gpt2_pretrain_wikitext_subset import train, train_flight_check
  train_flight_check(wandb_project="my-proj", wandb_name="flight")
"""

from __future__ import annotations

from alien_ink.exp.recipe import Gpt2PretrainExperiment, run_main
from alien_ink.hf.ds import wikitext_103_subset

EXPERIMENT = Gpt2PretrainExperiment(
    run_name="gpt2-pretrain-wikitext-subset",
    title="GPT-2 from scratch on WikiText-103 (20k subset)",
    spot_check_title="GPT-2 WikiText subset — Spot Check",
    data_factory=wikitext_103_subset,
    module_description=(
        "Pretrain GPT-2 on a materialized WikiText-103 subset "
        "or spot-check a saved checkpoint."
    ),
    max_steps=-1,
    num_train_epochs=3,
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
