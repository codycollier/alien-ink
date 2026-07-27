#!/usr/bin/env python
"""Pretrain GPT-2 on a small materialized C4 (English) subset.

Uses the first ~20k train docs (plus 1k validation) instead of the full
streamed corpus. Thin entrypoint over
:class:`alien_ink.exp.recipe.Gpt2PretrainExperiment`.

Run from an installed environment (CLI)::

  python -m alien_ink.exp.gpt2_pretrain_c4_subset --train
  python -m alien_ink.exp.gpt2_pretrain_c4_subset --flight-check
  python -m alien_ink.exp.gpt2_pretrain_c4_subset --spot-check

Compose ablations without new modules::

  from alien_ink.exp.gpt2_pretrain_c4_subset import EXPERIMENT
  EXPERIMENT.with_data(block_size=512).variant(run_name="c4-sub-b512").train()
"""

from __future__ import annotations

from alien_ink.exp.recipe import Gpt2PretrainExperiment, module_api, run_main
from alien_ink.hf.ds import c4_english_subset

EXPERIMENT = Gpt2PretrainExperiment(
    run_name="gpt2-pretrain-c4-subset",
    title="GPT-2 from scratch on C4 English (20k subset)",
    spot_check_title="GPT-2 C4 subset — Spot Check",
    data_factory=c4_english_subset,
    module_description=(
        "Pretrain GPT-2 on a materialized C4 (English) subset "
        "or spot-check a saved checkpoint."
    ),
    max_steps=-1,
    num_train_epochs=3,
    warmup_steps=200,
)

base_config, train, train_flight_check, spot_check, build_parser, main = module_api(
    EXPERIMENT
)


if __name__ == "__main__":
    run_main(EXPERIMENT)
