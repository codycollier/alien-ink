#!/usr/bin/env python
"""Pretrain GPT-2 on a small materialized WikiText-103 subset.

Uses the first ~20k train docs (plus 1k validation) instead of the full
streamed corpus. Thin entrypoint over
:class:`alien_ink.exp.recipe.Gpt2PretrainExperiment`.

Run from an installed environment (CLI)::

  python -m alien_ink.exp.gpt2_pretrain_wikitext_subset --train
  python -m alien_ink.exp.gpt2_pretrain_wikitext_subset --flight-check
  python -m alien_ink.exp.gpt2_pretrain_wikitext_subset --spot-check

Compose ablations without new modules::

  from alien_ink.exp.gpt2_pretrain_wikitext_subset import EXPERIMENT
  EXPERIMENT.with_arch(n_layer=6).variant(run_name="wt-sub-l6").train()
"""

from __future__ import annotations

from alien_ink.exp.recipe import Gpt2PretrainExperiment, module_api, run_main
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

config, train, train_flight_check, spot_check, build_parser, main = module_api(
    EXPERIMENT
)
base_config = config  # alias


if __name__ == "__main__":
    run_main(EXPERIMENT)
