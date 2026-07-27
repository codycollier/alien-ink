"""Experiment entrypoints shipped with alien-ink.

Corpus modules bind a ``data_factory`` into
:class:`~alien_ink.exp.recipe.Gpt2PretrainExperiment`. Compose ablations with
``variant`` / ``with_arch`` / ``with_data`` / ``with_trainer_knobs`` rather than
adding one module per hyperparameter combination.
"""
