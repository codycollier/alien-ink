"""Archive of training manifests for record and reuse.

Each module is an explicit :class:`~alien_ink.hf.manifest.Manifest` literal
plus a thin ``main`` that calls ``MANIFEST.train()``. Spell out every field
in place — do not rely on dataclass defaults.

**Naming (convention only, not enforced):** prefer filenames that mirror
``run_name`` with underscores instead of hyphens, e.g.
``pre_gemma_c4_5k_mist.py`` ↔ ``pre-gemma-c4-5k-mist``. Suggested grammar:
``{stage}_{family}_{corpus}_{budget}_{host}``. Slug tokens are labels for
humans/W&B/``output/<run_name>/``; training reads structured manifest fields
(``data``, ``model``, ``schedule``, ``hardware``, ``stage``), not the name.
Arbitrary names (e.g. ``baz_baseline.py`` / ``baz-baseline``) are fine.

Hyphenated families (``gpt-2``, ``gpt-neox``) cannot use ``python -m``; run
them as scripts instead.
"""
