"""Archive of training manifests for local Mist / RTX 3070 runs.

Each program is a :class:`~alien_ink.hf.manifest.Manifest` literal plus a thin
``main`` that calls ``MANIFEST.train()``. Kept for historical record and reuse.

Filenames are stage-prefixed (``pre_…`` / ``sft_…``) to match ``Manifest.stage``
and the ``run_name`` grammar ``{stage}-{family}-{corpus}-{budget}-{host}``.
"""
