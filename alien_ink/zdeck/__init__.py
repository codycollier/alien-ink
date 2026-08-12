"""Archive of training manifests for record and reuse.

Each module is an explicit :class:`~alien_ink.hf.manifest.Manifest` literal
plus a thin ``main`` that calls ``MANIFEST.train()``. Spell out every field
in place — do not rely on dataclass defaults.

**Naming (convention only, not enforced):** prefer filenames that mirror
``run_name`` with underscores instead of hyphens, e.g.
``pre_gemma_c4_5k_mist.py`` ↔ ``pre-gemma-c4-5k-mist``. Suggested grammar:
``{stage}_{family}_{corpus}_{budget}_{host}``. Slug tokens are labels for
humans/W&B/``output/train/<run_name>/``; training reads structured manifest fields
(``data``, ``model``, ``schedule``, ``hardware``, ``stage``), not the name.
Arbitrary names (e.g. ``baz_baseline.py`` / ``baz-baseline``) are fine.

Hyphenated families (``gpt-2``, ``gpt-neox``) cannot use ``python -m``; run
them as scripts instead. :func:`load_zdeck` / :func:`load_manifest` resolve
either form for tooling (chat, eval).
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alien_ink.hf.manifest import Manifest

__all__ = ["available_zdecks", "load_manifest", "load_zdeck"]

ZDECK_DIR = Path(__file__).resolve().parent


def available_zdecks() -> list[str]:
    """Short names of every zdeck program in the archive."""
    return sorted(
        path.stem for path in ZDECK_DIR.glob("*.py") if path.stem != "__init__"
    )


def _load_module_from_path(path: Path) -> tuple[ModuleType, str]:
    """Load a zdeck .py file (supports hyphenated filenames)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    cwd = Path.cwd().resolve()
    label = str(path.relative_to(cwd)) if path.is_relative_to(cwd) else str(path)
    spec = importlib.util.spec_from_file_location(f"zdeck_script_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load zdeck script {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, label


def load_zdeck(name: str) -> tuple[ModuleType, str]:
    """Load a zdeck program; return ``(module, display_label)``.

    ``name`` may be a short archive name (``pre_gpt-2_wikitext_5k_mist``), a
    ``.py`` path (resolved against the current working directory), or a module
    path (``alien_ink.zdeck.baseline_perf_gemma_mist``). Hyphenated filenames cannot
    be imported as modules, so archive files are tried first.
    """
    name = name.strip()
    if name.endswith(".py") or "/" in name:
        return _load_module_from_path(Path(name).expanduser().resolve())

    script = ZDECK_DIR / f"{name}.py"
    if script.is_file():
        return _load_module_from_path(script)

    module_name = name
    if name.startswith("zdeck."):
        module_name = f"alien_ink.{name}"
    elif not name.startswith("alien_ink.zdeck."):
        module_name = f"alien_ink.zdeck.{name}"

    try:
        return importlib.import_module(module_name), module_name
    except ModuleNotFoundError as exc:
        listing = ", ".join(available_zdecks()) or "(none)"
        raise ModuleNotFoundError(
            f"Unknown zdeck {name!r}. Available: {listing}"
        ) from exc


def load_manifest(name: str) -> tuple[Manifest, str]:
    """Load a zdeck and return its ``MANIFEST`` plus a display label."""
    # Lazy: keep ``import alien_ink.zdeck`` free of the transformers stack.
    from alien_ink.hf.manifest import Manifest

    mod, label = load_zdeck(name)
    manifest = getattr(mod, "MANIFEST", None)
    if not isinstance(manifest, Manifest):
        raise ValueError(f"{label} has no Manifest named MANIFEST")
    return manifest, label
