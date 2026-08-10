#!/usr/bin/env python
"""Post-training completion eval against a zdeck checkpoint on Mist (RTX 3070).

Points at a zdeck, resolves the trained model under ``output/train/<run_name>/``,
loads an external eval JSON (prompt + expected completion), runs greedy
completions, and writes results under ``output/train/<run_name>/evals/``.

  ./bin/model-eval-mist.py sft_smollm2-135m_geo_mist \\
    --evals /tmp/population-exact/geo-us-states.json

  ./bin/model-eval-mist.py pre_gpt-neox_wikitext_3ep_mist \\
    --evals /path/to/eval.json --max-new-tokens 64

Eval file contents are never copied into the run output — only scores and
predicted text are stored.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType

from alien_ink.com.device import device_info
from alien_ink.hf.eval import run_completion_eval
from alien_ink.hf.manifest import Manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
ZDECK_DIR = REPO_ROOT / "alien_ink" / "zdeck"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a post-training completion eval against a zdeck checkpoint."
        ),
    )
    parser.add_argument(
        "zdeck",
        help="Zdeck program name, module, or script (e.g. sft_smollm2-135m_geo_mist)",
    )
    parser.add_argument(
        "--evals",
        required=True,
        type=Path,
        help="Path to eval JSON (list of {slug, prompt, completion})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Max new tokens per completion (default: longest expected "
            "completion length + cushion)"
        ),
    )
    return parser.parse_args(argv)


def _load_module_from_path(path: Path) -> tuple[ModuleType, str]:
    """Load a zdeck .py file (supports hyphenated filenames)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    label = (
        str(path.relative_to(REPO_ROOT))
        if path.is_relative_to(REPO_ROOT)
        else str(path)
    )
    spec = importlib.util.spec_from_file_location(f"zdeck_script_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load zdeck script {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, label


def load_zdeck(name: str) -> tuple[ModuleType, str]:
    """Load a zdeck module or hyphenated script by short name / path / module path."""
    name = name.strip()
    if name.endswith(".py") or "/" in name:
        return _load_module_from_path((REPO_ROOT / name).resolve())

    script = ZDECK_DIR / f"{name}.py"
    if "-" in name or script.is_file():
        try:
            return _load_module_from_path(script)
        except FileNotFoundError:
            pass

    module_name = name
    if name.startswith("zdeck."):
        module_name = f"alien_ink.{name}"
    elif not name.startswith("alien_ink.zdeck."):
        module_name = f"alien_ink.zdeck.{name}"

    try:
        return importlib.import_module(module_name), module_name
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Unknown zdeck {name!r}. "
            "Try sft_smollm2-135m_geo_mist, sft_pythia-160m_geo_mist, "
            "pre_gpt-2_wikitext_5k_mist, pre_gpt-neox_wikitext_3ep_mist."
        ) from exc


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)

    evals_path = args.evals.expanduser()
    if not evals_path.is_file():
        raise SystemExit(f"evals file not found: {evals_path}")

    mod, label = load_zdeck(args.zdeck)
    manifest = getattr(mod, "MANIFEST", None)
    if not isinstance(manifest, Manifest):
        raise SystemExit(f"{label} has no Manifest named MANIFEST")

    prefer_bf16 = manifest.hardware.prefer_bf16
    prefer_fp16 = manifest.hardware.prefer_fp16
    device, _, _ = device_info(prefer_bf16=prefer_bf16, prefer_fp16=prefer_fp16)

    try:
        run_completion_eval(
            manifest=manifest,
            evals_path=evals_path,
            device=device,
            zdeck_label=label,
            max_new_tokens=args.max_new_tokens,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
