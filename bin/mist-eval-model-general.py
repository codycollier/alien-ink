#!/usr/bin/env python
"""Post-training completion eval against a zdeck checkpoint on Mist (RTX 3070).

Points at a zdeck, resolves the trained model under ``output/train/<run_name>/``,
loads an external eval JSON (prompt + expected completion), runs greedy
completions, and writes results under ``output/train/<run_name>/evals/``.

  ./bin/mist-eval-model-general.py sft_smollm2-135m_geo_mist \\
    --evals /tmp/population-exact/geo-us-states.json

  ./bin/mist-eval-model-general.py pre_gpt-neox_wikitext_3ep_mist \\
    --evals /path/to/eval.json --max-new-tokens 64

Eval file contents are never copied into the run output — only scores and
predicted text are stored.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from alien_ink.com.device import device_info
from alien_ink.hf.eval import run_completion_eval
from alien_ink.zdeck import load_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)

    evals_path = args.evals.expanduser()
    if not evals_path.is_file():
        raise SystemExit(f"evals file not found: {evals_path}")

    try:
        manifest, label = load_manifest(args.zdeck)
    except (ModuleNotFoundError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

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
