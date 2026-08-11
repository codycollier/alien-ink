#!/usr/bin/env python
"""Interactive completion against a trained zdeck checkpoint on Mist (RTX 3070).

Each turn is a fresh prompt — no history. Type a sentence starter or fragment;
the model continues it in plain text (suited to base LMs pretrained on raw corpus).

  ./bin/model-chat-mist.py pre_gpt-2_wikitext_5k_mist
  ./bin/model-chat-mist.py pre_gemma_c4_5k_mist --max-new-tokens 120

Requires a finished (or checkpointed) run under ``output/train/<run_name>/``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
import textwrap
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

from alien_ink.com.device import collect_accelerator_info, device_info
from alien_ink.com.log import banner, blank, detail, get_logger, header, step
from alien_ink.hf.gen import CompletionResult, generate_chat_completions
from alien_ink.hf.manifest import Manifest
from alien_ink.hf.model import find_checkpoint_path, load_pretrained_model


log = get_logger("bin.chat")

_LINE_WIDTH = 79
_RULE = "-" * _LINE_WIDTH
_WAIT_WIDTH = 14
_WAIT_INTERVAL = 0.07
_BLUE = "\033[34m"
_RESET = "\033[0m"

REPO_ROOT = Path(__file__).resolve().parent.parent
ZDECK_DIR = REPO_ROOT / "alien_ink" / "zdeck"


@contextmanager
def wait_anim() -> Iterator[None]:
    """Slide a soft marker left↔right on one line; clear when done."""
    stop = threading.Event()

    def _run() -> None:
        pos = 0
        direction = 1
        while not stop.is_set():
            track = ["·"] * _WAIT_WIDTH
            track[pos] = "●"
            sys.stdout.write(f"\r  {_BLUE}{''.join(track)}{_RESET}")
            sys.stdout.flush()
            pos += direction
            if pos <= 0 or pos >= _WAIT_WIDTH - 1:
                direction *= -1
            time.sleep(_WAIT_INTERVAL)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive text completion against a zdeck checkpoint.",
    )
    parser.add_argument(
        "zdeck",
        help="Zdeck program name, module, or script (e.g. pre_gpt-2_wikitext_5k_mist)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=120,
        help="Max new tokens per completion (default: 120)",
    )
    return parser.parse_args(argv)


def _load_module_from_path(path: Path) -> tuple[ModuleType, str]:
    """Load a zdeck .py file (supports hyphenated filenames)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    label = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
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

    # Hyphenated zdeck filenames cannot be imported as packages.
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
            "Try pre_gpt-2_wikitext_5k_mist, pre_gpt-2_wikipedia_5k_mist, "
            "pre_gemma_c4_5k_mist, pre_gemma_c4_50k_mist, "
            "pre_gpt-neox_wikitext_4ep_mist."
        ) from exc


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)

    mod, label = load_zdeck(args.zdeck)
    manifest = getattr(mod, "MANIFEST", None)
    if not isinstance(manifest, Manifest):
        raise SystemExit(f"{label} has no Manifest named MANIFEST")

    header(logger=log)
    banner(f"interactive completion — {manifest.run_name}", logger=log)

    prefer_bf16 = manifest.hardware.prefer_bf16
    prefer_fp16 = manifest.hardware.prefer_fp16
    device, _, _ = device_info(prefer_bf16=prefer_bf16, prefer_fp16=prefer_fp16)
    accel = collect_accelerator_info(prefer_bf16=prefer_bf16, prefer_fp16=prefer_fp16)
    gpu = accel.gpu_name or device
    step(f"Device: {device} ({gpu})", logger=log)
    detail(f"zdeck:    {label}", logger=log)
    detail(f"family:   {manifest.model.family}", logger=log)
    detail(f"run_name: {manifest.run_name}", logger=log)

    model_path = find_checkpoint_path(manifest.output_dir())
    model, tokenizer = load_pretrained_model(
        model_path,
        device,
        family=manifest.model.family,
    )
    model.config.use_cache = True
    gen = manifest.gen_config(max_new_tokens=args.max_new_tokens)

    blank(logger=log)
    step("Ready. Each turn is a fresh completion (no history).", logger=log)
    detail("Shows 4 candidates: greedy, then T=0.5 / 0.8 / 1.2.", logger=log)
    detail("Ctrl-C to exit.", logger=log)
    detail("Type a sentence starter, e.g. The capital of Texas is", logger=log)
    blank(logger=log)

    while True:
        print(_RULE)
        try:
            prompt = input("input› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            step("Bye.", logger=log)
            return

        if not prompt:
            continue

        try:
            with wait_anim():
                results = generate_chat_completions(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    gen,
                )
        except Exception as exc:
            log.error("generation failed: %s", exc)
            continue

        _print_completions(results)


def _print_completions(results: list[CompletionResult]) -> None:
    for index, result in enumerate(results, start=1):
        text = result.text if result.text else "∅"
        prefix = f"[{index}] "
        body = f"{text}  ({result.stats_label()})"
        print(
            textwrap.fill(
                body,
                width=_LINE_WIDTH,
                initial_indent=prefix,
                subsequent_indent=" " * len(prefix),
                break_long_words=True,
                break_on_hyphens=False,
            )
        )


if __name__ == "__main__":
    main()
