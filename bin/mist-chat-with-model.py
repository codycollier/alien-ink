#!/usr/bin/env python
"""Interactive completion against a trained zdeck checkpoint on Mist (RTX 3070).

Each turn is a fresh prompt — no history. Type a sentence starter or fragment;
the model continues it in plain text (suited to base LMs pretrained on raw corpus).

  ./bin/mist-chat-with-model.py pre_gpt-2_wikitext_5k_mist
  ./bin/mist-chat-with-model.py sft_pythia-70m_geo_100ep_mist --max-new-tokens 120

Requires a finished (or checkpointed) run under ``output/train/<run_name>/``.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from alien_ink.com.device import collect_accelerator_info, device_info
from alien_ink.com.log import banner, blank, detail, get_logger, header, step
from alien_ink.hf.eval import load_model_for_eval
from alien_ink.hf.gen import CompletionResult, generate_chat_completions
from alien_ink.hf.model import PretrainedLmConfig
from alien_ink.zdeck import load_manifest


log = get_logger("bin.chat")

_LINE_WIDTH = 79
_RULE = "-" * _LINE_WIDTH
_WAIT_WIDTH = 14
_WAIT_INTERVAL = 0.07
_BLUE = "\033[34m"
_RESET = "\033[0m"

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)

    try:
        manifest, label = load_manifest(args.zdeck)
    except (ModuleNotFoundError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    header(logger=log)
    banner(f"interactive completion — {manifest.run_name}", logger=log)

    prefer_bf16 = manifest.hardware.prefer_bf16
    prefer_fp16 = manifest.hardware.prefer_fp16
    device, _, _ = device_info(prefer_bf16=prefer_bf16, prefer_fp16=prefer_fp16)
    accel = collect_accelerator_info(prefer_bf16=prefer_bf16, prefer_fp16=prefer_fp16)
    gpu = accel.gpu_name or device
    step(f"Device: {device} ({gpu})", logger=log)
    detail(f"zdeck:    {label}", logger=log)
    if isinstance(manifest.model, PretrainedLmConfig):
        detail(f"base:     {manifest.model.model_name}", logger=log)
    else:
        detail(f"family:   {manifest.model.family}", logger=log)
    detail(f"run_name: {manifest.run_name}", logger=log)

    try:
        model, tokenizer, _ = load_model_for_eval(manifest, device)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
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
