#!/usr/bin/env python
"""Interactive completion against a trained zdeck checkpoint on Mist (RTX 3070).

Each turn is a fresh prompt — no history. Type a sentence starter or fragment;
the model continues it in plain text (suited to base LMs pretrained on raw corpus).

  ./bin/model-chat-mist.py gpt2_wikitext_5k
  ./bin/model-chat-mist.py alien_ink.zdeck.gemma_c4_5k --max-new-tokens 120

Requires a finished (or checkpointed) run under ``output/<run_name>/``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alien_ink.com.device import collect_accelerator_info, device_info
from alien_ink.com.log import banner, blank, detail, get_logger, header, step
from alien_ink.hf.gen import generate_completion
from alien_ink.hf.manifest import Manifest
from alien_ink.hf.model import find_checkpoint_path, load_pretrained_model

COMPLETION_STOP_STRINGS = ("\n\n",)

log = get_logger("bin.chat")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive text completion against a zdeck checkpoint.",
    )
    parser.add_argument(
        "zdeck",
        help="Zdeck program name or module (e.g. gpt2_wikitext_5k)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=120,
        help="Max new tokens per completion (default: 120)",
    )
    return parser.parse_args(argv)


def resolve_zdeck_module(name: str) -> str:
    """Accept ``gpt2_wikitext_5k`` or a full ``alien_ink.zdeck.*`` path."""
    name = name.strip()
    if name.startswith("alien_ink.zdeck."):
        return name
    if name.startswith("zdeck."):
        return f"alien_ink.{name}"
    return f"alien_ink.zdeck.{name}"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)

    module_name = resolve_zdeck_module(args.zdeck)
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Unknown zdeck module {module_name!r}. "
            "Try gpt2_wikitext_5k, gpt2_wikipedia_5k, gemma_c4_5k, gemma_c4_50k."
        ) from exc

    manifest = getattr(mod, "MANIFEST", None)
    if not isinstance(manifest, Manifest):
        raise SystemExit(f"{module_name} has no Manifest named MANIFEST")

    header(logger=log)
    banner(f"interactive completion — {manifest.run_name}", logger=log)

    prefer_bf16 = manifest.hardware.prefer_bf16
    prefer_fp16 = manifest.hardware.prefer_fp16
    device, _, _ = device_info(prefer_bf16=prefer_bf16, prefer_fp16=prefer_fp16)
    accel = collect_accelerator_info(prefer_bf16=prefer_bf16, prefer_fp16=prefer_fp16)
    gpu = accel.gpu_name or device
    step(f"Device: {device} ({gpu})", logger=log)
    detail(f"zdeck:    {module_name}", logger=log)
    detail(f"family:   {manifest.model.family}", logger=log)
    detail(f"run_name: {manifest.run_name}", logger=log)

    model_path = find_checkpoint_path(manifest.output_dir())
    model, tokenizer = load_pretrained_model(
        model_path,
        device,
        family=manifest.model.family,
    )
    model.config.use_cache = True

    blank(logger=log)
    step("Ready. Each turn is a fresh completion (no history).", logger=log)
    detail("Commands: /quit  /exit  /q   or Ctrl-C / Ctrl-D", logger=log)
    detail("Type a sentence starter, e.g. The capital of Texas is", logger=log)
    blank(logger=log)

    while True:
        print("--------------------------------------------------------------")
        try:
            prompt = input("input› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            step("Bye.", logger=log)
            return

        if not prompt:
            continue
        if prompt.lower() in {"/quit", "/exit", "/q"}:
            step("Bye.", logger=log)
            return

        step("Completing with model...", logger=log)

        try:
            completion = generate_completion(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=args.max_new_tokens,
                stop_strings=COMPLETION_STOP_STRINGS,
            )
        except Exception as exc:
            log.error("generation failed: %s", exc)
            continue

        if completion:
            print(f"model> {completion}")


if __name__ == "__main__":
    main()
