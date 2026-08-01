#!/usr/bin/env python
"""Interactive completion against a trained zdeck checkpoint on Mist (RTX 3070).

Each turn is a fresh completion — no chat history. The hard-coded system prompt
plus your typed line are sent as a single prompt; the model reply is printed
clearly so you can spot-check quality.

  ./bin/chat_mist.py gpt2_wikitext_5k
  ./bin/chat_mist.py alien_ink.zdeck.gemma_c4_5k --max-new-tokens 120

Requires a finished (or checkpointed) run under ``output/<run_name>/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant written in alien ink. "
    "Answer clearly and directly in plain language."
)

RULE = "-" * 70
STAR = "* " * 35


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive fresh-completion chat against a zdeck checkpoint.",
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top_p (default: 0.95)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (default: 50)",
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


def build_prompt(user_text: str) -> str:
    """Combine the hard-coded system prompt with a fresh user turn."""
    return (
        f"System: {SYSTEM_PROMPT}\n"
        f"User: {user_text.strip()}\n"
        f"Assistant:"
    )


def print_you(text: str) -> None:
    print()
    print(STAR.rstrip())
    print("  YOU")
    print(STAR.rstrip())
    for line in text.splitlines() or [""]:
        print(f"  {line}")
    print()


def print_model(text: str) -> None:
    print(RULE)
    print("  MODEL")
    print(RULE)
    body = text.strip() if text.strip() else "(empty completion)"
    for line in body.splitlines() or [""]:
        print(f"  {line}")
    print(RULE)
    print()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Resolve paths before importing the training stack.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)

    import importlib

    from alien_ink.com.device import collect_accelerator_info, device_info
    from alien_ink.com.log import banner, blank, detail, get_logger, header, step
    from alien_ink.hf.gen import generate_completion
    from alien_ink.hf.manifest import Manifest
    from alien_ink.hf.model import find_checkpoint_path, load_pretrained_model

    log = get_logger("bin.chat")

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
    # Training saves with use_cache=False for grad checkpointing.
    model.config.use_cache = True

    blank(logger=log)
    step("Ready. Each turn is a fresh completion (no history).", logger=log)
    detail("Commands: /quit  /exit  /q   or Ctrl-C / Ctrl-D", logger=log)
    detail(f"System: {SYSTEM_PROMPT}", logger=log)
    blank(logger=log)

    turn = 0
    while True:
        try:
            user_text = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            step("Bye.", logger=log)
            return

        if not user_text:
            continue
        if user_text.lower() in {"/quit", "/exit", "/q"}:
            step("Bye.", logger=log)
            return

        turn += 1
        print_you(user_text)
        step(f"Completing turn {turn}...", logger=log)

        prompt = build_prompt(user_text)
        try:
            completion = generate_completion(
                model,
                tokenizer,
                prompt,
                device,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
            )
        except Exception as exc:
            log.error("generation failed: %s", exc)
            continue

        print_model(completion)


if __name__ == "__main__":
    main()
