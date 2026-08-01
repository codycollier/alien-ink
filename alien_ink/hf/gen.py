"""Text generation and prompt sampling helpers for spot-checks."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from alien_ink.com.device import device_info, torch_device
from alien_ink.com.log import banner, blank, get_logger, header, step
from alien_ink.hf.ds import HubTextSource, load_text_prompts
from alien_ink.hf.model import ModelFamily, find_checkpoint_path, load_pretrained_model

log = get_logger("hf.gen")


DEFAULT_PROMPTS = [
    "The capital of France is",
    "In the year 1969,",
    "The theory of relativity explains",
    "Python is a programming language that",
    "The human genome contains",
    "During the Renaissance, artists such as",
    "Climate change refers to",
    "The Olympic Games were first held",
    "In computer science, an algorithm is",
    "The Pacific Ocean is the",
]

_SENTENCE_END_STOP_STRINGS = (".", "!", "?")


@dataclass(frozen=True)
class GenConfig:
    """Family-aware generation defaults for base causal LMs."""

    max_new_tokens: int = 80
    do_sample: bool = False
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 0.8
    stop_strings: tuple[str, ...] = _SENTENCE_END_STOP_STRINGS
    add_special_tokens: bool = True


def _default_gen_config() -> GenConfig:
    return GenConfig(add_special_tokens=True)


def _gemma_gen_config() -> GenConfig:
    # Match mid-sequence continuation; Gemma tokenizers prepend BOS by default.
    return GenConfig(add_special_tokens=False)


_GEN_CONFIG_BY_FAMILY: dict[ModelFamily, GenConfig] = {
    "gpt2": _default_gen_config(),
    "gpt_neox": _default_gen_config(),
    "gemma": _gemma_gen_config(),
}


def gen_config_for_family(family: ModelFamily, **overrides) -> GenConfig:
    """Return generation defaults for ``family``, with optional field overrides."""
    try:
        base = _GEN_CONFIG_BY_FAMILY[family]
    except KeyError as exc:
        raise ValueError(f"unsupported family: {family!r}") from exc
    if not overrides:
        return base
    return replace(base, **overrides)


@dataclass(frozen=True)
class SpotCheckConfig:
    num_samples: int = 5
    max_new_tokens: int = 80
    seed: int = 101
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 0.8


def sample_prompts(
    pool: list[str],
    *,
    count: int,
    seed: int,
) -> list[str]:
    """Return ``count`` prompts sampled from ``pool`` (or the whole pool if smaller)."""
    if len(pool) <= count:
        return list(pool)
    rng = random.Random(seed)
    return rng.sample(pool, count)


def pick_prompts(
    *,
    count: int,
    seed: int,
    extra_prompts: list[str] | None = None,
    defaults: list[str] | None = None,
) -> list[str]:
    """Combine default prompts with extras, then sample ``count`` of them."""
    pool = list(defaults if defaults is not None else DEFAULT_PROMPTS)
    if extra_prompts:
        pool.extend(extra_prompts)
    return sample_prompts(pool, count=count, seed=seed)


@torch.inference_mode()
def generate_completion(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: str,
    gen: GenConfig,
) -> str:
    """Generate a single completion for ``prompt`` and return only the new text."""
    target = torch_device(device)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=gen.add_special_tokens,
    ).to(target)
    input_len = inputs["input_ids"].shape[1]
    gen_kwargs: dict = {
        "max_new_tokens": gen.max_new_tokens,
        "do_sample": gen.do_sample,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if gen.do_sample:
        gen_kwargs.update(
            top_k=gen.top_k,
            top_p=gen.top_p,
            temperature=gen.temperature,
        )
    if gen.stop_strings:
        gen_kwargs["stop_strings"] = list(gen.stop_strings)
        gen_kwargs["tokenizer"] = tokenizer
    output_ids = model.generate(**inputs, **gen_kwargs)
    new_ids = output_ids[0, input_len:]
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)
    return completion.strip()


def collect_spot_check_prompts(
    spot: SpotCheckConfig,
    *,
    text_source: HubTextSource | None = None,
) -> list[str]:
    """Build a prompt pool from defaults plus optional corpus excerpts."""
    extra: list[str] = []
    if text_source is not None:
        try:
            extra = load_text_prompts(
                text_source,
                count=spot.num_samples,
                seed=spot.seed,
            )
        except Exception as exc:
            log.warning("corpus prompts skipped: %s", exc)
    return pick_prompts(
        count=spot.num_samples,
        seed=spot.seed,
        extra_prompts=extra,
    )


def run_spot_check(
    *,
    output_dirs: list[Path] | tuple[Path, ...],
    family: ModelFamily = "gpt2",
    spot: SpotCheckConfig | None = None,
    text_source: HubTextSource | None = None,
    title: str = "Causal LM spot check",
    prefer_bf16: bool = True,
    prefer_fp16: bool = True,
) -> None:
    """Load the newest checkpoint under ``output_dirs`` and log sample completions."""
    spot = spot or SpotCheckConfig()
    gen = gen_config_for_family(
        family,
        max_new_tokens=spot.max_new_tokens,
        do_sample=spot.do_sample,
        top_k=spot.top_k,
        top_p=spot.top_p,
        temperature=spot.temperature,
    )

    header(logger=log)
    banner(title, logger=log)

    device, _, _ = device_info(prefer_bf16=prefer_bf16, prefer_fp16=prefer_fp16)
    step(f"Device: {device}", logger=log)

    model_path = find_checkpoint_path(*output_dirs)
    model, tokenizer = load_pretrained_model(model_path, device, family=family)

    blank(logger=log)
    step(
        f"Sampling {spot.num_samples} random prompts (seed={spot.seed})...",
        logger=log,
    )
    prompts = collect_spot_check_prompts(spot, text_source=text_source)
    blank(logger=log)

    for index, prompt in enumerate(prompts, start=1):
        completion = generate_completion(model, tokenizer, prompt, device, gen)
        log.info("--- sample %s ---", index)
        log.info("PROMPT:     %s", prompt)
        log.info("COMPLETION: %s", completion)
        blank(logger=log)

    log.info("Done.")
