"""GPT-2 model construction and checkpoint loading via Hugging Face Transformers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


@dataclass(frozen=True)
class Gpt2ArchConfig:
    """Architecture + tokenizer identity for a from-scratch GPT-2."""

    tokenizer_name: str = "gpt2"
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    use_cache: bool = False


def load_gpt2_tokenizer(name_or_path: str | Path) -> GPT2Tokenizer:
    """Load a GPT-2 tokenizer and ensure a pad token is set (eos)."""
    tokenizer = GPT2Tokenizer.from_pretrained(str(name_or_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_gpt2_from_scratch(
    tokenizer: GPT2Tokenizer,
    arch: Gpt2ArchConfig | None = None,
    *,
    verbose: bool = True,
) -> GPT2LMHeadModel:
    """Initialize a GPT-2 LM head model with random weights from config."""
    arch = arch or Gpt2ArchConfig()
    if verbose:
        print(">> Initializing GPT-2 from config with random weights...")

    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=arch.n_positions,
        n_embd=arch.n_embd,
        n_layer=arch.n_layer,
        n_head=arch.n_head,
    )
    model = GPT2LMHeadModel(model_config)
    # Incompatible with gradient checkpointing and unused during training; also
    # frees the KV cache memory during eval forward passes.
    model.config.use_cache = arch.use_cache

    if verbose:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   parameters: {param_count:,}")
    return model


def build_model_and_tokenizer(
    arch: Gpt2ArchConfig | None = None,
    *,
    verbose: bool = True,
) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """Load tokenizer and initialize a GPT-2 model from scratch."""
    arch = arch or Gpt2ArchConfig()
    if verbose:
        print(f">> Loading GPT-2 tokenizer ({arch.tokenizer_name})...")
    tokenizer = load_gpt2_tokenizer(arch.tokenizer_name)
    model = build_gpt2_from_scratch(tokenizer, arch, verbose=verbose)
    return model, tokenizer


def resolve_checkpoint_path(output_dir: Path) -> Path:
    """Prefer a final save under ``output_dir``, else the latest ``checkpoint-*``."""
    if (output_dir / "config.json").exists():
        return output_dir

    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError(f"No trained model found under {output_dir}.")


def find_checkpoint_path(*output_dirs: Path) -> Path:
    """Return the first resolvable checkpoint among candidate output dirs."""
    errors: list[str] = []
    for output_dir in output_dirs:
        try:
            return resolve_checkpoint_path(output_dir)
        except FileNotFoundError as exc:
            errors.append(str(exc))
    detail = " ".join(errors) if errors else "no candidates given"
    raise FileNotFoundError(
        f"No trained model found. Run training first. ({detail})"
    )


def load_pretrained_model(
    model_path: Path,
    device: str,
    *,
    verbose: bool = True,
) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """Load a saved GPT-2 checkpoint onto ``device`` in eval mode."""
    if verbose:
        print(f">> Loading model from {model_path}...")
    tokenizer = load_gpt2_tokenizer(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer
