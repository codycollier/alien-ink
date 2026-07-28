"""From-scratch causal LM construction and checkpoint loading.

Supported families (extend via :data:`MODEL_BUILDERS` / :data:`TOKENIZER_LOADERS`):

- ``gpt2`` — GPT-2 (default small ~124M)
- ``gpt_neox`` — GPT-NeoX (default small, Pythia-style dims)
- ``gemma`` — Gemma (default tiny; tokenizer may require HF access)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import (
    AutoTokenizer,
    GemmaConfig,
    GemmaForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from alien_ink.device import move_module_to_device
from alien_ink.log import detail, get_logger, step

log = get_logger("hf.model")

# Families shipped today; register more with :func:`register_model_family`.
SUPPORTED_FAMILIES = ("gpt2", "gpt_neox", "gemma")


@dataclass(frozen=True)
class ModelArchConfig:
    """Architecture + tokenizer identity for a from-scratch causal LM.

    Shared field names map onto each family's HF config. Defaults target a
    Mist-friendly GPT-2 small (~124M) that fits an RTX 3070 at
    ``block_size=1024`` with batch 2 / accum 16.
    """

    family: str = "gpt2"
    tokenizer_name: str | None = None
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    intermediate_size: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    use_cache: bool = False

    def resolved_tokenizer_name(self) -> str:
        if self.tokenizer_name:
            return self.tokenizer_name
        return _DEFAULT_TOKENIZERS[self.family]

    def validate(self) -> None:
        if self.family not in MODEL_BUILDERS:
            known = ", ".join(sorted(MODEL_BUILDERS))
            raise ValueError(
                f"unknown model family {self.family!r}; known: {known}"
            )
        if self.n_positions < 1:
            raise ValueError(f"n_positions must be >= 1, got {self.n_positions}")
        if self.n_embd < 1:
            raise ValueError(f"n_embd must be >= 1, got {self.n_embd}")
        if self.n_layer < 1:
            raise ValueError(f"n_layer must be >= 1, got {self.n_layer}")
        if self.n_head < 1:
            raise ValueError(f"n_head must be >= 1, got {self.n_head}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.intermediate_size is not None and self.intermediate_size < 1:
            raise ValueError(
                f"intermediate_size must be >= 1 when set, "
                f"got {self.intermediate_size}"
            )
        if self.num_key_value_heads is not None and self.num_key_value_heads < 1:
            raise ValueError(
                f"num_key_value_heads must be >= 1 when set, "
                f"got {self.num_key_value_heads}"
            )
        if self.head_dim is not None and self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1 when set, got {self.head_dim}")


# Backward-compatible alias used by older call sites / tests.
Gpt2ArchConfig = ModelArchConfig

_DEFAULT_TOKENIZERS: dict[str, str] = {
    "gpt2": "gpt2",
    "gpt_neox": "EleutherAI/pythia-70m",
    "gemma": "google/gemma-2b",
}


def _ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def load_tokenizer(name_or_path: str | Path, *, family: str = "gpt2") -> PreTrainedTokenizerBase:
    """Load a tokenizer for ``family`` and ensure a pad token is set."""
    loader = TOKENIZER_LOADERS.get(family)
    if loader is not None:
        return loader(str(name_or_path))
    tokenizer = AutoTokenizer.from_pretrained(str(name_or_path))
    return _ensure_pad_token(tokenizer)


def load_gpt2_tokenizer(name_or_path: str | Path) -> GPT2Tokenizer:
    """Load a GPT-2 tokenizer and ensure a pad token is set (eos)."""
    tokenizer = GPT2Tokenizer.from_pretrained(str(name_or_path))
    return _ensure_pad_token(tokenizer)  # type: ignore[return-value]


def _build_gpt2(tokenizer: PreTrainedTokenizerBase, arch: ModelArchConfig) -> GPT2LMHeadModel:
    model_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=arch.n_positions,
        n_embd=arch.n_embd,
        n_layer=arch.n_layer,
        n_head=arch.n_head,
    )
    model = GPT2LMHeadModel(model_config)
    model.config.use_cache = arch.use_cache
    return model


def _build_gpt_neox(
    tokenizer: PreTrainedTokenizerBase,
    arch: ModelArchConfig,
) -> GPTNeoXForCausalLM:
    intermediate = arch.intermediate_size or (4 * arch.n_embd)
    model_config = GPTNeoXConfig(
        vocab_size=len(tokenizer),
        hidden_size=arch.n_embd,
        num_hidden_layers=arch.n_layer,
        num_attention_heads=arch.n_head,
        intermediate_size=intermediate,
        max_position_embeddings=arch.n_positions,
        use_cache=arch.use_cache,
    )
    return GPTNeoXForCausalLM(model_config)


def _build_gemma(
    tokenizer: PreTrainedTokenizerBase,
    arch: ModelArchConfig,
) -> GemmaForCausalLM:
    head_dim = arch.head_dim or (arch.n_embd // arch.n_head)
    kv_heads = arch.num_key_value_heads or arch.n_head
    # Gemma default MLP ratio is ~8× with gated GeGLU; keep a compact default.
    intermediate = arch.intermediate_size or (arch.n_embd * 4)
    model_config = GemmaConfig(
        vocab_size=len(tokenizer),
        hidden_size=arch.n_embd,
        intermediate_size=intermediate,
        num_hidden_layers=arch.n_layer,
        num_attention_heads=arch.n_head,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        max_position_embeddings=arch.n_positions,
        use_cache=arch.use_cache,
    )
    return GemmaForCausalLM(model_config)


ModelBuilder = Callable[[PreTrainedTokenizerBase, ModelArchConfig], PreTrainedModel]
TokenizerLoader = Callable[[str], PreTrainedTokenizerBase]

MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "gpt2": _build_gpt2,
    "gpt_neox": _build_gpt_neox,
    "gemma": _build_gemma,
}

TOKENIZER_LOADERS: dict[str, TokenizerLoader] = {
    "gpt2": load_gpt2_tokenizer,
    "gpt_neox": lambda name: _ensure_pad_token(AutoTokenizer.from_pretrained(name)),
    "gemma": lambda name: _ensure_pad_token(AutoTokenizer.from_pretrained(name)),
}


def register_model_family(
    name: str,
    *,
    builder: ModelBuilder,
    tokenizer_loader: TokenizerLoader | None = None,
    default_tokenizer: str | None = None,
) -> None:
    """Register an additional from-scratch model family."""
    MODEL_BUILDERS[name] = builder
    if tokenizer_loader is not None:
        TOKENIZER_LOADERS[name] = tokenizer_loader
    if default_tokenizer is not None:
        _DEFAULT_TOKENIZERS[name] = default_tokenizer


def build_model_from_scratch(
    tokenizer: PreTrainedTokenizerBase,
    arch: ModelArchConfig | None = None,
    *,
    verbose: bool = True,
) -> PreTrainedModel:
    """Initialize a causal LM with random weights from ``arch``."""
    arch = arch or ModelArchConfig()
    arch.validate()
    if verbose:
        step(
            f"Initializing {arch.family} from config with random weights...",
            logger=log,
        )
    model = MODEL_BUILDERS[arch.family](tokenizer, arch)
    if verbose:
        param_count = sum(p.numel() for p in model.parameters())
        detail(f"parameters: {param_count:,}", logger=log)
    return model


def build_gpt2_from_scratch(
    tokenizer: PreTrainedTokenizerBase,
    arch: ModelArchConfig | None = None,
    *,
    verbose: bool = True,
) -> PreTrainedModel:
    """Initialize a GPT-2 LM head model with random weights from config."""
    arch = arch or ModelArchConfig(family="gpt2")
    if arch.family != "gpt2":
        raise ValueError(f"build_gpt2_from_scratch requires family='gpt2', got {arch.family!r}")
    return build_model_from_scratch(tokenizer, arch, verbose=verbose)


def build_model_and_tokenizer(
    arch: ModelArchConfig | None = None,
    *,
    verbose: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load tokenizer and initialize a model from scratch."""
    arch = arch or ModelArchConfig()
    arch.validate()
    tok_name = arch.resolved_tokenizer_name()
    if verbose:
        step(f"Loading {arch.family} tokenizer ({tok_name})...", logger=log)
    tokenizer = load_tokenizer(tok_name, family=arch.family)
    model = build_model_from_scratch(tokenizer, arch, verbose=verbose)
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
    detail_msg = " ".join(errors) if errors else "no candidates given"
    raise FileNotFoundError(
        f"No trained model found. Run training first. ({detail_msg})"
    )


def load_pretrained_model(
    model_path: Path,
    device: str,
    *,
    family: str | None = None,
    verbose: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a saved causal-LM checkpoint onto ``device`` in eval mode."""
    from transformers import AutoConfig, AutoModelForCausalLM

    if verbose:
        step(f"Loading model from {model_path}...", logger=log)

    resolved_family = family
    if resolved_family is None:
        try:
            cfg = AutoConfig.from_pretrained(model_path)
            model_type = getattr(cfg, "model_type", None)
            if model_type in MODEL_BUILDERS:
                resolved_family = model_type
            elif model_type == "gpt_neox":
                resolved_family = "gpt_neox"
        except Exception:
            resolved_family = "gpt2"

    tokenizer = load_tokenizer(model_path, family=resolved_family or "gpt2")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    move_module_to_device(model, device)
    model.eval()
    return model, tokenizer
