"""Causal-LM construction and checkpoint loading via Hugging Face Transformers.

Supports from-scratch GPT-2, GPT-NeoX, and Gemma architectures sized for a
local ~8 GB GPU. Add a new ``family`` branch in ``build_model_from_scratch`` to
extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from transformers import (
    AutoTokenizer,
    GemmaConfig,
    GemmaForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from alien_ink.com.device import move_module_to_device
from alien_ink.com.log import detail, get_logger, step

log = get_logger("hf.model")

ModelFamily = Literal["gpt-2", "gpt-neox", "gemma"]
AttentionImplementation = Literal["eager", "sdpa"]

# Mist / RTX 3070 (~8 GB) friendly defaults for from-scratch pretraining.
_MIST_GPT2 = dict(n_positions=1024, n_embd=768, n_layer=12, n_head=12)
_MIST_GPT_NEOX = dict(
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    hidden_act="gelu",
    hidden_dropout=0.0,
    attention_dropout=0.0,
    rope_theta=10_000.0,
    rotary_pct=0.25,
    tie_word_embeddings=False,
)
_MIST_GEMMA = dict(
    n_positions=1024,
    n_embd=512,
    n_layer=8,
    n_head=8,
    head_dim=64,
    intermediate_size=2048,
    hidden_act="gelu_pytorch_tanh",
    hidden_dropout=0.0,
    attention_dropout=0.0,
    norm_epsilon=1e-6,
    rope_theta=10_000.0,
    tie_word_embeddings=True,
    num_key_value_heads=8,
)


@dataclass(frozen=True)
class CausalLmArchConfig:
    """Architecture + tokenizer identity for from-scratch causal LM pretraining.

    Field names follow the GPT-2 convention (``n_embd`` / ``n_layer`` / …) and
    are mapped onto each family's HF config. Defaults are Mist-sized.
    """

    family: ModelFamily = "gpt-2"
    tokenizer_name: str = "gpt2"
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    head_dim: int | None = None
    intermediate_size: int | None = None
    hidden_act: str = "gelu_new"
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    rope_theta: float | None = None
    rotary_pct: float | None = None
    tie_word_embeddings: bool = True
    num_key_value_heads: int | None = None
    # SDPA dispatches to PyTorch's fused Flash / memory-efficient kernels when
    # the GPU, dtype, and shape permit it (including the RTX 3070).  ``eager``
    # remains available as a compatibility escape hatch.
    attention_implementation: AttentionImplementation = "sdpa"
    use_cache: bool = False

    def validate(self) -> None:
        if self.family not in {"gpt-2", "gpt-neox", "gemma"}:
            raise ValueError(
                f"family must be one of gpt-2, gpt-neox, gemma; got {self.family!r}"
            )
        if self.n_positions < 1:
            raise ValueError(f"n_positions must be >= 1, got {self.n_positions}")
        if self.n_embd < 1:
            raise ValueError(f"n_embd must be >= 1, got {self.n_embd}")
        if self.n_layer < 1:
            raise ValueError(f"n_layer must be >= 1, got {self.n_layer}")
        if self.n_head < 1:
            raise ValueError(f"n_head must be >= 1, got {self.n_head}")
        if self.family != "gemma" and self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.head_dim is not None and self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1 when set, got {self.head_dim}")
        if self.head_dim is not None and self.n_embd != self.n_head * self.head_dim:
            raise ValueError(
                f"n_embd ({self.n_embd}) must equal n_head * head_dim "
                f"({self.n_head * self.head_dim})"
            )
        if self.intermediate_size is not None and self.intermediate_size < 1:
            raise ValueError(
                "intermediate_size must be >= 1 when set, "
                f"got {self.intermediate_size}"
            )
        if not self.hidden_act.strip():
            raise ValueError("hidden_act must be a non-empty string")
        for name, value in (
            ("hidden_dropout", self.hidden_dropout),
            ("attention_dropout", self.attention_dropout),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")
        if self.norm_epsilon <= 0:
            raise ValueError(f"norm_epsilon must be > 0, got {self.norm_epsilon}")
        if self.initializer_range <= 0:
            raise ValueError(
                f"initializer_range must be > 0, got {self.initializer_range}"
            )
        if self.rope_theta is not None and self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be > 0 when set, got {self.rope_theta}")
        if self.rotary_pct is not None and not 0.0 < self.rotary_pct <= 1.0:
            raise ValueError(f"rotary_pct must be in (0, 1], got {self.rotary_pct}")
        if self.family != "gpt-neox" and self.rotary_pct is not None:
            raise ValueError("rotary_pct is only supported for gpt-neox")
        if self.num_key_value_heads is not None:
            if self.family != "gemma":
                raise ValueError("num_key_value_heads is only supported for gemma")
            if self.num_key_value_heads < 1:
                raise ValueError(
                    "num_key_value_heads must be >= 1 when set, "
                    f"got {self.num_key_value_heads}"
                )
            if self.n_head % self.num_key_value_heads != 0:
                raise ValueError(
                    f"n_head ({self.n_head}) must be divisible by "
                    f"num_key_value_heads ({self.num_key_value_heads})"
                )
        if self.family == "gemma" and self.hidden_dropout != 0.0:
            raise ValueError("gemma does not support hidden_dropout; use 0.0")
        if self.attention_implementation not in {"eager", "sdpa"}:
            raise ValueError(
                "attention_implementation must be 'eager' or 'sdpa', got "
                f"{self.attention_implementation!r}"
            )


def gpt2_arch(**overrides) -> CausalLmArchConfig:
    """Mist-sized GPT-2 (124M-class) from-scratch config."""
    return CausalLmArchConfig(
        family="gpt-2",
        tokenizer_name=overrides.pop("tokenizer_name", "gpt2"),
        **{**_MIST_GPT2, **overrides},
    )


def gpt_neox_arch(**overrides) -> CausalLmArchConfig:
    """Mist-sized GPT-NeoX from-scratch config."""
    return CausalLmArchConfig(
        family="gpt-neox",
        tokenizer_name=overrides.pop("tokenizer_name", "EleutherAI/gpt-neox-20b"),
        **{**_MIST_GPT_NEOX, **overrides},
    )


def gemma_arch(**overrides) -> CausalLmArchConfig:
    """Mist-sized Gemma from-scratch config (small; not full Gemma-2B)."""
    return CausalLmArchConfig(
        family="gemma",
        tokenizer_name=overrides.pop("tokenizer_name", "google/gemma-2b"),
        **{**_MIST_GEMMA, **overrides},
    )


def load_tokenizer(name_or_path: str | Path) -> PreTrainedTokenizerBase:
    """Load a tokenizer and ensure a pad token is set (eos fallback)."""
    tokenizer = AutoTokenizer.from_pretrained(str(name_or_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model_from_scratch(
    tokenizer: PreTrainedTokenizerBase,
    arch: CausalLmArchConfig | None = None,
    *,
    verbose: bool = True,
) -> PreTrainedModel:
    """Initialize a causal LM with random weights from ``arch``."""
    arch = arch or gpt2_arch()
    arch.validate()
    if verbose:
        step(
            f"Initializing {arch.family} from config with random weights...",
            logger=log,
        )

    vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else tokenizer.vocab_size
    special_token_ids = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if arch.family == "gpt-2":
        model_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=arch.n_positions,
            n_embd=arch.n_embd,
            n_layer=arch.n_layer,
            n_head=arch.n_head,
            n_inner=arch.intermediate_size,
            activation_function=arch.hidden_act,
            resid_pdrop=arch.hidden_dropout,
            embd_pdrop=arch.hidden_dropout,
            attn_pdrop=arch.attention_dropout,
            layer_norm_epsilon=arch.norm_epsilon,
            initializer_range=arch.initializer_range,
            tie_word_embeddings=arch.tie_word_embeddings,
            attn_implementation=arch.attention_implementation,
            use_cache=arch.use_cache,
            **special_token_ids,
        )
        model: PreTrainedModel = GPT2LMHeadModel(model_config)
    elif arch.family == "gpt-neox":
        intermediate = arch.intermediate_size or (4 * arch.n_embd)
        model_config = GPTNeoXConfig(
            vocab_size=vocab_size,
            max_position_embeddings=arch.n_positions,
            hidden_size=arch.n_embd,
            num_hidden_layers=arch.n_layer,
            num_attention_heads=arch.n_head,
            intermediate_size=intermediate,
            hidden_act=arch.hidden_act,
            hidden_dropout=arch.hidden_dropout,
            attention_dropout=arch.attention_dropout,
            layer_norm_eps=arch.norm_epsilon,
            initializer_range=arch.initializer_range,
            rotary_emb_base=arch.rope_theta or 10_000.0,
            rotary_pct=arch.rotary_pct or 0.25,
            tie_word_embeddings=arch.tie_word_embeddings,
            attn_implementation=arch.attention_implementation,
            use_cache=arch.use_cache,
            **special_token_ids,
        )
        model = GPTNeoXForCausalLM(model_config)
    elif arch.family == "gemma":
        head_dim = arch.head_dim or (arch.n_embd // arch.n_head)
        intermediate = arch.intermediate_size or (4 * arch.n_embd)
        # GemmaConfig defaults num_key_value_heads=16 even when
        # num_attention_heads is overridden; mismatch breaks SDPA.
        rope_kwargs = (
            {
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": arch.rope_theta or 10_000.0,
                }
            }
            if "rope_parameters" in getattr(GemmaConfig, "__annotations__", {})
            else {"rope_theta": arch.rope_theta or 10_000.0}
        )
        model_config = GemmaConfig(
            vocab_size=vocab_size,
            max_position_embeddings=arch.n_positions,
            hidden_size=arch.n_embd,
            num_hidden_layers=arch.n_layer,
            num_attention_heads=arch.n_head,
            num_key_value_heads=arch.num_key_value_heads or arch.n_head,
            head_dim=head_dim,
            intermediate_size=intermediate,
            hidden_act=arch.hidden_act,
            attention_dropout=arch.attention_dropout,
            rms_norm_eps=arch.norm_epsilon,
            initializer_range=arch.initializer_range,
            tie_word_embeddings=arch.tie_word_embeddings,
            attn_implementation=arch.attention_implementation,
            use_cache=arch.use_cache,
            **special_token_ids,
            **rope_kwargs,
        )
        model = GemmaForCausalLM(model_config)
    else:
        raise ValueError(f"unsupported family: {arch.family!r}")

    # Incompatible with gradient checkpointing and unused during training.
    model.config.use_cache = arch.use_cache

    if verbose:
        param_count = sum(p.numel() for p in model.parameters())
        detail(f"parameters: {param_count:,}", logger=log)
    return model


def build_model_and_tokenizer(
    arch: CausalLmArchConfig | None = None,
    *,
    verbose: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load tokenizer and initialize a causal LM from scratch."""
    arch = arch or gpt2_arch()
    if verbose:
        step(f"Loading tokenizer ({arch.tokenizer_name})...", logger=log)
    tokenizer = load_tokenizer(arch.tokenizer_name)
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
    family: ModelFamily = "gpt-2",
    verbose: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a saved causal-LM checkpoint onto ``device`` in eval mode."""
    if verbose:
        step(f"Loading model from {model_path}...", logger=log)
    tokenizer = load_tokenizer(model_path)
    if family == "gpt-2":
        model: PreTrainedModel = GPT2LMHeadModel.from_pretrained(model_path)
    elif family == "gpt-neox":
        model = GPTNeoXForCausalLM.from_pretrained(model_path)
    elif family == "gemma":
        model = GemmaForCausalLM.from_pretrained(model_path)
    else:
        raise ValueError(f"unsupported family: {family!r}")
    move_module_to_device(model, device)
    model.eval()
    return model, tokenizer
