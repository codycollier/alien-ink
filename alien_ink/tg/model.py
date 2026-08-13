"""GPT-2 in tinygrad, matched to Hugging Face ``GPT2LMHeadModel`` (Mist / 124M).

Same block as Alien Ink ``family='gpt-2'``: learned positions, pre-LN,
``gelu_new``, resid/attn/embd dropout, tied ``wte``/``lm_head``, tokenizer
vocab (no 50,304 pad). From-scratch init follows the GPT-2 residual scaling.
"""

from __future__ import annotations

import math

from tinygrad import Tensor, nn
from tinygrad.nn.state import get_parameters

from alien_ink.hf.metrics import ModelSize
from alien_ink.hf.model import CausalLmArchConfig

__all__ = [
    "GPT2",
    "build_gpt2",
    "count_gpt2_params",
    "gelu_new",
]


def gelu_new(x: Tensor) -> Tensor:
    """OpenAI GPT-2 GELU (HF ``gelu_new``): tanh approximation, not erf."""
    return 0.5 * x * (1.0 + (math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))).tanh())


def _dropout(x: Tensor, p: float) -> Tensor:
    if p <= 0.0:
        return x
    return x.dropout(p)


def _act(name: str):
    if name == "gelu_new":
        return gelu_new
    if name == "gelu":
        return lambda x: x.gelu()
    raise ValueError(f"tinygrad GPT-2 hidden_act must be 'gelu_new' or 'gelu', got {name!r}")


class CausalSelfAttention:
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ):
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.n_embd = n_embd
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def __call__(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = c // self.n_head
        q = q.reshape(b, t, self.n_head, hs).transpose(1, 2)
        k = k.reshape(b, t, self.n_head, hs).transpose(1, 2)
        v = v.reshape(b, t, self.n_head, hs).transpose(1, 2)
        attn_p = self.attn_pdrop if Tensor.training else 0.0
        y = q.scaled_dot_product_attention(k, v, dropout_p=attn_p, is_causal=True)
        y = y.transpose(1, 2).reshape(b, t, c)
        return _dropout(self.c_proj(y), self.resid_pdrop)


class MLP:
    def __init__(
        self,
        n_embd: int,
        n_inner: int,
        hidden_act: str,
        resid_pdrop: float,
    ):
        self.c_fc = nn.Linear(n_embd, n_inner)
        self.c_proj = nn.Linear(n_inner, n_embd)
        self.act = _act(hidden_act)
        self.resid_pdrop = resid_pdrop

    def __call__(self, x: Tensor) -> Tensor:
        return _dropout(self.c_proj(self.act(self.c_fc(x))), self.resid_pdrop)


class Block:
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_inner: int,
        hidden_act: str,
        norm_eps: float,
        attn_pdrop: float,
        resid_pdrop: float,
    ):
        self.ln_1 = nn.LayerNorm(n_embd, eps=norm_eps)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ln_2 = nn.LayerNorm(n_embd, eps=norm_eps)
        self.mlp = MLP(n_embd, n_inner, hidden_act, resid_pdrop)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT2:
    """Causal GPT-2; ``__call__(idx, targets=idx)`` returns ``(logits, loss)``."""

    def __init__(
        self,
        *,
        vocab_size: int,
        n_positions: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        n_inner: int,
        hidden_act: str = "gelu_new",
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        norm_epsilon: float = 1e-5,
        tie_word_embeddings: bool = True,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.tie_word_embeddings = tie_word_embeddings
        self.embd_pdrop = hidden_dropout
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.h = [
            Block(
                n_embd,
                n_head,
                n_inner,
                hidden_act,
                norm_epsilon,
                attention_dropout,
                hidden_dropout,
            )
            for _ in range(n_layer)
        ]
        self.ln_f = nn.LayerNorm(n_embd, eps=norm_epsilon)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.wte.weight

    def __call__(self, idx: Tensor, targets: Tensor | None = None):
        _b, t = idx.shape
        if t > self.n_positions:
            raise ValueError(
                f"sequence length {t} exceeds n_positions ({self.n_positions})"
            )
        pos = Tensor.arange(t, device=idx.device)
        x = _dropout(self.wte(idx) + self.wpe(pos), self.embd_pdrop)
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits
        # Gather-based NLL: avoid sparse_categorical_crossentropy's (B,T,V) one-hot.
        log_probs = logits[:, :-1, :].log_softmax()
        loss = -log_probs.gather(2, targets[:, 1:].unsqueeze(-1)).squeeze(-1).mean()
        return logits, loss


def _normal(shape: tuple[int, ...], std: float) -> Tensor:
    return Tensor.normal(*shape, mean=0.0, std=std)


def _init_linear(linear: nn.Linear, std: float) -> None:
    linear.weight = _normal(tuple(linear.weight.shape), std)
    if linear.bias is not None:
        linear.bias = Tensor.zeros(*linear.bias.shape)


def _init_gpt2(model: GPT2, initializer_range: float) -> None:
    """HF GPT-2 init: N(0, std), residual ``c_proj`` scaled by ``1/sqrt(2L)``."""
    std = initializer_range
    n_layer = len(model.h)
    resid_std = std / math.sqrt(2.0 * n_layer) if n_layer > 0 else std
    model.wte.weight = _normal(tuple(model.wte.weight.shape), std)
    model.wpe.weight = _normal(tuple(model.wpe.weight.shape), std)
    for block in model.h:
        _init_linear(block.attn.c_attn, std)
        _init_linear(block.attn.c_proj, resid_std)
        _init_linear(block.mlp.c_fc, std)
        _init_linear(block.mlp.c_proj, resid_std)
    if model.tie_word_embeddings:
        model.lm_head.weight = model.wte.weight
    else:
        _init_linear(model.lm_head, std)


def build_gpt2(arch: CausalLmArchConfig, vocab_size: int) -> GPT2:
    """Build a randomly initialized tinygrad GPT-2 from ``CausalLmArchConfig``."""
    arch.validate()
    if arch.family != "gpt-2":
        raise ValueError(
            "tinygrad backend only supports family='gpt-2', "
            f"got {arch.family!r}"
        )
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
    n_inner = arch.intermediate_size or (4 * arch.n_embd)
    model = GPT2(
        vocab_size=vocab_size,
        n_positions=arch.n_positions,
        n_embd=arch.n_embd,
        n_layer=arch.n_layer,
        n_head=arch.n_head,
        n_inner=n_inner,
        hidden_act=arch.hidden_act,
        hidden_dropout=arch.hidden_dropout,
        attention_dropout=arch.attention_dropout,
        norm_epsilon=arch.norm_epsilon,
        tie_word_embeddings=arch.tie_word_embeddings,
    )
    _init_gpt2(model, arch.initializer_range)
    return model


def count_gpt2_params(model: GPT2) -> ModelSize:
    """Param counts with tied ``wte``/``lm_head`` counted once."""
    unique: list[Tensor] = []
    seen: set[int] = set()
    for param in get_parameters(model):
        if id(param) in seen:
            continue
        seen.add(id(param))
        unique.append(param)
    total = int(sum(param.numel() for param in unique))

    embed = 0
    embed_seen: set[int] = set()
    for weight in (model.wte.weight, model.wpe.weight, model.lm_head.weight):
        if id(weight) in embed_seen:
            continue
        embed_seen.add(id(weight))
        embed += int(weight.numel())
    return ModelSize(
        total_params=total,
        trainable_params=total,
        non_embedding_params=max(0, total - embed),
    )
