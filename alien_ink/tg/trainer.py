"""Tinygrad GPT-2 train step: AdamW, cosine+warmup, grad clip, TinyJit."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from tinygrad import Tensor, TinyJit, dtypes, nn
from tinygrad.nn.optim import Optimizer, OptimizerGroup
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save

from alien_ink.com.log import detail, get_logger, step
from alien_ink.tg.model import GPT2

log = get_logger("tg.trainer")

__all__ = [
    "build_optimizer",
    "clip_grad_norm_",
    "cosine_lr",
    "evaluate_loss",
    "make_train_step",
    "save_gpt2",
    "set_optimizer_lr",
]


def cosine_lr(
    step: int,
    *,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
) -> float:
    """HF cosine-with-warmup: linear warmup, then cosine to 0."""
    if max_steps < 1:
        return learning_rate
    if warmup_steps > 0 and step < warmup_steps:
        return learning_rate * float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(
    model: GPT2,
    *,
    learning_rate: float,
    adam_beta1: float,
    adam_beta2: float,
    weight_decay: float,
) -> Optimizer:
    """AdamW with HF Trainer decay groups: 2D weights decay, 1D do not."""
    unique: list[Tensor] = []
    seen: set[int] = set()
    for param in get_parameters(model):
        if id(param) in seen:
            continue
        seen.add(id(param))
        unique.append(param)
    decay = [p for p in unique if p.ndim >= 2]
    nodecay = [p for p in unique if p.ndim < 2]
    opts: list[Optimizer] = []
    if decay:
        opts.append(
            nn.optim.AdamW(
                decay,
                lr=learning_rate,
                b1=adam_beta1,
                b2=adam_beta2,
                weight_decay=weight_decay,
            )
        )
    if nodecay:
        opts.append(
            nn.optim.AdamW(
                nodecay,
                lr=learning_rate,
                b1=adam_beta1,
                b2=adam_beta2,
                weight_decay=0.0,
            )
        )
    if not opts:
        raise ValueError("model has no parameters")
    if len(opts) == 1:
        return opts[0]
    return OptimizerGroup(*opts)


def set_optimizer_lr(optimizer: Optimizer, learning_rate: float) -> None:
    opts = (
        optimizer.optimizers
        if isinstance(optimizer, OptimizerGroup)
        else (optimizer,)
    )
    for opt in opts:
        opt.lr.assign(
            Tensor([learning_rate], device=opt.lr.device, dtype=opt.lr.dtype)
        ).realize()


def clip_grad_norm_(params: Sequence[Tensor], max_norm: float) -> Tensor | None:
    """In-place global-norm clip. Returns the unclipped norm, or None."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return None
    total = grads[0].float().square().sum()
    for grad in grads[1:]:
        total = total + grad.float().square().sum()
    total = total.sqrt()
    clip_coef = (max_norm / (total + 1e-6)).clip(0.0, 1.0)
    for param in params:
        if param.grad is not None:
            grad = param.grad
            param.grad.assign(grad * clip_coef.cast(grad.dtype))
    return total


def _optimizer_params(optimizer: Optimizer) -> list[Tensor]:
    return list(optimizer.params)


def _zero_grad_bufs(params: Sequence[Tensor]) -> None:
    """Allocate zero grad buffers so every microbatch uses the accumulate path."""
    for param in params:
        param.grad = Tensor.zeros(
            *param.shape, device=param.device, dtype=param.dtype
        )
    Tensor.realize(*[param.grad for param in params if param.grad is not None])


def make_train_step(
    model: GPT2,
    optimizer: Optimizer,
    *,
    max_grad_norm: float,
    accum_steps: int = 1,
) -> Callable[[Tensor], tuple[Tensor, float | None]]:
    """TinyJit train step.

    ``accum_steps == 1``: ``idx`` is ``(B, T)``.
    ``accum_steps > 1``: ``idx`` is ``(accum, B, T)`` with a fixed microbatch size.

    Only a single microbatch forward/backward is jitted. Accumulating inside one
    TinyJit keeps all microbatch activations alive and OOMs at GPT-2 Mist scale.

    Returns ``(mean_loss, grad_norm)`` where ``grad_norm`` is the unclipped
    global norm (or ``None`` when clipping is disabled).
    """
    params = _optimizer_params(optimizer)
    accum = max(1, int(accum_steps))
    scale = 1.0 / float(accum)

    @TinyJit
    @Tensor.train()
    def micro_step(idx: Tensor) -> Tensor:
        _, loss = model(idx, idx)
        backward_loss = loss if accum <= 1 else loss * scale
        backward_loss.backward()
        grads = [p.grad for p in params if p.grad is not None]
        return loss.realize(*grads)

    def step(idx: Tensor) -> tuple[Tensor, float | None]:
        optimizer.zero_grad()
        _zero_grad_bufs(params)
        with Tensor.train():
            if accum <= 1:
                mean_loss = micro_step(idx)
            else:
                loss_acc: Tensor | None = None
                # Contiguous copies: TinyJit keys on input UOps, so idx[i] views
                # (different SHRINK offsets) would miss the captured graph.
                micros = [idx[i].contiguous() for i in range(accum)]
                Tensor.realize(*micros)
                for batch in micros:
                    loss = micro_step(batch)
                    loss_acc = loss if loss_acc is None else loss_acc + loss
                assert loss_acc is not None
                mean_loss = loss_acc * scale
            grad_norm_t: Tensor | None = None
            if max_grad_norm > 0:
                grad_norm_t = clip_grad_norm_(params, max_grad_norm)
            # float32 scalar so callers can use .item() under bf16 training.
            mean_loss_f = mean_loss.cast(dtypes.float)
            extras = list(optimizer.schedule_step())
            if grad_norm_t is not None:
                extras.append(grad_norm_t)
            mean_loss_f.realize(*extras)
            grad_norm = float(grad_norm_t.item()) if grad_norm_t is not None else None
            return mean_loss_f, grad_norm

    return step


def train_step_variable(
    model: GPT2,
    optimizer: Optimizer,
    microbatches: Sequence[np.ndarray],
    *,
    max_grad_norm: float,
) -> tuple[float, float | None]:
    """Non-jitted optimizer step for a ragged last group of microbatches."""
    params = _optimizer_params(optimizer)
    scale = 1.0 / float(len(microbatches))
    optimizer.zero_grad()
    _zero_grad_bufs(params)
    loss_acc: Tensor | None = None
    with Tensor.train():
        for arr in microbatches:
            tokens = Tensor(arr)
            _, loss = model(tokens, tokens)
            (loss * scale).backward()
            # Realize grads so this microbatch's activations can free.
            Tensor.realize(*[p.grad for p in params if p.grad is not None])
            loss_acc = loss if loss_acc is None else loss_acc + loss
        assert loss_acc is not None
        mean_loss = loss_acc * scale
        grad_norm_t: Tensor | None = None
        if max_grad_norm > 0:
            grad_norm_t = clip_grad_norm_(params, max_grad_norm)
        mean_loss_f = mean_loss.cast(dtypes.float)
        extras = list(optimizer.schedule_step())
        if grad_norm_t is not None:
            extras.append(grad_norm_t)
        mean_loss_f.realize(*extras)
        grad_norm = float(grad_norm_t.item()) if grad_norm_t is not None else None
    return float(mean_loss_f.item()), grad_norm


def evaluate_loss(
    model: GPT2,
    batches: Sequence[np.ndarray],
    *,
    max_batches: int | None = None,
) -> float:
    """Mean shifted LM loss over packed eval blocks (dropout off)."""
    if not batches:
        raise ValueError("evaluate_loss requires at least one batch")
    limit = len(batches) if max_batches is None else min(len(batches), max_batches)
    total = 0.0
    n = 0
    for arr in batches[:limit]:
        tokens = Tensor(arr)
        _, loss = model(tokens, tokens)
        total += float(loss.cast(dtypes.float).item())
        n += 1
    return total / n


def save_gpt2(model: GPT2, path) -> None:
    """Write tinygrad safetensors (tied weights appear once per unique tensor)."""
    step(f"Saving tinygrad weights to {path}", logger=log)
    safe_save(get_state_dict(model), str(path))
    detail(f"saved {path}", logger=log)


def iter_shuffled_microbatches(
    dataset,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
) -> list[np.ndarray]:
    """Materialize one shuffled epoch of ``(B, T)`` int32 batches (last may be short)."""
    n = len(dataset)
    if n < 1:
        raise ValueError("train dataset is empty")
    rng = np.random.RandomState(seed + epoch)
    order = rng.permutation(n)
    batches: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        rows = [dataset[int(i)]["input_ids"] for i in order[start : start + batch_size]]
        batches.append(np.asarray(rows, dtype=np.int32))
    return batches


def group_microbatches(
    microbatches: Sequence[np.ndarray],
    accum_steps: int,
) -> list[list[np.ndarray]]:
    groups: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    for arr in microbatches:
        current.append(arr)
        if len(current) == accum_steps:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def group_is_jit_ready(group: Sequence[np.ndarray], accum_steps: int) -> bool:
    if len(group) != accum_steps:
        return False
    shape0 = group[0].shape
    return all(arr.shape == shape0 for arr in group)
