# GPU memory during pretraining

Alien Ink trains on Mist (RTX 3070, **~8 GB** VRAM). This page explains what
fills that budget: model weights, optimizer state, activations, logits, and
overhead — and how the Mist zdeck knobs keep each family under the limit.

Formulas use GiB = 1024³ bytes. Training dtype is **bf16 or fp16** (2 bytes
per value) unless noted. Optimizer moments stay in **fp32** (4 bytes).

---

## At a glance

| Bucket | Scales with | Mist GPT-2 (order) | Mist Gemma (order) |
|---|---|---:|---:|
| Parameters (weights) | `P` | ~0.2 GiB | ~0.5 GiB |
| Gradients | `P` | ~0.2 GiB | ~0.5 GiB |
| AdamW state (+ master) | `P` | ~1.0–1.4 GiB | ~2.2–3.2 GiB |
| Logits (LM head output) | `B × S × V` | ~0.2–0.4 GiB | ~0.5–1.0 GiB |
| Activations (w/ checkpointing) | `B × S × H × L` | ~0.5–1.5 GiB | ~0.3–1.0 GiB |
| CUDA / allocator / temps | fixed-ish | ~0.5–1.0 GiB | ~0.5–1.0 GiB |
| **Peak (comfortable)** | | **~3–5 GiB** | **~5–7 GiB** |
| Micro-batch `B` | | **2** | **1** |

`P` = parameter count, `B` = `per_device_train_batch_size`, `S` = `block_size`,
`V` = vocab size, `H` = `n_embd`, `L` = `n_layer`.

Corpus size (WikiText vs C4) does **not** change GPU memory — streams stay on
CPU/disk. Only the **packed micro-batch** on device matters.

---

## The memory equation

Peak training VRAM is roughly:

```
VRAM ≈ params + grads + optimizer + activations + logits + overhead
```

| Term | Formula (bytes) | Notes |
|---|---|---|
| Params | `P × 2` | bf16/fp16 weights on GPU |
| Grads | `P × 2` | same shape as params |
| Optimizer | `P × 8` to `P × 12` | AdamW `m`+`v` in fp32 (=8); + fp32 master copy (=+4) under some AMP setups |
| Activations | see below | reduced by `gradient_checkpointing=True` |
| Logits | `B × S × V × bytes` | often the Gemma killer; CE may upcast to fp32 |
| Overhead | ~0.5–1 GiB | CUDA context, cuDNN workspace, fragmentation |

**Rule of thumb for the static (model + train state) part:**

```
bytes_per_param ≈ 12   # bf16 weights + bf16 grads + fp32 Adam (m, v)
                   16   # …plus fp32 master weights
static_GiB ≈ P × bytes_per_param / 1024³
```

| Params `P` | @ 12 B/param | @ 16 B/param |
|---:|---:|---:|
| 124M (GPT-2 Mist) | **1.4 GiB** | **1.8 GiB** |
| 125M (NeoX Mist) | **1.4 GiB** | **1.9 GiB** |
| 290M (Gemma Mist) | **3.2 GiB** | **4.3 GiB** |

Everything else (activations, logits, overhead) sits on top of that floor.

---

## What each contributor is

### 1. Model parameters (weights)

The network itself. Dominated by:

| Piece | Shape | Count |
|---|---|---|
| Token embeddings | `V × H` | (×2 if LM head untied) |
| Per-layer attention + MLP | ~`12 H²` (order of magnitude) | × `L` |
| Position / norms | small | — |

```
embed_params ≈ V × H          # tied
             ≈ 2 × V × H      # untied (Mist Gemma treated this way)
```

| Family | `V` | `H` | Embed (approx.) | Trunk | Total `P` |
|---|---:|---:|---:|---:|---:|
| GPT-2 Mist | ~50k | 768 | ~39M (tied) | ~85M | **~124M** |
| GPT-NeoX Mist | ~50k | 768 | ~39M | ~86M | **~125M** |
| Gemma Mist | ~256k | 512 | ~262M (untied) | ~20–40M | **~280–300M** |

On Mist, Gemma’s **vocabulary**, not depth, drives parameter memory.

Weights in bf16:

```
weight_GiB = P × 2 / 1024³
```

| Family | Weight memory |
|---|---:|
| GPT-2 | ~0.23 GiB |
| Gemma Mist | ~0.54 GiB |

### 2. Gradients

Full fine-tune (Alien Ink default): one gradient tensor per parameter → **same
size as weights** (~0.23 GiB GPT-2, ~0.54 GiB Gemma).

### 3. Optimizer state (AdamW)

Adam keeps two fp32 buffers per parameter (`m`, `v`):

```
adam_GiB = P × 8 / 1024³
```

| Family | Adam `m`+`v` |
|---|---:|
| GPT-2 | ~0.92 GiB |
| Gemma Mist | ~2.2 GiB |

If a master fp32 weight copy is present, add another `P × 4` (~0.5 / ~1.1 GiB).

**Gradient accumulation does not multiply optimizer memory.** Moments update
once per optimizer step; micro-batches only accumulate gradients into the same
grad buffers. That is why Gemma uses `batch=1`, `accum=32` instead of `batch=2`.

### 4. Training “data” on the GPU

Hub rows and the stream buffer live in **host RAM / disk**. What hits VRAM:

| Tensor | Shape | Size on Mist GPT-2 (`B=2`, `S=1024`) |
|---|---|---|
| `input_ids` / `labels` | `B × S` int64 | ~16 KiB — negligible |
| Attention mask | `B × S` | negligible |
| Hidden states / residuals | `B × S × H` per live activation | **large** |
| Attention scores / context | scales with `S²` per head (implementation-dependent) | **large** at long context |
| **Logits** | `B × S × V` | **often the largest data term** |

So “data memory” means **activations + logits for the current micro-batch**,
not the corpus.

#### Logits (LM head)

```
logits_bytes = B × S × V × elem_bytes
```

| Setup | `B` | `S` | `V` | fp16 (2 B) | fp32 (4 B, CE upcast) |
|---|---:|---:|---:|---:|---:|
| GPT-2 Mist | 2 | 1024 | ~50,257 | **~0.19 GiB** | **~0.39 GiB** |
| Gemma Mist | 1 | 1024 | ~256,000 | **~0.49 GiB** | **~0.98 GiB** |
| Gemma @ batch 2 | 2 | 1024 | ~256,000 | **~0.98 GiB** | **~1.95 GiB** |

The zdeck comment that Gemma `batch=2` materializes **~2 GiB** logits matches
the **fp32** column — cross-entropy often upcasts logits before the loss.

#### Activations (forward + backward)

Without checkpointing, every layer’s intermediates are retained for backward —
memory grows roughly **linear in `L`**, and attention can grow with **`S²`**.

With `gradient_checkpointing=True` (all Mist zdecks): only checkpoints are
kept; intermediates are recomputed on backward. Activation memory drops by
roughly an order of magnitude (≈√`L` or “one layer at a time”), at the cost of
extra compute (~20–30% slower is typical).

Order-of-magnitude with checkpointing on Mist:

```
act_GiB ≲ k × B × S × H × 2 / 1024³     # k ≈ a few×L (checkpointed)
```

| Family | `B` | `H` | `L` | Rough activations |
|---|---:|---:|---:|---|
| GPT-2 | 2 | 768 | 12 | ~0.5–1.5 GiB |
| Gemma | 1 | 512 | 8 | ~0.3–1.0 GiB |

Turning checkpointing **off** on 8 GB usually OOMs even when the static budget
looked fine.

### 5. Overhead

CUDA context, cuBLAS/cuDNN workspaces, allocator fragmentation, and transient
kernels: plan **~0.5–1.0 GiB** that never shows up in the param math. An “8 GB”
card often has only ~7.2–7.5 GiB usable after the driver reserve.

---

## Worked Mist budgets

### GPT-2 Mist (`gpt-2_wikitext_5k` / `gpt-2_wikipedia_5k`)

`P ≈ 124M`, `B=2`, `S=1024`, `V≈50k`, checkpointing on, bf16/fp16.

```
                    GiB
Params (bf16)       ████░░░░░░░░░░░░  0.23
Grads               ████░░░░░░░░░░░░  0.23
Adam m+v (fp32)     ████████████████  0.92
(+ master, if any)  ████████░░░░░░░░  0.46
Logits (fp16→fp32)  ██████░░░░░░░░░░  0.2–0.4
Activations (ckpt)  ████████████░░░░  0.5–1.5
Overhead            ████████░░░░░░░░  0.5–1.0
                    ────────────────
Peak (typical)      ~3–5 GiB  of  ~8 GiB
```

Headroom is why GPT-2 can use **micro-batch 2**. Effective batch 32 comes from
`gradient_accumulation_steps=16` without extra VRAM for Adam.

### Gemma Mist (`gemma_c4_*`, `gemma_wikitext_4ep`)

`P ≈ 290M`, `B=1`, `S=1024`, `V≈256k`, checkpointing on.

```
                    GiB
Params (bf16)       ████████░░░░░░░░  0.54
Grads               ████████░░░░░░░░  0.54
Adam m+v (fp32)     ████████████████████████  2.2
(+ master, if any)  ████████████░░░░  1.1
Logits @ B=1        ████████████░░░░  0.5–1.0
Logits @ B=2        ████████████████████████  ~2.0 (fp32)
Activations (ckpt)  ████████░░░░░░░░  0.3–1.0
Overhead            ████████░░░░░░░░  0.5–1.0
                    ────────────────
Peak @ B=1          ~5–7 GiB  of  ~8 GiB
Peak @ B=2          OOM risk (logits alone ~2 GiB on top of ~4+ GiB static)
```

Same **32,768 tokens/optimizer step** as GPT-2 via `accum=32`, without doubling
the logit tensor.

### Side-by-side

| | GPT-2 Mist | Gemma Mist |
|---|---:|---:|
| Dominant cost | Adam + activations | Adam + **vocab / logits** |
| Static (12–16 B/`P`) | ~1.4–1.8 GiB | ~3.2–4.3 GiB |
| Logits @ zdeck `B` | ~0.2–0.4 GiB | ~0.5–1.0 GiB |
| Zdeck `B` × accum | 2 × 16 | 1 × 32 |
| Tokens / step | 32,768 | 32,768 |
| Fits 8 GB? | Comfortably | Tight; `B=2` OOMs |

---

## What moves the needle

| Knob | Effect on VRAM | Mist practice |
|---|---|---|
| `per_device_train_batch_size` | Linear in activations + logits | GPT-2: 2; Gemma: 1 |
| `gradient_accumulation_steps` | **None** on peak (same grad buffer) | Raise to keep effective batch |
| `block_size` / `n_positions` | Linear in act/logits; attention worse than linear in `S` | Keep 1024 |
| Vocab `V` | Linear in embed params + logits | Gemma’s main pressure |
| `n_embd`, `n_layer` | Params + activations | Mist Gemma is shallow/narrow |
| `gradient_checkpointing` | Large ↓ activations, ↑ time | **Always on** in zdecks |
| `prefer_bf16` / `prefer_fp16` | ~2× vs fp32 weights/acts | Prefer bf16, else fp16 |
| Dataset / `max_steps` | No GPU effect | Stream on host |
| `dataloader_num_workers` | Host RAM / CPU, not VRAM | 2 |

### Effective batch without extra VRAM

```
tokens_per_step = B × gradient_accumulation_steps × block_size
```

| | `B` | accum | `block_size` | tokens/step |
|---|---:|---:|---:|---:|
| GPT-2 | 2 | 16 | 1024 | 32,768 |
| Gemma | 1 | 32 | 1024 | 32,768 |

---

## Quick estimate recipe

1. Count or approximate `P` (or use the Mist architecture tables for each family).
2. Static floor: `P × 12 / 1024³` (or ×16 with master weights).
3. Add logits: `B × S × V × 2` (or ×4 if loss upcasts).
4. Add ~1–2 GiB for checkpointed activations + CUDA overhead on Mist-scale models.
5. Leave **≥1 GiB** slack — fragmentation and peak intermediates exceed averages.

```
estimate_GiB ≈ P×12/1024³ + B×S×V×4/1024³ + 1.5
```

| Family | Plug-in | Estimate | 8 GB? |
|---|---|---:|---|
| GPT-2, B=2 | 124e6×12 + 2×1024×5e4×4 + 1.5 | ~3.3 | yes |
| Gemma, B=1 | 290e6×12 + 1×1024×256e3×4 + 1.5 | ~5.7 | yes (tight) |
| Gemma, B=2 | 290e6×12 + 2×1024×256e3×4 + 1.5 | ~7.6 | often OOM |

---

## Mapping to Alien Ink knobs

All of the above are controlled from the manifest:

```python
hardware=HardwareConfig(
    per_device_train_batch_size=...,  # B — activation + logit memory
    gradient_accumulation_steps=...,  # effective batch; not peak VRAM
    prefer_bf16=True,
    prefer_fp16=True,
    gradient_checkpointing=True,     # activation memory
)
# data.block_size → S
# model.n_embd / n_layer / tokenizer → H, L, V → P and logits
```
