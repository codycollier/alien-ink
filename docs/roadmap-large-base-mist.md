# Roadmap: large pretrained bases on Mist

**Status: speculative.** This is a map, not a commitment. It sketches how
Alien Ink could take a larger open source pretrained model, fine-tune it with
PEFT/LoRA on Mist (RTX 3070, 8 GB VRAM), and then quantize everything —
including the base model — so the result runs inference on the same card.

The 3070 is the accepted bottleneck. The plan does not try to escape it with
cloud GPUs or CPU offload heroics; it designs the whole loop to fit inside it.

## The loop

```
┌────────────────┐   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
│ 1. choose base │ → │ 2. QLoRA train │ → │ 3. merge       │ → │ 4. quantize    │
│ pretrained,    │   │ 4-bit base +   │   │ adapter into   │   │ merged model   │
│ 1B–8B, open    │   │ LoRA adapter   │   │ full-precision │   │ for inference  │
│ weights        │   │ on Mist        │   │ base (on CPU)  │   │ (GGUF/GPTQ/…)  │
└────────────────┘   └────────────────┘   └────────────────┘   └────────────────┘
                                                                       │
                                                              ┌────────▼───────┐
                                                              │ 5. inference   │
                                                              │ on Mist, 8 GB: │
                                                              │ weights + KV   │
                                                              └────────────────┘
```

Two different quantizations appear in this loop, and keeping them distinct is
most of the mental model:

| | Training-time (QLoRA) | Inference-time (GGUF/GPTQ/AWQ) |
|---|---|---|
| What is quantized | Frozen base weights (NF4 via bitsandbytes) | The merged, fine-tuned model |
| Why | Fit base + adapter + activations in 8 GB | Fit weights + KV cache in 8 GB |
| Precision of the trained part | LoRA adapter stays bf16/fp16 | Everything quantized (typically 4–5 bit) |
| Reversible | Yes — base checkpoint untouched | No — a derived artifact |

## Phase 1: choose the base model

The interesting question is how large a base Mist can *train against* with
QLoRA, since inference-only limits are looser. Rough tiers:

| Tier | Size | Candidates | QLoRA on Mist | Quantized inference |
|---|---:|---|---|---|
| Comfortable | 1–2B | Llama-3.2-1B, SmolLM2-1.7B, Qwen3-1.7B-Base | Easy; room for S=1024+, larger microbatch | Trivial |
| Workable | 3–4B | Llama-3.2-3B, Qwen2.5-3B, Gemma-class ~4B | Fits with checkpointing, B=1, moderate S | Comfortable |
| Ceiling | 7–8B | Qwen2.5-7B, Mistral-7B, Llama-3.1-8B | Tight; short sequences, careful measurement | Fits, KV cache is the constraint |

Speculative recommendation: run the loop end-to-end at the **comfortable tier
first** (SmolLM2-1.7B or Llama-3.2-1B), where every phase has slack and
failures are cheap. Only then attempt a 7B-class run, which is the genuinely
motivating target — a model class far beyond anything Mist could pretrain.

Selection criteria beyond size:

- open weights with a license compatible with local fine-tuning;
- a tokenizer/vocab that doesn't blow up the logits term (Llama 3's 128k and
  Qwen's ~152k vocabularies make `B × S × V` logits real costs again — the
  same lesson as Mist Gemma, at larger scale);
- GQA (fewer KV heads) — this pays off twice, in training activations and in
  the inference KV cache;
- an established GGUF/llama.cpp conversion path, so phase 4 is boring.

## Phase 2: QLoRA training on Mist

This is Track C2 of the [model learning plan](model-learning-plan.md), taken
to its intended scale. The base model loads in 4-bit NF4 via bitsandbytes and
is frozen; only the LoRA adapter trains, in bf16, with its own small optimizer
state.

### Why the memory equation changes

Full fine-tuning pays `P × 12–16` bytes for weights + grads + Adam (see
[gpu-memory.md](gpu-memory.md)). QLoRA replaces that with:

```
static ≈ P × 0.5          # NF4 base weights (frozen, no grads, no Adam)
       + P_lora × 12–16   # adapter weights + grads + Adam (P_lora ≪ P)
```

| Term | 7B example (bytes → GiB) |
|---|---|
| NF4 base weights (+ double-quant constants) | `7e9 × ~0.55` → **~3.6 GiB** |
| LoRA adapter, r=16 on attn+MLP (~40M params) | bf16 weights + grads + fp32 Adam → **~0.5 GiB** |
| Activations (checkpointing on, B=1) | ∝ `B × S × H × L` → **~1–2 GiB** at S=512–1024 |
| Logits | `B × S × V × 4` (fp32 CE) → **~0.5–0.6 GiB** at S=1024, V≈128–152k |
| CUDA / dequant workspaces / overhead | **~1 GiB** |
| **Peak** | **~6.5–7.5 GiB** — at the edge of the card |

```
QLoRA peak vs 8 GiB card (speculative estimates)

1.7B  S=1024  ██████████░░░░░░░░░░░░░░░░░░░░░░  ~2.5–3.5   comfortable
3B    S=1024  ██████████████████░░░░░░░░░░░░░░  ~4.5–5.5   workable
7B    S=512   ██████████████████████████░░░░░░  ~6.5–7.5   ceiling; measure
              0    1    2    3    4    5    6    7    8 GiB
```

The 7B row is the whole reason the tiering in phase 1 matters: it likely fits,
but only with short sequences, microbatch 1, and paged optimizer states as a
spike absorber. Treat OOM at S=1024 as expected, not as failure.

### Training configuration (expected shape)

- NF4 quantization with double quantization, compute dtype bf16;
- gradient checkpointing on, always, at every tier;
- `B=1` with gradient accumulation for the effective batch — same principle
  as the Gemma zdecks: accumulation is free, microbatch is not;
- paged AdamW (`paged_adamw_8bit` or similar) so optimizer spikes page to
  host RAM instead of OOMing;
- sequence length is the primary escape valve: 1024 → 512 → 256 before
  giving up on a model size;
- LoRA rank 8–16 to start, targeting attention projections first, attention +
  MLP as the measured comparison. Rank is a quality knob, barely a memory one
  at these sizes.

Throughput will be poor by pretraining standards — 4-bit dequant on the fly
plus checkpointing recompute. That is the accepted trade: this loop optimizes
for *what can be trained at all*, not tokens/sec. Mist's 16 cores and 94 GB
RAM stay useful for data loading and paged optimizer state.

### Prerequisites in the repository

In learning-plan terms, this phase sits behind C0 (LoRA infrastructure) and
C2 (4-bit loading), which are not yet implemented. The SFT stage
(`Manifest.train(stage="sft")`, `PretrainedLmConfig`) already provides generic
Hub checkpoint loading, the shared eval path, and manifest recording — the
adapter and quantization config layer on top of that rather than replacing it.

## Phase 3: merge the adapter

The training artifact is a small adapter (tens of MB), never a second copy of
the base. For inference-time quantization, the adapter merges into a
full-precision copy of the base:

1. load the base in bf16/fp16 **on CPU** — a 7B model at bf16 is ~14 GB,
   which does not fit in 8 GB VRAM but fits easily in Mist's 94 GB RAM;
2. apply and merge the LoRA deltas (`W' = W + BA·α/r`);
3. save the merged model as an ordinary Hugging Face checkpoint.

Do **not** merge into the NF4 base. Merging into quantized weights compounds
quantization error; the merge target is the original full-precision
checkpoint, and quantization happens once, after the merge.

Keep both artifacts in the archive: the adapter (tiny, composable, re-usable
against the pristine base) and the merged checkpoint (input to phase 4). The
merged bf16 model is a large intermediate; it can be deleted after phase 4 if
disk pressure matters, since it is reproducible from base + adapter.

## Phase 4: quantize the merged model for inference

| Format | Tooling | Character | Fit for Mist |
|---|---|---|---|
| **GGUF (Q4_K_M / Q5_K_M)** | llama.cpp `convert` + `quantize` | CPU-side one-shot quantization, no GPU needed, no calibration data required for K-quants | **Recommended first path** — boring, well-trodden, CPU/GPU-split fallback for free |
| GPTQ | AutoGPTQ / GPTQModel | Calibration-based, better accuracy per bit in some regimes, GPU-resident | Second experiment; calibration adds a step and a data choice |
| AWQ | AutoAWQ | Activation-aware calibration; strong 4-bit quality | Same tier as GPTQ; try if GPTQ disappoints |
| bitsandbytes `load_in_4bit` | transformers | Runtime quantization of the merged checkpoint | Zero extra artifacts, but slower inference; fine for smoke tests |

Speculative recommendation: **GGUF Q4_K_M** as the canonical inference
artifact. It requires no calibration dataset, runs entirely on CPU (so the
14 GB merged model never has to touch VRAM), and llama.cpp's partial-offload
means a mis-estimated fit degrades to slower instead of failing.

Evaluate the quantized model against the merged bf16 model (on CPU, slowly)
on the same held-out set and fixed prompt set used during training. Perplexity
delta between bf16 and Q4_K_M is the number that says whether 4-bit was free
or costly for this particular fine-tune; Q5_K_M is the fallback if Q4 is
measurably worse and the memory budget allows it.

## Phase 5: inference budget on 8 GB

Inference VRAM is weights + KV cache + compute buffers. No gradients, no
optimizer, no logits-for-every-position — a different equation from training:

```
VRAM ≈ quantized weights + KV cache + activation/compute buffers + overhead
KV bytes/token ≈ 2 × L × n_kv_heads × head_dim × elem_bytes
```

Worked speculative example, Qwen2.5-7B-class (28 layers, 4 KV heads via GQA,
head_dim 128, fp16 cache):

| Component | Size |
|---|---:|
| Q4_K_M weights | ~4.4 GiB |
| KV cache @ 8k context | 2 × 28 × 4 × 128 × 2 B ≈ 57 KB/token → ~0.45 GiB |
| Compute buffers + CUDA overhead | ~1–1.5 GiB |
| **Total** | **~6–6.5 GiB** — fits with headroom |

```
Inference on Mist, Q4_K_M + 8k context (speculative)

1.7B  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░  ~2 GiB     trivial
3B    ██████████░░░░░░░░░░░░░░░░░░░░░░  ~3 GiB     easy
7B    ████████████████████████░░░░░░░░  ~6–6.5     fits; KV limits context
      0    1    2    3    4    5    6    7    8 GiB
```

A model without GQA, or with a fat KV head count (Llama-3.1-8B has 8 KV
heads, ~131 KB/token), pays roughly double for context — which is why KV
geometry belongs in the phase-1 selection criteria. Quantized KV cache (q8_0)
buys the context back at some quality cost if needed.

## Alien Ink integration (speculative)

Following the existing manifest philosophy — everything that varies lives in
the manifest, functions over classes, quantization separate from LoRA so
failures localize:

- a LoRA config block (rank, alpha, dropout, target modules) alongside
  `PretrainedLmConfig`, plus a quantization config block (4-bit on/off, quant
  type, compute dtype) that composes with it rather than being fused into it;
- trainable/frozen parameter reporting extended to show adapter share;
- adapter-only checkpoint saving; the base model is referenced by name and
  revision in the manifest, never copied;
- zdecks in the established naming style, e.g.
  `qlora_llama-3.2-1b_geo_mist.py` … `qlora_qwen2.5-7b_*_mist.py`;
- phases 3–4 (merge, GGUF conversion, quantization) as small CPU-side
  scripts or `bin/` entries rather than trainer machinery — they are one-shot
  artifact transformations, not training;
- the manifest records the full artifact chain: base name+revision → adapter
  → merged checkpoint hash → quantized artifact, so any inference model in
  the zdeck's history is reproducible.

New optional dependency set: `peft`, `bitsandbytes` (CUDA-only), and llama.cpp
as an external tool rather than a Python dependency.

## Risks and open questions

- **The 7B QLoRA fit is unproven on this card.** Published QLoRA numbers for
  7B are usually quoted against ~10 GB cards. S=512 and paged optimizers may
  or may not close the gap; the comfortable-tier run exists to make this an
  experiment rather than a blocker.
- **bitsandbytes on the local CUDA/driver stack** is a classic source of
  version friction; pin and smoke-test before designing around it.
- **Double quantization of quality**: NF4 during training *and* Q4 for
  inference are two independent lossy steps. The bf16-merged model evaluated
  on CPU is the ground truth that separates "the fine-tune is weak" from "the
  quantization hurt".
- **Chat templates and label masking** (Track B future work) matter more for
  instruction-tuned bases than they did for small-model SFT; a base model
  plus completion-style data sidesteps this initially.
- **Throughput on the ceiling tier** may make 7B QLoRA runs multi-day for
  even modest token budgets. That is acceptable for a personal library, but
  worth knowing before choosing a dataset size.

## Suggested sequence

1. Implement LoRA (C0) against a small known base and existing SFT data.
2. Add 4-bit loading (C2) and verify frozen-base/trainable-adapter reporting.
3. Full loop at the comfortable tier: QLoRA SmolLM2-1.7B or Llama-3.2-1B →
   merge on CPU → GGUF Q4_K_M → evaluate quantized vs merged-bf16.
4. Repeat at 3B; record peak VRAM, tokens/sec, and quality deltas at each
   phase.
5. Attempt the 7B ceiling with S=512, B=1, paged AdamW; treat the first run
   as a memory measurement, not a training run.
6. Only after the loop is measured end-to-end, iterate on quality: rank,
   target modules, data, and GPTQ/AWQ as inference-quantization comparisons.

## Sources

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — NF4, double quantization, paged optimizers.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — adapter formulation and merge math.
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft) — LoRA config, adapter checkpoints, merge/unload.
- [Hugging Face bitsandbytes integration](https://huggingface.co/docs/transformers/quantization/bitsandbytes) — 4-bit loading and compute dtype.
- [llama.cpp quantization](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) — GGUF K-quant formats and conversion flow.
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) / [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) — calibration-based inference quantization.
