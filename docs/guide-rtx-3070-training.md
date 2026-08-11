# RTX 3070 training playbook

How to train well on Mist: an Ampere RTX 3070 with 8 GB of VRAM,
third-generation Tensor Cores, and CUDA capability 8.6. Treat 7–7.5 GiB as
the usable training budget; the driver, CUDA context, and temporary workspaces
consume the rest. This guide gives the per-family batch settings, the
reasoning behind them, and a disciplined benchmark sequence for changing them.

The checked-in manifests already use the sound baseline: bf16 when available
(fp16 fallback), TF32, fused AdamW, fixed 1024-token blocks, and an effective
batch of 32. `CausalLmArchConfig` selects PyTorch SDPA by default. On a
CUDA shape and dtype supported by the hardware, SDPA automatically chooses its
best fused backend (Flash or memory-efficient); otherwise it safely falls back.
This makes the speed/memory improvement portable and requires no FlashAttention
build.

## Recommended settings

| Family | Mist model | Safe starting microbatch × accumulation | Quality-focused run | Notes |
|---|---:|---:|---:|---|
| GPT-2 | ~124M | 4 × 8 | 75–100k steps | Fastest family; keep the 4-token microbatch only after an OOM smoke test. |
| GPT-NeoX | ~163M | 4 × 8 | 100k steps | Its untied output head costs extra static VRAM; use 2 × 16 if the 4-token run is close to the limit. |
| Pythia 70M / 160M | 70M / 160M | 4 × 8 | 100k steps | NeoX architecture at published shapes; same settings as NeoX. 70M is the fast-iteration option. |
| Llama (SmolLM2-135M) | ~135M | 2 × 16 | 100k steps | 30 layers carry more activation memory than the 12-layer baselines; enable checkpointing before shrinking further. |
| Gemma | ~165M | 1 × 32 | 100k steps | Leave checkpointing on; its 256k vocabulary makes logits and embeddings the limiting cost. |

At a 1024 block size, every one of these schedules sees 32,768 tokens per
optimizer update. Therefore 50k steps is 1.64B tokens and 100k is 3.28B.
For a from-scratch ~124–165M parameter model, 50k is a useful short run but
not a quality endpoint: the classic compute-optimal reference point is about
20 tokens per parameter (roughly 76–101k steps for these manifests). This is
a planning heuristic, not a promise that a small single-GPU training budget
will reach frontier-model quality.

Do not trade a workable microbatch for excessive accumulation. Once a larger
microbatch fits, it generally has better GPU utilization at the same effective
batch. Conversely, if Gemma or NeoX OOMs, reduce the microbatch first and raise
accumulation to preserve the 32-block effective batch.

## Family-specific quality work

### Gemma

The current model is a Gemma-shaped 165M-parameter model, not Gemma-2B. About
131M parameters are the tied 256k-token embedding/head table, leaving only
about 34M in the transformer trunk. This is why it is expensive but not
proportionally capable on English-only data.

If Gemma is intended only for English, the deeper architectural improvement is
a separately trained 32k–64k tokenizer and a correspondingly smaller
embedding/head, not shrinking `vocab_size` while still using Gemma's
256k-token tokenizer (that would produce invalid token ids).

### GPT-NeoX

NeoX is the best architecture comparison to GPT-2 here, but it starts with an
untied LM head. Tie input and output embeddings only as an explicit experiment:
it can recover about one 50k×768 matrix of parameters and optimizer state, but
it changes the architecture and should be compared with equal token budgets.
Use SDPA and start at 4 × 8; if peak memory is too close to 8 GB, use 2 × 16
instead of disabling mixed precision or TF32.

### GPT-2

GPT-2 remains a useful throughput and dataset-quality baseline. Its 50k-vocab
head is much cheaper than Gemma's, so put available capacity into more tokens
and better data rather than a wider model. Train it on the exact filtered data
mixture used by the other families, preserve document separators when packing,
and compare validation perplexity at equal *tokens seen*, not equal epochs.

## A disciplined benchmark sequence

1. Run 100 optimizer steps for each family with the production tokenizer,
   sequence length, and evaluation loop. Record peak allocated/reserved VRAM,
   tokens/sec, loss, and whether SDPA selected a fused backend.
2. Select the largest non-fragile microbatch (leave at least ~0.5 GiB reserve),
   then change accumulation so the effective batch stays at 32 blocks.
3. Run a 5k-step learning-rate sweep around `3e-4`, `4.5e-4`, and `6e-4`.
   Keep the same warmup ratio (about 4% for the existing schedules) and reject
   configurations with diverging loss or materially worse validation loss.
4. Promote only the winning configuration to 75–100k steps. Evaluate the same
   held-out corpus and a small fixed prompt set for every run.

`torch.compile` can help after warm-up, but it is workload- and version-
dependent and adds compilation time/memory. Keep it enabled in the current
fixed-shape manifests, but benchmark it against `torch_compile=False` on the
actual CUDA environment; do not assume it wins for short 5k-step experiments.

## Sources

- [NVIDIA RTX 3070 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3070-3070ti/) — 8 GB GDDR6, Ampere, compute capability 8.6.
- [Hugging Face: efficient single-GPU training](https://huggingface.co/docs/transformers/v4.45.2/perf_train_gpu_one) — checkpointing trade-off, mixed precision, and TF32 guidance.
- [Hugging Face: mixed precision training](https://huggingface.co/docs/transformers/mixed_precision_training) — bf16/TF32 behavior on Ampere.
- [PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention) — automatic fused attention-kernel selection.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — the compute-optimal token/model-size analysis.
