# Pythia pretraining on Mist

Practical from-scratch Pythia options for Mist: one RTX 3070 with 8 GB of
VRAM. The recommendation is **Pythia-70M for the best complete experiment**
or **Pythia-160M when model scale matters more than turnaround time**.
Pythia-410M is the full-parameter ceiling, requires 512-token blocks, and is
not a comfortable long pretraining run. Models at 1B and above do not fit with
ordinary full-parameter AdamW and are out of scope for pretraining on Mist.

This guide is for random initialization and pretraining. Chinchilla's scaling
rule does not specify a fine-tuning budget for an existing checkpoint.

## What “Chinchilla optimal” means here

The Chinchilla paper studies how to divide a fixed training-compute budget
between parameter count and training data. Its central result is that model
parameters and training tokens should grow at approximately the same rate.
The paper does **not** state a universal `D = 20N` law. It fits three
compute-optimal frontiers that agree that parameters and data should scale in
roughly equal proportions but differ in their exact coefficients. Its
Approach 1 projections give 8.0B tokens for 400M parameters and 20.2B for 1B;
Approach 2 gives 7.7B and 20.0B; Approach 3 gives 9.2B and 27.1B. Chinchilla
itself used 1.4T tokens for 70B parameters. These results motivate the commonly
used planning approximation:

```text
training tokens D ≈ 20 × model parameters N
training FLOPs C ≈ 6 × N × D
```

This is a rule of thumb, not a stopping theorem or a claim that the optimum is
exactly 20. The fitted optimum depends on the architecture, tokenizer, corpus
quality, and compute accounting. The paper assumes an “infinite data” regime:
the number of training tokens is smaller than the available corpus. Repeating
a small corpus to reach 20:1 falls outside that assumption. Use the ratio to
choose a defensible run budget, then compare held-out loss at checkpoints
before deciding whether more data is worthwhile.

The paper uses `C ≈ 6ND` to derive its parametric efficient frontier, while
its detailed experimental FLOP accounting includes embeddings and other
operations. `6ND` is useful for scale intuition, not for predicting Mist wall
time—especially for small Pythia models, where the vocabulary matrices are a
larger fraction of the model and hardware utilization differs sharply from
large TPU runs.

Pythia's published names count the embedding and unembedding parameters,
which is the appropriate count for Mist's memory planning and is a reasonable
input to the 20:1 approximation: Appendix F says the Chinchilla analysis also
counts embedding matrices in both parameters and FLOPs. The original Pythia
suite did **not** follow this budget: every main model saw about 300B tokens,
so its smaller models were trained far beyond the 20:1 planning ratio.

The paper's experiments range from below 70M to 16B parameters, but most runs
are above 500M, and the authors report larger fit residuals at low compute.
Consequently, the 14M and 31M rows below are especially uncertain
extrapolations; even 70M sits near the lower edge of the evidence.

## Token and step budgets

Alien Ink's standard effective batch is 32 packed blocks. At a 1024-token
block size that is 32,768 tokens per optimizer step. The safe 410M setup uses
64 × 512-token blocks and therefore preserves the same tokens per step.

```text
optimizer steps = ceil(target tokens / 32,768)
```

| Pythia size | 20:1 planning target | Optimizer steps | Mist status |
|---:|---:|---:|---|
| 14M | 280M tokens | 8,545 | Would fit, but Alien Ink has no 14M factory today |
| 31M | 620M tokens | 18,921 | Would fit, but Alien Ink has no 31M factory today |
| **70M** | **1.4B tokens** | **42,725** | **Recommended; supported and comfortable** |
| **160M** | **3.2B tokens** | **97,657** | **Supported; safe but substantially slower** |
| 410M | 8.2B tokens | 250,245 | Edge configuration; technically plausible, operationally poor |
| 1B | 20B tokens | 610,352 | Does not fit full-parameter AdamW in 8 GB |

These are optimizer steps, not forward/backward microsteps. For example,
Pythia-70M at microbatch 4 and accumulation 8 performs eight microsteps per
optimizer step, or 341,800 microsteps for the full 1.4B-token run.

The 1.4B-token budget in the question is exactly the 20:1 target for 70M. It
is only 44% of the corresponding 160M target and 17% of the 410M target.

## Configurations that protect the 8 GB ceiling

Start with the largest listed microbatch. If a 100-step production-shape
smoke test leaves less than 0.5 GiB unused or OOMs during evaluation or the
first optimizer update, move one row to the right. Reducing microbatch and
raising accumulation preserves the token budget and learning schedule.

| Model | Block size | Preferred | Conservative | Last-resort safe shape |
|---|---:|---:|---:|---:|
| Pythia-70M | 1024 | `B=4, accum=8`, checkpointing off | `B=2, accum=16` | `B=1, accum=32`, checkpointing on |
| Pythia-160M | 1024 | `B=4, accum=8`, checkpointing off | `B=2, accum=16`, checkpointing on | `B=1, accum=32`, checkpointing on |
| Pythia-410M | **512** | — | — | `B=1, accum=64`, checkpointing on, compile off |

For every row, use:

- bf16 when CUDA supports it, with fp16 fallback;
- PyTorch SDPA, TF32, and fused AdamW;
- evaluation batch no larger than the training microbatch;
- fixed-length packed data, so short documents do not waste most blocks;
- streaming for Wikipedia or C4, avoiding a full corpus materialization;
- `torch_compile=False` for the first smoke test, enabling it only after an
  A/B benchmark proves both a speed win and adequate memory headroom.

The 410M limit is based on Mist's checked-in full-fine-tuning experiment: a
1024-token block OOMs on the full-vocabulary cross-entropy buffer, while batch
one at 512 with checkpointing is the intended fit. Treat that as the hardware
ceiling, not a guarantee across PyTorch/CUDA versions. Pythia-1B's parameter,
gradient, and Adam state alone exceed the usable budget before activations and
logits, so gradient accumulation cannot make it fit.

## Recommended runs

### Option A: Pythia-70M, complete 20:1 run

```text
model                 Pythia-70M from scratch
corpus                C4 English, Wikipedia English, or a documented mixture
block size            1024
microbatch × accum    4 × 8
tokens/update         32,768
optimizer steps       42,725
target tokens         1.4B
```

This is the default recommendation. It reaches the 20:1 planning target in a
single-GPU-sized project and leaves enough VRAM margin to diagnose data and
training behavior without every change becoming a memory exercise.

### Option B: Pythia-160M, complete 20:1 run

```text
model                 Pythia-160M from scratch
corpus                preferably C4 or a C4/Wikipedia mixture
block size            1024
microbatch × accum    4 × 8; fall back to 2 × 16 with checkpointing
tokens/update         32,768
optimizer steps       97,657
target tokens         3.2B
```

This is the larger supported comparison, but it consumes more than twice the
tokens and does more work per token. Choose it for a model-size experiment,
not because 160M is automatically a better use of one 3070.

### Option C: fixed 1.4B-token comparison

Train both 70M and 160M for 42,725 steps on exactly the same token stream.
This is a useful controlled comparison, but only the 70M run is at the 20:1
planning point. Label the 160M result **budget-matched**, not
Chinchilla-optimal.

### Option D: Pythia-410M ceiling experiment

Use 512-token blocks, batch one, accumulation 64, gradient checkpointing, and
250,245 optimizer steps for 8.2B tokens. Run 100 steps, then 5,000 steps,
before committing. Even if memory is stable, the likely multi-week runtime
makes 410M a poor practical choice on Mist. Fine-tuning a published 410M
checkpoint is much more sensible than pretraining it here.

## Corpus choice

| Corpus | Fit for these runs | Caution |
|---|---|---|
| Wikipedia English | Excellent for a 70M 1.4B-token run | Narrow encyclopedic domain; one full pass is several billion tokenizer tokens, so stop by tokens rather than epochs |
| C4 English | Enough fresh data for every feasible target | Noisier web text; streaming and deterministic shuffling are important |
| WikiText-103 | Smoke tests and learning curves | About 0.1B words; reaching 1.4B tokens requires many repeats and is not equivalent to 1.4B fresh tokens |
| Wikipedia + C4 | Best general recommendation | Record the mixture and phase boundaries explicitly in the manifest |

For a general English base, a reasonable 70M plan is a minority Wikipedia
phase for high-quality encyclopedic prose and a majority C4 phase for breadth.
The 20:1 heuristic concerns total tokens seen, not the mixture, and does not
tell us what the Wikipedia/C4 proportions should be. The paper trained
Chinchilla on a MassiveText mixture and adjusted its subset distribution for
the longer run. Keep a held-out slice from each local domain and compare both
losses; one aggregate loss can hide a regression in the smaller domain.

## Time planning

Wall time must be derived from a production-shape benchmark on Mist. Measure
training tokens/second after warm-up and use:

```text
hours = target tokens / measured training tokens_per_second / 3,600
```

For the 70M target, illustrative sustained throughputs are:

| Sustained throughput | 1.4B-token compute time |
|---:|---:|
| 5k tokens/s | 77.8 h |
| 10k tokens/s | 38.9 h |
| 15k tokens/s | 25.9 h |
| 20k tokens/s | 19.4 h |

Add 10–25% for evaluation, checkpointing, dataloader stalls, and restarts.
Do not transfer the 70M estimate to 160M or 410M: throughput falls as the
model grows, and 410M also pays for checkpoint recomputation and shorter
blocks. Benchmark at least 100 optimizer steps with the final block size,
microbatch, accumulation, tokenizer, evaluation, and checkpoint settings.

## Preflight before a long run

1. Tokenize and pack the real corpus; confirm that reported tokens per
   optimizer step equal 32,768.
2. Run 100 optimizer steps including at least one evaluation and save. Record
   peak allocated and reserved VRAM, tokens/second, and loss.
3. Require at least about 0.5 GiB headroom. CUDA kernels, evaluation, and the
   first Adam update can peak at different times.
4. Extrapolate wall time from measured training tokens/second and include
   overhead.
5. Match the cosine learning-rate schedule horizon to the intended token
   budget. The paper identifies this as important when comparing stopping
   points; an intermediate checkpoint from a much longer schedule is not an
   equivalent shorter run.
6. Save by tokens seen, not merely epochs, and include an early checkpoint so
   an unstable learning rate does not waste the full budget.
7. Compare validation loss at equal tokens for model-size experiments.

## Sources

- [Hoffmann et al., *Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556) — primary Chinchilla scaling study.
- [Google DeepMind: empirical analysis of compute-optimal training](https://deepmind.google/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/) — authors' overview of the result.
- [EleutherAI Pythia repository](https://github.com/EleutherAI/pythia) — published sizes, shapes, batch, sequence length, and 300B-token training budget.
- [Pythia-160M training configuration](https://github.com/EleutherAI/pythia/blob/main/models/160M/pythia-160m.yml) — reference architecture and training settings.
- [Alien Ink GPU memory reference](reference-gpu-memory.md) — Mist's parameter, optimizer, activation, and logits budget.
- [Alien Ink RTX 3070 playbook](guide-rtx-3070-training.md) — shared benchmarking and OOM procedure.
