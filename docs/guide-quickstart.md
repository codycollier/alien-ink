# Quickstart

The daily commands: setup, running zdeck programs, completions, and evals.

Alien Ink does local GPU training on Mist (RTX 3070, ~8 GB) in two stages:
from-scratch pretraining (`pre`) of **GPT-2**, **GPT-NeoX**, **Pythia**,
**Llama/SmolLM2**, and **Gemma** architectures, and full-parameter fine-tuning
(`sft`) of pretrained checkpoints. Corpora: WikiText-103, English Wikipedia,
C4, and geo-us-states — as a stream, a materialized subset, or a complete
materialized split.

## Setup

```bash
./bin/setup.sh
source .venv/bin/activate
cp -n .env.example .env   # set HF_TOKEN and WANDB_API_KEY
```

## Zdeck

Each program is a fully explicit `Manifest` (every data / model / hardware /
wandb / schedule field spelled out, including `stage`) plus a thin `main`.
Filenames mirror `run_name` (underscores vs hyphens), including the host
token — e.g. `pre_gemma_c4_5k_mist.py` ↔ `pre-gemma-c4-5k-mist`. W&B identity
is set on the manifest (no package defaults). Swap or retune `hardware` when
moving GPUs; clone a zdeck program when preserving an ablation over time.

Representative programs (see `alien_ink/zdeck/` for the full deck):

```bash
python alien_ink/zdeck/pre_gpt-2_wikitext_5k_mist.py        # GPT-2, 5k steps, WikiText stream
python -m alien_ink.zdeck.pre_gemma_c4_50k_mist             # Gemma (Mist-sized), 50k steps, C4 stream
python alien_ink/zdeck/pre_pythia-160m_wikitext_4ep_mist.py # Pythia-160M, 4 epochs, WikiText complete
python alien_ink/zdeck/pre_smollm2-135m_wikitext_4ep_mist.py # SmolLM2-135M, 4 epochs, WikiText complete
python alien_ink/zdeck/pre_gpt-neox_curriculum_geo_mist.py  # GPT-NeoX, curriculum: C4 then geo-us-states
python alien_ink/zdeck/sft_pythia-160m_geo_mist.py          # SFT pythia-160m on geo-us-states
./bin/mist-train-baseline-perf.sh gpt-2                    # comparable 0.25-epoch perf baselines
./bin/mist-train-baseline-perf.sh gpt-neox
./bin/mist-train-baseline-perf.sh pythia                   # Pythia-160M
./bin/mist-train-baseline-perf.sh gpt-2-tinygrad           # same GPT-2 knobs, tinygrad backend
```

Or background one with `./bin/pretrain-mist.sh` (edit the script to select a
module). Checkpoints land under `output/train/<run_name>/`.

## Completions and evals

Interactive completions against a checkpoint:

```bash
./bin/model-chat-mist.py pre_gpt-2_wikitext_5k_mist
```

Score a checkpoint against an external eval JSON (prompt/completion pairs):

```bash
./bin/model-eval-mist.py sft_pythia-160m_geo_mist --evals /path/to/eval.json
```

## Manifest composition

```python
from alien_ink.zdeck.pre_gemma_c4_5k_mist import MANIFEST

# One-off tweaks via composition; for lasting ablations, copy a zdeck module
MANIFEST.with_schedule(learning_rate=3e-4).variant(run_name="pre-wt-lr3e-4-mist").train()
MANIFEST.with_hardware(per_device_train_batch_size=2, gradient_accumulation_steps=16).train()
```

See the [pretraining reference](reference-pretraining.md) for data flow and
manifests, [model families](reference-model-families.md) for architectures,
[datasets](reference-datasets.md) for corpora, and
[completions and evals](reference-completions-and-eval.md) for generation and
scoring details.
