## alien-ink - references for usage

Local GPU pretraining (Mist / RTX 3070, ~8 GB) for small language models from
scratch: **GPT-2**, **GPT-NeoX**, and **Gemma**. Datasets: English Wikipedia,
WikiText-103, and C4 — as a stream, a materialized subset, or a complete
materialized split.

### Setup

```bash
./bin/setup.sh
source .venv/bin/activate
cp -n .env.example .env   # set HF_TOKEN and WANDB_API_KEY
```

### Samples

Each sample is a fully explicit `Recipe` (every data / model / hardware /
wandb / schedule field spelled out) plus a thin `main`. W&B identity is set
on the recipe (no package defaults). Swap or retune `hardware` when moving
GPUs; clone a sample when preserving an ablation over time.

```bash
python -m alien_ink.samples.gpt2_wikipedia_5k   # GPT-2, 5k steps, Wikipedia stream
python -m alien_ink.samples.gpt2_wikitext_5k    # GPT-2, 5k steps, WikiText stream
python -m alien_ink.samples.gemma_c4_50k        # Gemma (Mist-sized), 50k steps, C4 stream
```

Or background one with `./bin/pretrain_mist.sh`.

```python
from alien_ink.samples.gpt2_wikitext_5k import RECIPE

# One-off tweaks via composition; for lasting ablations, copy a sample module
RECIPE.with_schedule(learning_rate=3e-4).variant(run_name="wt-lr3e-4").train()
RECIPE.with_hardware(per_device_train_batch_size=4, gradient_accumulation_steps=8).train()
```
