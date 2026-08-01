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

### Zdeck

Each program is a fully explicit `Manifest` (every data / model / hardware /
wandb / schedule field spelled out) plus a thin `main`. W&B identity is set
on the manifest (no package defaults). Swap or retune `hardware` when moving
GPUs; clone a zdeck program when preserving an ablation over time.

```bash
python alien_ink/zdeck/gpt-2_wikipedia_5k.py      # GPT-2, 5k steps, Wikipedia stream
python alien_ink/zdeck/gpt-2_wikitext_5k.py       # GPT-2, 5k steps, WikiText stream
python -m alien_ink.zdeck.gemma_c4_5k            # Gemma (Mist-sized), 5k steps, C4 stream
python -m alien_ink.zdeck.gemma_c4_50k           # Gemma (Mist-sized), 50k steps, C4 stream
python -m alien_ink.zdeck.gemma_wikitext_4ep     # Gemma (Mist-sized), 4 epochs, WikiText complete
python alien_ink/zdeck/gpt-neox_wikitext_4ep.py  # GPT-NeoX, 4 epochs, WikiText complete
python alien_ink/zdeck/gpt-neox_wikitext_baseperf.py  # GPT-NeoX, 0.125 epochs, WikiText complete
```

Or background one with `./bin/pretrain_mist.sh` (edit the script to select a module).

Completions against a checkpoint: `./bin/model-chat-mist.py gpt-2_wikitext_5k`

```python
from alien_ink.zdeck.gemma_c4_5k import MANIFEST

# One-off tweaks via composition; for lasting ablations, copy a zdeck module
MANIFEST.with_schedule(learning_rate=3e-4).variant(run_name="wt-lr3e-4").train()
MANIFEST.with_hardware(per_device_train_batch_size=4, gradient_accumulation_steps=8).train()
```

See `docs/` for data flow, families, corpora, and completion details.
