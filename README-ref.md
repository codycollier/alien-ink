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

W&B entity / project / name are set explicitly in each sample (no package defaults).

```bash
python -m alien_ink.samples.gpt2_wikipedia_5k   # GPT-2, 5k steps, Wikipedia stream
python -m alien_ink.samples.gpt2_wikitext_5k    # GPT-2, 5k steps, WikiText stream
python -m alien_ink.samples.gemma_c4_50k        # Gemma (Mist-sized), 50k steps, C4 stream
```

Or background one with `./bin/pretrain_mist.sh`.
