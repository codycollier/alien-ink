## alien-ink

[![Project Status: WIP](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![PyPI version](https://badge.fury.io/py/alien-ink.svg)](https://badge.fury.io/py/alien-ink)

Local from-scratch language model pretraining for **Mist** (RTX 3070, ~8 GB).

Supports GPT-2, GPT-NeoX, and Gemma architectures, plus English Wikipedia,
WikiText-103, and C4 datasets in streaming, materialized subset, or fully
materialized modes. Families and corpora are registry-based and extendable.


## Setup (Mist)

```bash
./bin/setup.sh
source .venv/bin/activate
cp .env.example .env   # add HF_TOKEN / WANDB_API_KEY as needed
```


## Samples

Two runnable sample programs ship under `alien_ink.samples`:

| Sample | Steps | Data |
|---|---|---|
| `alien_ink.samples.pretrain_wikipedia_5k` | 5 000 | streamed English Wikipedia |
| `alien_ink.samples.pretrain_wikitext_50k` | 50 000 | streamed WikiText-103 |

```bash
python -m alien_ink.samples.pretrain_wikipedia_5k
python -m alien_ink.samples.pretrain_wikitext_50k --wandb
```

Batch defaults are Mist-oriented: microbatch `2`, gradient accumulation `16`
(effective batch 32) at `block_size=1024`.


## Library sketch

```python
from pathlib import Path
from alien_ink.hf.ds import wikipedia_english, wikitext_103_subset, c4_english
from alien_ink.hf.model import ModelArchConfig
from alien_ink.hf.pretrain import PretrainConfig, pretrain
from alien_ink.hf.trainer import CausalLmTrainerConfig

cfg = PretrainConfig(
    data=wikipedia_english(load_mode="streaming"),  # or "subset" / "complete"
    arch=ModelArchConfig(family="gpt2"),            # or "gpt_neox" / "gemma"
    trainer=CausalLmTrainerConfig(
        output_dir=Path("output/my-run"),
        run_name="my-run",
        max_steps=5_000,
    ),
)
pretrain(cfg, use_wandb=False)
```

Dataset helpers:

- Streaming: `wikipedia_english()`, `wikitext_103()`, `c4_english()`
- Subset: `*_subset()` (default 20k train + 1k eval)
- Complete: `*_complete()` (full non-streaming Hub load)

Register more corpora with `alien_ink.hf.ds.register_dataset`, and more model
families with `alien_ink.hf.model.register_model_family`.


## Tests

```bash
uv pip install -e ".[hf,test]"
pytest
```
