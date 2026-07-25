## alien-ink

[![Project Status: WIP](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![PyPI version](https://badge.fury.io/py/alien-ink.svg)](https://badge.fury.io/py/alien-ink)




...

Software for an audience of one.


## Running experiments

Experiment entrypoints live under `alien_ink/exp/`. Each module supports a short
**flight check** (smoke test) and a **full training** run, plus an optional
spot-check of the newest checkpoint.

| Experiment | Module |
|---|---|
| WikiText-103 | `alien_ink.exp.gpt2_pretrain_wikitext` |
| English Wikipedia | `alien_ink.exp.gpt2_pretrain_wikipedia_english` |
| C4 (English) | `alien_ink.exp.gpt2_pretrain_c4` |

Install with the Hugging Face extras (`pip install -e ".[hf]"`), then run from a
working directory that has (or will create) `.env` and `output/`. Credentials
and artifacts resolve relative to `cwd`.

Optional `.env` keys are listed in `.env.example` (`HF_TOKEN` /
`HUGGING_FACE_HUB_TOKEN`, `WANDB_API_KEY`). Copy it to `.env` and fill in values.

### Local (CLI)

Flight check (fast end-to-end smoke test):

```bash
python -m alien_ink.exp.gpt2_pretrain_wikitext --flight-check
```

Full training run:

```bash
python -m alien_ink.exp.gpt2_pretrain_wikitext --train
```

Spot-check completions from the latest saved checkpoint:

```bash
python -m alien_ink.exp.gpt2_pretrain_wikitext --spot-check
```

Swap the module name for Wikipedia or C4. Mode flags (`--train`,
`--flight-check`, `--spot-check`) are mutually exclusive.

Background Wikipedia pretrain with a timestamped log (see `bin/`):

```bash
./bin/gpt2_pretrain_wikipedia_english.sh
```

### Weights & Biases (project / run name)

Set project and run name with CLI flags or function kwargs only (not via
environment variables). Defaults: project `alien-ink`, run name from the
experiment config.

```bash
python -m alien_ink.exp.gpt2_pretrain_wikitext --train \
  --wandb-project my-proj --wandb-name my-run

./bin/gpt2_pretrain_wikipedia_english.sh \
  --wandb-project my-proj --wandb-name my-run
```

### Notebook / REPL

```python
from alien_ink.exp.gpt2_pretrain_wikitext import train, train_flight_check, spot_check

train_flight_check()  # smoke test
# train()             # full run
# spot_check()        # sample from newest checkpoint

train(
    wandb_project="my-proj",
    wandb_name="my-run",
)
```

Same pattern for the other modules (`gpt2_pretrain_wikipedia_english`,
`gpt2_pretrain_c4`). Ensure the notebook kernel’s working directory is where you
want `.env` and `output/` to live.
