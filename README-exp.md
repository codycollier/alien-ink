## Running experiments

Experiment entrypoints live under `alien_ink/exp/`. Each module supports a short
**flight check** (smoke test) and a **full training** run, plus an optional
spot-check of the newest checkpoint.

| Experiment | Module |
|---|---|
| WikiText-103 | `alien_ink.exp.gpt2_pretrain_wikitext` |
| English Wikipedia | `alien_ink.exp.gpt2_pretrain_wikipedia_english` |
| C4 (English) | `alien_ink.exp.gpt2_pretrain_c4` |

Mode flags (`--train`, `--flight-check`, `--spot-check`) are mutually exclusive.
Flight-check W&B / Trainer run names are `{run_name}-flight-check` (for example
`gpt2-pretrain-wikitext-flight-check`).

Credentials and artifacts resolve relative to `cwd` at call time (`.env`,
`output/`). Optional `.env` keys are listed in `.env.example`.

### Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[hf]"
cp .env.example .env   # then fill in HF_TOKEN / WANDB_API_KEY
```

Or:

```bash
./bin/exp-setup.sh
cp .env.example .env   # then fill in HF_TOKEN / WANDB_API_KEY
```

### Local (CLI)

Flight check (fast end-to-end smoke test):

```bash
python -m alien_ink.exp.gpt2_pretrain_wikitext --flight-check
python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --flight-check
python -m alien_ink.exp.gpt2_pretrain_c4 --flight-check
```

Full training run:

```bash
python -m alien_ink.exp.gpt2_pretrain_wikitext --train
python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --train
python -m alien_ink.exp.gpt2_pretrain_c4 --train
```

Spot-check completions from the latest saved checkpoint:

```bash
python -m alien_ink.exp.gpt2_pretrain_wikitext --spot-check
python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --spot-check
python -m alien_ink.exp.gpt2_pretrain_c4 --spot-check
```

Background Wikipedia pretrain with a timestamped log (see `bin/`):

```bash
./bin/gpt2_pretrain_wikipedia_english.sh
```

### Weights & Biases (entity / project / run name)

W&B layout is organization → team (entity) → project → runs. Set entity,
project, and run name with CLI flags or function kwargs only (not via
environment variables). Defaults: entity `logbook`, project `ink-explore`,
run name from the experiment config. Pass `--no-wandb` (or `use_wandb=False`)
to skip W&B entirely.

```bash
python -m alien_ink.exp.gpt2_pretrain_wikitext --train \
  --wandb-entity logbook --wandb-project ink-explore --wandb-name my-run
python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --train \
  --wandb-entity logbook --wandb-project ink-explore --wandb-name my-run
python -m alien_ink.exp.gpt2_pretrain_c4 --train \
  --wandb-entity logbook --wandb-project ink-explore --wandb-name my-run

python -m alien_ink.exp.gpt2_pretrain_wikitext --flight-check --no-wandb
python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --flight-check --no-wandb
python -m alien_ink.exp.gpt2_pretrain_c4 --flight-check --no-wandb

./bin/gpt2_pretrain_wikipedia_english.sh
```

### Training overrides / resume

Optional CLI knobs for `--train` and `--flight-check`:

```bash
python -m alien_ink.exp.gpt2_pretrain_wikitext --train \
  --max-steps 1000 \
  --learning-rate 3e-4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 32 \
  --resume-from-checkpoint

python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --train \
  --max-steps 1000 \
  --learning-rate 3e-4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 32 \
  --resume-from-checkpoint

python -m alien_ink.exp.gpt2_pretrain_c4 --train \
  --max-steps 1000 \
  --learning-rate 3e-4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 32 \
  --resume-from-checkpoint
```

`--resume-from-checkpoint` alone auto-picks the latest checkpoint under the run
`output_dir`; pass a path to resume from a specific directory.

### Notebook / REPL

```python
from alien_ink.exp.gpt2_pretrain_wikitext import train, train_flight_check, spot_check
# from alien_ink.exp.gpt2_pretrain_wikipedia_english import train, train_flight_check, spot_check
# from alien_ink.exp.gpt2_pretrain_c4 import train, train_flight_check, spot_check

train_flight_check(use_wandb=False)  # smoke test
# train(resume_from_checkpoint=True)
# spot_check()

train(
    wandb_entity="logbook",
    wandb_project="ink-explore",
    wandb_name="my-run",
)
```

Ensure the notebook kernel’s working directory is where you want `.env` and
`output/` to live (paths are resolved when you call the functions, not at
import time).

### Tests

```bash
uv pip install -e ".[hf,test]"
pytest
```
