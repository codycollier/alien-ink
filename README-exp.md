## Running experiments

Experiment entrypoints live under `alien_ink/exp/`. Each module supports a short
**flight check** (smoke test) and a **full training** run, plus an optional
spot-check of the newest checkpoint.

| Experiment | Module |
|---|---|
| WikiText-103 | `alien_ink.exp.gpt2_pretrain_wikitext` |
| WikiText-103 (20k subset) | `alien_ink.exp.gpt2_pretrain_wikitext_subset` |
| English Wikipedia | `alien_ink.exp.gpt2_pretrain_wikipedia_english` |
| English Wikipedia (20k subset) | `alien_ink.exp.gpt2_pretrain_wikipedia_english_subset` |
| C4 (English) | `alien_ink.exp.gpt2_pretrain_c4` |
| C4 English (20k subset) | `alien_ink.exp.gpt2_pretrain_c4_subset` |

Subset experiments materialize a small prefix (~20k train + 1k eval) instead of
streaming the full corpus, and use shorter default training (`max_steps=2000`).

Mode flags (`--train`, `--flight-check`, `--spot-check`) are mutually exclusive.
Flight-check W&B / Trainer run names are `{run_name}-flight-check` (for example
`gpt2-pretrain-wikitext-flight-check`).

Credentials and artifacts resolve relative to `cwd` at call time (`.env`,
`output/`). Optional `.env` keys are listed in `.env.example`.

## Running locally with a GPU

Use this path when you have a machine with a local GPU and want to drive
experiments from the shell (or a background `bin/` script).

### Setup

```bash
uv venv
source .venv/bin/activate
# PyPI torch is CUDA 13.0; cu126 works with CUDA 12.x drivers (e.g. 12.2).
UV_TORCH_BACKEND=cu126 uv pip install -e ".[hf]"
cp .env.example .env   # then fill in HF_TOKEN / WANDB_API_KEY
```

Or:

```bash
./bin/exp-setup.sh
cp .env.example .env   # then fill in HF_TOKEN / WANDB_API_KEY
```

### CLI

Flight check (fast end-to-end smoke test):

```bash
python -m alien_ink.exp.gpt2_pretrain_wikitext --flight-check
python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --flight-check
python -m alien_ink.exp.gpt2_pretrain_c4 --flight-check
python -m alien_ink.exp.gpt2_pretrain_wikitext_subset --flight-check
python -m alien_ink.exp.gpt2_pretrain_wikipedia_english_subset --flight-check
python -m alien_ink.exp.gpt2_pretrain_c4_subset --flight-check
```

Full training run:

```bash
python -m alien_ink.exp.gpt2_pretrain_wikitext --train
python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --train
python -m alien_ink.exp.gpt2_pretrain_c4 --train
python -m alien_ink.exp.gpt2_pretrain_wikitext_subset --train
python -m alien_ink.exp.gpt2_pretrain_wikipedia_english_subset --train
python -m alien_ink.exp.gpt2_pretrain_c4_subset --train
```

Spot-check completions from the latest saved checkpoint:

```bash
python -m alien_ink.exp.gpt2_pretrain_wikitext --spot-check
python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --spot-check
python -m alien_ink.exp.gpt2_pretrain_c4 --spot-check
python -m alien_ink.exp.gpt2_pretrain_wikitext_subset --spot-check
python -m alien_ink.exp.gpt2_pretrain_wikipedia_english_subset --spot-check
python -m alien_ink.exp.gpt2_pretrain_c4_subset --spot-check
```

Background pretrain with a timestamped log (see `bin/`):

```bash
./bin/gpt2_pretrain_wikitext.sh
./bin/gpt2_pretrain_wikipedia_english.sh
./bin/gpt2_pretrain_c4.sh
./bin/gpt2_pretrain_wikitext_subset.sh
./bin/gpt2_pretrain_wikipedia_english_subset.sh
./bin/gpt2_pretrain_c4_subset.sh
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
python -m alien_ink.exp.gpt2_pretrain_wikipedia_english_subset --train \
  --wandb-entity logbook --wandb-project ink-explore \
  --wandb-name gpt2-pretrain-wpe-subset

python -m alien_ink.exp.gpt2_pretrain_wikitext --flight-check --no-wandb
python -m alien_ink.exp.gpt2_pretrain_wikipedia_english_subset --flight-check --no-wandb

./bin/gpt2_pretrain_wikitext.sh
./bin/gpt2_pretrain_wikipedia_english_subset.sh
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

### Run artifacts (repro + speed comparison)

Each training run writes two JSON files under the run `output_dir`
(for example `output/gpt2-pretrain-wikitext/`):

| File | When | Contents |
|---|---|---|
| `run_config.json` | before training | Fully resolved recipe (data / arch / trainer), accelerator fingerprint (GPU name, VRAM, CUDA, precision, world size), software versions, tokens/step |
| `run_summary.json` | when training stops | Steps, tokens trained, wall time, tokens/sec, estimated TFLOP/s + MFU (Kaplan 6N), train loss, status (`completed` / `interrupted` / `failed`) |

The same summary fields are pushed to the W&B run summary when W&B is enabled.
Use identical recipe settings (especially `block_size`, batch size, and
`gradient_accumulation_steps`) across machines to compare GPU speed fairly;
`tokens_per_sec` and `tflops_per_sec` are the main cross-hardware metrics.

### Tests

```bash
UV_TORCH_BACKEND=cu126 uv pip install -e ".[hf,test]"
pytest
```

## Running in a notebook (Google Colab)

Ensure secrets from .env are set as notebook secrets (key icon to left).

### Install (GPU / CPU)

```python
%pip install -q -U "alien-ink[hf]"
```

```
import alien_ink

print(alien_ink.stars)
```

### Install (TPU)

1. Runtime → Change runtime type → **TPU**.
2. Keep the runtime's PyTorch/XLA stack; install alien-ink deps without replacing
   `torch` / `torch_xla`:

```python
%pip install -q python-dotenv "accelerate>=1.1.0" "datasets>=2.14" "transformers>=4.40" "wandb>=0.16"
%pip install -q --no-deps "alien-ink"
```

If `torch_xla` is missing on the runtime, install a matching pair first
(see [PyTorch/XLA](https://docs.pytorch.org/xla/master/)):

```python
%pip install -q torch torch_xla[tpu]
```

Verify:

```python
import torch
import torch_xla
import torch_xla.runtime as xr
from alien_ink.device import resolve_device

print("torch", torch.__version__, "xla", torch_xla.__version__)
print("device_type", xr.device_type(), "→", resolve_device())
```

Expect `resolve_device()` → `xla`.

### Call the experiment functions

```python
# from alien_ink.exp.gpt2_pretrain_wikipedia_english_subset import (
from alien_ink.exp.gpt2_pretrain_wikitext import train, train_flight_check, spot_check


# train(
train_flight_check(
    use_wandb=True,
    wandb_entity="logbook",
    wandb_project="ink-explore",
    wandb_name="my-run",
)

```

On a TPU notebook, `train` / `train_flight_check` auto-wrap with Accelerate's
`notebook_launcher` (multi-core). Override with `tpu_launch=False` or set
`tpu_num_processes=N`. That path returns `(None, None)`; check
`output/.../run_summary.json` for metrics.

Gradient checkpointing is auto-disabled on XLA/TPU: PyTorch's native
`torch.utils.checkpoint` crashes with `module 'torch' has no attribute 'xla'`.
TPU HBM is usually enough for GPT-2 small without it.

### CLI on a Cloud TPU VM

Launch multi-core training with XLA spawn (do not run bare `python -m` if you
want all cores):

```bash
python -m torch_xla.distributed.xla_spawn --num_cores=8 \
  -m alien_ink.exp.gpt2_pretrain_wikitext --flight-check
```
