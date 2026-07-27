
# Overview


## Running Experiments - General

| Experiment | Module |
|---|---|
| WikiText-103 | `alien_ink.exp.gpt2_pretrain_wikitext` |
| WikiText-103 (20k subset) | `alien_ink.exp.gpt2_pretrain_wikitext_subset` |
| English Wikipedia | `alien_ink.exp.gpt2_pretrain_wikipedia_english` |
| English Wikipedia (20k subset) | `alien_ink.exp.gpt2_pretrain_wikipedia_english_subset` |
| C4 (English) | `alien_ink.exp.gpt2_pretrain_c4` |
| C4 English (20k subset) | `alien_ink.exp.gpt2_pretrain_c4_subset` |


- Experiment entrypoints live under `alien_ink/exp/`
- Each module supports: `--flight-check`, `--train`, `--spot-check`
- Subset experiments materialize a small prefix (~20k train + 1k eval; 3 epochs)
- Non-subset experiments stream the corpus and run for max steps
- Log and eval steps are dynamically configured as a convenience


## Accelerator profiles (batch + run names)

| Environment | Assumed hardware | Train batch / accum | Run suffix |
|---|---|---|---|
| Local CUDA (≤12 GB) | RTX (e.g. 3070 8 GB) | `2` / `16` | `-gpu` |
| Colab GPU / L4 / ≥20 GB | Colab **G4** (NVIDIA L4 ~24 GB) | `32` / `1` | `-gpu` |
| XLA/TPU | Colab **TPU v6e-1** (1×32 GB HBM) | `64` / `1` | `-tpu` |


- Flight checks stay tiny (`batch=1`, `accum=1`, short `block_size`) and use `{run_name}-flight-check-{gpu|tpu}`.
- Override batch/accum anytime via kwargs or CLI (`--per-device-train-batch-size`, `--gradient-accumulation-steps`).


---

# Quick References


## Google Colab - Runtime: GPU (default: G4)

- Set notebook secrets (key icon) to match `.env`: `HF_TOKEN`, `WANDB_API_KEY`.
- **Runtime** → Colab **G4** (NVIDIA L4)
- Keep Colab's CUDA `torch`; install HF deps + `alien-ink` with `--no-deps`

```python
# install, setup, verification
%pip install -q -U python-dotenv "accelerate>=1.1.0" "datasets>=2.14" "transformers>=4.40" "wandb>=0.16"
%pip install -q --no-deps -U "alien-ink"

import torch
import alien_ink
from alien_ink.hf.hardware import resolve_accelerator_profile

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print(alien_ink.stars)
print(resolve_accelerator_profile())  # label=colab-g4, batch=32, accum=1
```

```python
# run an experiment (use train_flight_check(...) for a smoke test)
from alien_ink.exp.gpt2_pretrain_wikipedia_english_subset import (
    train,
    train_flight_check,
    spot_check,
)

train(
    use_wandb=True,
    wandb_entity="logbook",
    wandb_project="ink-explore",
    wandb_name="gpt2-pretrain-wpe-subset-nb-gpu",
)
```


## Google Colab - Runtime: TPU (default: v6e1)

- Set notebook secrets (key icon) to match `.env`: `HF_TOKEN`, `WANDB_API_KEY`.
- **Runtime** → Colab **TPU v6e-1** (single chip)
- Keep the runtime's PyTorch/XLA stack; do not replace `torch` / `torch_xla`

```python
# install, setup, verification
%pip install -q -U python-dotenv "accelerate>=1.1.0" "datasets>=2.14" "transformers>=4.40" "wandb>=0.16"
%pip install -q --no-deps -U "alien-ink"

import torch
import torch_xla
import torch_xla.runtime as xr
from alien_ink.device import resolve_device
from alien_ink.hf.hardware import resolve_accelerator_profile

print("torch", torch.__version__, "xla", torch_xla.__version__)
print("device_type", xr.device_type(), "→", resolve_device())
print(resolve_accelerator_profile())  # label=colab-tpu-v6e1, batch=64, accum=1
# Expect resolve_device() → xla and profile.tpu_num_processes == 1
```

```python
# run an experiment (use train_flight_check(...) for a smoke test)
# Auto-wraps with Accelerate notebook_launcher (1 process on v6e-1).
# Returns (None, None); metrics in output/...-tpu/run_summary.json
from alien_ink.exp.gpt2_pretrain_wikipedia_english_subset import (
    train,
    train_flight_check,
    spot_check,
)

train(
    use_wandb=True,
    wandb_entity="logbook",
    wandb_project="ink-explore",
    wandb_name="gpt2-pretrain-wpe-subset-nb-tpu",
    # tpu_num_processes=1,  # default for v6e-1; set N for multi-chip
    # tpu_launch=False,     # skip notebook_launcher (debug / already launched)
)
```

XLA notes (applied automatically):

- Gradient checkpointing off (native torch checkpoint breaks on XLA)
- Optimizer forced to `adamw_torch` (fused AdamW rejects XLA)
- Prefer large `per_device_train_batch_size` and `gradient_accumulation_steps=1`



## Running locally with a GPU (mist)

Use this path when you have a machine with a local GPU and want to drive
experiments from the shell (or a background `bin/` script). Defaults stay
conservative for an ~8 GB RTX 3070.


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


### Run

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

Background pretrain on Mist (local RTX 3070) with a timestamped log. Edit
`bin/gpt2_pretrain_mist.sh` to uncomment the mode (`--train` /
`--flight-check`) and dataset, then:

```bash
./bin/gpt2_pretrain_mist.sh
```



---

# Various customization options and dev details


### Weights & Biases (entity / project / run name)

W&B layout is organization → team (entity) → project → runs. Set entity,
project, and run name with CLI flags or function kwargs only (not via
environment variables). Defaults: entity `logbook`, project `ink-explore`,
run name from the experiment config **plus** `-gpu` / `-tpu`. Pass
`--no-wandb` (or `use_wandb=False`) to skip W&B entirely.

```bash
python -m alien_ink.exp.gpt2_pretrain_wikitext --train \
  --wandb-entity logbook --wandb-project ink-explore --wandb-name my-run
# → W&B / output name: my-run-gpu (on CUDA)

python -m alien_ink.exp.gpt2_pretrain_wikipedia_english_subset --train \
  --wandb-entity logbook --wandb-project ink-explore

python -m alien_ink.exp.gpt2_pretrain_wikitext --flight-check --no-wandb
python -m alien_ink.exp.gpt2_pretrain_wikipedia_english_subset --flight-check --no-wandb
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
```

`--resume-from-checkpoint` alone auto-picks the latest checkpoint under the run
`output_dir`; pass a path to resume from a specific directory.

### Run artifacts (repro + speed comparison)

Each training run writes two JSON files under the run `output_dir`
(for example `output/gpt2-pretrain-wikitext-gpu/`):

| File | When | Contents |
|---|---|---|
| `run_config.json` | before training | Fully resolved recipe (data / arch / trainer), accelerator fingerprint (GPU/TPU name, VRAM, CUDA, precision, world size), software versions, tokens/step |
| `run_summary.json` | when training stops | Steps, tokens trained, wall time, tokens/sec, estimated TFLOP/s + MFU (Kaplan 6N), train loss, status (`completed` / `interrupted` / `failed`) |

The same summary fields are pushed to the W&B run summary when W&B is enabled.
`tokens_per_sec` and `tflops_per_sec` are the main cross-hardware metrics; note
batch/accum differ by profile, so compare utilization (MFU) carefully.

### Tests

```bash
UV_TORCH_BACKEND=cu126 uv pip install -e ".[hf,test]"
pytest
```


---
