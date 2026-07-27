
# Overview

Model training configurations and recipes are `experiments`.


## Experiment options

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
# (if upgrades happen, restart session manually)
%pip install -q -U python-dotenv "accelerate>=1.1.0" "datasets>=2.14" "transformers>=4.40" "wandb>=0.16"
%pip install -q --no-deps -U "alien-ink"

import alien_ink

print(alien_ink.stars)
print(alien_ink.device.introspect())
```

```python
# run a training experiment
from alien_ink.exp.gpt2_pretrain_wikipedia_english_subset import train

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
# (if upgrades happen, restart session manually)
%pip install -q -U python-dotenv "accelerate>=1.1.0" "datasets>=2.14" "transformers>=4.40" "wandb>=0.16"
%pip install -q --no-deps -U "alien-ink"

import alien_ink

print(alien_ink.stars)
print(alien_ink.device.introspect())
```

```python
# Run a training experiment
from alien_ink.exp.gpt2_pretrain_wikipedia_english_subset import train

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



## Mist - Running locally (RTX 3070)

Use this path when you have a machine with a local GPU and want to drive
experiments from the shell (or a background `bin/` script). Defaults stay
conservative for an ~8 GB RTX 3070.


```bash
# Setup
./bin/exp-setup.sh
source .venv/bin/activate
```

```bash
# open and edit as needed

# Run a model training / experiment
./bin/gpt2_pretrain_mist.sh
```


### Manual examples

Run short flight checks

```bash
python -m alien_ink.exp.gpt2_pretrain_wikitext --flight-check
python -m alien_ink.exp.gpt2_pretrain_wikipedia_english --flight-check
python -m alien_ink.exp.gpt2_pretrain_c4 --flight-check
python -m alien_ink.exp.gpt2_pretrain_wikitext_subset --flight-check
python -m alien_ink.exp.gpt2_pretrain_wikipedia_english_subset --flight-check
python -m alien_ink.exp.gpt2_pretrain_c4_subset --flight-check
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




---

# Appendix - Customizations and Dev Details


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
