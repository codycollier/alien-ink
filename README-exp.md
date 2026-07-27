
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
- Log and eval steps are dynamically configured as a convenience (subsets: ~5 evals per epoch, including epoch end)
- Ablations compose from a base `EXPERIMENT` — do not clone modules for LR / depth / block_size grids

```python
from alien_ink.exp.gpt2_pretrain_wikitext_subset import EXPERIMENT

# Hundreds of combinations = product of variants, not new files
ablations = [
    EXPERIMENT.variant(run_name="wt-sub-lr3e4", learning_rate=3e-4),
    EXPERIMENT.with_arch(n_layer=6).variant(run_name="wt-sub-l6"),
    EXPERIMENT.with_data(block_size=512).with_trainer_knobs(weight_decay=0.05),
]
for exp in ablations:
    exp.train(use_wandb=True)
```


## Training machine profiles (batch + run names)

Recipes pick an `AcceleratorProfile` from `alien_ink.hf.hardware` based on the
live device. Multiples are relative to the Mist **RTX 3070** baseline
(8 GB, 40.6 TFLOPS FP16). Cloud profiles prefer large microbatches and
`gradient_accumulation_steps=1` so HBM/VRAM stays busy.

| Profile | Hardware | Mem | Peak | vs 3070 | Mem× | Train batch / accum | Eff. batch | Suffix |
|---|---|---|---|---|---|---|---|---|
| `local-rtx` | RTX 3070 (Mist) | 8 GB | 40.6 TFLOPS FP16 | 1.0× | 1× | `2` / `16` | 32 | `-gpu` |
| `colab-g4` | **G4** RTX PRO 6000 Blackwell | 96 GB | 500 TFLOPS FP16 | 12.3× | 12× | `64` / `1` | 64 | `-gpu` |
| `colab-a100-40gb` | **A100** 40 GB (Ampere) | 40 GB | 312 TFLOPS FP16 | 7.7× | 5× | `32` / `1` | 32 | `-gpu` |
| `colab-tpu-v6e1` | **TPU v6e-1** Trillium (1-chip) | 32 GB | 918 TFLOPS BF16 | 22.6× | 4× | `32` / `1` | 32 | `-tpu` |
| `colab-l4` | L4 / other ≥20 GB mid-tier | 24 GB | 121 TFLOPS FP16 | 3.0× | 3× | `32` / `1` | 32 | `-gpu` |

Scaling notes:

- **Local / Mist** keeps the proven small-microbatch + accum recipe (fits ~8 GB).
- **G4** microbatch is raised with the 12× memory headroom (64 vs local’s effective 32). Drop to `32` if you want local-parity tokens/step.
- **A100 40 GB** uses batch 32 (40 GB is under the ~51G needed for batch 64 @ 1024).
- **TPU v6e-1** stays at 32: batch 64 OOMs GPT-2 @ `block_size=1024` (~51G HBM).
- **L4** is auto-selected for mid-VRAM GPUs so they never inherit the G4 96 GB recipe.
- Flight checks stay tiny (`batch=1`, `accum=1`, short `block_size`) and use `{run_name}-flight-check-{gpu|tpu}`.
- Override batch/accum anytime via kwargs or CLI (`--per-device-train-batch-size`, `--gradient-accumulation-steps`).



---

# Quick References


## Google Colab - Runtime: GPU (default: G4)

- Set notebook secrets (key icon) to match `.env`: `HF_TOKEN`, `WANDB_API_KEY`.
- **Runtime** → Colab **G4** (RTX PRO 6000 Blackwell, 96 GB). A100 40 GB and L4 map to their own profiles.
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


## Google Colab - Runtime: TPU (default: v6e-1)

- Set notebook secrets (key icon) to match `.env`: `HF_TOKEN`, `WANDB_API_KEY`.
- **Runtime** → Colab **TPU v6e-1** Trillium (single chip, 32 GB HBM)
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
experiments from the shell (or a background `bin/` script). Resolves to the
`local-rtx` profile — conservative for an ~8 GB RTX 3070 (`batch=2`, `accum=16`).


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
