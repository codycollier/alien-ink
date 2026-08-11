# Pretraining

Alien Ink trains **base causal language models** from scratch on raw text
corpora. There is no chat template, no system prompt, and no instruction
format — the model learns to predict the next token in ordinary prose. This
document covers how training data is configured, how raw text becomes
training blocks, and how manifests and the zdeck turn all of it into
reproducible programs.

Completions and post-training evals live in
[completions and evals](reference-completions-and-eval.md); corpus details in
[datasets](reference-datasets.md); architectures in
[model families](reference-model-families.md).

---

## The pipeline at a glance

```
Hub corpus (text rows)
        │
        ▼
  PretrainDataConfig     ← what to load, how much, block size
        │
        ▼
  tokenize + chunk       ← fixed-length blocks (labels = input_ids)
        │
        ▼
  CausalLmArchConfig     ← family, tokenizer, architecture
        │
        ▼
  train → output/train/<run_name>/
        │
        ▼
  generate_completion    ← family-aware GenConfig from manifest
```

Every training run is described by a **manifest** in the zdeck
(`alien_ink/zdeck/`). The manifest binds a dataset config, a model family,
hardware knobs, and schedule into one reproducible program.

---

## Training data configuration

Training data is configured with two dataclasses in `alien_ink.hf.ds`.

### `HubTextSource`

Identifies a text split on the Hugging Face Hub:

| Field | Meaning |
|---|---|
| `dataset` | Hub dataset id (e.g. `Salesforce/wikitext`) |
| `name` | Config name, if the dataset has variants (e.g. `wikitext-103-v1`, `en`) |
| `split` | Split to read (`train`, `validation`, …) |
| `text_column` | Column containing raw text (almost always `"text"`) |

### `PretrainDataConfig`

Controls loading, eval hold-out, and token packing:

| Field | Typical value | Notes |
|---|---|---|
| `source` | `HubTextSource(...)` | Training corpus |
| `eval_source` | optional second `HubTextSource` | Dedicated eval split; if omitted, N deterministically shuffled train rows are held out |
| `mode` | `"stream"` | See load modes below |
| `max_eval_samples` | `1000` | Eval is always a bounded map-style dataset |
| `max_train_samples` | `None` for stream | Required for `subset` mode |
| `stream_shuffle_buffer` | `10000` | Shuffle buffer when streaming |
| `block_size` | `1024` | Tokens per training example; must be ≤ `model.n_positions` |
| `respect_document_boundaries` | `True` | Chunk each Hub row independently; set `False` to pack across rows (needed for short-row corpora like WikiText) |
| `tokenizer_num_proc` | `4` | Parallel workers for map-style preprocessing |
| `seed` | `101` | Shuffle / hold-out reproducibility |

### Load modes

| Mode | Train data | When to use |
|---|---|---|
| `stream` | `IterableDataset` — never fully downloaded | Default for zdeck runs on Mist; scales to large corpora |
| `subset` | Materialized prefix of N rows | Fast experiments, CI, debugging |
| `complete` | Full split downloaded | Smaller corpora (e.g. WikiText-103 entire train split) |

Factory helpers (`wikitext_103()`, `c4_english()`, `wikipedia_english()`,
`geo_us_states()`, and the generic `hub_text()`) return ready-made
`PretrainDataConfig` objects with sensible defaults. A `Curriculum`
(`alien_ink.hf.curriculum`) sequences several configs into one run; see the
[curricula section of the datasets reference](reference-datasets.md#curricula).

All zdeck programs today read a single **`text` column** per row — metadata
columns (titles, URLs, timestamps) are dropped during tokenization. Corpus
sizes, character, and Mist-scale planning live in
[datasets](reference-datasets.md).

---

## From raw text to training blocks

Preprocessing is **identical across model families** — only the tokenizer
differs.

1. **Load** rows via `load_train_eval(data)`.
2. **Tokenize** without tokenizer-added special tokens, then append one EOS token to every document.
3. **Chunk** token streams into contiguous blocks of `block_size` tokens.
4. **Labels** are a copy of `input_ids` (standard causal LM: predict token *t* from tokens *0…t−1*).

By default (`respect_document_boundaries=True`):

- Each Hub row is chunked **independently** — a training block never spans two documents.
- Long documents become multiple consecutive blocks; a per-document remainder shorter than `block_size` is dropped.
- Rows shorter than `block_size` produce **no** blocks (they are skipped). Prefer document-sized rows (articles, pages) for this mode.

```
  doc A (3000 tok) ──► [1024] [1024]   (remainder dropped)
  doc B (500 tok)  ──► (skipped)
  doc C (2048 tok) ──► [1024] [1024]
```

With `respect_document_boundaries=False` (classic packing — used by WikiText
factories/zdecks):

- EOS-delimited token streams are **concatenated** across rows and map batches, then sliced.
- Document boundaries are **not** respected — a block may span two articles or lines.
- Short rows pack with neighbors; remainders carry into the next map batch. Only the final incomplete worker-shard remainder is dropped.

```
  doc A tokens ──┐
  doc B tokens ──┼──► [1024] [1024] [1024] …
  doc C tokens ──┘
```

Every document ends in exactly one EOS token. Tokenizer-added BOS/EOS tokens
are disabled during preprocessing, preventing both unmarked joins and repeated
BOS tokens inside packed blocks.

---

## Model families

Architecture and tokenizer are set in `CausalLmArchConfig`
(`alien_ink.hf.model`); every family initializes with random weights. See
[model families](reference-model-families.md) for full architecture,
tokenizer, and VRAM detail.

| Family | HF model class | Default tokenizer | Zdeck dims |
|---|---|---|---|
| `gpt-2` | `GPT2LMHeadModel` | `gpt2` | 12 layers, 768 embd, 12 heads, 1024 positions |
| `gpt-neox` | `GPTNeoXForCausalLM` | `EleutherAI/gpt-neox-20b` | same as GPT-2 |
| `pythia` | `GPTNeoXForCausalLM` | `EleutherAI/pythia-70m` / `-160m` | published EleutherAI shapes |
| `llama` | `LlamaForCausalLM` | `HuggingFaceTB/SmolLM2-135M` | SmolLM2-135M shape (30 × 576) |
| `gemma` | `GemmaForCausalLM` | `google/gemma-2b` | 8 layers, 512 embd, 8 heads, 1024 positions |

**VRAM note:** Gemma's ~256k vocabulary makes logits memory-heavy; Gemma
zdecks use batch size 1 × grad accum 32 on the 8 GB GPU, versus 4 × 8 for the
~50k-vocab families. See [GPU memory](reference-gpu-memory.md).

---

## Manifests and the zdeck

A zdeck module is a Python file exporting a `MANIFEST`. Filenames mirror
`run_name` (underscores vs hyphens), including the host token:

| Layer | Pattern | Example |
|---|---|---|
| File | `{stage}_{family}_{corpus}_{budget}_{host}.py` | `pre_gemma_c4_5k_mist.py` |
| `run_name` / W&B | `{stage}-{family}-{corpus}-{budget}-{host}` | `pre-gemma-c4-5k-mist` |

`stage` is a Manifest field: `"pre"` (from-scratch pretrain, takes a
`CausalLmArchConfig`) or `"sft"` (full-parameter supervised fine-tune of a
pretrained checkpoint, takes a `PretrainedLmConfig` — see
[fine-tuning pretrained checkpoints](reference-model-families.md#fine-tuning-pretrained-checkpoints)).
The host token (e.g. `mist`) is the short machine profile used in names; full
GPU details live on `hardware.label` (e.g. `mist-rtx-3070`).

Archived schedules use `warmup_steps=None, warmup_ratio=0.04`, so epoch-based
warmup follows the resolved packed dataset length. The schedule seed controls
model/training randomness; the data seed separately controls shuffling,
evaluation sampling, and `TrainingArguments.data_seed`.

```python
MANIFEST = Manifest(
    run_name="pre-gpt-2-wikitext-5k-mist",
    title="GPT-2 from scratch on WikiText-103 (5k steps)",
    stage="pre",
    data=PretrainDataConfig(...),
    model=CausalLmArchConfig(family="gpt-2", tokenizer_name="gpt2", ...),
    hardware=HardwareConfig(...),
    wandb=WandbConfig(...),
    schedule=ScheduleConfig(max_steps=5_000, ...),
)
```

Checkpoints are written to:

```
output/train/<run_name>/
  config.json
  model.safetensors   (or pytorch_model.bin)
  tokenizer files
  checkpoint-*/        (intermediate saves)
  evals/               (post-training eval results)
```

### Current zdeck programs

Pretraining (`stage="pre"`):

| Module | Model | Corpus |
|---|---|---|
| `pre_gpt-2_wikitext_5k_mist` | GPT-2 | WikiText-103 (stream, 5k steps) |
| `pre_gpt-2_wikipedia_5k_mist` | GPT-2 | English Wikipedia (stream, 5k steps) |
| `pre_gpt-neox_wikitext_3ep_mist` | GPT-NeoX | WikiText-103 (complete, 3 epochs) |
| `pre_gpt-neox_wikitext_4ep_mist` | GPT-NeoX | WikiText-103 (complete, 4 epochs) |
| `pre_gpt-neox_c4_5k_mist` | GPT-NeoX | C4 English (stream, 5k steps) |
| `pre_gpt-neox_curriculum_geo_mist` | GPT-NeoX | Curriculum: C4 then geo-us-states |
| `pre_pythia-70m_wikitext_4ep_mist` | Pythia-70M | WikiText-103 (complete, 4 epochs) |
| `pre_pythia-160m_wikitext_4ep_mist` | Pythia-160M | WikiText-103 (complete, 4 epochs) |
| `pre_smollm2-135m_wikitext_4ep_mist` | SmolLM2-135M | WikiText-103 (complete, 4 epochs) |
| `pre_gemma_c4_5k_mist` | Gemma | C4 English (stream, 5k steps) |
| `pre_gemma_c4_50k_mist` | Gemma | C4 English (stream, 50k steps) |
| `pre_gemma_wikitext_4ep_mist` | Gemma | WikiText-103 (complete, 4 epochs) |
| `baseline_perf_mist` | GPT-NeoX | WikiText-103 (complete, 0.25 epochs — perf baseline) |
| `baseline_perf_gemma_mist` | Gemma | WikiText-103 (complete, 0.25 epochs — perf baseline) |

Supervised fine-tuning (`stage="sft"`):

| Module | Base checkpoint | Corpus |
|---|---|---|
| `sft_pythia-160m_geo_mist` | `EleutherAI/pythia-160m` | geo-us-states |
| `sft_smollm2-135m_geo_mist` | `HuggingFaceTB/SmolLM2-135M` | geo-us-states |

### Running training

From the repo root:

```bash
python alien_ink/zdeck/pre_gpt-2_wikitext_5k_mist.py
python -m alien_ink.zdeck.pre_gemma_c4_5k_mist
```

Or in the background on Mist:

```bash
./bin/pretrain-mist.sh   # edit the script to select a zdeck module
```

---

## Adding a new training program

1. Copy an existing zdeck module under `alien_ink/zdeck/`, naming it after the run (`{stage}_{family}_{corpus}_{budget}_{host}.py`).
2. Set `stage` and a matching `run_name` (`{stage}-{family}-{corpus}-{budget}-{host}`).
3. Set `data` to a `PretrainDataConfig` (or factory like `wikitext_103()`) pointing at your corpus.
4. Set `model.family` and `model.tokenizer_name` to match.
5. Ensure `data.block_size ≤ model.n_positions`.
6. Run `python -m alien_ink.zdeck.your_program` and complete `./bin/model-chat-mist.py your_program` when checkpoints exist.

For a new Hub corpus, define a `HubTextSource` with the correct `text_column`
name — most text datasets use `"text"`, but always verify on the Hub dataset
card.
