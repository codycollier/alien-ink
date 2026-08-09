# Pretraining data and completion

Alien Ink trains **base causal language models** from scratch on raw text corpora. There is no chat template, no system prompt, and no instruction format — the model learns to predict the next token in ordinary prose. Completions at inference time use the same plain-text continuation style.

This document covers how training data is structured, how it flows through the pipeline for each model family, and how to run interactive completions against a trained checkpoint.

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
  train → output/<run_name>/
        │
        ▼
  generate_completion    ← family-aware GenConfig from manifest
```

Every training run is described by a **manifest** in the zdeck (`alien_ink/zdeck/`). The manifest binds a dataset config, a model family, hardware knobs, and schedule into one reproducible program.

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

Factory helpers (`wikitext_103()`, `c4_english()`, `wikipedia_english()`) return ready-made `PretrainDataConfig` objects with sensible defaults.

---

## Built-in corpora and raw text shape

All zdeck programs today read a single **`text` column** per row. The model never sees metadata columns (titles, URLs, timestamps) — those are dropped during tokenization.

### WikiText-103 (`Salesforce/wikitext`, `wikitext-103-v1`)

Used by: `gpt-2_wikitext_5k`

- **Train split:** curated Wikipedia articles, one row per article or paragraph block.
- **Eval split:** dedicated `validation` split (not a hold-out from train).
- **Text character:** clean encyclopedic prose, relatively short rows compared to web crawl.

Example row shape:

```
text: "The Eiffel Tower is a wrought @-@ iron lattice tower on the Champ de Mars in Paris , France . It is named after the engineer Gustave Eiffel , whose company designed and built the tower ."
```

WikiText uses `@-@` as a hyphen placeholder — the tokenizer sees those characters as ordinary text.

### English Wikipedia (`wikimedia/wikipedia`, `20231101.en`)

Used by: `gpt-2_wikipedia_5k`

- **Single train split** — no bundled validation set.
- **Eval:** `max_eval_samples` rows selected by a deterministic bounded-buffer shuffle, then skipped for training.
- **Text character:** full article bodies; longer and more varied than WikiText.

Example row shape:

```
text: "The Apollo program, also known as Project Apollo, was the United States human spaceflight program carried out by the National Aeronautics and Space Administration ..."
```

### C4 English (`allenai/c4`, `en`)

Used by: `gemma_c4_5k`, `gemma_c4_50k`

- **Train + validation splits** on the Hub (validation used for eval when `eval_source` is set).
- **Text character:** web crawl snippets — noisy, diverse, conversational and formal mix. Requires accepting Hugging Face dataset terms for `allenai/c4`.
- **Gemma tokenizer:** requires Hugging Face access to `google/gemma-2b` (used for vocabulary only; weights are random at init).

Example row shape:

```
text: "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The China–Nepal border runs across its summit point."
```

---

## From raw text to training blocks

Preprocessing is **identical across model families** — only the tokenizer differs.

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

With `respect_document_boundaries=False` (classic packing — used by WikiText factories/zdecks):

- EOS-delimited token streams are **concatenated** across rows and map batches, then sliced.
- Document boundaries are **not** respected — a block may span two articles or lines.
- Short rows pack with neighbors; remainders carry into the next map batch. Only the final incomplete worker-shard remainder is dropped.

```
  doc A tokens ──┐
  doc B tokens ──┼──► [1024] [1024] [1024] …
  doc C tokens ──┘
```

Every document ends in exactly one EOS token. Tokenizer-added BOS/EOS tokens are disabled during preprocessing, preventing both unmarked joins and repeated BOS tokens inside packed blocks.

---

## Model families

Architecture and tokenizer are set in `CausalLmArchConfig` (`alien_ink.hf.model`). Field names follow GPT-2 conventions and are mapped onto each Hugging Face config class internally.

| Family | HF model class | Default tokenizer | Mist-sized dims (zdeck) |
|---|---|---|---|
| `gpt-2` | `GPT2LMHeadModel` | `gpt2` | 12 layers, 768 embd, 12 heads, 1024 positions |
| `gpt-neox` | `GPTNeoXForCausalLM` | `EleutherAI/gpt-neox-20b` | same as GPT-2 |
| `gemma` | `GemmaForCausalLM` | `google/gemma-2b` | 8 layers, 512 embd, 8 heads, 1024 positions |

### GPT-2 (`family="gpt-2"`)

- Byte-pair tokenizer, vocab ~50k. No beginning-of-sequence token.
- Training uses plain text plus explicit trailing EOS separators; generation has
  its own family-aware special-token policy.
- Typical zdeck pairings: WikiText-103, English Wikipedia.

### Gemma (`family="gemma"`)

- SentencePiece tokenizer, vocab ~256k. Defines `<bos>`, `<eos>`, and pad tokens.
- During training, tokenizer-added special tokens are disabled and each Hub row receives one trailing EOS separator.
- **VRAM note:** the large vocabulary makes logits memory-heavy; Gemma zdecks use batch size 1 × grad accum 32 on an 8 GB GPU.
- Typical zdeck pairings: C4 English.

### GPT-NeoX (`family="gpt-neox"`)

- Typical zdeck pairings: WikiText-103 (complete / epoch mode). Same data syntax as GPT-2.

---

## Manifests and the zdeck

A zdeck module is a Python file exporting a `MANIFEST`. Filenames mirror
`run_name` (underscores vs hyphens), including the host token:

| Layer | Pattern | Example |
|---|---|---|
| File | `{stage}_{family}_{corpus}_{budget}_{host}.py` | `pre_gemma_c4_5k_mist.py` |
| `run_name` / W&B | `{stage}-{family}-{corpus}-{budget}-{host}` | `pre-gemma-c4-5k-mist` |

`stage` is a Manifest field: `"pre"` (from-scratch pretrain) or `"sft"`
(supervised fine-tune; reserved — `train()` not implemented yet). The host
token (e.g. `mist`) is the short machine profile used in names; full GPU
details live on `hardware.label` (e.g. `mist-rtx-3070`).

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
output/<run_name>/
  config.json
  model.safetensors   (or pytorch_model.bin)
  tokenizer files
  checkpoint-*/        (intermediate saves)
```

### Current zdeck programs

| Module | Model family | Corpus |
|---|---|---|
| `alien_ink/zdeck/pre_gpt-2_wikitext_5k_mist.py` | GPT-2 | WikiText-103 (stream) |
| `alien_ink/zdeck/pre_gpt-2_wikipedia_5k_mist.py` | GPT-2 | English Wikipedia (stream) |
| `alien_ink.zdeck.pre_gemma_c4_5k_mist` | Gemma | C4 English (stream) |
| `alien_ink.zdeck.pre_gemma_c4_50k_mist` | Gemma | C4 English (stream, 50k steps) |
| `alien_ink.zdeck.pre_gemma_wikitext_4ep_mist` | Gemma | WikiText-103 (complete, 4 epochs) |
| `alien_ink/zdeck/pre_gpt-neox_wikitext_4ep_mist.py` | GPT-NeoX | WikiText-103 (complete, 4 epochs) |
| `alien_ink.zdeck.baseline_perf_mist` | GPT-NeoX | WikiText-103 (complete, 0.25 epochs) |

### Running training

From the repo root:

```bash
python alien_ink/zdeck/pre_gpt-2_wikitext_5k_mist.py
python -m alien_ink.zdeck.pre_gemma_c4_5k_mist
```

Or in the background on Mist:

```bash
./bin/pretrain_mist.sh   # edit the script to select a zdeck module
```

---

## Running completions

These checkpoints are **base LMs**, not instruction-tuned chat models. Prompt them with a sentence starter or fragment; the model continues in plain text.

Do **not** use chat roles (`System:`, `User:`, `Assistant:`) — the model was never trained on that format.

### Interactive REPL

```bash
./bin/model-chat-mist.py pre_gpt-2_wikitext_5k_mist
./bin/model-chat-mist.py pre_gemma_c4_5k_mist
./bin/model-chat-mist.py alien_ink.zdeck.pre_gemma_c4_5k_mist --max-new-tokens 120
```

The script loads the manifest, resolves `output/<run_name>/`, and drives generation from `manifest.model.family`.

Each turn prints **four** candidates on separate lines: greedy (deterministic) first, then sampled at temperatures `0.5`, `0.8`, and `1.2`. A counter and decoding stats sit beside each completion.

Example session:

```
input› The capital of Texas is
[1] Austin .  (greedy T=0, 2 tok)
[2] Austin, Texas.  (T=0.5 top_p=0.95, 4 tok)
[3] a city known for live music.  (T=0.8 top_p=0.95, 7 tok)
[4] located in the southern United States.  (T=1.2 top_p=0.95, 8 tok)
```

Type **Ctrl-C** to exit.

### Family-aware generation config

Completion settings live in `GenConfig` (`alien_ink.hf.gen`), selected by model family:

```python
gen = manifest.gen_config(max_new_tokens=120)
completion = generate_completion(model, tokenizer, prompt, device, gen)
```

| Setting | GPT-2 / GPT-NeoX | Gemma |
|---|---|---|
| `add_special_tokens` | `True` | `False` (avoids prepending BOS on continuation) |
| `do_sample` | `False` (greedy, deterministic) | same |
| `stop_strings` | `.`, `!`, `?` via `model.generate` | same |

Stop strings are passed to Hugging Face `generate()` as native stopping criteria — generation halts when the model emits sentence-ending punctuation, not by trimming the output afterward.

### Programmatic spot checks

```python
from alien_ink.hf.gen import run_spot_check, SpotCheckConfig

run_spot_check(
    output_dirs=[Path("output/pre-gpt-2-wikitext-5k-mist")],
    family="gpt-2",
    spot=SpotCheckConfig(num_samples=5, do_sample=True),
)
```

Default prompt seeds (`"The capital of France is"`, etc.) are short sentence starters suited to base LM continuation.

---

## Quick reference: family × dataset

| | WikiText-103 | Wikipedia EN | C4 EN |
|---|---|---|---|
| **GPT-2** | ✓ zdeck | ✓ zdeck | supported |
| **Gemma** | ✓ zdeck | supported | ✓ zdeck |
| **GPT-NeoX** | ✓ zdeck | supported | supported |

“Supported” means the data loaders and training stack work; only cells marked **zdeck** have a checked-in manifest program today.

---

## Adding a new training program

1. Copy an existing zdeck module under `alien_ink/zdeck/`, naming it after the run (`{stage}_{family}_{corpus}_{budget}_{host}.py`).
2. Set `stage` and a matching `run_name` (`{stage}-{family}-{corpus}-{budget}-{host}`).
3. Set `data` to a `PretrainDataConfig` (or factory like `wikitext_103()`) pointing at your corpus.
4. Set `model.family` and `model.tokenizer_name` to match.
5. Ensure `data.block_size ≤ model.n_positions`.
6. Run `python -m alien_ink.zdeck.your_program` and complete `./bin/model-chat-mist.py your_program` when checkpoints exist.

For a new Hub corpus, define a `HubTextSource` with the correct `text_column` name — most text datasets use `"text"`, but always verify on the Hub dataset card.
