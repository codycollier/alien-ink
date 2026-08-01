# Datasets

Alien Ink pretrains base causal LMs on raw text corpora from the Hugging Face
Hub. Every zdeck program today reads a single `text` column; metadata such as
titles, URLs, and timestamps is dropped during tokenization.

Factories and loaders live in `alien_ink.hf.ds`. This page describes the
**corpora themselves** — size, character, splits, and how they behave under
Alien Ink’s packing pipeline.

---

## At a glance

| Corpus | Hub id | Config | Train rows | Word / token scale | ~Steps / epoch | ~Time / epoch (Mist) | Eval | Zdeck |
|---|---|---|---:|---|---:|---|---|---|
| WikiText-103 | `Salesforce/wikitext` | `wikitext-103-v1` | 1,801,350 | ~103M words / ~120–150M BPE | **~4k** | **~4 h** | dedicated `validation` | `gpt2_wikitext_5k` |
| English Wikipedia | `wikimedia/wikipedia` | `20231101.en` | ~6.4M | ~3–5B words / ~4–7B BPE | **~120k–210k** | **~5–8.5 d** | hold-out from train | `gpt2_wikipedia_5k` |
| C4 English | `allenai/c4` | `en` | 364,868,892 | ~156B SpaCy / ~150–200B subword | **~5M–6M** | **~6–8 mo** | dedicated `validation` | `gemma_c4_5k`, `gemma_c4_50k` |

Row counts for WikiText and C4 come from Hub `dataset_info`. Wikipedia’s
~6.4M is the published count for `20231101.en`. Word/token scales are
authoritative for WikiText-103 and C4; Wikipedia is an estimate (see below).
Steps and times assume Mist zdeck throughput (below).

**Tokenizers change the count.** WikiText’s classic “103M tokens” are
whitespace word tokens. GPT-2 BPE and Gemma SentencePiece typically produce
~1.2–1.5× more subword tokens on the same English prose. Treat subword figures
and step counts as order-of-magnitude unless you re-tokenize the corpus.

---

## How Alien Ink consumes text

Regardless of corpus:

1. Rows are loaded (`stream`, `subset`, or `complete` — see
   [`pretraining-and-completion.md`](pretraining-and-completion.md)).
2. The family’s tokenizer encodes the `text` column.
3. Token streams are concatenated and sliced into fixed `block_size` blocks
   (default **1024**).
4. Labels copy `input_ids` (next-token prediction).

Document boundaries are **not** preserved across blocks. Short rows are packed
with neighbors; a leftover shorter than one block is dropped.

### Mist zdeck throughput

All current zdeck manifests use `block_size=1024` and the same **effective
tokens per optimizer step**:

```
tokens_per_step = per_device_batch × gradient_accumulation × block_size
```

| Zdeck hardware | Batch × accum | `tokens_per_step` |
|---|---|---:|
| GPT-2 (`gpt2_wikitext_5k`, `gpt2_wikipedia_5k`) | 2 × 16 | **32,768** |
| Gemma (`gemma_c4_5k`, `gemma_c4_50k`) | 1 × 32 | **32,768** |

Gemma uses a smaller micro-batch because the ~256k vocab makes logits heavy;
grad accum is raised so GPT-2 and Gemma still see the same tokens per step.

### Steps to finish one epoch

One **epoch** here means one pass over the packed train tokens (streaming
wraps; step-capped zdecks usually stop earlier):

```
steps_per_epoch ≈ corpus_subword_tokens / 32_768
```

| Corpus | Subword tokens (est.) | Steps / epoch | ~Wall clock / epoch (Mist) | Current zdeck `max_steps` |
|---|---:|---:|---|---:|
| WikiText-103 | ~120–150M (GPT-2 BPE) | **~3.7k–4.6k** | **~3.5–4.5 h** | 5,000 (`gpt2_wikitext_5k`) |
| Wikipedia EN | ~4–7B (GPT-2 BPE) | **~120k–210k** | **~5–8.5 days** | 5,000 (`gpt2_wikipedia_5k`) |
| C4 EN | ~150–200B (Gemma / GPT-2) | **~4.6M–6.1M** | **~6–8 months** | 5,000 / 50,000 (`gemma_c4_*`) |

### Wall clock on Mist (RTX 3070)

Calibration from a Mist Gemma C4 run: **~50,000 steps ≈ 48 hours**
(≈3.5 s/step at the zdeck effective batch). Scale linearly with step count:

```
hours ≈ steps × (48 / 50_000)
```

Midpoint planning numbers:

| Corpus | Planning tokens | Steps / epoch | Time / epoch (Mist) |
|---|---:|---:|---|
| WikiText-103 | 135M | **~4,100** | **~4 hours** |
| Wikipedia EN | 5.5B | **~170,000** | **~7 days** |
| C4 EN | 175B | **~5,300,000** | **~7 months** |

GPT-2 / NeoX Mist stacks use the same tokens/step as Gemma; wall clock can
drift a bit with model size and micro-batch shape, so treat times as ±~25%.

So `gpt2_wikitext_5k` is roughly **one full WikiText pass** (slightly over,
~5 hours). The Wikipedia and C4 5k/50k programs are short slices: useful for
Mist smoke-runs, not full corpus coverage. A 50k-step C4 run (~48 h) covers
about **1%** of one C4 epoch.

Tokens seen at common step budgets (any Mist zdeck above):

| Steps | Tokens seen | ~Wall clock (Mist) |
|---:|---:|---|
| 5,000 | ~164M | ~5 hours |
| 50,000 | ~1.64B | ~48 hours |
| ~4,100 | ~134M ≈ one WikiText epoch | ~4 hours |
| ~170,000 | ~5.6B ≈ one Wikipedia epoch | ~7 days |
| ~5,300,000 | ~174B ≈ one C4 epoch | ~7 months |

Default subset mode (for experiments / CI): **20,000** train rows + **1,000**
eval rows (`DEFAULT_SUBSET_*` in `alien_ink.hf.ds`).
---

## WikiText-103

| | |
|---|---|
| **Hub** | `Salesforce/wikitext`, config `wikitext-103-v1` |
| **Factory** | `wikitext_103()`, `wikitext_103_subset()`, `wikitext_103_complete()` |
| **License** | Creative Commons Attribution-ShareAlike |
| **Paper** | Merity et al., *Pointer Sentinel Mixture Models* (2016) |
| **Zdeck** | `alien_ink.zdeck.gpt2_wikitext_5k` |

### Character

Curated English Wikipedia: only Good and Featured articles. Clean encyclopedic
prose with original casing, punctuation, and numbers. Designed for long-range
language modeling (full articles, not shuffled sentences).

The Hub `wikitext-103-v1` split is **line-oriented**: many rows are blank
section separators or short paragraphs, not one row per article. Empty strings
are common; the tokenizer still sees them as (near-)empty encodings.

Hyphens appear as the placeholder `@-@` (e.g. `wrought @-@ iron`). That is
intentional preprocessing from the original release — the model learns those
characters as ordinary text.

### Size (authoritative word-level stats)

From Merity et al. (whitespace / word tokens, including newlines):

| Split | Articles | Word tokens | Vocab (word-level) |
|---|---:|---:|---:|
| Train | 28,475 | **103,227,021** | 267,735 |
| Validation | 60 | 217,646 | — |
| Test | 60 | 245,569 | — |

Hub map-style row counts (`wikitext-103-v1`):

| Split | Rows | Approx. stored text |
|---|---:|---|
| Train | 1,801,350 | ~545 MB |
| Validation | 3,760 | ~1.2 MB |
| Test | 4,358 | ~1.3 MB |

Rough conversions for planning:

| Measure | Train estimate |
|---|---|
| Words | ~103M |
| Characters | ~500–550M (matches Hub `num_bytes`) |
| GPT-2 BPE tokens | ~120–150M |
| Packed 1024-token blocks | ~120k–150k |
| Mist steps / epoch (`2×16×1024`) | **~3.7k–4.6k** (~4.1k midpoint) |
| Mist wall clock / epoch | **~3.5–4.5 hours** (~4 h midpoint) |

Average non-empty Hub row is short (tens to low hundreds of words). After
packing, the model mostly sees contiguous article fragments spanning multiple
Hub rows.

### Coverage under `gpt2_wikitext_5k`

That manifest runs **5,000** steps → ~164M tokens (~5 hours on Mist). Against
~120–150M GPT-2 tokens in train, that is about **1.1–1.4 epochs** — the only
Mist zdeck that roughly finishes its corpus.

### Eval in Alien Ink

Manifests set `eval_source` to the Hub `validation` split and cap with
`max_eval_samples=1000` (validation has 3,760 rows; only the first 1,000 are
materialized for eval).

### When to use it

Fast, high-quality English for GPT-2-sized runs. Small enough that
`mode="complete"` is practical, and small enough that a ~4k–5k Mist schedule
can cover a full epoch. Too homogeneous (encyclopedia only) to stress
web-scale robustness — prefer C4 for that.

---

## English Wikipedia (`20231101.en`)

| | |
|---|---|
| **Hub** | `wikimedia/wikipedia`, config `20231101.en` |
| **Factory** | `wikipedia_english()`, `wikipedia_english_subset()`, `wikipedia_english_complete()` |
| **License** | CC BY-SA 3.0 / GFDL (Wikimedia dump terms) |
| **Zdeck** | `alien_ink.zdeck.gpt2_wikipedia_5k` |

### Character

Full cleaned article bodies from the **2023-11-01** English dump: markup
stripped, reference-like sections removed via the Hub pipeline
(`mwparserfromhell`). One Hub row ≈ one article (`id`, `url`, `title`, `text`).
Alien Ink uses only `text`.

Compared to WikiText-103: far more articles, wider quality mix (not only
Featured/Good), longer average documents, and more topical diversity — still
encyclopedic, not web crawl noise.

### Size (estimates)

| Measure | Estimate | Notes |
|---|---|---|
| Articles (Hub rows) | **~6.4M** | Published for `20231101.en` |
| Words | **~3–5B** | Live EN wiki is ~5B+ words today; Nov 2023 cleaned dump is a bit lower |
| Characters | ~20–30B | Rough; depends on cleaning |
| GPT-2 BPE tokens | **~4–7B** | ~1.3 tokens/word is a common English BPE rule of thumb |
| Packed 1024-token blocks | ~4–7M | Order of magnitude |
| Mist steps / epoch (`2×16×1024`) | **~120k–210k** (~170k midpoint) |
| Mist wall clock / epoch | **~5–8.5 days** (~7 days midpoint) |
| Avg. words / article | ~500–800 | Heavy skew: many stubs, few very long pages |

There is **no official single token count** for this Hub config. Figures above
are planning estimates, not measured Alien Ink re-tokenizations.

`mode="complete"` downloads the full English split — large on disk; prefer
`stream` on Mist (as the zdeck does).

### Coverage under `gpt2_wikipedia_5k`

That manifest runs **5,000** steps → ~164M tokens ≈ **2–4%** of one Wikipedia
epoch (~5 hours on Mist). Finishing the dump at Mist GPT-2 throughput would take
on the order of **~120k–210k steps** (~**5–8.5 days** of continuous Mist time).

### Eval in Alien Ink

No bundled validation split. With `eval_source=None`, the loader holds out the
first `max_eval_samples` (default 1,000) stream rows, then skips them for
training so train/eval do not overlap.

### When to use it

Broader Wikipedia coverage than WikiText without leaving the encyclopedia
domain. Good default for GPT-2 Mist runs when you want more data diversity
than WikiText-103 but not C4’s license friction or noise. Plan for **~170k
steps (~7 days on Mist)** if the goal is a full pass.

---

## C4 English

| | |
|---|---|
| **Hub** | `allenai/c4`, config `en` |
| **Factory** | `c4_english()`, `c4_english_subset()`, `c4_english_complete()` |
| **License / access** | Requires accepting the Hub dataset terms |
| **Origin** | Common Crawl (Apr 2019), cleaned as in Raffel et al. (T5) |
| **Zdeck** | `gemma_c4_5k`, `gemma_c4_50k` |

### Character

Web pages filtered for English natural language: short lines and boilerplate
removed, language-ID threshold, blocklist filtering, etc. Result is **noisy and
diverse** — news, forums, docs, marketing, how-tos — mixed register and quality.

Each row is one URL’s extracted text, plus `timestamp` and `url` (unused by
Alien Ink). Snippets are usually longer than WikiText lines but shorter and
messier than full Wikipedia articles.

### Size (authoritative)

From Dodge et al. / AllenAI hosting of C4.en (SpaCy English tokenizer):

| Split | Documents | SpaCy tokens | Compressed size |
|---|---:|---:|---|
| Train | **364,868,892** | **~156B** (corpus total ≈ this) | ~305 GB |
| Validation | **364,608** | ~proportional (~0.1% of train) | — |

Hub `dataset_info` for `en`:

| Split | Rows | `num_bytes` (approx. text payload) |
|---|---:|---:|
| Train | 364,868,892 | ~829 GB |
| Validation | 364,608 | ~826 MB |

Derived averages (train):

| Measure | Approx. |
|---|---|
| SpaCy tokens / doc | ~430 |
| Words (≈ SpaCy tokens) | ~156B |
| GPT-2 / Gemma subword tokens | ~150–200B (order of magnitude) |
| Packed 1024-token blocks | ~150–200M |
| Mist steps / epoch (`1×32×1024`) | **~4.6M–6.1M** (~5.3M midpoint) |
| Mist wall clock / epoch | **~6–8 months** (~7 months midpoint) |
| Chars / doc | ~2–3k |

Other Hub variants exist (`en.noblocklist`, `en.noclean`, `realnewslike`,
`multilingual`) but Alien Ink’s factory uses **`en`** only.

### Coverage under `gemma_c4_5k` / `gemma_c4_50k`

Both manifests share Mist Gemma hardware (`batch=1`, `accum=32`,
`block_size=1024` → 32,768 tokens/step):

| Program | `max_steps` | Tokens seen | ~Wall clock (Mist) |
|---|---:|---:|---|
| `gemma_c4_5k` | 5,000 | ~164M | ~5 hours |
| `gemma_c4_50k` | 50,000 | ~1.64B | ~48 hours |

A full C4 epoch at this throughput is **~5–6 million steps (~6–8 months on
Mist)** — not a practical Mist-local target; treat the zdecks as long smoke /
early-loss runs on a stream sample.

### Eval in Alien Ink

Manifests point `eval_source` at Hub `validation` and materialize up to 1,000
rows. Never download the full train split on Mist — always `mode="stream"`
(or a small `subset`).

### Access note

Gemma zdecks also need Hub access to `google/gemma-2b` for the **tokenizer**
(weights are random at init). Accept C4 terms and authenticate with `HF_TOKEN`
before the first stream.

### When to use it

Scale and diversity. Appropriate when exploring Gemma (or any family) against
web text. Overkill for tiny debugging runs — use `c4_english_subset()` or
WikiText instead.

---

## Comparing the three

| | WikiText-103 | Wikipedia EN | C4 EN |
|---|---|---|---|
| **Domain** | Curated wiki | Full wiki dump | Web crawl |
| **Cleanliness** | Very clean | Cleaned wiki markup | Noisy |
| **Long-form structure** | Strong (articles) | Strong | Weak / mixed |
| **Scale** | ~0.1B words | ~3–5B words | ~156B tokens |
| **Mist steps / epoch** | ~4k | ~170k | ~5.3M |
| **Mist time / epoch** | ~4 hours | ~7 days | ~7 months |
| **Fits in RAM complete?** | Yes | No (practical) | No |
| **Dedicated val split?** | Yes | No | Yes |
| **Typical Mist pairing** | GPT-2 | GPT-2 | Gemma |

Relative train size (order of magnitude, word/SpaCy token scale):

```
WikiText-103    ████                                         ~0.1B   (~4k steps, ~4 h)
Wikipedia EN    ████████████████████████                     ~4B    (~170k steps, ~7 d)
C4 EN           ████████████████████████████████████████…    ~156B  (~5.3M steps, ~7 mo)
```

---

## Adding another corpus

1. Confirm the Hub card: config name, splits, and the text column name.
2. Add a `HubTextSource` (and optionally a factory next to `wikitext_103` /
   `c4_english` / `wikipedia_english`).
3. Prefer a dedicated validation split when one exists; otherwise rely on
   hold-out eval.
4. Keep `block_size ≤ model.n_positions`.
5. For huge corpora, default `mode="stream"` in zdeck manifests.

See also: [`pretraining-and-completion.md`](pretraining-and-completion.md) for
load modes and packing; [`model-families.md`](model-families.md) for which
family is paired with which corpus today.
