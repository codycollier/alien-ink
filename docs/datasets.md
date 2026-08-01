# Datasets

Alien Ink pretrains base causal LMs on raw text corpora from the Hugging Face
Hub. Every zdeck program today reads a single `text` column; metadata such as
titles, URLs, and timestamps is dropped during tokenization.

Factories and loaders live in `alien_ink.hf.ds`. This page describes the
**corpora themselves** — size, character, splits, and how they behave under
Alien Ink’s packing pipeline.

---

## At a glance

| Corpus | Hub id | Config | Train rows | Word / token scale | Eval in Alien Ink | Zdeck users |
|---|---|---|---:|---|---|---|
| WikiText-103 | `Salesforce/wikitext` | `wikitext-103-v1` | 1,801,350 | ~103M words | dedicated `validation` | `gpt2_wikitext_5k` |
| English Wikipedia | `wikimedia/wikipedia` | `20231101.en` | ~6.4M | ~3–5B words | hold-out from train | `gpt2_wikipedia_5k` |
| C4 English | `allenai/c4` | `en` | 364,868,892 | ~156B SpaCy tokens | dedicated `validation` | `gemma_c4_5k`, `gemma_c4_50k` |

Row counts for WikiText and C4 come from Hub `dataset_info`. Wikipedia’s
~6.4M is the published count for `20231101.en`. Word/token scales are
authoritative for WikiText-103 and C4; Wikipedia is an estimate (see below).

**Tokenizers change the count.** WikiText’s classic “103M tokens” are
whitespace word tokens. GPT-2 BPE and Gemma SentencePiece typically produce
~1.2–1.5× more subword tokens on the same English prose. Treat subword figures
as order-of-magnitude unless you re-tokenize the corpus.

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

### Tokens seen in a Mist zdeck run

Effective tokens per optimizer step:

```
per_device_batch × gradient_accumulation × block_size
```

| Family (typical Mist hardware) | Effective tokens / step | 5k steps | 50k steps |
|---|---:|---:|---:|
| GPT-2 (`batch=2`, `accum=16`) | 32,768 | ~164M | — |
| Gemma (`batch=1`, `accum=32`) | 32,768 | ~164M | ~1.64B |

So a 5k-step Mist run sees on the order of **one WikiText-103** of subword
tokens — a small slice of Wikipedia or a tiny fraction of C4.

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

Average non-empty Hub row is short (tens to low hundreds of words). After
packing, the model mostly sees contiguous article fragments spanning multiple
Hub rows.

### Eval in Alien Ink

Manifests set `eval_source` to the Hub `validation` split and cap with
`max_eval_samples=1000` (validation has 3,760 rows; only the first 1,000 are
materialized for eval).

### When to use it

Fast, high-quality English for GPT-2-sized runs. Small enough that
`mode="complete"` is practical. Too homogeneous (encyclopedia only) to stress
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
| Avg. words / article | ~500–800 | Heavy skew: many stubs, few very long pages |

There is **no official single token count** for this Hub config. Figures above
are planning estimates, not measured Alien Ink re-tokenizations.

`mode="complete"` downloads the full English split — large on disk; prefer
`stream` on Mist (as the zdeck does).

### Eval in Alien Ink

No bundled validation split. With `eval_source=None`, the loader holds out the
first `max_eval_samples` (default 1,000) stream rows, then skips them for
training so train/eval do not overlap.

### When to use it

Broader Wikipedia coverage than WikiText without leaving the encyclopedia
domain. Good default for GPT-2 Mist runs when you want more data diversity
than WikiText-103 but not C4’s license friction or noise.

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
| Chars / doc | ~2–3k |

Other Hub variants exist (`en.noblocklist`, `en.noclean`, `realnewslike`,
`multilingual`) but Alien Ink’s factory uses **`en`** only.

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
| **Fits in RAM complete?** | Yes | No (practical) | No |
| **Dedicated val split?** | Yes | No | Yes |
| **Typical Mist pairing** | GPT-2 | GPT-2 | Gemma |

Relative train size (order of magnitude, word/SpaCy token scale):

```
WikiText-103    ████                                         ~0.1B
Wikipedia EN    ████████████████████████                     ~4B
C4 EN           ████████████████████████████████████████…    ~156B
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
