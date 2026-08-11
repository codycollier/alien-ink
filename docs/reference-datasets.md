# Datasets

Alien Ink pretrains base causal LMs on raw text corpora from the Hugging Face
Hub. Every zdeck program today reads a single `text` column; metadata such as
titles, URLs, and timestamps is dropped during tokenization.

Factories live in `alien_ink.hf.ds`. This page covers the corpora — size,
character, splits, and Mist-scale planning.

---

## Overview

### Corpora at a glance

| | WikiText-103 | Wikipedia EN | C4 EN |
|---|---|---|---|
| **Hub** | `Salesforce/wikitext` | `wikimedia/wikipedia` | `allenai/c4` |
| **Config** | `wikitext-103-v1` | `20231101.en` | `en` |
| **Domain** | Curated wiki | Full wiki dump | Web crawl |
| **Cleanliness** | Very clean | Cleaned markup | Noisy / diverse |
| **Train scale** | ~103M words | ~3–5B words | ~156B SpaCy tokens |
| **Subword (est.)** | ~120–150M | ~4–7B | ~150–200B |
| **Mist steps / epoch** | **~4k** | **~170k** | **~5.3M** |
| **Mist time / epoch** | **~4 h** | **~7 d** | **~7 mo** |
| **Dedicated val?** | Yes | No (hold-out) | Yes |
| **`complete` practical?** | Yes | No | No |
| **Typical Mist pairing** | GPT-2 / NeoX | GPT-2 | Gemma |
| **Zdeck** | `gpt-2_wikitext_5k` | `gpt-2_wikipedia_5k` | `gemma_c4_5k`, `gemma_c4_50k` |

WikiText and C4 row counts come from Hub `dataset_info`; Wikipedia’s ~6.4M is
the published `20231101.en` count. Word/token scales are authoritative for
WikiText-103 and C4; Wikipedia is an estimate. Tokenizers change the count —
whitespace “words” ≠ GPT-2 BPE / Gemma SentencePiece (~1.2–1.5× on English
prose). Treat subword figures and step counts as order-of-magnitude unless you
re-tokenize.

### Relative train size

Order of magnitude (word / SpaCy token scale), with Mist epoch midpoints:

```
WikiText-103    ████                                         ~0.1B   (~4k steps, ~4 h)
Wikipedia EN    ████████████████████████                     ~4B    (~170k steps, ~7 d)
C4 EN           ████████████████████████████████████████…    ~156B  (~5.3M steps, ~7 mo)
```

### Mist throughput

All current zdeck manifests use `block_size=1024` and the same effective tokens
per optimizer step:

```
tokens_per_step = per_device_batch × gradient_accumulation × block_size
                = 32,768
```

| Zdeck | Batch × accum | Why |
|---|---|---|
| GPT-2 / NeoX / Pythia | 4 × 8 | Larger micro-batch; checkpointing off on Mist |
| SmolLM2-135M | 2 × 16 | 30-layer stack carries more activation memory |
| Gemma (`gemma_c4_*`, `gemma_wikitext_4ep`) | 1 × 32 | Smaller micro-batch (256k vocab logits); ckpt on |

Calibration (Mist Gemma C4): **~50,000 steps ≈ 48 h** (≈3.5 s/step). Scale
linearly: `hours ≈ steps × (48 / 50_000)`. GPT-2 / NeoX wall clock can drift
±~25% with model size and micro-batch shape.

| Steps | Tokens seen | ~Wall clock | Coverage |
|---:|---:|---|---|
| 5,000 | ~164M | ~5 h | ~1.1–1.4 WikiText epochs; ~2–4% Wikipedia; ≪1% C4 |
| 50,000 | ~1.64B | ~48 h | ~1% of one C4 epoch |
| ~4,100 | ~134M | ~4 h | ≈ one WikiText epoch |
| ~170,000 | ~5.6B | ~7 d | ≈ one Wikipedia epoch |
| ~5,300,000 | ~174B | ~7 mo | ≈ one C4 epoch |

So `gpt-2_wikitext_5k` roughly finishes its corpus; Wikipedia and C4 5k/50k
programs are short slices for Mist smoke / early-loss runs.

Default subset mode (experiments / CI): **20,000** train + **1,000** eval rows
(`DEFAULT_SUBSET_*` in `alien_ink.hf.ds`).

### How Alien Ink consumes text

1. Load rows (`stream`, `subset`, or `complete`).
2. Encode the `text` column with the family tokenizer.
3. Slice into fixed `block_size` blocks (default **1024**).
4. Labels copy `input_ids` (next-token prediction).

By default (`respect_document_boundaries=True`), each Hub row is chunked
independently — blocks never span two documents. Short rows (< `block_size`)
produce no blocks; long rows become multiple blocks with a dropped remainder.

Set `respect_document_boundaries=False` to concatenate across rows before
slicing (classic packing). WikiText factories and zdecks do this because Hub
rows are line-oriented and usually shorter than one block. Each row receives
one EOS separator, and incomplete tails carry across map batches; only a final
incomplete worker-shard tail is discarded.

---

## WikiText-103

| | |
|---|---|
| **Hub** | `Salesforce/wikitext`, config `wikitext-103-v1` |
| **Factory** | `wikitext_103()`, `_subset()`, `_complete()` |
| **License** | CC BY-SA |
| **Paper** | Merity et al., *Pointer Sentinel Mixture Models* (2016) |
| **Train** | 1,801,350 rows · **103,227,021** word tokens · ~545 MB |
| **Val / test** | 3,760 / 4,358 rows · 217k / 246k word tokens |
| **GPT-2 BPE (est.)** | ~120–150M · ~120k–150k packed blocks |
| **Mist / epoch** | **~3.7k–4.6k steps** · **~3.5–4.5 h** |
| **Eval** | Hub `validation`, capped at `max_eval_samples=1000` |
| **Zdeck** | `gpt-2_wikitext_5k` (5k steps ≈ **1.1–1.4 epochs**) |

Curated English Wikipedia (Good / Featured only): clean encyclopedic prose,
original casing and punctuation. Hub split is **line-oriented** — many blank
section separators and short paragraphs; empty strings are common. Hyphens
appear as `@-@` (original preprocessing). Average non-empty row is short; after
packing the model mostly sees contiguous article fragments spanning Hub rows
(`respect_document_boundaries=False`).

**Use when:** Fast, high-quality English for GPT-2 / NeoX-sized runs.
`mode="complete"` is practical; a ~4k–5k Mist schedule covers a full epoch.
Too homogeneous for web-scale robustness — prefer C4 for that.

---

## English Wikipedia (`20231101.en`)

| | |
|---|---|
| **Hub** | `wikimedia/wikipedia`, config `20231101.en` |
| **Factory** | `wikipedia_english()`, `_subset()`, `_complete()` |
| **License** | CC BY-SA 3.0 / GFDL (Wikimedia dump terms) |
| **Train** | **~6.4M** articles (one row ≈ one article) |
| **Scale (est.)** | ~3–5B words · ~4–7B GPT-2 BPE · ~4–7M packed blocks |
| **Mist / epoch** | **~120k–210k steps** · **~5–8.5 d** (~170k / ~7 d midpoint) |
| **Eval** | No val split — hold out a seeded shuffled sample of `max_eval_samples` rows |
| **Zdeck** | `gpt-2_wikipedia_5k` (5k steps ≈ **2–4%** of one epoch) |

Full cleaned article bodies from the 2023-11-01 English dump (markup stripped,
reference sections removed). Compared to WikiText: far more articles, wider
quality mix, longer average documents, more topical diversity — still
encyclopedic. No official single token count; figures are planning estimates.
Prefer `stream` on Mist; `complete` is large on disk.

**Use when:** Broader Wikipedia coverage without leaving the encyclopedia
domain. Plan **~170k steps (~7 days on Mist)** for a full pass.

---

## C4 English

| | |
|---|---|
| **Hub** | `allenai/c4`, config `en` |
| **Factory** | `c4_english()`, `_subset()`, `_complete()` |
| **License** | Requires accepting Hub dataset terms |
| **Origin** | Common Crawl (Apr 2019), cleaned as in Raffel et al. (T5) |
| **Train** | **364,868,892** docs · **~156B** SpaCy tokens · ~305 GB compressed / ~829 GB text |
| **Val** | 364,608 docs |
| **Subword (est.)** | ~150–200B · ~150–200M packed blocks |
| **Mist / epoch** | **~4.6M–6.1M steps** · **~6–8 mo** (~5.3M / ~7 mo midpoint) |
| **Eval** | Hub `validation`, capped at 1,000 rows |
| **Zdeck** | `gemma_c4_5k` (~5 h) · `gemma_c4_50k` (~48 h, ~1% of an epoch) |

Web pages filtered for English natural language (boilerplate / language-ID /
blocklist). Noisy and diverse — news, forums, docs, marketing. One row ≈ one
URL’s extracted text. Always `mode="stream"` (or a small `subset`) on Mist —
never download the full train split. Gemma zdecks also need Hub access to
`google/gemma-2b` for the tokenizer; accept C4 terms and set `HF_TOKEN` first.

Other Hub variants exist (`en.noblocklist`, `en.noclean`, …); Alien Ink uses
**`en`** only.

**Use when:** Scale and diversity, especially with Gemma. Overkill for tiny
debugging — use `c4_english_subset()` or WikiText instead.

---

## geo-us-states

| | |
|---|---|
| **Hub** | `codycollier/geo-us-states` |
| **Factory** | `geo_us_states()` (or `hub_text(...)` for any personal corpus) |
| **Train** | **56** rows (one long document per US state / territory) · ~3.9 MB text |
| **Scale (est.)** | ~17k tokens/row · ~880 packed blocks at 1024 |
| **Eval** | No val split — hold out `max_eval_samples=4` rows (default) |
| **Zdeck** | `gpt-neox_curriculum_geo` (curriculum followup phase) |

Tiny custom corpus of long documents. Always `mode="complete"`; a Mist epoch
is ~27 steps, so give it a curriculum phase budget of ~100 steps for 3–4
epochs. Keep the eval hold-out small — the standard 1,000-row default would
swallow the whole corpus. `hub_text()` builds a `PretrainDataConfig` for any
similar Hub text dataset without writing a dedicated factory.

---

## Curricula

A `Curriculum` (`alien_ink.hf.curriculum`) sequences datasets or subsets into
one pretraining run: an ordered tuple of phases, each a `PretrainDataConfig`
plus an optimizer-step budget. It materializes as a single streaming dataset
whose phase boundaries land exactly on optimizer steps — the trainer sees a
normal dataset, so one run, one LR schedule, one W&B run.

```python
CURRICULUM = Curriculum(
    phases=(
        CurriculumPhase(data=C4_DATA, steps=5_000, label="c4"),
        CurriculumPhase(data=GEO_DATA, steps=100, label="geo"),
    ),
    eval_data={"geo": GEO_DATA, "c4": C4_DATA},
)
```

Semantics:

- **Step budgets.** A phase of `steps` contributes exactly
  `steps × per_device_batch × grad_accum × world_size` blocks. Small
  materialized phases repeat (multiple epochs) to fill their budget; streamed
  phases must be large enough to supply theirs.
- **Schedule.** Set `schedule.max_steps=CURRICULUM.total_steps()` (validated).
  Epoch mode is not supported. Align `save_steps` with phase boundaries so a
  checkpoint exists exactly at each switch (unaligned boundaries warn) —
  useful for re-running followup phases from the same base.
- **Eval.** Fixed for the whole run: default (first phase's config), one
  explicit config, or named sets as above (reported as `eval_<name>_loss`;
  the first name drives best-model selection and early stopping). Reusing a
  phase's config as an eval entry is safe for hold-out corpora — the seeded
  shuffle holds out the same rows in both places.
- **Learning rate caveat.** Cosine-to-zero pairs badly with small late
  phases: the LR is nearly spent when the followup data arrives, which is the
  main caveat in the mid-training literature. Consider
  `lr_scheduler_type="warmup_stable_decay"` or a shorter decay if the
  followup phases matter most.

See `pre_gpt-neox_curriculum_geo_mist.py` for a complete manifest.

---

## Adding another corpus

1. Confirm the Hub card: config name, splits, and text column name.
2. Add a `HubTextSource` (and optionally a factory next to the existing three).
3. Prefer a dedicated validation split when one exists; otherwise hold-out eval.
4. Keep `block_size ≤ model.n_positions`. Prefer document-sized rows so
   `respect_document_boundaries=True` (default) yields useful blocks; use
   `False` only for short-row corpora like WikiText.
5. For huge corpora, default `mode="stream"` in zdeck manifests.
