# Roadmap: full fine-tuning for downstream tasks on Mist

**Status: active path.** Pretraining from scratch (Track A) taught how models
form. This roadmap teaches how they *specialize*: full-parameter — and then
partial-parameter — fine-tuning of published base checkpoints for concrete
downstream tasks, entirely within Mist (RTX 3070, 8 GB VRAM, 16 cores, 94 GB
RAM). No LoRA, no adapters, no quantized bases — that is the
[large-base roadmap](roadmap-large-base-mist.md). Here every trained weight
is a real weight, so optimizer state, gradient flow, freezing, and forgetting
stay fully visible.

```
        Track A (done)              this roadmap                    later
  ┌────────────────────┐   ┌──────────────────────────┐   ┌──────────────────┐
  │ pretrain from      │ → │ full / partial fine-tune │ → │ PEFT · LoRA ·    │
  │ scratch, 70–165M   │   │ published bases 70M–0.6B │   │ QLoRA, 1B–8B     │
  └────────────────────┘   └──────────────────────────┘   └──────────────────┘
```

Two destination task shapes, both served by the same causal-LM machinery:

| | Shape I — judge | Shape II — sage |
|---|---|---|
| Task class | Supervised: classification, labeling, extraction | Generative: free-form QA over a small corpus |
| Output space | Closed — one of k short labels | Open — sentences grounded in the corpus |
| Training form | prompt → short completion, prompt tokens masked | corpus absorption, then QA-pair SFT |
| Primary metric | exact-match rate | token F1 / ROUGE-L + teacher-forced ppl |
| Harness support | `exact_rate` in `alien_ink.hf.eval` — already built | `token_f1` / `rouge_l` / `mean_ppl` — already built |

---

## Where the repository already stands

Track B0 of the [learning plan](roadmap-learning-plan.md) is implemented,
so this roadmap starts warm, not cold:

- `Manifest(stage="sft")` loads any Hub or local checkpoint via
  `PretrainedLmConfig` + `AutoModelForCausalLM` (`alien_ink.hf.sft`);
- two checked-in SFT zdecks (`sft_pythia-160m_geo_mist`,
  `sft_smollm2-135m_geo_mist`) fine-tune on geo-us-states;
- the completion eval harness (`alien_ink.hf.eval`) scores exact/prefix
  match, char similarity, token F1, ROUGE-L, and teacher-forced loss/ppl
  against an external JSON prompt/completion file;
- trainable-parameter counting and reporting are wired into every run.

The one structural gap: the SFT data path still reuses pretraining's packed
blocks with `labels = input_ids`. Downstream tasks need **prompt/completion
pairs with the prompt masked out of the loss**. That gap is phase F1, and it
is the only new machinery this roadmap requires — everything after is
experiments and zdecks.

---

## The memory line for full fine-tuning

Full fine-tuning pays the full static bill from
[the GPU memory reference](reference-gpu-memory.md): `P × 12–16` bytes for bf16 weights + grads +
fp32 Adam, plus logits `B × S × V × 4` and ~1.5 GiB of activations/overhead.
That draws a hard line across the model zoo:

```
Full fine-tune static floor (P × 12) + logits @ B=1, S=1024, fp32 CE

Pythia-70M      ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   ~1.0 GiB   trivial
SmolLM2-135M    █████░░░░░░░░░░░░░░░░░░░░░░░░░░░   ~1.7      comfortable
Pythia-160M     ██████░░░░░░░░░░░░░░░░░░░░░░░░░░   ~2.0      comfortable
Gemma-3-270M    ████████████████░░░░░░░░░░░░░░░░   ~4.0      workable (256k vocab logits)
Granite-4-350M  █████████████████░░░░░░░░░░░░░░░   ~4.3      workable — B=1, ckpt on
SmolLM2-360M    ███████████████████░░░░░░░░░░░░░   ~4.7      tight — B=1, ckpt on
Pythia-410M     █████████████████████░░░░░░░░░░░   ~5.3      ceiling — measure first
Qwen2.5-0.5B    █████████████████████████░░░░░░░   ~6.1+     over the line w/ fp32 Adam
Qwen3-0.6B      ████████████████████████████████   ~7.2+     over the line w/ fp32 Adam
                0    1    2    3    4    5    6    7    8 GiB
```

Two levers move the line without leaving full fine-tuning:

| Lever | What it does | Cost |
|---|---|---|
| **8-bit Adam** (`optim="adamw_8bit"`) | m+v drop from 8 to ~2 B/param → static ~6 B/param. Qwen2.5-0.5B falls to ~2.8 GiB static; Qwen3-0.6B to ~3.4 GiB | Slightly noisier optimizer; still every weight trains |
| **Partial freezing** | Frozen params need no grads and no Adam: static ≈ `P_train × 12 + P_frozen × 2`. Freezing Gemma-3-270M's ~170M-param embedding table erases ~1.9 GiB of state | Frozen layers stop adapting — which is the experiment |

Freezing is therefore not a consolation prize: it is both the partial
fine-tuning curriculum *and* the memory lever that makes the 0.5–0.6B class
honest on this card. Sequence length remains the escape valve
(1024 → 512 → 256), same as everywhere else on Mist.

---

## Base model scout

Bases, not instruct-tunes: instruction-tuned variants bake in chat templates
and alignment that confound small clean experiments. All rows verified
available on the Hub as of August 2026.

| Base | Params | Arch notes | Vocab | Ctx | License | Role on Mist |
|---|---:|---|---:|---:|---|---|
| [`EleutherAI/pythia-70m`](https://huggingface.co/EleutherAI/pythia-70m) | 70M | NeoX, partial RoPE, untied | ~50k | 2k | Apache-2.0 | Fast iteration; debugging the masking path |
| [`EleutherAI/pythia-160m`](https://huggingface.co/EleutherAI/pythia-160m) | 160M | NeoX; matches Mist NeoX dims | ~50k | 2k | Apache-2.0 | Anchor base — already the first SFT zdeck |
| [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | 135M | Llama block: RoPE, SwiGLU, RMSNorm, GQA | ~49k | 8k | Apache-2.0 | Modern-arch anchor; already a zdeck |
| [`HuggingFaceTB/SmolLM2-360M`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) | 360M | Same block, 32 layers | ~49k | 8k | Apache-2.0 | First scale step; strong ≤400M English base |
| [`ibm-granite/granite-4.0-350m-base`](https://huggingface.co/ibm-granite/granite-4.0-350m-base) | 350M | Dense transformer: GQA, SwiGLU, RMSNorm, tied embeds; 28 × 1024 | ~100k | 32k | Apache-2.0 | 2025-vintage 350M-class alternative to SmolLM2-360M; a hybrid-Mamba2 `-h-` twin exists for arch ablations |
| [`EleutherAI/pythia-410m`](https://huggingface.co/EleutherAI/pythia-410m) | 410M | NeoX, 24 × 1024 | ~50k | 2k | Apache-2.0 | Scale step inside one suite — clean size ablations vs 70M/160M |
| [`google/gemma-3-270m`](https://huggingface.co/google/gemma-3-270m) | 270M | ~170M of it is the embedding table | ~262k | 32k | Gemma terms (gated) | Purpose-built for task fine-tuning; the embed-freeze case study |
| [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) | 0.49B | GQA, tied embeddings | ~152k | 32k | Apache-2.0 | First over-the-line target; needs 8-bit Adam or freezing |
| [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | 0.6B | 28 layers, GQA; transformers ≥ 4.51 | ~152k | 32k | Apache-2.0 | The ceiling; 2025-consensus best tunability under 1B |
| [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | 0.8B | Hybrid Gated DeltaNet + gated attention; **natively multimodal** (vision encoder); latest transformers only | ~248k (tied) | 262k | Apache-2.0 | 2026 successor above the ceiling — needs 8-bit Adam *and* embed-freeze; verify a text-only `AutoModelForCausalLM` load path before designing around it |
| [`LiquidAI/LFM2-350M`](https://huggingface.co/LiquidAI/LFM2-350M) | 350M | Hybrid conv + attention | ~65k | 32k | LFM Open License | Optional curiosity — tops 2025/26 tunability benchmarks; nonstandard arch, verify `AutoModel` path first. Succeeded by LFM2.5 (its explicit `-Base` checkpoint is 1.2B — over this line) |

Beyond the line — `Llama-3.2-1B` (1.23B, 128k vocab), `OLMo-2-1B`,
`SmolLM2-1.7B`, `MiniCPM5-1B-Base` (1.08B, standard Llama arch),
`LFM2.5-1.2B-Base`, `granite-4.0-1b-base` (really 1.6B) — full Adam state
alone exceeds the card. They stay in the
[large-base roadmap](roadmap-large-base-mist.md) where QLoRA pays their rent.

Scouting rules that generalize past this table:

- prefer **suites** (Pythia, SmolLM2, Qwen, Granite) — size ablations inside
  one family isolate scale from architecture;
- small vocab (~50k) keeps logits cheap; 152–262k vocabs re-run the Mist
  Gemma lesson at fine-tuning time — budget `B × S × V × 4` before starting;
- check the base's pretraining data recency if the task needs world
  knowledge; Pythia (2023 Pile) knows less than Qwen3 (2025 corpus);
- mind the **transformers floor**: the repo pins `>=4.46`; Qwen3 needs
  ≥ 4.51, SmolLM3 ≥ 4.53, and 2026 hybrids (Qwen3.5, LFM2.5) want a current
  release — check the model card before adding a zdeck;
- prefer plain dense transformers for clean experiments; 2026's hybrid
  (DeltaNet / Mamba2 / conv) and multimodal bases are interesting but add
  load-path and PEFT-targeting risk that has nothing to do with the task.

---

## Where instruction tuning fits

Nothing here is missing for task-specific outcomes: F1's prompt/completion
path with masked loss *is* the mechanism instruction tuning uses, applied to
one instruction instead of thousands. Three positions, in decreasing
relevance to this roadmap:

- **Task-specific SFT (this roadmap).** One task, one template, loss on the
  completion. All model capacity goes to the task, and `exact_rate` /
  token F1 measure it directly. At 70M–0.6B this beats prompting a
  generalist.
- **Base vs instruct ablation (allowed override).** The scout table excludes
  instruct-tunes because chat templates and alignment confound clean
  experiments — a learning rule, not a law. When a task is purely
  outcome-driven and the base run underperforms, fine-tune the instruct
  variant of the same base (`Qwen3-0.6B` vs `Qwen3-0.6B-Base`) on the same
  task data: same manifest, different `model_name`. The delta is the
  measured value of the instruction prior for that task.
- **General instruction tuning (later, Track C era).** Broad multi-task
  instruction data (SmolTalk, Dolly, Tulu-style mixtures) plus a chat
  template produces a model that follows *novel* instructions without
  retraining. Once F1 exists, an instruction mixture is just another dataset
  through the same masked path — but sub-1B instruction generality is weak,
  so it earns its slot only after the per-task recipes are measured.

---

## Phases

### F0 — Baseline what exists

Run both checked-in SFT zdecks to completion; write a geo eval JSON and score
it with `run_completion_eval` against base and fine-tuned checkpoints.
Record: trainable %, peak VRAM, tokens/sec, eval deltas. This is the control
group for everything below, and it costs one evening.

### F1 — The task data path (the real code work)

Add prompt/completion supervised data alongside — not replacing — the packed
path:

- a prompt/completion source (`prompt_column` / `completion_column`, or a
  small template over dataset fields) parallel to `HubTextSource`;
- tokenize prompt and completion separately; `labels = -100` over prompt and
  padding, real ids over completion;
- pad-to-longest batches instead of packing; keep `block_size` as truncation;
- an explicit `loss_on_prompt: bool` — completion-only vs full-sequence loss
  is itself a worthwhile measured ablation at small data sizes;
- manifest records the template and column mapping verbatim.

Verify on Pythia-70M with a 100-example toy set: masked positions contribute
zero loss, the model memorizes the toy set, resume works. Functions over
classes; the collator is a function, not a framework.

### F2 — Shape I: supervised classification as generation

Render classification as next-token prediction: `"{text}\nlabel:"` →
`" world"`. Greedy decode, score with `exact_rate`.

| Dataset | Hub | Task | Size | Why it fits |
|---|---|---|---:|---|
| AG News | `fancyzhx/ag_news` | 4-way news topic | 120k | Clean, boring, well-studied — perfect first target |
| SST-2 | `stanfordnlp/sst2` | binary sentiment | 67k | Single-token labels; fastest signal |
| TREC | `CogComp/trec` | 6-way question type | 5.4k | Small enough to overfit deliberately and observe it |

Protocol: Pythia-160M and SmolLM2-135M on AG News, LR 1e-5–5e-5, 1–3 epochs,
warmup ratio 0.03, cosine or constant. Sweep nothing until the first run
lands. Then the custom classification task swaps in through the same source +
template — that is the entire point of the manifest.

### F3 — Shape II: generative QA on a small corpus

Two stages, two manifests, one lineage — this is where "continued
pretraining" and "SFT" become distinct tools instead of synonyms:

```
  ┌─────────────────────────┐      ┌─────────────────────────┐
  │ F3a: absorb             │      │ F3b: answer             │
  │ stage="sft", packed     │  →   │ stage="sft", masked     │
  │ corpus (existing path)  │      │ QA pairs (F1 path)      │
  │ base + geo-us-states    │      │ + QA eval JSON          │
  └─────────────────────────┘      └─────────────────────────┘
        domain knowledge                 answering behavior
```

- **F3a** is exactly the existing geo zdecks: packed causal-LM on the corpus.
- **F3b** fine-tunes the F3a checkpoint (local path as `model_name`) on
  QA pairs over the same corpus — a few hundred, hand-written or generated
  from the documents and then hand-checked.
- Eval: held-out QA items scored on token F1 / ROUGE-L / teacher-forced ppl;
  hold out whole documents, not just questions, to test composition over
  absorbed knowledge rather than recall of training pairs.
- Ablate the ordering: base→F3b directly (no absorption) vs F3a→F3b. The
  delta is the measured value of continued pretraining.

Scaffolding sets before the custom corpus: `rajpurkar/squad` (extractive QA
formatting at scale), `databricks/databricks-dolly-15k` (15k human-written
instruction/response pairs, CC BY-SA, includes closed-book QA over provided
context — the closest public analog to "QA over my documents").

### F4 — Partial fine-tuning: the freeze ladder

Add a freeze spec to the manifest — declarative, recorded, reported:

```python
model=PretrainedLmConfig(
    model_name="HuggingFaceTB/SmolLM2-360M",
    freeze=("embeddings", "layers:0-15"),   # patterns → requires_grad=False
)
```

Then descend the ladder on one fixed task/seed/budget from F2 or F3:

| Rung | Trainable | Hypothesis under test |
|---|---|---|
| Full fine-tune | 100% | Reference quality and reference VRAM |
| Freeze embeddings | ~70–95% | Token geometry is reusable; big static savings on fat-vocab models |
| Freeze bottom half | ~50% | Low layers hold general features; tasks live high |
| Top-2 layers + head | ~10–15% | The cheap-adaptation floor — how much task fits in the head |

Record quality vs trainable % vs peak VRAM on one plot per task. This ladder
is the conceptual bridge to PEFT: LoRA is the same question — *which
parameters need to move?* — answered with different machinery.

### F5 — Scale to the ceiling

Take the best F2/F4 recipe up the scout table: SmolLM2-360M → Pythia-410M →
Qwen2.5-0.5B → Qwen3-0.6B-Base, adding `adamw_8bit` and/or embedding-freeze
where the line demands it. Treat each first run as a memory measurement.
Add a **forgetting probe** from F5 onward: fixed WikiText validation
perplexity plus a fixed general prompt set, scored before and after every
fine-tune. Full fine-tuning at LR 3e-5 for 3 epochs can quietly lobotomize a
0.5B base; the probe makes it a number instead of a vibe.

---

## Alien Ink adaptation checklist

In keeping with the house rules — variables in the manifest, functions over
classes, quantization and adapters out of scope here:

- [ ] prompt/completion data source + masking collator (F1) beside the
      packed path, sharing tokenizer handling;
- [ ] `loss_on_prompt` flag, recorded in `run_config.json`;
- [ ] `freeze` patterns on `PretrainedLmConfig` (F4), applied at load,
      reflected in the existing trainable-params report;
- [ ] eval extension: per-class accuracy for Shape I (aggregate `exact_rate`
      already exists);
- [ ] forgetting probe as a second eval JSON + a general-ppl helper (F5);
- [ ] zdecks in the established grammar as each phase lands:
      `sft_pythia-160m_agnews_mist`, `sft_smollm2-360m_geoqa_mist`,
      `sft_qwen3-0.6b_agnews_frozen-embed_mist`;
- [ ] optional dep check: `bitsandbytes` only if/when `adamw_8bit` enters
      (F5); transformers ≥ 4.51 gate for Qwen3.

---

## Operating rules

Carried over from the learning plan's evaluation rules, plus fine-tuning
specifics:

- **LR discipline**: 1e-5–5e-5, ~20× below pretraining; warmup 3%;
  `adam_beta2=0.999`. The existing geo zdecks already encode this.
- **Hold constant per comparison**: base checkpoint + revision, data, seed,
  token budget, sequence length, eval set. Change one thing.
- **Record always**: trainable params and %, frozen-pattern spec, optimizer
  variant, peak allocated/reserved VRAM, tokens/sec, task metric, and (from
  F5) the forgetting probe.
- **Correctness before throughput**: `torch_compile` off and checkpointing
  on for every first run of a new base or new data path.
- **Overfit on purpose once per task**: a model that cannot memorize 100
  examples has a bug, not a capacity problem.

## Suggested sequence

1. F0 — run and eval both existing geo SFT zdecks; archive the numbers.
2. F1 — masked prompt/completion path; verify on Pythia-70M + toy set.
3. F2 — AG News on Pythia-160M and SmolLM2-135M; first exact-match numbers.
4. F3 — geo QA: absorb → answer; ablate skipping absorption.
5. F4 — freeze ladder on the best F2 task; quality vs trainable % vs VRAM.
6. F5 — SmolLM2-360M, then Pythia-410M; add the forgetting probe.
7. F5+ — Qwen2.5-0.5B / Qwen3-0.6B-Base with 8-bit Adam or frozen
   embeddings; first runs are memory measurements.
8. Swap in the custom datasets/tasks/evals through the same manifests —
   by now that is a data change, not a code change.
9. Hand off to Track C: re-run one finished task with LoRA and compare.

## Sources

- [Pythia: a suite for analyzing LLMs across training and scaling](https://arxiv.org/abs/2304.01373) — the reference suite; sizes 70M–12B.
- [SmolLM2 (HuggingFaceTB)](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) — compact Llama-block bases with published recipes.
- [Gemma 3 270M](https://developers.googleblog.com/en/introducing-gemma-3-270m/) — a small base explicitly positioned for task-specific fine-tuning.
- [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) — current consensus pick for sub-1B tunability.
- [distil labs small-model fine-tuning benchmark (2026)](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning/) — tunability rankings behind the scout table.
- [Granite 4.0 Nano (IBM, Oct 2025)](https://huggingface.co/blog/ibm-granite/granite-4-nano) — 350M/1B base checkpoints in dense and hybrid-Mamba2 variants, Apache-2.0.
- [Qwen3.5 (Feb–Mar 2026)](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) — multimodal hybrid-DeltaNet successors to Qwen3; base variants at 0.8B / 2B / 4B.
- [8-bit optimizers (bitsandbytes)](https://arxiv.org/abs/2110.02861) — the Adam-state lever.
- [Universal Language Model Fine-tuning (ULMFiT)](https://arxiv.org/abs/1801.06146) — gradual unfreezing; the intellectual ancestor of the F4 ladder.
