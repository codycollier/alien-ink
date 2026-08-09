# Small language model learning plan

This plan moves Alien Ink through four increasingly practical stages on Mist:
an RTX 3070 with 8 GB VRAM, 16 CPU cores, and 94 GB RAM.

1. pretraining randomly initialized models;
2. full-parameter fine-tuning of small models;
3. parameter-efficient fine-tuning (PEFT), beginning with LoRA;
4. quantized PEFT, primarily QLoRA, for larger base models.

The stages share data preparation and evaluation, but their memory requirements
are different. Full-parameter training is the best way to understand optimizer
state and gradient flow, while PEFT is the practical route to adapting larger
models on an 8 GB GPU.

## Current repository baseline

Alien Ink currently implements three from-scratch families:

| Family | Size | Role | Recommendation |
|---|---:|---|---|
| GPT-2 | ~124M | Simple, cheap, well-understood baseline | Keep as the debugging and throughput baseline |
| GPT-NeoX | ~163M | RoPE-based modern architecture comparison | Keep; it is the best current comparison to GPT-2 |
| Mist-sized Gemma | ~165M | Modern block and large SentencePiece vocabulary experiment | Keep as an optional experiment, not the main English model |

The family definitions live in [`alien_ink/hf/model.py`](../alien_ink/hf/model.py),
and the architecture comparison is documented in
[`model-families.md`](model-families.md).

The Gemma experiment is especially expensive for its size: its approximately
256k vocabulary makes the embedding/head table and training logits large. The
current Gemma model is a small Gemma-shaped architecture, not a pretrained
Gemma-2B checkpoint.

The current training playbook recommends 1024-token blocks and an effective
batch of 32 blocks. GPT-2 and NeoX generally start at microbatch 4 with
accumulation 8; Gemma starts at microbatch 1 with accumulation 32 and gradient
checkpointing enabled. See [`rtx-3070-training.md`](rtx-3070-training.md).

## Track A: from-scratch pretraining

### A0. Establish the existing baseline

- Run a short GPT-2 124M smoke test on WikiText-103.
- Run the same test with GPT-NeoX.
- Record peak allocated/reserved VRAM, tokens per second, training loss, and
  validation loss.
- Compare models at equal tokens seen, not equal epochs.
- Keep the existing 5k-step programs for debugging; treat 75k–100k steps as a
  more meaningful quality-oriented target for these model sizes.

### A1. Add Pythia as the first new family/reference suite

Try:

- `EleutherAI/pythia-70m`
- `EleutherAI/pythia-160m`

Pythia is GPT-NeoX-based and is particularly useful for education and research:
the suite provides multiple sizes, consistent data ordering, and many
intermediate checkpoints. Pythia-160M is close to the existing NeoX scale, while
Pythia-70M gives much faster iteration.

Use Pythia in two ways:

- initialize a matching architecture from scratch and train on the local
  corpus;
- inspect published checkpoints to understand how loss and behavior evolve
  during training.

Reference: [Pythia-160M model card](https://huggingface.co/EleutherAI/pythia-160m).

Implementation work:

- Add an explicit `pythia` family only if exact configuration/reproducibility is
  required; otherwise document it as a NeoX-compatible variant.
- Match tokenizer vocabulary size, rotary settings, normalization, embedding
  tying, and initialization rather than assuming that equal width/depth means
  identical Pythia behavior.
- Add a Pythia zdeck using the same dataset, block size, and token budget as the
  GPT-2 and NeoX baselines.

### A2. Add a modern compact Llama-style experiment

Try `HuggingFaceTB/SmolLM2-135M` first, followed by
`HuggingFaceTB/SmolLM2-360M` if the smaller model is stable.

SmolLM2 provides compact 135M, 360M, and 1.7B variants. The 135M model is the
right first target for Mist; the 360M model should use shorter blocks or
microbatch 1 with gradient checkpointing until measured safe.

Reference: [SmolLM2-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
[SmolLM2-360M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M).

Implementation work:

- Add a Llama-style architecture path using the appropriate Hugging Face
  configuration rather than routing it through GPT-2 or NeoX.
- Preserve the model's tokenizer identity and vocabulary size.
- Add model-specific generation defaults and a zdeck.
- Compare GPT-2, NeoX/Pythia, and SmolLM2 at equal parameter counts or equal
  token budgets where possible.

### A3. Keep Gemma as a controlled ablation

Complete at least one Gemma run to learn the effect of vocabulary size and
SentencePiece tokenization. For English-only experiments, do not interpret the
165M parameter count as equivalent capacity to a 165M model with a 32k–50k
vocabulary: much of the Gemma parameter budget is in the tied embedding/head
table.

A separately trained 32k–64k tokenizer would be a more meaningful English
Gemma experiment than shrinking the model while retaining Gemma's 256k
tokenizer.

## Track B: full-parameter fine-tuning

Alien Ink currently has the `sft` manifest stage as a reserved concept, but
`Manifest.train()` does not implement it yet. Complete the fine-tuning runtime
before adding many pretrained models. The first implementation should support
ordinary full-parameter SFT, without quantization or adapters, so the training
mechanics remain visible and easy to debug.

### B0. Implement full-parameter SFT

Add the following in small, composable pieces:

- generic `AutoModelForCausalLM` checkpoint loading;
- model/tokenizer metadata recorded in the manifest;
- supervised dataset formatting and chat-template support;
- labels masking for prompt versus completion tokens;
- evaluation and generation against a fixed held-out set;
- full-parameter checkpoint saving and resume;
- a small SFT zdeck with an explicit `stage="sft"`;
- clear reporting of total and trainable parameter counts.

Do not make the fine-tuning path depend on the three hard-coded architecture
branches used by the current from-scratch loader. Model families such as Qwen
and SmolLM2 should be loadable through their serialized Hugging Face config.

### B1. Full fine-tune an Alien Ink checkpoint

Before fine-tuning a large public base model, fine-tune one of the models
trained by Alien Ink itself:

1. GPT-2 124M;
2. Pythia/NeoX 160M;
3. SmolLM2-135M, once its architecture path is implemented.

Use a tiny, known dataset first. Verify that:

- training loss decreases;
- the model improves on examples from the task;
- validation loss does not immediately diverge;
- all intended parameters receive gradients;
- a saved checkpoint resumes with equivalent results.

The first full fine-tuning run should use microbatch 1–2, gradient
checkpointing, and a short sequence length if necessary. Do not optimize for
throughput until labels, masking, and resume behavior are correct.

### B2. Full fine-tune a public small base

Use `Qwen/Qwen2.5-0.5B` only after the 124M–160M full fine-tuning path is
verified. It is a 0.49B-parameter base model with RoPE, SwiGLU, RMSNorm, GQA,
tied word embeddings, and a 32k context window.

Attempt this as a full fine-tune only with short sequences, microbatch 1,
gradient checkpointing, and measured VRAM headroom. If optimizer state or
activations exceed the card's budget, stop at the smaller Alien Ink models and
move to the PEFT stage rather than silently offloading the experiment.

Reference: [Qwen2.5-0.5B model card](https://huggingface.co/Qwen/Qwen2.5-0.5B).

## Track C: PEFT and LoRA fine-tuning

PEFT means parameter-efficient fine-tuning: the base model is mostly frozen and
only a small number of additional or selected parameters are trained. LoRA is
the first PEFT method to implement because it is widely supported and exposes a
clear rank/quality/memory tradeoff.

### C0. Add a PEFT layer without changing the SFT data path

Keep the dataset formatting, masking, evaluation, and manifest structure from
full-parameter SFT. Add only the adaptation strategy:

- optional `peft` dependency;
- LoRA configuration in the manifest;
- target-module selection by model family;
- trainable versus frozen parameter reporting;
- adapter-only checkpoint saving;
- loading a base checkpoint plus adapter for generation;
- merge/unmerge behavior tested explicitly.

The first LoRA experiment should use GPT-2 124M or Pythia-160M. Compare full
fine-tuning and LoRA on the same task, dataset, seed, and token budget. Record
whether the smaller trainable parameter count changes convergence or final
quality.

### C1. LoRA on modern compact bases

Try the following in order:

1. `HuggingFaceTB/SmolLM2-135M`;
2. `HuggingFaceTB/SmolLM2-360M`;
3. `Qwen/Qwen2.5-0.5B`;
4. `Qwen/Qwen3-0.6B-Base`.

SmolLM2 has compact 135M, 360M, and 1.7B variants and published SFT recipes.
The 135M and 360M models are useful for comparing full fine-tuning against
LoRA at relatively manageable sizes.

References: [SmolLM2-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
[SmolLM2-360M model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M).

Qwen2.5-0.5B is a practical modern base for multilingual and code-oriented
experiments. Qwen3-0.6B-Base is a newer comparison with 28 layers, GQA, and a
32k context window. Qwen3 requires Transformers 4.51 or newer; raise the
project's declared minimum if it becomes a supported example.

References: [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B),
[Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base).

### C2. Add quantized PEFT / QLoRA

After ordinary LoRA works, add 4-bit loading and QLoRA. Keep quantization
separate from the LoRA implementation so failures can be localized.

Add:

- an optional CUDA dependency set for 4-bit loading;
- explicit compute dtype and quantization configuration;
- tests that confirm base parameters are frozen;
- VRAM and throughput measurements for full-precision LoRA versus QLoRA;
- adapter checkpoints that never require saving a second full copy of the base
  model.

Use QLoRA for the first 1B-class experiment:

- `HuggingFaceTB/SmolLM2-1.7B`;
- `allenai/OLMo-2-0425-1B`;
- TinyLlama 1.1B.

These should be treated as QLoRA targets on the 8 GB card. Full Adam
fine-tuning is not the appropriate first experiment at this scale.

References: [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B),
[OLMo 2 1B](https://huggingface.co/allenai/OLMo-2-0425-1B),
[TinyLlama 1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T).

### C3. PEFT methods after LoRA

Once LoRA and QLoRA are measured and reproducible, add other PEFT methods as
controlled comparisons:

- adapters;
- IA3;
- prefix tuning;
- prompt tuning;
- selective unfreezing of embeddings, normalization, or final layers.

Do not add these methods before the LoRA baseline is stable. The purpose of the
comparison is to understand which parameters need to move for a task, not to
accumulate adapter types without a common evaluation protocol.

## Suggested sequence

1. GPT-2 124M pretraining smoke test and baseline.
2. GPT-NeoX 163M pretraining comparison at equal tokens.
3. Pythia-70M from-scratch pretraining run.
4. Pythia-160M comparison and checkpoint inspection.
5. SmolLM2-135M from-scratch architecture experiment.
6. Optional Gemma vocabulary experiment.
7. Implement generic checkpoint loading and full-parameter SFT.
8. Full fine-tune an Alien Ink GPT-2, NeoX/Pythia, or SmolLM2 checkpoint.
9. Full fine-tune Qwen2.5-0.5B only if measured VRAM permits it.
10. Add PEFT infrastructure and reproduce the same task with LoRA.
11. Compare full fine-tuning and LoRA on 135M–500M bases.
12. Add QLoRA and fine-tune Qwen2.5-0.5B or Qwen3-0.6B-Base.
13. Try SmolLM2-1.7B, OLMo-2 1B, or TinyLlama 1.1B with QLoRA.
14. Add other PEFT methods after LoRA/QLoRA are stable.

## Evaluation rules

For every from-scratch comparison, hold constant:

- dataset and data split;
- tokenizer-specific token budget;
- block size;
- effective batch size;
- optimizer and learning-rate schedule;
- random seed;
- validation data and prompt set.

Record:

- parameter count and non-embedding parameter count;
- tokenizer vocabulary size and tokens-per-character ratio;
- peak allocated and reserved VRAM;
- tokens per second;
- training and validation loss;
- total tokens seen;
- checkpoint size and wall-clock time.

For fine-tuning, additionally record:

- fine-tuning mode: full parameters, LoRA, or another PEFT method;
- trainable parameter count and percentage;
- adapter rank and target modules;
- quantization mode;
- sequence length and packing behavior;
- base-model versus adapter storage size;
- task-specific held-out metrics.

The goal is not to find one universally best model. The goal is to build a
measured understanding of how tokenizer, architecture, data, parameter count,
and adaptation method interact under Mist's memory budget.
