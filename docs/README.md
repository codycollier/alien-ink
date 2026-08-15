# Docs

AI generated reference material for Alien Ink present and possible futures.

Alien Ink trains small causal language models on Mist (a local RTX 3070,
~8 GB VRAM). Every training run is a fully explicit `Manifest` — data, model,
hardware, wandb, schedule — archived as a program in the zdeck
(`alien_ink/zdeck/`). Two stages exist today: `pre` (from-scratch pretraining
of GPT-2, GPT-NeoX, Pythia, Llama/SmolLM2, and Gemma architectures) and `sft`
(full-parameter fine-tuning of pretrained checkpoints).

## Doc types

| Prefix | Meaning |
|---|---|
| `guide-` | Opinionated operating advice — how to run things well |
| `reference-` | How the code and data work today — kept in sync with the repo |
| `roadmap-` | Plans and speculation — what could come next, in decreasing certainty |
| `xp-` | Analysis of completed experiments and recommendations for follow-up work |

## Guides

| Doc | Contents |
|---|---|
| [Quickstart](guide-quickstart.md) | Setup, running zdeck programs, completions, evals — the daily commands |
| [RTX 3070 training](guide-rtx-3070-training.md) | Mist playbook — batch/accumulation settings per family, benchmark discipline |
| [Pythia on Mist](guide-pythia-chinchilla-mist.md) | OOM-safe Pythia options and Chinchilla token/step budgets for the RTX 3070 |

## References

| Doc | Contents |
|---|---|
| [Pretraining](reference-pretraining.md) | Pipeline, data config, load modes, packing, manifests, the zdeck |
| [Datasets](reference-datasets.md) | WikiText-103, English Wikipedia, C4, geo-us-states, curricula — sizes, character, Mist steps/epoch |
| [Model families](reference-model-families.md) | GPT-2, GPT-NeoX, Pythia, Gemma, Llama/SmolLM2 — architecture, tokenizers, VRAM, zdeck pairings; SFT of pretrained checkpoints |
| [GPU memory](reference-gpu-memory.md) | VRAM budget — params, Adam, activations, logits; Mist worked examples |
| [Completions and evals](reference-completions-and-eval.md) | Generation REPL, family-aware gen config, spot checks, the completion eval harness |

## Experimental analysis

| Doc | Contents |
|---|---|
| [Memorization experiments — 2026-08-14](xp-ai-analysis-2026-08-14.md) | Results from direct-completion and corpus memorization runs, evaluation caveats, and prioritized QA follow-up experiments |

## Roadmaps

Ordered from most to least certain:

| Doc | Status | Contents |
|---|---|---|
| [Learning plan](roadmap-learning-plan.md) | Living — Tracks A and B0 done | The master arc: pretraining → full SFT → LoRA → QLoRA |
| [Full fine-tuning on Mist](roadmap-full-finetune-mist.md) | Active | Full/partial fine-tuning of 70M–0.6B bases for downstream tasks; base-model scout, freeze ladder |
| [Large bases on Mist](roadmap-large-base-mist.md) | Speculative | QLoRA a 1B–8B pretrained base on the 3070, merge, quantize everything, run inference in 8 GB |

## Gaps

Not yet written, would be useful:

- **Manifest and schedule reference** — every `Manifest` / `ScheduleConfig` /
  `HardwareConfig` field, defaults, and validation rules in one place.
- **Outputs and artifacts** — the `output/train/<run_name>/` layout,
  checkpoint resume, W&B conventions, and what is safe to delete.
