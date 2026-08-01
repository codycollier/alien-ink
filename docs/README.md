# Docs

Reference material for Alien Ink pretraining: corpora, model families, GPU
memory, and how data flows from Hub text into checkpoints and completions.

| Doc | Contents |
|---|---|
| [Datasets](datasets.md) | WikiText-103, English Wikipedia, C4 — sizes, character, Mist steps/epoch |
| [Model families](model-families.md) | GPT-2, GPT-NeoX, Gemma — architecture, tokenizers, VRAM, zdeck pairings |
| [GPU memory](gpu-memory.md) | VRAM budget — params, Adam, activations, logits; Mist worked examples |
| [Pretraining and completion](pretraining-and-completion.md) | Pipeline, load modes, packing, manifests, generation REPL |

Start with **datasets** or **model families** for reference tables; use
**GPU memory** when sizing batch / accum for 8 GB; use **pretraining and
completion** for end-to-end workflow.
