# Docs

AI generated reference material for Alien Ink model training.


| Doc | Contents |
|---|---|
| [Datasets](datasets.md) | WikiText-103, English Wikipedia, C4 — sizes, character, Mist steps/epoch |
| [Model families](model-families.md) | GPT-2, GPT-NeoX, Pythia, Gemma, Llama/SmolLM2 — architecture, tokenizers, VRAM, zdeck pairings; SFT of pretrained checkpoints |
| [Model learning plan](model-learning-plan.md) | From-scratch curriculum, recommended base models, and the path to LoRA/QLoRA |
| [Roadmap: full fine-tuning on Mist](roadmap-full-finetune-mist.md) | Active — full/partial fine-tuning of 70M–0.6B bases for downstream tasks; base-model scout, freeze ladder, example tasks/evals |
| [Roadmap: large bases on Mist](roadmap-large-base-mist.md) | Speculative — QLoRA a 1B–8B pretrained base on the 3070, merge, quantize everything, run inference in 8 GB |
| [GPU memory](gpu-memory.md) | VRAM budget — params, Adam, activations, logits; Mist worked examples |
| [Pretraining and completion](pretraining-and-completion.md) | Pipeline, load modes, packing, manifests, generation REPL |

