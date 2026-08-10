"""Hugging Face integrations for alien-ink.

Submodules:
  ds       — dataset streaming / subset / complete Hub text corpora
  tok      — tokenize + pack LM blocks
  model    — GPT-2 / GPT-NeoX / Pythia / Gemma / Llama build / checkpoint load
  trainer  — TrainingArguments / Trainer builders
  metrics  — FLOPs / throughput / run_config + run_summary artifacts
  manifest — Manifest dataclass (data / model / hardware / wandb / schedule)
  pretrain — causal-LM pretrain entrypoint (optional W&B via alien_ink.com.wb)
  sft      — full-parameter fine-tune entrypoint for pretrained checkpoints
  gen      — generation / spot-check
"""
