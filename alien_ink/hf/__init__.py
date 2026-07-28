"""Hugging Face integrations for alien-ink.

Submodules:
  ds       — dataset streaming / Hub text corpora
  tok      — tokenize + pack LM blocks
  model    — GPT-2 build / checkpoint load
  trainer  — TrainingArguments / Trainer builders
  metrics  — FLOPs / throughput / run_config + run_summary artifacts
  hardware — AcceleratorProfile + get_profile (batch / run-name defaults)
  pretrain — GPT-2 pretrain config + runner (optional W&B via alien_ink.wb)
  gen      — generation / spot-check
"""
