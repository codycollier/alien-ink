#!/usr/bin/env bash
# Background GPT-2 pretrain on Mist (local RTX 3070). Logs under output/ with a timestamp.
#
# Uncomment one mode and one dataset block below, then:
#   ./bin/gpt2_pretrain_mist.sh
# Optional CLI overrides are forwarded (e.g. --max-steps 1000 --no-wandb).
#
# W&B defaults: entity logbook, project ink-explore.
# Run names get a -gpu suffix from alien_ink (e.g. …-mist-gpu).
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p output

entity="logbook"
project="ink-explore"

# --- mode (uncomment one) -----------------------------------------------------
mode="--train"
# mode="--flight-check"

# --- dataset (uncomment one module + name pair) -------------------------------
# WikiText-103 (full stream)
# module="alien_ink.exp.gpt2_pretrain_wikitext"
# name="gpt2-pretrain-wikitext-mist"

# WikiText-103 (20k subset)
# module="alien_ink.exp.gpt2_pretrain_wikitext_subset"
# name="gpt2-pretrain-wikitext-subset-mist"

# English Wikipedia (full stream)
# module="alien_ink.exp.gpt2_pretrain_wikipedia_english"
# name="gpt2-pretrain-wpe-mist"

# English Wikipedia (20k subset)
module="alien_ink.exp.gpt2_pretrain_wikipedia_english_subset"
name="gpt2-pretrain-wpe-subset-mist"

# C4 English (full stream)
# module="alien_ink.exp.gpt2_pretrain_c4"
# name="gpt2-pretrain-c4-mist"

# C4 English (20k subset)
# module="alien_ink.exp.gpt2_pretrain_c4_subset"
# name="gpt2-pretrain-c4-subset-mist"

# ------------------------------------------------------------------------------
stamp=$(date +%Y%m%d-%H%M%S)
log="output/${name}-${stamp}.log"

nohup python -m "$module" \
  "$mode" --wandb-entity "$entity" --wandb-project "$project" --wandb-name "$name" "$@" \
  >"$log" 2>&1 &

echo "pid $!  log $log  mode $mode  module $module"
echo "wandb_entity $entity  wandb_project $project  wandb_name $name"
