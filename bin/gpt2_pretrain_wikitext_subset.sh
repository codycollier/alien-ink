#!/usr/bin/env bash
# Background GPT-2 WikiText-103 subset pretrain; logs under output/ with a timestamp.
#
# Defaults: W&B entity logbook, project ink-explore, run name gpt2-pretrain-wikitext-subset
# (flight-check run name: gpt2-pretrain-wikitext-subset-flight-check).
# Optional overrides:
#   ./bin/gpt2_pretrain_wikitext_subset.sh --wandb-entity other --wandb-project other-proj
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p output

stamp=$(date +%Y%m%d-%H%M%S)
entity="logbook"
project="ink-explore"
name="gpt2-pretrain-wikitext-subset-mist"
log="output/${name}-${stamp}.log"

nohup python -m alien_ink.exp.gpt2_pretrain_wikitext_subset \
  --train --wandb-entity "$entity" --wandb-project "$project" --wandb-name "$name" "$@" \
  >"$log" 2>&1 &

echo "pid $!  log $log  wandb_entity $entity  wandb_project $project  wandb_name $name"
