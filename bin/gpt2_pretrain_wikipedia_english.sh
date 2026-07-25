#!/usr/bin/env bash
# Background GPT-2 Wikipedia pretrain; logs under output/ with a timestamp.
#
# Optional W&B overrides (CLI flags only):
#   ./bin/gpt2_pretrain_wikipedia_english.sh --wandb-project my-proj --wandb-name my-run
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p output

stamp=$(date +%Y%m%d-%H%M%S)
name="gpt2-pretrain-wiki-eng-${stamp}"
log="output/${name}.log"

nohup python -m alien_ink.exp.gpt2_pretrain_wikipedia_english \
  --train --wandb-name "$name" "$@" \
  >"$log" 2>&1 &

echo "pid $!  log $log  wandb_name $name"
