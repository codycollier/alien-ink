#!/usr/bin/env bash
# Background sample pretrain on Mist (local RTX 3070). Logs under output/.
#
# Uncomment one sample below, then:
#   ./bin/pretrain_mist.sh
# Extra args are forwarded (e.g. --wandb).
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p output

# --- sample (uncomment one) ---------------------------------------------------
module="alien_ink.samples.pretrain_wikipedia_5k"
name="sample-gpt2-wikipedia-5k"

# module="alien_ink.samples.pretrain_wikitext_50k"
# name="sample-gpt2-wikitext-50k"
# ------------------------------------------------------------------------------

stamp=$(date +%Y%m%d-%H%M%S)
log="output/${name}-${stamp}.log"

nohup python -m "$module" "$@" >"$log" 2>&1 &

echo "pid $!  log $log  module $module"
