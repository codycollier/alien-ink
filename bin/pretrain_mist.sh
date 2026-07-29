#!/usr/bin/env bash
# Background pretrain sample on Mist (local RTX 3070). Logs under output/.
#
# Uncomment one sample block below, then:
#   ./bin/pretrain_mist.sh
#
# W&B entity / project / name are set inside each sample module.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p output

# --- sample (uncomment one) ---------------------------------------------------
# module="alien_ink.samples.gpt2_wikipedia_5k"
# name="gpt2-wikipedia-5k-mist"

# module="alien_ink.samples.gpt2_wikitext_5k"
# name="gpt2-wikitext-5k-mist"

module="alien_ink.samples.gemma_c4_5k"
name="gemma-c4-5k-mist"
# ------------------------------------------------------------------------------

stamp=$(date +%Y%m%d-%H%M%S)
log="output/${name}-${stamp}.log"

nohup python -m "$module" "$@" >"$log" 2>&1 &

echo "pid $!  log $log  module $module"
