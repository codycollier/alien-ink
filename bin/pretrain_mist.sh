#!/usr/bin/env bash
# Background pretrain zdeck program on Mist (local RTX 3070). Logs under output/.
#
# Uncomment one zdeck block below, then:
#   ./bin/pretrain_mist.sh
#
# W&B entity / project / name are set inside each zdeck module.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p output

# --- zdeck (uncomment one) ---------------------------------------------------
# module="alien_ink.zdeck.gpt2_wikipedia_5k"
# name="gpt2-wikipedia-5k-mist"

# module="alien_ink.zdeck.gpt2_wikitext_5k"
# name="gpt2-wikitext-5k-mist"

# module="alien_ink.zdeck.gemma_c4_5k"
# name="gemma-c4-5k-mist"

module="alien_ink.zdeck.gemma_c4_50k"
name="gemma-c4-50k-mist"

# module="alien_ink.zdeck.gemma_wikitext_4ep"
# name="gemma-wikitext-4ep-mist"

# module="alien_ink.zdeck.gpt_neox_wikitext_4ep"
# name="gpt-neox-wikitext-4ep-mist"
# ------------------------------------------------------------------------------

stamp=$(date +%Y%m%d-%H%M%S)
log="output/${name}-${stamp}.log"

nohup python -m "$module" "$@" >"$log" 2>&1 &

echo "pid $!  log $log  module $module"
