#!/usr/bin/env bash
# Background pretrain zdeck program on Mist (local RTX 3070). Logs under output/.
#
# Uncomment one zdeck block below, then:
#   ./bin/pretrain_mist.sh
#
# W&B entity / project / name are set inside each zdeck module.
# Use `module=...` for importable packages, or `script=...` for hyphenated
# zdeck filenames that cannot be run with `python -m`.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p output

# --- zdeck (uncomment one) ---------------------------------------------------
module=""
script=""

# script="alien_ink/zdeck/gpt-2_wikipedia_5k.py"
# name="gpt-2-wikipedia-5k-mist"

# script="alien_ink/zdeck/gpt-2_wikitext_5k.py"
# name="gpt-2-wikitext-5k-mist"

# module="alien_ink.zdeck.gemma_c4_5k"
# name="gemma-c4-5k-mist"

# module="alien_ink.zdeck.gemma_c4_50k"
# name="gemma-c4-50k-mist"

module="alien_ink.zdeck.gemma_wikitext_4ep"
name="gemma-wikitext-4ep-mist"

# script="alien_ink/zdeck/gpt-neox_wikitext_4ep.py"
# name="gpt-neox-wikitext-4ep-mist"

# script="alien_ink/zdeck/gpt-neox_wikitext_baseperf.py"
# name="gpt-neox-wikitext-baseperf-mist"

# ------------------------------------------------------------------------------

stamp=$(date +%Y%m%d-%H%M%S)
log="output/${name}-${stamp}.log"

if [[ -n "$script" ]]; then
  nohup python "$script" "$@" >"$log" 2>&1 &
  target="$script"
elif [[ -n "$module" ]]; then
  nohup python -m "$module" "$@" >"$log" 2>&1 &
  target="$module"
else
  echo "error: set module= or script= to a zdeck program" >&2
  exit 1
fi

echo "pid $!  log $log  target $target"
