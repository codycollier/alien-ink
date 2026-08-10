#!/usr/bin/env bash
# Background pretrain zdeck program on Mist (local RTX 3070). Logs under output/train/.
#
# Uncomment one zdeck block below, then:
#   ./bin/pretrain_mist.sh
#
# W&B entity / project / name are set inside each zdeck module.
# Use `module=...` for importable packages, or `script=...` for hyphenated
# zdeck filenames that cannot be run with `python -m`.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p output/train

# --- zdeck (uncomment one) ---------------------------------------------------
module=""
script=""

# script="alien_ink/zdeck/pre_gpt-2_wikipedia_5k_mist.py"
# name="pre-gpt-2-wikipedia-5k-mist"

# script="alien_ink/zdeck/pre_gpt-2_wikitext_5k_mist.py"
# name="pre-gpt-2-wikitext-5k-mist"

# module="alien_ink.zdeck.pre_gemma_c4_5k_mist"
# name="pre-gemma-c4-5k-mist"

# module="alien_ink.zdeck.pre_gemma_c4_50k_mist"
# name="pre-gemma-c4-50k-mist"

# module="alien_ink.zdeck.pre_gemma_wikitext_4ep_mist"
# name="pre-gemma-wikitext-4ep-mist"

# script="alien_ink/zdeck/pre_gpt-neox_wikitext_4ep_mist.py"
# name="pre-gpt-neox-wikitext-4ep-mist"

# script="alien_ink/zdeck/pre_gpt-neox_wikitext_3ep_mist.py"
# name="pre-gpt-neox-wikitext-3ep-mist"

# module="alien_ink.zdeck.baseline_perf_mist"
# name="baseline-perf-mist"

module="alien_ink.zdeck.baseline_perf_gemma_mist"
name="baseline-perf-gemma-mist"

# ------------------------------------------------------------------------------

stamp=$(date +%Y%m%d-%H%M%S)
log="output/train/${name}-${stamp}.log"

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
