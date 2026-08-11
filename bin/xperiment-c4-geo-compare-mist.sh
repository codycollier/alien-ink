#!/usr/bin/env bash
# Background C4 baseline vs curriculum-geo compare on Mist (local RTX 3070).
# Runs the full sequence into one log under output/logs/:
#
#   1. train pre_gpt-neox_c4_5k_mist
#   2. eval population-exact geo-us-states (baseline)
#   3. train pre_gpt-neox_curriculum_geo_mist
#   4. eval population-exact geo-us-states (curriculum)
#
#   ./bin/c4-geo-compare-mist.sh
#
# Override the eval file with EVALS=/path/to.json if needed.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p output/logs

evals="${EVALS:-/tmp/population-exact/geo-us-states.json}"
if [[ ! -f "$evals" ]]; then
  echo "error: evals file not found: $evals" >&2
  exit 1
fi

name="c4-geo-compare-mist"
stamp=$(date +%Y%m%d-%H%M%S)
log="output/logs/${name}-${stamp}.log"

nohup bash -c '
set -euo pipefail
evals="$1"

banner() {
  printf "\n======== %s ========\n" "$1"
}

#banner "1/4 train baseline (C4 5k)"
#python alien_ink/zdeck/pre_gpt-neox_c4_5k_mist.py

banner "2/4 eval baseline — population-exact geo-us-states"
./bin/model-eval-mist.py pre_gpt-neox_c4_5k_mist --evals "$evals"

#banner "3/4 train curriculum (C4 5k + geo 100)"
#python alien_ink/zdeck/pre_gpt-neox_curriculum_geo_mist.py

banner "4/4 eval curriculum — population-exact geo-us-states"
./bin/model-eval-mist.py pre_gpt-neox_curriculum_geo_mist --evals "$evals"

banner "done"
' _ "$evals" >"$log" 2>&1 &

echo "pid $!  log $log"
