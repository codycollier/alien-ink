#!/usr/bin/env bash
# Background baseline-perf zdeck program on Mist (local RTX 3070). Logs under output/train/.
#
#   ./bin/mist-train-baseline-perf.sh
#
# GPT-NeoX from scratch, 0.25 epochs on WikiText-103 (complete). W&B entity /
# project / name are set inside the zdeck module.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p output/train

module="alien_ink.zdeck.baseline_perf_mist"
name="baseline-perf-mist"

stamp=$(date +%Y%m%d-%H%M%S)
log="output/train/${name}-${stamp}.log"

nohup python -m "$module" "$@" >"$log" 2>&1 &

echo "pid $!  log $log  target $module"
