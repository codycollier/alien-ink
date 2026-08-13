#!/usr/bin/env bash
# Background baseline-perf zdeck program on Mist (local RTX 3070). Logs under output/train/.
#
#   ./bin/mist-train-baseline-perf.sh gpt-2
#   ./bin/mist-train-baseline-perf.sh gpt-neox
#   ./bin/mist-train-baseline-perf.sh pythia
#
# Each model runs from scratch for 0.25 epochs on the same complete WikiText-103
# dataset. W&B entity / project / name are set inside each zdeck manifest.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p output/train

if [[ $# -lt 1 ]]; then
    echo "usage: $0 {gpt-2|gpt-neox|pythia} [training args...]" >&2
    exit 2
fi
family="$1"
shift

case "$family" in
    gpt-2) module="alien_ink/zdeck/baseline_perf_gpt-2_mist.py"; name="baseline-perf-gpt-2-mist" ;;
    gpt-neox) module="alien_ink/zdeck/baseline_perf_gpt-neox_mist.py"; name="baseline-perf-gpt-neox-mist" ;;
    pythia) module="alien_ink/zdeck/baseline_perf_pythia-160m_mist.py"; name="baseline-perf-pythia-160m-mist" ;;
    *) echo "unknown family: $family (expected gpt-2, gpt-neox, or pythia)" >&2; exit 2 ;;
esac

stamp=$(date +%Y%m%d-%H%M%S)
log="output/train/${name}-${stamp}.log"

nohup python "$module" "$@" >"$log" 2>&1 &

echo "pid $!  log $log  target $module"
