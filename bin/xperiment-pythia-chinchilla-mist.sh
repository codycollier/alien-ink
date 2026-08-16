#!/usr/bin/env bash
# Background Pythia Chinchilla-scale pretraining experiment on Mist.
# Trains the three Wikipedia -> WikiText-103 manifests sequentially:
#
#   1. Pythia-14M  (8,545 steps)
#   2. Pythia-31M (18,921 steps)
#   3. Pythia-70M (42,725 steps)
#
# The sequence and each individual run are logged under output/logs/. W&B and
# model outputs remain controlled by the manifests under output/train/.
#
#   ./bin/xperiment-pythia-chinchilla-mist.sh
#
# To skip models already completed by an earlier invocation:
#
#   START_AT=31m ./bin/xperiment-pythia-chinchilla-mist.sh
#   START_AT=70m ./bin/xperiment-pythia-chinchilla-mist.sh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p output/logs output/train

start_at="${START_AT:-14m}"
case "$start_at" in
  14m | 31m | 70m) ;;
  *)
    echo "error: START_AT must be 14m, 31m, or 70m; got: $start_at" >&2
    exit 2
    ;;
esac

if [[ "${1:-}" != "--foreground" ]]; then
  stamp=$(date +%Y%m%d-%H%M%S)
  log="output/logs/pythia-chinchilla-mist-${stamp}.log"

  nohup env START_AT="$start_at" XPERIMENT_STAMP="$stamp" \
    "$0" --foreground >"$log" 2>&1 &

  echo "pid $!  log $log  start_at $start_at"
  exit 0
fi

stamp="${XPERIMENT_STAMP:-$(date +%Y%m%d-%H%M%S)}"
current_run="setup"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

banner() {
  printf '\n======== [%s] %s ========\n' "$(timestamp)" "$1"
}

on_exit() {
  status=$?
  if (( status == 0 )); then
    banner "experiment complete"
  else
    banner "experiment failed during ${current_run} (exit ${status})"
    printf 'Restart at this model with: START_AT=%s %s\n' \
      "$current_run" "$0"
  fi
}
trap on_exit EXIT

run_manifest() {
  size="$1"
  script="$2"
  run_name="$3"
  run_log="output/logs/${run_name}-${stamp}.log"
  current_run="$size"

  banner "train ${size} — ${script}"
  printf 'run log: %s\n' "$run_log"
  python "$script" 2>&1 | tee "$run_log"
  banner "finished ${size}"
}

started=false
for size in 14m 31m 70m; do
  if [[ "$size" == "$start_at" ]]; then
    started=true
  fi
  if [[ "$started" != true ]]; then
    continue
  fi

  case "$size" in
    14m)
      run_manifest \
        "$size" \
        "alien_ink/zdeck/pre_pythia-14m_wikipedia_wikitext_chinchilla_mist.py" \
        "pre-pythia-14m-wikipedia-wikitext-chinchilla-mist"
      ;;
    31m)
      run_manifest \
        "$size" \
        "alien_ink/zdeck/pre_pythia-31m_wikipedia_wikitext_chinchilla_mist.py" \
        "pre-pythia-31m-wikipedia-wikitext-chinchilla-mist"
      ;;
    70m)
      run_manifest \
        "$size" \
        "alien_ink/zdeck/pre_pythia-70m_wikipedia_wikitext_chinchilla_mist.py" \
        "pre-pythia-70m-wikipedia-wikitext-chinchilla-mist"
      ;;
  esac
done
