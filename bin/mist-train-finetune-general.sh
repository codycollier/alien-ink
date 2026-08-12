#!/usr/bin/env bash
# Background fine-tune (sft) zdeck program on Mist (local RTX 3070). Logs under output/train/.
#
# Uncomment one zdeck block below, then:
#   ./bin/mist-train-finetune-general.sh
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

# script="alien_ink/zdeck/sft_pythia-160m_geo_mist.py"
# name="sft-pythia-160m-geo-mist"

# script="alien_ink/zdeck/sft_smollm2-135m_geo_mist.py"
# name="sft-smollm2-135m-geo-mist"

# Full-document geo corpus with exact in-corpus sentence evaluation.
# script="alien_ink/zdeck/sft_pythia-70m_geo_sentence_memorize_mist.py"
# name="sft-pythia-70m-geo-sentence-memorize-mist"

# Full-document geo corpus at Mist's full-fine-tuning ceiling.
script="alien_ink/zdeck/sft_pythia-410m_geo_sentence_memorize_mist.py"
name="sft-pythia-410m-geo-sentence-memorize-mist"

# Completion-trained synthetic facts (population-basic).
# script="alien_ink/zdeck/sft_pythia-70m_geo_fact_memorize_mist.py"
# name="sft-pythia-70m-geo-fact-memorize-mist"

# Completion-trained synthetic facts at Mist's full-fine-tuning ceiling.
# script="alien_ink/zdeck/sft_pythia-410m_geo_fact_memorize_mist.py"
# name="sft-pythia-410m-geo-fact-memorize-mist"

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
