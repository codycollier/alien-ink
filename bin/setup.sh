#!/usr/bin/env bash
#
# Create a local venv and install alien-ink (training deps are in the main set).
#
set -euo pipefail

echo "-------------------------------------------------------------------------"
echo ":: Alien Ink — setup (Mist / local GPU)"
echo "-------------------------------------------------------------------------"

cd "$(dirname "$0")/.."

echo "-------------------------------------------------------------------------"
echo ":: Creating and activating virt"
echo "-------------------------------------------------------------------------"
uv venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "-------------------------------------------------------------------------"
echo ":: Installing alien-ink and dependencies"
echo "-------------------------------------------------------------------------"
uv pip install -e .

echo "-------------------------------------------------------------------------"
echo ":: Ready"
echo "-------------------------------------------------------------------------"
echo
echo "> Environment is ready"
echo "> Do:"
echo "  source .venv/bin/activate"
echo "  cp -n .env.example .env   # then fill HF_TOKEN / WANDB_API_KEY"
echo "  python alien_ink/zdeck/pre_gpt-2_wikitext_5k_mist.py"
echo
