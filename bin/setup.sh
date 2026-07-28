#!/usr/bin/env bash
#
# Local Mist setup for alien-ink training.
#

echo "-------------------------------------------------------------------------"
echo ":: Alien-Ink — setup (Mist)"
echo "-------------------------------------------------------------------------"

cd "$(dirname "$0")/.."

echo "-------------------------------------------------------------------------"
echo ":: Creating and activating virt"
echo "-------------------------------------------------------------------------"
uv venv
source .venv/bin/activate

echo "-------------------------------------------------------------------------"
echo ":: Installing alien-ink and dependencies"
echo "-------------------------------------------------------------------------"
uv pip install -e ".[hf]"

echo "-------------------------------------------------------------------------"
echo ":: Ready"
echo "-------------------------------------------------------------------------"
echo
echo "> Environment is ready"
echo "> Do:"
echo "$ source .venv/bin/activate"
echo "$ python -m alien_ink.samples.pretrain_wikipedia_5k"
echo
