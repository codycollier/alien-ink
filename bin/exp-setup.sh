#!/usr/bin/env bash
#
#
#
#

echo "-------------------------------------------------------------------------"
echo ":: Alien-Ink Experiments - setup"
echo "-------------------------------------------------------------------------"


# Go to root of repository
cd $(dirname $0)
cd ../

# Ensure environment
echo "-------------------------------------------------------------------------"
echo ":: Creating and activating virt"
echo "-------------------------------------------------------------------------"
uv venv
source .venv/bin/activate

# Ensure deps and install ink
echo "-------------------------------------------------------------------------"
echo ":: Installing alien-ink and depedencies"
echo "-------------------------------------------------------------------------"
# PyPI torch is CUDA 13.0; use CUDA 12.6 wheels for older NVIDIA drivers.
UV_TORCH_BACKEND=cu126 uv pip install -e ".[hf]"

# Instruction
echo "-------------------------------------------------------------------------"
echo ":: ..."
echo "-------------------------------------------------------------------------"
echo
echo "> Environment is ready"
echo "> Do:"
echo "$ source .venv/bin/activate"
echo "$ python -m alien_ink.exp.gpt2_pretrain_wikitext --flight-check"
echo
echo

