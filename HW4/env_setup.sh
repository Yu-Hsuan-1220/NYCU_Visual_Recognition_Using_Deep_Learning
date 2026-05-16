#!/usr/bin/env bash
set -e

# Create conda env (skip if it already exists)
if ! conda env list | awk '{print $1}' | grep -qx VRDL_4; then
    conda create -y -n VRDL_4 python=3.10
fi

# Activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate VRDL_4

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install einops==0.7.0 pillow tqdm wandb scikit-image
pip install "numpy<2"
