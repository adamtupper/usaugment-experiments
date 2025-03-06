# #!/bin/bash
# Download model weights from GitHub and Huggingface Hub using Git LFS
# For more information, see https://huggingface.co/docs/hub/en/models-downloading

# Usage: bash scripts/download/download_models.sh <output_dir>

if [ -z "$1" ]
then
    echo "Missing positional argument <output_dir>"
    exit 1
fi

# EfficientNet weights
wget -c https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth -P $1/hub/checkpoints
wget -c https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth -P $1/hub/checkpoints

# MiT-B2 weights
model_dir=$1/nvidia/mit-b2
mkdir -p $model_dir
git clone git@hf.co:nvidia/mit-b2 $model_dir