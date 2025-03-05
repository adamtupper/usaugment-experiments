# #!/bin/bash
# Download model weights from Huggingface Hub using Git LFS
# For more information, see https://huggingface.co/docs/hub/en/models-downloading

# Usage: bash scripts/download/download_models.sh <output_dir>

if [ -z "$1" ]
then
    echo "Missing positional argument <output_dir>"
    exit 1
fi

model_dir=$1/nvidia/mit-b2
mkdir -p $model_dir
git clone git@hf.co:nvidia/mit-b2 $model_dir