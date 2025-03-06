#!/bin/bash
# Setup Slurm compute node

# Usage: source scripts/slurm/setup_node.sh <dataset> <models_dir>

if [ -z "$1" ]
then
    echo "Missing positional argument <dataset>"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Missing positional argument <models_dir>"
    exit 1
fi

# Purge modules
module purge

# Load modules
module load gcc arrow git-lfs python/3.11 cuda cudnn rust httpproxy opencv

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_HOME=$SLURM_TMPDIR
export NO_ALBUMENTATIONS_UPDATE=1
export HYDRA_FULL_ERROR=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=$SLURM_TMPDIR/huggingface/hub

# Copy data, pre-trained models and code to compute node
# $1 is the dataset name, $2 is the directory containing the model checkpoints
tar -xf $project/data/$1.tar.gz -C $SLURM_TMPDIR
cp -r $project/models/$2 $SLURM_TMPDIR
rsync -a $project/usaugment-experiments $SLURM_TMPDIR --exclude-from=$project/usaugment-experiments/.gitignore

# Create virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install package and dependencies
pip install --no-index --upgrade pip
pip install --no-index -r $SLURM_TMPDIR/usaugment-experiments/requirements_cc.txt
pip install --no-index -e $SLURM_TMPDIR/usaugment-experiments

# Change to working directory
cd $SLURM_TMPDIR
    