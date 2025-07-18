#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=ALL

# Perform a grid search over the learning rate and weight decay values 
# using Optuna for a particular model on a task.

# Usage: sbatch --job-name <name> usaugment-experiments/scripts/slurm/tune.sh <dataset> <task> <model> <trials_per_job>

# Print Job info
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo ""

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

if [ -z "$1" ]; then
    echo "Missing positional argument <dataset>"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Missing positional argument <task>"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Missing positional argument <model>"
    exit 1
fi

if [ -z "$4" ]
then
    echo "Missing positional argument <trials_per_job>"
    exit 1
fi

# Copy data, pre-trained models and code to compute node ($1 is the dataset name)
tar -xf $project/data/$1.tar.gz -C $SLURM_TMPDIR
cp -r $project/models/* $SLURM_TMPDIR
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

# Create output directory
mkdir -p $scratch/$1/$2/$3

# Run trial
# NOTE: We override the batch size and epochs for model ablations
python usaugment-experiments/src/usaugment/tune_size_arch_ablation.py \
    output_dir=$scratch/$1/$2/$3 \
    data_dir=$SLURM_TMPDIR \
    models_dir=$SLURM_TMPDIR \
    task=$2 \
    model=$3 \
    trials_per_job=$4 \
    batch_size=64 \
    epochs=200
    