#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:v100:1
#SBATCH --array 3-15%8
#SBATCH --mail-type=ALL

# Perform 10 repetitions using TrivialAugment with the Top-N
# augmentations, where N is in the range {2, 14}, for a particular model
# on a task. Each job in the array tests a different N.

# Usage: sbatch --job-name <name> usaugment-experiments/scripts/slurm/train_trivial_augment.sh <dataset> <task> <model> <lr> <weight_decay>

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

if [ -z "1" ]; then
    echo "Missing positional argument <dataset>"
    exit 1
fi

if [ -z "2" ]; then
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
    echo "Missing positional argument <lr>"
    exit 1
fi

if [ -z "$5" ]
then
    echo "Missing positional argument <weight_decay>"
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

# Configure TrivialAugment
augmentation=trivial_augment_${2}_${3}
top_n_augmentations=$SLURM_ARRAY_TASK_ID

for seed in {1..10}; do
    if [ ! -d $scratch/$1/$2/$3/$augmentation/$top_n_augmentations/$seed ] ; then
        # Run for this seed does not exist
        
        # Create output directory
        mkdir -p $scratch/$1/$2/$3/$augmentation/$top_n_augmentations/$seed
    
        # Train model
        python usaugment-experiments/src/usaugment/train.py \
            output_dir=$scratch/$1/$2/$3/$augmentation/$top_n_augmentations/$seed \
            data_dir=$SLURM_TMPDIR \
            models_dir=$SLURM_TMPDIR \
            task=$2 \
            model=$3 \
            augmentation=$augmentation \
            seed=$seed \
            batch_size=64 \
            epochs=200 \
            lr=$4 \
            weight_decay=$5 \
            top_n_augmentations=$top_n_augmentations
    fi
done
    