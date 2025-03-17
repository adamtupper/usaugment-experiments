#!/bin/bash

sbatch --time=04:00:00 --job-name=aul_mass_classification usaugment-experiments/scripts/slurm/train_trivial_augment.sh aul_mass_v5 aul_mass_classification mitb2_multiclass 0.00001 0.0001
sbatch --time=04:00:00 --job-name=aul_mass_classification usaugment-experiments/scripts/slurm/train_trivial_augment.sh aul_mass_v5 aul_mass_classification efficientnetb5_multiclass 0.000215443 0.00001
sbatch --time=04:00:00 --job-name=aul_mass_segmentation usaugment-experiments/scripts/slurm/train_trivial_augment.sh aul_mass_v5 aul_mass_segmentation segformerb2_binary 0.000215443 0.000001
sbatch --time=04:00:00 --job-name=aul_mass_segmentation usaugment-experiments/scripts/slurm/train_trivial_augment.sh aul_mass_v5 aul_mass_segmentation efficientnetb5_unet_binary 0.001 0.0001

sbatch --time=32:00:00 --job-name=camus_classification usaugment-experiments/scripts/slurm/train_trivial_augment.sh camus_v4 camus_classification mitb2_multiclass 0.00001 0.000001
sbatch --time=32:00:00 --job-name=camus_classification usaugment-experiments/scripts/slurm/train_trivial_augment.sh camus_v4 camus_classification efficientnetb5_multiclass 0.00001 0.000001
# sbatch --time=80:00:00 --job-name=camus_segmentation usaugment-experiments/scripts/slurm/train_trivial_augment.sh camus_v4 camus_segmentation segformerb2_multiclass 0.0000464159 0.00001
# sbatch --time=80:00:00 --job-name=camus_segmentation usaugment-experiments/scripts/slurm/train_trivial_augment.sh camus_v4 camus_segmentation efficientnetb5_unet_multiclass 0.000215443 0.0000316228

sbatch --time=4:00:00 --job-name=mmotu_classification usaugment-experiments/scripts/slurm/train_trivial_augment.sh mmotu_v2 mmotu_classification mitb2_multiclass 0.000215443 0.0001
sbatch --time=4:00:00 --job-name=mmotu_classification usaugment-experiments/scripts/slurm/train_trivial_augment.sh mmotu_v2 mmotu_classification efficientnetb5_multiclass 0.000215443 0.001
sbatch --time=5:00:00 --job-name=mmotu_segmentation usaugment-experiments/scripts/slurm/train_trivial_augment.sh mmotu_v2 mmotu_segmentation segformerb2_binary 0.000215443 0.00001
sbatch --time=5:00:00 --job-name=mmotu_segmentation usaugment-experiments/scripts/slurm/train_trivial_augment.sh mmotu_v2 mmotu_segmentation efficientnetb5_unet_binary 0.001 0.000001
