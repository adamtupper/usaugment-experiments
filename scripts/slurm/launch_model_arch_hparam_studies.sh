#!/bin/bash

# sbatch --time=02:00:00 --array 1-5%1 --job-name aul_mass_class_mit usaugment-experiments/scripts/slurm/tune.sh aul_mass_v5 aul_mass_classification mitb2_multiclass 10
sbatch --time=02:00:00 --array 1-5%1 --job-name aul_mass_class_eff usaugment-experiments/scripts/slurm/tune.sh aul_mass_v5 aul_mass_classification efficientnetb5_multiclass 10

# sbatch --time=03:00:00 --array 1-49%4 --job-name camus_class_mit usaugment-experiments/scripts/slurm/tune.sh camus_v4 camus_classification mitb2_multiclass 1
sbatch --time=05:00:00 --array 1-49%4 --job-name camus_class_eff usaugment-experiments/scripts/slurm/tune.sh camus_v4 camus_classification efficientnetb5_multiclass 1

sbatch --time=02:00:00 --array 1-9%1 --job-name mmotu_class_mit usaugment-experiments/scripts/slurm/tune.sh mmotu_v2 mmotu_classification mitb2_multiclass 6
sbatch --time=02:00:00 --array 1-10%1 --job-name mmotu_class_eff usaugment-experiments/scripts/slurm/tune.sh mmotu_v2 mmotu_classification efficientnetb5_multiclass 5

sbatch --time=02:00:00 --array 1-10%1 --job-name aul_mass_seg_seg usaugment-experiments/scripts/slurm/tune.sh aul_mass_v5 aul_mass_segmentation segformerb2_binary 5
sbatch --time=02:00:00 --array 1-10%1 --job-name aul_mass_seg_eff usaugment-experiments/scripts/slurm/tune.sh aul_mass_v5 aul_mass_segmentation efficientnetb5_unet_binary 5

sbatch --time=05:00:00 --array 1-49%4 --job-name camus_seg_seg usaugment-experiments/scripts/slurm/tune.sh camus_v4 camus_segmentation segformerb2_multiclass 1
sbatch --time=06:00:00 --array 1-49%4 --job-name camus_seg_eff usaugment-experiments/scripts/slurm/tune.sh camus_v4 camus_segmentation efficientnetb5_unet_multiclass 1

sbatch --time=02:00:00 --array 1-13%1 --job-name mmotu_seg_seg usaugment-experiments/scripts/slurm/tune.sh mmotu_v2 mmotu_segmentation segformerb2_binary 4
sbatch --time=02:00:00 --array 1-13%1 --job-name mmotu_seg_eff usaugment-experiments/scripts/slurm/tune.sh mmotu_v2 mmotu_segmentation efficientnetb5_unet_binary 4