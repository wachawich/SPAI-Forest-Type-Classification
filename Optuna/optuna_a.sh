#!/bin/bash
#SBATCH -p gpu                # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 16                 # Specify number of nodes and processors per task
#SBATCH --gpus-per-task=1               # Specify the number of GPUs
#SBATCH --ntasks-per-node=1        # Specify tasks per node
#SBATCH -t 14:00:00            # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt900293            # Specify project name
#SBATCH -J opppp        # Specify job name

module reset
module load Mamba
module load cudatoolkit/23.3_12.0

conda deactivate
conda activate /lustrefs/disk/project/lt900114-ai24o2/wachi/mytorch

srun python3 optuna_a.py