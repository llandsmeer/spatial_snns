#!/bin/bash
#SBATCH --job-name=spatial2
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --array=0-20

module load 2024
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

unset LD_LIBRARY_PATH

A_values=(0 1 2 3 4 5 None)
B_values=(100 200 300)

A_index=$((SLURM_ARRAY_TASK_ID / 3))
B_index=$((SLURM_ARRAY_TASK_ID % 3))

A=${A_values[$A_index]}
B=${B_values[$B_index]}

python3 train.py "$A" "$B"

