#!/bin/bash
#SBATCH --job-name=exp_w_init
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --array=0-1
#SBATCH --mem=4G

module load 2024
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

unset LD_LIBRARY_PATH

CMDS=(
'python3 train_yy.py --ndim None --nhidden 30 --lr 0.0005 --dt 0.01 --batch_size 150 --ttrain 200 --wseed 10 --bseed 25 --layer --force'
)

eval ${CMDS[$SLURM_ARRAY_TASK_ID]}
