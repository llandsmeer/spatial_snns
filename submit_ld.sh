#!/bin/bash
#SBATCH --job-name=layer_delay_new
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=logs_ld_new/%x_%j.out
#SBATCH --error=logs_ld_new/%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=logs_ld_new/%x_%j.out
#SBATCH --array=0-49
#SBATCH --mem=4G

module load 2024
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

unset LD_LIBRARY_PATH

CMDS=(
# 'python3 train_yy.py --ndim None --nhidden 5 --lr 0.001 --dt 0.01 --batch_size 150 --beta 8 --wseed 20 --bseed 0 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 5 --lr 0.001 --dt 0.01 --batch_size 150 --beta 8 --wseed 21 --bseed 1 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 5 --lr 0.001 --dt 0.01 --batch_size 150 --beta 8 --wseed 22 --bseed 2 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 5 --lr 0.001 --dt 0.01 --batch_size 150 --beta 8 --wseed 23 --bseed 3 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 5 --lr 0.001 --dt 0.01 --batch_size 150 --beta 8 --wseed 24 --bseed 4 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 5 --lr 0.001 --dt 0.01 --batch_size 150 --beta 8 --wseed 25 --bseed 5 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 5 --lr 0.001 --dt 0.01 --batch_size 150 --beta 8 --wseed 26 --bseed 6 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 5 --lr 0.001 --dt 0.01 --batch_size 150 --beta 8 --wseed 27 --bseed 7 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 5 --lr 0.001 --dt 0.01 --batch_size 150 --beta 8 --wseed 28 --bseed 8 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 5 --lr 0.001 --dt 0.01 --batch_size 150 --beta 8 --wseed 29 --bseed 9 --layer --force'
'python3 train_yy.py --ndim None --nhidden 10 --lr 0.0008 --dt 0.01 --batch_size 150 --beta 8 --wseed 20 --bseed 0 --layer --force'
'python3 train_yy.py --ndim None --nhidden 10 --lr 0.0008 --dt 0.01 --batch_size 150 --beta 8 --wseed 21 --bseed 1 --layer --force'
'python3 train_yy.py --ndim None --nhidden 10 --lr 0.0008 --dt 0.01 --batch_size 150 --beta 8 --wseed 22 --bseed 2 --layer --force'
'python3 train_yy.py --ndim None --nhidden 10 --lr 0.0008 --dt 0.01 --batch_size 150 --beta 8 --wseed 23 --bseed 3 --layer --force'
'python3 train_yy.py --ndim None --nhidden 10 --lr 0.0008 --dt 0.01 --batch_size 150 --beta 8 --wseed 24 --bseed 4 --layer --force'
'python3 train_yy.py --ndim None --nhidden 10 --lr 0.0008 --dt 0.01 --batch_size 150 --beta 8 --wseed 25 --bseed 5 --layer --force'
'python3 train_yy.py --ndim None --nhidden 10 --lr 0.0008 --dt 0.01 --batch_size 150 --beta 8 --wseed 26 --bseed 6 --layer --force'
'python3 train_yy.py --ndim None --nhidden 10 --lr 0.0008 --dt 0.01 --batch_size 150 --beta 8 --wseed 27 --bseed 7 --layer --force'
'python3 train_yy.py --ndim None --nhidden 10 --lr 0.0008 --dt 0.01 --batch_size 150 --beta 8 --wseed 28 --bseed 8 --layer --force'
'python3 train_yy.py --ndim None --nhidden 10 --lr 0.0008 --dt 0.01 --batch_size 150 --beta 8 --wseed 29 --bseed 9 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 15 --lr 0.0007 --dt 0.01 --batch_size 150 --beta 8 --wseed 20 --bseed 0 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 15 --lr 0.0007 --dt 0.01 --batch_size 150 --beta 8 --wseed 21 --bseed 1 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 15 --lr 0.0007 --dt 0.01 --batch_size 150 --beta 8 --wseed 22 --bseed 2 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 15 --lr 0.0007 --dt 0.01 --batch_size 150 --beta 8 --wseed 23 --bseed 3 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 15 --lr 0.0007 --dt 0.01 --batch_size 150 --beta 8 --wseed 24 --bseed 4 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 15 --lr 0.0007 --dt 0.01 --batch_size 150 --beta 8 --wseed 25 --bseed 5 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 15 --lr 0.0007 --dt 0.01 --batch_size 150 --beta 8 --wseed 26 --bseed 6 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 15 --lr 0.0007 --dt 0.01 --batch_size 150 --beta 8 --wseed 27 --bseed 7 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 15 --lr 0.0007 --dt 0.01 --batch_size 150 --beta 8 --wseed 28 --bseed 8 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 15 --lr 0.0007 --dt 0.01 --batch_size 150 --beta 8 --wseed 29 --bseed 9 --layer --force'
'python3 train_yy.py --ndim None --nhidden 20 --lr 0.0006 --dt 0.01 --batch_size 150 --beta 8 --wseed 20 --bseed 0 --layer --force'
'python3 train_yy.py --ndim None --nhidden 20 --lr 0.0006 --dt 0.01 --batch_size 150 --beta 8 --wseed 21 --bseed 1 --layer --force'
'python3 train_yy.py --ndim None --nhidden 20 --lr 0.0006 --dt 0.01 --batch_size 150 --beta 8 --wseed 22 --bseed 2 --layer --force'
'python3 train_yy.py --ndim None --nhidden 20 --lr 0.0006 --dt 0.01 --batch_size 150 --beta 8 --wseed 23 --bseed 3 --layer --force'
'python3 train_yy.py --ndim None --nhidden 20 --lr 0.0006 --dt 0.01 --batch_size 150 --beta 8 --wseed 24 --bseed 4 --layer --force'
'python3 train_yy.py --ndim None --nhidden 20 --lr 0.0006 --dt 0.01 --batch_size 150 --beta 8 --wseed 25 --bseed 5 --layer --force'
'python3 train_yy.py --ndim None --nhidden 20 --lr 0.0006 --dt 0.01 --batch_size 150 --beta 8 --wseed 26 --bseed 6 --layer --force'
'python3 train_yy.py --ndim None --nhidden 20 --lr 0.0006 --dt 0.01 --batch_size 150 --beta 8 --wseed 27 --bseed 7 --layer --force'
'python3 train_yy.py --ndim None --nhidden 20 --lr 0.0006 --dt 0.01 --batch_size 150 --beta 8 --wseed 28 --bseed 8 --layer --force'
'python3 train_yy.py --ndim None --nhidden 20 --lr 0.0006 --dt 0.01 --batch_size 150 --beta 8 --wseed 29 --bseed 9 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 25 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 20 --bseed 0 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 25 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 21 --bseed 1 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 25 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 22 --bseed 2 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 25 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 23 --bseed 3 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 25 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 24 --bseed 4 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 25 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 25 --bseed 5 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 25 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 26 --bseed 6 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 25 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 27 --bseed 7 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 25 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 28 --bseed 8 --layer --force'
# 'python3 train_yy.py --ndim None --nhidden 25 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 29 --bseed 9 --layer --force'
'python3 train_yy.py --ndim None --nhidden 30 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 20 --bseed 0 --layer --force'
'python3 train_yy.py --ndim None --nhidden 30 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 21 --bseed 1 --layer --force'
'python3 train_yy.py --ndim None --nhidden 30 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 22 --bseed 2 --layer --force'
'python3 train_yy.py --ndim None --nhidden 30 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 23 --bseed 3 --layer --force'
'python3 train_yy.py --ndim None --nhidden 30 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 24 --bseed 4 --layer --force'
'python3 train_yy.py --ndim None --nhidden 30 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 25 --bseed 5 --layer --force'
'python3 train_yy.py --ndim None --nhidden 30 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 26 --bseed 6 --layer --force'
'python3 train_yy.py --ndim None --nhidden 30 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 27 --bseed 7 --layer --force'
'python3 train_yy.py --ndim None --nhidden 30 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 28 --bseed 8 --layer --force'
'python3 train_yy.py --ndim None --nhidden 30 --lr 0.0005 --dt 0.01 --batch_size 150 --beta 8 --wseed 29 --bseed 9 --layer --force'
'python3 train_yy.py --ndim None --nhidden 45 --lr 0.0002 --dt 0.01 --batch_size 150 --beta 8 --wseed 20 --bseed 0 --layer --force'
'python3 train_yy.py --ndim None --nhidden 45 --lr 0.0002 --dt 0.01 --batch_size 150 --beta 8 --wseed 21 --bseed 1 --layer --force'
'python3 train_yy.py --ndim None --nhidden 45 --lr 0.0002 --dt 0.01 --batch_size 150 --beta 8 --wseed 22 --bseed 2 --layer --force'
'python3 train_yy.py --ndim None --nhidden 45 --lr 0.0002 --dt 0.01 --batch_size 150 --beta 8 --wseed 23 --bseed 3 --layer --force'
'python3 train_yy.py --ndim None --nhidden 45 --lr 0.0002 --dt 0.01 --batch_size 150 --beta 8 --wseed 24 --bseed 4 --layer --force'
'python3 train_yy.py --ndim None --nhidden 45 --lr 0.0002 --dt 0.01 --batch_size 150 --beta 8 --wseed 25 --bseed 5 --layer --force'
'python3 train_yy.py --ndim None --nhidden 45 --lr 0.0002 --dt 0.01 --batch_size 150 --beta 8 --wseed 26 --bseed 6 --layer --force'
'python3 train_yy.py --ndim None --nhidden 45 --lr 0.0002 --dt 0.01 --batch_size 150 --beta 8 --wseed 27 --bseed 7 --layer --force'
'python3 train_yy.py --ndim None --nhidden 45 --lr 0.0002 --dt 0.01 --batch_size 150 --beta 8 --wseed 28 --bseed 8 --layer --force'
'python3 train_yy.py --ndim None --nhidden 45 --lr 0.0002 --dt 0.01 --batch_size 150 --beta 8 --wseed 29 --bseed 9 --layer --force'
'python3 train_yy.py --ndim None --nhidden 60 --lr 0.0001 --dt 0.01 --batch_size 150 --beta 8 --wseed 20 --bseed 0 --layer --force'
'python3 train_yy.py --ndim None --nhidden 60 --lr 0.0001 --dt 0.01 --batch_size 150 --beta 8 --wseed 21 --bseed 1 --layer --force'
'python3 train_yy.py --ndim None --nhidden 60 --lr 0.0001 --dt 0.01 --batch_size 150 --beta 8 --wseed 22 --bseed 2 --layer --force'
'python3 train_yy.py --ndim None --nhidden 60 --lr 0.0001 --dt 0.01 --batch_size 150 --beta 8 --wseed 23 --bseed 3 --layer --force'
'python3 train_yy.py --ndim None --nhidden 60 --lr 0.0001 --dt 0.01 --batch_size 150 --beta 8 --wseed 24 --bseed 4 --layer --force'
'python3 train_yy.py --ndim None --nhidden 60 --lr 0.0001 --dt 0.01 --batch_size 150 --beta 8 --wseed 25 --bseed 5 --layer --force'
'python3 train_yy.py --ndim None --nhidden 60 --lr 0.0001 --dt 0.01 --batch_size 150 --beta 8 --wseed 26 --bseed 6 --layer --force'
'python3 train_yy.py --ndim None --nhidden 60 --lr 0.0001 --dt 0.01 --batch_size 150 --beta 8 --wseed 27 --bseed 7 --layer --force'
'python3 train_yy.py --ndim None --nhidden 60 --lr 0.0001 --dt 0.01 --batch_size 150 --beta 8 --wseed 28 --bseed 8 --layer --force'
'python3 train_yy.py --ndim None --nhidden 60 --lr 0.0001 --dt 0.01 --batch_size 150 --beta 8 --wseed 29 --bseed 9 --layer --force'
)

eval ${CMDS[$SLURM_ARRAY_TASK_ID]}
