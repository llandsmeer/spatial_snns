#!/bin/bash
#SBATCH --job-name=layer_weight
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --array=0-60
#SBATCH --mem=4G

module load 2024
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

unset LD_LIBRARY_PATH

CMDS=(
'python3 train_yy.py --ndim 0 --nhidden 5 --lr 0.002 --dt 0.01 --batch_size 150 --beta 10 --wseed 40 --bseed 95 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 5 --lr 0.002 --dt 0.01 --batch_size 150 --beta 10 --wseed 41 --bseed 96 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 5 --lr 0.002 --dt 0.01 --batch_size 150 --beta 10 --wseed 42 --bseed 97 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 5 --lr 0.002 --dt 0.01 --batch_size 150 --beta 10 --wseed 43 --bseed 98 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 5 --lr 0.002 --dt 0.01 --batch_size 150 --beta 10 --wseed 44 --bseed 99 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 15 --lr 0.0015 --dt 0.01 --batch_size 150 --beta 10 --wseed 40 --bseed 95 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 15 --lr 0.0015 --dt 0.01 --batch_size 150 --beta 10 --wseed 41 --bseed 96 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 15 --lr 0.0015 --dt 0.01 --batch_size 150 --beta 10 --wseed 42 --bseed 97 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 15 --lr 0.0015 --dt 0.01 --batch_size 150 --beta 10 --wseed 43 --bseed 98 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 15 --lr 0.0015 --dt 0.01 --batch_size 150 --beta 10 --wseed 44 --bseed 99 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 30 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 40 --bseed 95 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 30 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 41 --bseed 96 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 30 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 42 --bseed 97 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 30 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 43 --bseed 98 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 30 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 44 --bseed 99 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 45 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 40 --bseed 95 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 45 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 41 --bseed 96 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 45 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 42 --bseed 97 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 45 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 43 --bseed 98 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 45 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 44 --bseed 99 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 60 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 40 --bseed 95 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 60 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 41 --bseed 96 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 60 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 42 --bseed 97 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 60 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 43 --bseed 98 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 60 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 44 --bseed 99 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 75 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 40 --bseed 95 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 75 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 41 --bseed 96 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 75 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 42 --bseed 97 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 75 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 43 --bseed 98 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 75 --lr 0.001 --dt 0.01 --batch_size 150 --beta 10 --wseed 44 --bseed 99 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 90 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 40 --bseed 95 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 90 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 41 --bseed 96 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 90 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 42 --bseed 97 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 90 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 43 --bseed 98 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 90 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 44 --bseed 99 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 105 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 40 --bseed 95 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 105 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 41 --bseed 96 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 105 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 42 --bseed 97 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 105 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 43 --bseed 98 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 105 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 44 --bseed 99 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 120 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 40 --bseed 95 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 120 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 41 --bseed 96 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 120 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 42 --bseed 97 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 120 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 43 --bseed 98 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 120 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 44 --bseed 99 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 150 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 40 --bseed 95 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 150 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 41 --bseed 96 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 150 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 42 --bseed 97 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 150 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 43 --bseed 98 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 150 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 44 --bseed 99 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 30 --lr 0.00300 --dt 0.01 --batch_size 150 --beta 10 --wseed 20 --bseed 70 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 30 --lr 0.00200 --dt 0.01 --batch_size 150 --beta 10 --wseed 20 --bseed 70 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 30 --lr 0.00150 --dt 0.01 --batch_size 150 --beta 10 --wseed 20 --bseed 70 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 30 --lr 0.00100 --dt 0.01 --batch_size 150 --beta 10 --wseed 20 --bseed 70 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 30 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 20 --bseed 70 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 90 --lr 0.00300 --dt 0.01 --batch_size 150 --beta 10 --wseed 20 --bseed 70 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 90 --lr 0.00200 --dt 0.01 --batch_size 150 --beta 10 --wseed 20 --bseed 70 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 90 --lr 0.00150 --dt 0.01 --batch_size 150 --beta 10 --wseed 20 --bseed 70 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 90 --lr 0.00100 --dt 0.01 --batch_size 150 --beta 10 --wseed 20 --bseed 70 --layer --force'
'python3 train_yy.py --ndim 0 --nhidden 90 --lr 0.00055 --dt 0.01 --batch_size 150 --beta 10 --wseed 20 --bseed 70 --layer --force'
)

eval ${CMDS[$SLURM_ARRAY_TASK_ID]}
