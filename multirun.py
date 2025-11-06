import itertools
import glob
import os
import json
import shlex
import uuid

template = '''#!/bin/bash
#SBATCH --job-name=jobs_JOBNAME
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=logs/JOBNAME_%x_%j.out
#SBATCH --error=logs/JOBNAME_%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --time=14:00:00
#SBATCH --array=0-JOBCOUNT
#SBATCH --mem=4G

cd /home/amovahedin/spatial_delays

module load 2024
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

unset LD_LIBRARY_PATH

CMDS=(
CMDSCMDSCMDSCMDS
)

eval ${CMDS[$SLURM_ARRAY_TASK_ID]}
'''

def get_args():
    o = []
    for fn in glob.glob('saved/*/log.jsons'):
        args = None
        with open(fn, 'rb') as f:
            for line in f:
                row = json.loads(line)
                if row['table'] == 'args':
                    args = row
                    break
            f.seek(0, os.SEEK_END)
            size = f.tell()
            seek_pos = max(size - 8*1024, 0)
            f.seek(seek_pos)
            iilast = 0
            for line in f:
                try:
                    row = json.loads(line)
                except:
                    continue
                if row['table'] == 'epoch':
                    iilast = row['i']
            if iilast == 29 or 1:
                o.append(args)
    return o

existing = get_args()

def exists(config):
    for row in existing:
        # print('----')
        if all(row.get(k) == v for k, v in config.items()):
            print('#EXIST')
            return True
        # for k, v in config.items(): print( k, ':', row.get(k), '==', v )
    return False

def generate_calls(prod, randomize=True, const='python3 train.py --skip'):
    lines = []
    for config in itertools.product(*prod.values()):
        config = dict(zip(prod.keys(), config))
        if exists(config):
            continue
        line = ' '.join(f'--{k} {v}' for k, v in config.items())
        lines.append(const + ' ' + line)
    return lines

def make_sbatch(lines):
    cmd_lines = '\n'.join(shlex.quote(line) for line in lines)
    s = template
    # print('nlines', len(lines))
    s = s.replace('JOBCOUNT', str(len(lines)))
    s = s.replace('CMDSCMDSCMDSCMDS', cmd_lines)
    s = s.replace('JOBNAME', str(uuid.uuid4()))
    return s

def make_bash(lines):
    cmd_lines = '\n'.join(line for line in lines)
    return cmd_lines

# print(make_sbatch(exp_sizes_nets := generate_calls({
#     # 'nhidden': [10, 20, 30, 40, 50, 60, 70, 80, 90,],
#     'nhidden': [100, 150, 200, 250, 300],
#     'net': ['0', '1', '2', '3', '4', '5', 'inf'],
#     'seed': [0, 1, 2, 3, 4, 6],
#     'dt': [0.5],
#     'batch_size': [32],
#     'nepochs': [30],
#     'tgtfreq': [10]
#     # 'delaygradscale': [1],
# })))

# pos_check_mu_sigma = generate_calls({
#     'nhidden': [30, 100],
#     'net': [
#             '2',
#            ],
#     'seed': [0],
#     'dt': [0.5],
#     'possigma': [10, 20, 30],
#     'delaygradscale': [0, 1, 10],
# })
# 
# delay_check_mu_sigma = generate_calls({
#     'nhidden': [30, 100],
#     'net': [
#             '0','inf'
#             ],
#     'seed': [0],
#     'dt': [0.5],
#     'delaysigma': [0.5, 1, 2, 5],
#     'delaymu': [4, 8, 20, 30],
#     'delaygradscale': [0, 1, 10],
# })
# 
# exp_eps = []
# exp_eps = generate_calls({
#     'nhidden': [30, 100, 300],
#     'net': [
#             '2e0.0', '2e0.1', '2e0.2', '2e0.3', '2e0.4',
#             '2e0.5', '2e0.6', '2e0.7', '2e0.8', '2e0.9',
#             '2e1.0', '2e1.1', '2e1.2', '2e1.3', '2e1.4',
#             '2e1.5', '2e1.6', '2e1.7', '2e1.8', '2e1.9',
#             '2e2.0',
#             ],
#     'seed': [0, 1, 2, 3, 4, 5,
#              6, 7, 8, 9, 10],
#     'dt': [0.5],
# })

# o = make_bash(exp_eps)
# o = make_sbatch(exp_sizes_nets + exp_eps)

#o = make_sbatch(
#pos_check_mu_sigma + delay_check_mu_sigma
#)
#print(o)

# exp_eps = []
# exp_eps = generate_calls({
#     'nhidden': [30, 100, 300],
#     'net': [
#             '2e0.0', '2e0.1', '2e0.2', '2e0.3', '2e0.4',
#             '2e0.5', '2e0.6', '2e0.7', '2e0.8', '2e0.9',
#             '2e1.0', '2e1.1', '2e1.2', '2e1.3', '2e1.4',
#             '2e1.5', '2e1.6', '2e1.7', '2e1.8', '2e1.9',
#             '2e2.0',
#             ],
#     'seed': [0, 1, 2, 3, 4, 5,
#              6, 7, 8, 9, 10],
#     'dt': [0.5],
# })

print(make_sbatch(exp_sizes_nets := generate_calls({
    'nhidden': [10, 30, 50],
    # 'nhidden': [100, 150, 200, 250, 300],
    'net': [
            '3e0.00',
            '3e0.05',
            '3e0.10',
            '3e0.15',
            '3e0.20',
            '3e0.25',
            '3e0.30',
            '3e0.35',
            '3e0.40',
            '3e0.45',
            '3e0.50',
            ],
    'seed': [0, 1, 2, 3, 4, 6],
    'dt': [0.5],
    'batch_size': [32],
    'nepochs': [30],
    'tgtfreq': [10]
    # 'delaygradscale': [1],
})))
