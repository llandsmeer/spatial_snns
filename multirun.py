import itertools
import glob
import os
import json
import shlex
import uuid

template = '''
#!/bin/bash
#SBATCH --job-name=jobs_JOBNAME
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=logs/JOBNAME_%x_%j.out
#SBATCH --error=logs/JOBNAME_%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --time=14:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --array=0-JOBCOUNT
#SBATCH --mem=4G

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
                if row['table'] == 'step':
                    iilast = row['i']
            if iilast == 19999:
                o.append(args)
    return o

existing = get_args()

def exists(config):
    for row in existing:
        # print('----')
        if all(row.get(k) == v for k, v in config.items()):
            return True
        # for k, v in config.items(): print( k, ':', row.get(k), '==', v )
    return False

def generate_calls(prod, randomize=True, const='python3 train.py --skip --batch_size=4'):
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

exp_sizes_nets = generate_calls({
    'nhidden': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300],
    'net': [
            '0',
            '1', '2', '3', '4', '5', '2e0.3',
            'inf'],
    'seed': [0, 1, 2],
    'dt': [0.5],
})

exp_eps = generate_calls({
    'nhidden': [30, 100, 300],
    'net': [
            '2e0.0', '2e0.1', '2e0.2', '2e0.3', '2e0.4',
            '2e0.5', '2e0.6', '2e0.7', '2e0.8', '2e0.9',
            '2e1.0', '2e1.1', '2e1.2', '2e1.3', '2e1.4',
            '2e1.5', '2e1.6', '2e1.7', '2e1.8', '2e1.9',
            '2e2.0',
            ],
    'seed': [0, 1, 2, 3, 4, 5],
    'dt': [0.5],
})

# exp_eps = generate_calls({
#     'nhidden': [30, 100, 300],
#     'net': [
#             '2e0.0', '2e0.1', '2e0.2', '2e0.3', '2e0.4',
#             '3e0.0', '3e0.1', '3e0.2', '3e0.3', '3e0.4',
#             '2e0.5', '2e0.6', '2e0.7', '2e0.8', '2e0.9',
#             '2e1.0', '2e1.1', '2e1.2', '2e1.3', '2e1.4',
#             '2e1.5', '2e1.6', '2e1.7', '2e1.8', '2e1.9',
#             '2e2.0',
#             '3e0.5', '3e0.6', '3e0.7', '3e0.8', '3e0.9',
#             '3e1.0', '3e1.1', '3e1.2', '3e1.3', '3e1.4',
#             '3e1.5', '3e1.6', '3e1.7', '3e1.8', '3e1.9',
#             '3e2.0',
#             ],
#     'seed': [0],
#     'dt': [0.5],
# })


# o = make_bash(exp_eps)
o = make_sbatch(exp_sizes_nets + exp_eps)

print(o)
