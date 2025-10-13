import itertools
import glob
import json
import shlex
import uuid

template = '''
#!/bin/bash
#SBATCH --job-name=jobs_JOBNAME
#SBATCH --output=logs/JOBNAME_%x_%j.out
#SBATCH --error=logs/JOBNAME_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --array=0-10

# --- pre ---
module load python/3.12
source venv/bin/activate

# --- commands ---
CMDS=(
CMDSCMDSCMDSCMDS
)

eval ${CMDS[$SLURM_ARRAY_TASK_ID]}

# --- post ---
'''



def get_args():
    args = []
    for fn in glob.glob('saved/*/log.jsons'):
        for line in open(fn):
            row = json.loads(line)
            if row['table'] == 'args':
                args.append(row)
    return args

existing = get_args()

def exists(config):
    for row in existing:
        print('----')
        if all(row.get(k) == v for k, v in config.items()):
            return True
        for k, v in config.items():
            print( k, ':', row.get(k), '==', v )
    return False

def generate_calls(prod, randomize=True, const = 'python3 train.py'):
    lines = []
    for config in itertools.product(*prod.values()):
        config = dict(zip(prod.keys(), config))
        line = ' '.join(f'--{k} {v}' for k, v in config.items())
        lines.append(const + ' ' + line)
    return lines

def make_sbatch(lines):
    cmd_lines = '\n'.join(shlex.quote(line) for line in lines)
    s = template
    s = s.replace('CMDSCMDSCMDSCMDS', cmd_lines)
    s = s.replace('JOBNAME', str(uuid.uuid4()))
    print('nlines', len(lines))
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
    'seed': [0, 1, 2, 4, 5],
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
    'seed': [0, 1, 2, 4, 5],
    'dt': [0.5],
})


o = make_bash(exp_sizes_nets + exp_eps)

print(o)
