import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

def wholelines(fn):
    s = None
    with open(fn) as f:
        for line in f:
            if s is None: s = line.strip()
            if line.startswith(' '):
                s = s + ' ' + line.strip()
            else:
                yield s
                s = line
    if s is not None:
        yield s

def parse_log(fpath):
    def tryfloat(x):
        try:
            return float(x)
        except ValueError:
            return x
    for line in wholelines(fn):
        line = re.sub(r'\[[^\]]+\]', '_', line)
        line = [tryfloat(x) for x in line.split()]
        yield line
    return 'a', 'b'


stats = []

META, LOG = 0, 1
for fn in glob.glob('*/log.txt'):
    case = fn.split('/')[0]
    state = META
    meta = {}
    idx = 0
    rows = []
    parts_test = None
    for parts in parse_log(fn):
        if state is META:
            if not parts: continue
            if parts[0] == 'TEST':
                state = LOG
            else:
                meta[parts[0]] = parts[1]
        if state is LOG:
            if parts[0] == 'TOP1':
                parts_test = parts
            elif parts[0] == 'TOP1train':
                assert parts_test is not None
                rows.append(dict(
                    idx=idx,
                    top1=parts_test[1],
                    top3=parts_test[3],
                    top1train=parts[1],
                    top3train=parts[3]
                    ))
                top1 = None
            elif parts[0] == 'TRAIN':
                idx = parts[2]
    df = pd.DataFrame(rows)
    if len(df) == 0:
        continue
    print(meta)
    print(len(df.top1.values))
    stats.append(dict(
        **meta,
        top1=df.top1.values[-10:].mean(),
        top3=df.top3.values[-10:].mean(),
        top1train=df.top1train.values[-10:].mean(),
        top3train=df.top1train.values[-10:].mean()
        ))
    # plt.plot(df.idx, df.top1)
    # plt.plot(df.idx, df.top1train)
    # plt.plot(df.idx, df.top3, '--')
    plt.show()

stats = pd.DataFrame(stats)
stats = stats.sort_values('ndim', key=lambda x: x.apply(lambda x: 1e10 if x == 'None' else x))
stats = stats.reset_index()

plt.plot(stats.top1, label='top1 (test)')
plt.plot(stats.top3, label='top3 (test)')

#plt.plot(stats.top1train)
#plt.plot(stats.top3train)

plt.xticks(np.arange(len(stats['ndim'])), [f'ndim={x}' for x in stats['ndim']], rotation=45)
plt.ylabel('TOP1 or TOP3 score (%)')

plt.tight_layout()
plt.show()
