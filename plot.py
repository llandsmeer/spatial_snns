
import matplotlib.pyplot as plt
import json, pandas as pd, glob, tqdm

tables = {}
for f in tqdm.tqdm(glob.glob('saved/*/log.jsons')):
    args, hparams = {}, {}
    with open(f) as fh:
        for l in fh:
            j = json.loads(l)
            t = j.pop('table')
            if t == 'args': args = j
            elif t == 'hyperparams': hparams = j
            else:
                j.update(args)
                j.update(hparams)
                tables.setdefault(t, []).append(j)
dfs = {t: pd.DataFrame(v) for t, v in tables.items()}
epoch = dfs['epoch']

epoch['plotdim'] = epoch.apply(lambda x:
                               6 if x.netspec == 'inf' else x['ndim'], 1)

final = epoch[(epoch.i == 29) & (epoch.dt == 0.5)]


tgtfreq = {f'{f}': sub.t1p.values for f, sub in final.groupby('tgtfreq')}
plt.boxplot(tgtfreq.values(), labels=tgtfreq.keys())
plt.xlabel('Ftarget (Hz)')
plt.ylabel('TOP1 (%)')
plt.show()


final = final[final.tgtfreq == 10]

for net, sub in final.groupby('netspec'):
    sub = sub.sort_values('nhidden') # type: ignore
    sub = sub[['nhidden', 't1p']].groupby('nhidden').median()
    plt.plot(sub.index, sub.t1p, label=net)
plt.legend(title='Network Dimensions', ncol=2)
plt.xlabel('RSNN Neurons (#)')
plt.ylabel('Accuracy')
plt.show()


for nh, sub in final.groupby('nhidden'):
    sub = sub.sort_values('plotdim') # type: ignore
    sub = sub[['plotdim', 't1p']].groupby('plotdim').median()
    plt.plot(sub.index, sub.t1p, label=nh)
plt.legend(title='Nhidden', ncol=2)
plt.xlabel('Network Dimension')
plt.ylabel('Accuracy')
plt.show()
df['eps'] = df['netspec'].str.split('e').str[1].astype(float)
for n, sub in finaleps.groupby('nhidden'):
    if n not in [10, 50]:
        continue
    plt.plot(sub.eps * 100, sub.t1p, label=n)
plt.xlabel('Epsilon (%)')
plt.ylabel('Test accuracy (%)')
plt.ylim(50, 100)
plt.legend()







