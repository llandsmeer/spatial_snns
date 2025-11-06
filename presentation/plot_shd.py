import sys
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'

sys.path.append('..')

import shd

train = shd.SHD.load('../shd_train.h5', limit=100)

seen = set()



for i in range(20):
    t = train.times[i]
    u = train.units[i]
    l = train.labels[i]
    if i not in seen:
        plt.plot(t, u, '.', color='black')
        plt.xlim(0, 1)
        plt.savefig(f'label_{l}.svg')
        plt.clf()
    seen.add(i)
