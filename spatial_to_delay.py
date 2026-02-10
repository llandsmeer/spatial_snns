import numpy as np


params = np.load('params.npz')

iw = params['iw']
rw = params['rw']
idelay = params['idelay']
rdelay = params['rdelay']

print(iw.shape, rw.shape, idelay.shape, rdelay.shape)