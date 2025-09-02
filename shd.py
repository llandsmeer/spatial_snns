import jax
import jax.numpy as jnp
import h5py
import gzip
import typing
import matplotlib.pyplot as plt
import tqdm

class SHD(typing.NamedTuple):
    units:  typing.List[jax.Array]
    times:  typing.List[jax.Array]
    labels: jax.Array
    tmax:   float
    indicator_cache: dict[tuple, jax.Array]
    @property
    def size(self):
        return len(self.times)
    @classmethod
    def load(cls, fn, limit=None):
        if fn == 'train':
            fn = 'shd_train.h5'
        elif fn == 'test':
            fn = 'shd_test.h5'
        mkhandle = gzip.open if fn.endswith('.gz') else open
        with mkhandle(fn, 'rb') as f:
            ds = h5py.File(f)
            take = ... if limit is None else slice(limit)
            times = [jnp.array(x) for x in tqdm.tqdm(ds['spikes/times'][take])] # type: ignore
            return cls(
                units=[jnp.array(x) for x in tqdm.tqdm(ds['spikes/units'][take])], # type: ignore
                times=times,
                labels=jnp.array(ds['labels'][take]), # type: ignore
                tmax=float(jnp.max(jnp.array([jnp.max(x) for x in times]))),
                indicator_cache=dict()
                )
    def plot(self, idx, color=None):
        if color is None:
            color = plt.get_cmap('tab20')(self.labels[idx]/19)
        plt.scatter(
            self.times[idx],
            self.units[idx],
            color=color
            )
    def indicator(self, idx, dt=0.05, tsextra=0, pad=False):
        key = idx, dt, tsextra, pad
        if key in self.indicator_cache:
            return self.indicator_cache[key]
        print('CACHE MIS', idx)
        t = jnp.round(1e3 * self.times[idx] / dt).astype(int)
        u = self.units[idx]
        if pad:
            tmax = int(jnp.round(1e3 * self.tmax / dt))
            n = tmax + 1 + tsextra
        else:
            n = t.max() + 1 + tsextra
        # x = jnp.zeros((700, n)).at[u,t].set(1)
        x = jnp.zeros((n, 700), dtype=bool).at[t, u].set(True)
        # print(n, x.shape)
        self.indicator_cache[key] = x
        return x
    def indicators_labels(self, idxs, dt=0.05, tsextra=0):
        return [self.indicator(int(idx), dt=dt, tsextra=tsextra, pad=True)
                for idx in idxs], \
               [self.labels[idx] for idx in idxs]

if __name__ == '__main__':
    db = SHD.load('train', limit=1)
    db.plot(0)
    plt.show()

