import jax
import os
import jax.numpy as jnp
import h5py
import gzip
import typing
import matplotlib.pyplot as plt
import tqdm
import functools

# @functools.partial(jax.jit, static_argnames=('n',))
# def build(n, t, u):
#     base = jnp.zeros((n, 700), dtype=bool)
#     x = base.at[t, u].set(True)
#     return x
@functools.partial(jax.jit, donate_argnames=('out',))
def set_at(out, idx, x):
    return out.at[idx, :, :].set(x)

@functools.partial(jax.jit, static_argnames=('n',))
def build32(n, t, u):
    word_idx = u // 32
    bit_idx = u % 32
    def mask(sel):
        m = jnp.uint32(bit_idx == sel) << sel
        x = jnp.zeros((n, 22), dtype='uint32')
        x = x.at[t, word_idx].add(m) # this actually works
        return x
    word = ((((mask( 0) | mask( 1)) | (mask( 2) | mask( 3)))  |
             ((mask( 4) | mask( 5)) | (mask( 6) | mask( 7))))  |
            (((mask( 8) | mask( 9)) | (mask(10) | mask(11)))  |
             ((mask(12) | mask(13)) | (mask(14) | mask(15)))))  | \
           ((((mask(16) | mask(17)) | (mask(18) | mask(19)))  |
             ((mask(20) | mask(21)) | (mask(22) | mask(23))))  |
            (((mask(24) | mask(25)) | (mask(26) | mask(27)))  |
             ((mask(28) | mask(29)) | (mask(30) | mask(31)))))
    return word

n = 32
t = jnp.arange(32)
u = jnp.arange(32)

n = 10
t = jnp.arange(10)
u = jnp.arange(10) * 32

@functools.partial(jax.jit, static_argnames=('n',))
def build_bits(n, t, u):
    words = (700 + 31) // 32  # 22
    base = jnp.zeros((n, words), dtype=jnp.uint32)
    # assert (t > 0).all()
    # assert (u > 0).all()
    t = t.astype('uint32')
    u = u.astype('uint32')
    word_idx = u // 32
    bit_idx = u % 32
    mask = jnp.uint32(1) << bit_idx
    x = base.at[t, word_idx].set(mask | base[t, word_idx])
    return x

class SHD(typing.NamedTuple):
    units:  typing.List[jax.Array]
    times:  typing.List[jax.Array]
    labels: jax.Array
    tmax:   float
    fn: str
    @property
    def size(self):
        return len(self.labels)
    @classmethod
    def load(cls, fn, limit=None, skip=False):
        if fn == 'train':
            fn = 'shd_train.h5'
        elif fn == 'test':
            fn = 'shd_test.h5'
        mkhandle = gzip.open if fn.endswith('.gz') else open
        with mkhandle(fn, 'rb') as f:
            ds = h5py.File(f)
            take = ... if limit is None else slice(limit)
            if skip:
                print("SKIP!")
                return cls(units=None, times=None,
                            labels=jnp.array(ds['labels'][take]),
                               tmax=None, fn=fn)
            times = [jnp.array(x) for x in tqdm.tqdm(ds['spikes/times'][take])] # type: ignore
            return cls(
                units=[jnp.array(x) for x in tqdm.tqdm(ds['spikes/units'][take])], # type: ignore
                times=times,
                labels=jnp.array(ds['labels'][take]), # type: ignore
                tmax=float(jnp.max(jnp.array([jnp.max(x) for x in times]))),
                fn=fn
                )
    def plot(self, idx, color=None):
        if color is None:
            color = plt.get_cmap('tab20')(self.labels[idx]/19)
        plt.scatter(
            self.times[idx],
            self.units[idx],
            color=color
            )
    def indicator32(self, idx, dt=0.05, tsextra=0, pad=False, justshape=False):
        import numpy as np
        cache_dir = os.path.expanduser('~/.cache/shd')
        os.makedirs(cache_dir, exist_ok=True)
        filename = f"{self.fn}_idx={idx}_dt={dt}_tsextra={tsextra}_pad={pad}.npz"
        filename = os.path.join(cache_dir, filename)
        #################
        if os.path.exists(filename):
            with np.load(filename) as f:
                X = jnp.array(f['arr'])
            if justshape:
                return X.shape
            return X
        print('CACHE MIS', idx)
        #################
        t = jnp.round(1e3 * self.times[idx] / dt).astype(int)
        u = self.units[idx]
        if pad:
            tmax = int(jnp.round(1e3 * self.tmax / dt))
            n = tmax + 1 + tsextra
        else:
            n = t.max() + 1 + tsextra
        n = int(n)
        if justshape:
            return (n, 22)
        #################
        padding = 0, jnp.ceil(len(u) / 1024).astype(int) * 1024 - len(u)
        u_pad = jnp.pad(u, padding, constant_values=-1)
        t_pad = jnp.pad(t, padding, constant_values=-1)
        x = build32(n, t_pad, u_pad)
        np.savez_compressed(filename, arr=np.array(x))
        return x
    def indicators_labels32(self, idxs, dt=0.05, tsextra=0):
        nd = len(idxs)
        nt, nu = self.indicator32(int(idxs[0]), dt=dt, tsextra=tsextra, pad=True, justshape=True)
        X = jnp.zeros((nd, nt, nu), dtype='uint32')
        for i, idx in enumerate(tqdm.tqdm(idxs)):
            x = self.indicator32(int(idx), dt=dt, tsextra=tsextra, pad=True)
            X = set_at(X, i, x)
        Y = jnp.array([self.labels[idx] for idx in idxs])
        return X, Y

if __name__ == '__main__':
    db = SHD.load('train', limit=1)
    db.plot(0)
    plt.show()

#         if out is None:
#             out = jnp.zeros((n, 22), dtype='uint32')
#             x = build32(n, t_pad, u_pad, out=out)
#             if use_cache:
#                 self.indicator_cache[key] = x
#         else:
#             x = build32(n, t_pad, u_pad, out=out[0], out_idx=out[1])
