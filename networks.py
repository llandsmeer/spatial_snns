import typing

import sim

import jax
import jax.numpy as jnp

QUEUE = sim.implementations.SingleSpike
QUEUE = sim.implementations.FIFORing.sized(5)

if QUEUE is sim.implementations.SingleSpike:
    print('QUEUE is SingleSpike!')

class HyperParameters(typing.NamedTuple):
    nhidden: int
    ninput: int = 700
    ndim: int | None = None
    ifactor: float = 1.
    rfactor: float = 1.
    def build(self, key: jax.Array|None=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        if self.ndim == 0:
            return NoDelayNetwork.make(self, key)
        elif self.ndim == float('inf') or self.ndim is None:
            return DelayNetwork.make(self, key)
        else:
            return SpatialNetwork.make(self, key)
    def random_pos(self, n, key):
        assert self.ndim is not None and self.ndim > 0
        pos = 10 * jax.random.normal(key, (n,  self.ndim))
        return pos
    def random_delay(self, a, b, key):
        delays = 4 + .5*jax.random.normal(key, (a,b)).flatten()
        return delays
    def random_weight(self, a, b, key, zero=False, factor=1):
        W = jax.random.normal(key, (a,b))
        if zero:
            assert a == b
            W = zero_diagonal(W)
        weight = factor/(a*b) * W**2
        return weight

class NoDelayNetwork(typing.NamedTuple):
    iw: jax.Array
    rw: jax.Array
    @classmethod
    def make(cls, hyper: HyperParameters, key: jax.Array):
        assert hyper.ndim == 0
        keys = jax.random.split(key, 2)
        return NoDelayNetwork(
            iw = hyper.random_weight(hyper.nhidden, hyper.ninput, keys[0], zero=False, factor=hyper.ifactor),
            rw = hyper.random_weight(hyper.nhidden, hyper.nhidden, keys[1], factor=hyper.rfactor)
            )
    def sim(self, iapp):
        return DelayNetwork(
                self.iw,
                self.rw,
                jnp.zeros_like(self.iw).flatten(),
                jnp.zeros_like(self.rw).flatten()).sim(iapp)
    def save(self, fn):
        import numpy
        numpy.savez_compressed(fn, iw=self.iw, rw=self.rw)

class DelayNetwork(typing.NamedTuple):
    iw: jax.Array
    rw: jax.Array
    idelay: jax.Array
    rdelay: jax.Array
    def save(self, fn):
        import numpy
        numpy.savez_compressed(fn, iw=self.iw, rw=self.rw, idelay=self.idelay, rdelay=self.rdelay)
    @classmethod
    def make(cls, hyper: HyperParameters, key: jax.Array):
        assert hyper.ndim == float('inf') or hyper.ndim is None
        keys = jax.random.split(key, 4)
        return DelayNetwork(
            iw = hyper.random_weight(hyper.nhidden, hyper.ninput, keys[0], zero=False, factor=hyper.ifactor),
            rw = hyper.random_weight(hyper.nhidden, hyper.nhidden, keys[1], factor=hyper.rfactor),
            idelay = hyper.random_delay(hyper.nhidden, hyper.ninput, keys[2]),
            rdelay = hyper.random_delay(hyper.nhidden, hyper.nhidden, keys[3])
            )
    def sim(self, ispikes):
        ninput = self.iw.shape[1]
        nhidden = self.rw.shape[0]
        return sim.sim(
            ninput=ninput,
            nhidden=nhidden,
            iweight=self.iw,
            rweight=self.rw,
            idelay=self.idelay,
            rdelay=self.rdelay,
            ispikes=ispikes,
            dt=0.05,
            tau_syn=2.,
            tau_mem=10.,
            vthres=1.0,
            max_delay_ms=20.,
            Q=QUEUE
            )

class SpatialNetwork(typing.NamedTuple):
    iw: jax.Array
    rw: jax.Array
    ipos: jax.Array
    rpos: jax.Array
    @classmethod
    def make(cls, hyper: HyperParameters, key: jax.Array):
        assert hyper.ndim is not None and hyper.ndim >= 0
        keys = jax.random.split(key, 4)
        return SpatialNetwork(
            iw = hyper.random_weight(hyper.nhidden, hyper.ninput, keys[0], zero=False, factor=hyper.ifactor),
            rw = hyper.random_weight(hyper.nhidden, hyper.nhidden, keys[1], factor=hyper.rfactor),
            ipos = hyper.random_pos(hyper.ninput, keys[2]),
            rpos = hyper.random_pos(hyper.nhidden, keys[3])
            )
    def sim(self, iapp):
        return DelayNetwork(
                self.iw,
                self.rw,
                sptial_to_delay(self.ipos),
                sptial_to_delay(self.rpos)).sim(iapp)
    def save(self, fn):
        import numpy
        numpy.savez_compressed(fn, iw=self.iw, rw=self.rw, ipos=self.ipos, rpos=self.rpos)

def zero_diagonal(arr):
    n = arr.shape[0]
    return arr * (1 - jnp.eye(n))

def diagonal_const(arr, c):
    n = arr.shape[0]
    return arr * (1 - jnp.eye(n)) + jnp.eye(n) * c

def sptial_to_delay(r):
    d = jax.vmap(lambda ri: jnp.sqrt(1e-2+((r - ri)**2).sum(axis=1)))(r)
    d = diagonal_const(d, 1000000)
    d = d.flatten()
    return d

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    params = HyperParameters(ndim=0, ninput=10, nhidden=10)
    net = params.build()
    s = jax.random.uniform(jax.random.PRNGKey(0), shape=(2000, 10)) > 0.999
    o = net.sim(s)
    plt.imshow(jnp.hstack([s, o]).T, aspect='auto')
    plt.show()
