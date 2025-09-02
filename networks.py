import typing

import sim

import jax
import jax.numpy as jnp

QUEUE = sim.implementations.SingleSpike
QUEUE = sim.implementations.FIFORing.sized(5)

if QUEUE is sim.implementations.SingleSpike:
    print('QUEUE is SingleSpike!')

class NetworkWithReadout(typing.NamedTuple):
    net: 'NoDelayNetwork | DelayNetwork | SpatialNetwork'
    w: jax.Array
    def sim(self, iapp, **kwargs):
        s, v = self.net.sim(iapp, **kwargs)
        del v
        breakpoint()
        x = w @ s
        breakpoint()
        return x.sum(0)
    def save(self, fn):
        import numpy
        self.net.save(fn)
        numpy.savez_compressed(fn+'_read', w=self.w)

class HyperParameters(typing.NamedTuple):
    nhidden: int
    ninput: int = 700
    noutput: int | None = None
    ndim: int | None = None
    ifactor: float = 1.
    rfactor: float = 1.
    def build(self, key: jax.Array|None=None):
        key, readkey = jax.random.split(key)
        if key is None:
            key = jax.random.PRNGKey(0)
        if self.ndim == 0:
            net = NoDelayNetwork.make(self, key)
        elif self.ndim == float('inf') or self.ndim is None:
            net = DelayNetwork.make(self, key)
        else:
            net = SpatialNetwork.make(self, key)
        if self.noutput is None:
            return net
        return NetworkWithReadout(net, self.random_weight(self.noutput, self.nhidden, key=readkey))
    def random_pos(self, n, key):
        assert self.ndim is not None and self.ndim > 0
        pos = 10 * jax.random.normal(key, (n,  self.ndim))
        return pos
    def random_delay(self, a, b, key):
        delays = 4 + .5*jax.random.normal(key, (a,b)).flatten()
        return delays
    def random_weight(self, a, b, key, zero=False, factor=1):
        W = jax.random.uniform(key=key, shape=(a,b), minval=-0.2, maxval=0.8)
        if zero:
            assert a == b
            W = zero_diagonal(W)
        weight = factor/(a*b) * W
        return jnp.abs(weight)

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
    def sim(self, iapp, **kwargs):
        return DelayNetwork(
                self.iw,
                self.rw,
                jnp.zeros_like(self.iw).flatten(),
                jnp.zeros_like(self.rw).flatten()).sim(iapp, **kwargs)
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
    def sim(self, ispikes, **kwargs):
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
            dt=kwargs.get('dt', 0.05),
            tau_syn=kwargs.get('tau_syn', 2.),
            tau_mem=kwargs.get('tau_mem', 10.),
            vthres=1.0,
            max_delay_ms=kwargs.get('max_delay_ms', 20.),
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
    def sim(self, iapp, **kwargs):
        return DelayNetwork(
                self.iw,
                self.rw,
                spatial_to_delay(self.rpos, self.ipos),
                spatial_to_delay(self.rpos)).sim(iapp, **kwargs)
    def save(self, fn):
        import numpy
        numpy.savez_compressed(fn, iw=self.iw, rw=self.rw, ipos=self.ipos, rpos=self.rpos)

def zero_diagonal(arr):
    n = arr.shape[0]
    return arr * (1 - jnp.eye(n))

def diagonal_const(arr, c):
    n = arr.shape[0]
    return arr * (1 - jnp.eye(n)) + jnp.eye(n) * c

def spatial_to_delay(r, from_=None):
    if from_ is not None:
        d = jax.vmap(lambda ri: jnp.sqrt(1e-2+((from_ - ri)**2).sum(axis=1)))(r)
        # d.shape is here (r.shape[0], from_.shape[0])
        d = d.flatten()
        return d
    else:
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
