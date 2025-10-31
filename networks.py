import typing

import sim

import jax
import jax.numpy as jnp

import numpy

class NetworkWithReadout(typing.NamedTuple):
    net: 'NoDelayNetwork | DelayNetwork | SpatialNetwork'
    w: jax.Array
    def sim(self, iapp, **kwargs):
        s, v, _ = self.net.sim(iapp, **kwargs)
        o = jnp.einsum('oh,th->o', self.w, s)
        # o = jax.tree.map(lambda x: sim.grad_modify(x), o)
        return o, v, s.mean(0)
    def save(self, fn):
        self.net.save(fn)
        numpy.savez_compressed(fn+'_read', w=self.w)
    def load(self, fn):
        if '.npz' in fn:
            fn_read = fn.replace('.npz', '_read.npz')
        else:
            fn_read = fn + '_read'
        with open(fn_read, 'rb') as f:
            f = numpy.load(f)
            w = f['w']
        return NetworkWithReadout(
                net=self.net.load(fn),
                w=jnp.array(w)
                )
    
class NetworkWithTTFS(typing.NamedTuple):
    net: 'NoDelayNetwork | DelayNetwork | SpatialNetwork'
    def sim(self, iapp, **kwargs):
        s, v, ttfs = self.net.sim(iapp, **kwargs)
        return ttfs, v, s.mean(0)
    def save(self, fn):
        self.net.save(fn)
        # numpy.savez_compressed(fn+'_read', w=self.w)
    def load(self, fn):
        return NetworkWithTTFS(
                net=self.net.load(fn)
                )

class HyperParameters(typing.NamedTuple):
    nhidden: int
    ninput: int = 700
    noutput: int | None = None
    ndim: int | None = None
    ifactor: float = 1.
    rfactor: float = 1.
    layer: bool = False
    def build(self, key: jax.Array|None=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key, readkey = jax.random.split(key)
        if self.layer:
            if self.ndim == 0:
                net = NoDelayLayerNetwork.make(self, key)
            elif self.ndim == float('inf') or self.ndim is None:
                net = DelayLayerNetwork.make(self, key)
        else:
            if self.ndim == 0:
                net = NoDelayNetwork.make(self, key)
            elif self.ndim == float('inf') or self.ndim is None:
                net = DelayNetwork.make(self, key)
            else:
                net = SpatialNetwork.make(self, key)
        if self.noutput is None:
            return net
        elif self.layer:
            return NetworkWithTTFS(net)
        return NetworkWithReadout(net, self.random_weight(self.noutput, self.nhidden, key=readkey))
    def random_pos(self, n, key):
        assert self.ndim is not None and self.ndim > 0
        pos = 10 * jax.random.normal(key, (n,  self.ndim))
        return pos
    def random_delay(self, a, b, key):
        delays = 1 + .5*jax.random.normal(key, (a,b)).flatten()
        return delays
    def random_weight(self, a, b, key, zero=False, factor=1):
        W = jax.random.uniform(key=key, shape=(a,b), minval=-0.2, maxval=0.8)
        if zero:
            assert a == b
            W = zero_diagonal(W)
        weight = factor/(a*b) * W
        return weight #jnp.abs(weight)
    def random_idelay(self, a, b, key):
        delays = 0.5 + 0.5*jax.random.normal(key, (a,b)).flatten()
        return delays
    def random_rdelay(self, a, b, key):
        delays = 0.5*jax.random.normal(key, (a,b)).flatten()
        return delays

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
        numpy.savez_compressed(fn, iw=self.iw, rw=self.rw)
    def load(self, fn):
        with open(fn, 'rb') as f:
            f = numpy.load(f)
            iw = jnp.array(f['iw'])
            rw = jnp.array(f['rw'])
        return NoDelayNetwork(
                iw=iw,
                rw=rw
                )
    
class NoDelayLayerNetwork(typing.NamedTuple):
    iw: jax.Array
    rw: jax.Array
    @classmethod
    def make(cls, hyper: HyperParameters, key: jax.Array):
        assert hyper.ndim == 0
        keys = jax.random.split(key, 2)
        return NoDelayLayerNetwork(
            iw = hyper.random_weight(hyper.nhidden, hyper.ninput, keys[0], zero=False, factor=hyper.ifactor),
            rw = hyper.random_weight(hyper.noutput, hyper.nhidden, keys[1], factor=hyper.rfactor)
            )
    def sim(self, iapp, **kwargs):
        key = jax.random.PRNGKey(11)
        idelay = 0.5 + 0.5*jax.random.normal(key, self.iw.shape).flatten() #jnp.full_like(self.iw, -100).flatten()
        rdelay = 0.5*jax.random.normal(key, self.rw.shape).flatten() #jnp.full_like(self.rw, -100).flatten()
        return DelayLayerNetwork(
                self.iw,
                self.rw,
                idelay,
                rdelay).sim(iapp, **kwargs)
    def save(self, fn):
        numpy.savez_compressed(fn, iw=self.iw, rw=self.rw)
    def load(self, fn):
        with open(fn, 'rb') as f:
            f = numpy.load(f)
            iw = jnp.array(f['iw'])
            rw = jnp.array(f['rw'])
        return NoDelayLayerNetwork(
                iw=iw,
                rw=rw
                )

class DelayNetwork(typing.NamedTuple):
    iw: jax.Array
    rw: jax.Array
    idelay: jax.Array
    rdelay: jax.Array
    def save(self, fn):
        numpy.savez_compressed(fn, iw=self.iw, rw=self.rw, idelay=self.idelay, rdelay=self.rdelay)
    def load(self, fn):
        with open(fn, 'rb') as f:
            f = numpy.load(f)
            iw = jnp.array(f['iw'])
            rw = jnp.array(f['rw'])
            id_ = jnp.array(f['idelay'])
            rd_ = jnp.array(f['rdelay'])
        return DelayNetwork(
                iw=iw,
                rw=rw,
                idelay=id_,
                rdelay=rd_
                )
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
            max_delay_timesteps=int(1+kwargs.get('max_delay_ms', 20.)/kwargs.get('dt', 0.01)),
            )
    
class DelayLayerNetwork(typing.NamedTuple):
    iw: jax.Array
    rw: jax.Array
    idelay: jax.Array
    rdelay: jax.Array
    def save(self, fn):
        numpy.savez_compressed(fn, iw=self.iw, rw=self.rw, idelay=self.idelay, rdelay=self.rdelay)
    def load(self, fn):
        with open(fn, 'rb') as f:
            f = numpy.load(f)
            iw = jnp.array(f['iw'])
            rw = jnp.array(f['rw'])
            id_ = jnp.array(f['idelay'])
            rd_ = jnp.array(f['rdelay'])
        return DelayLayerNetwork(
                iw=iw,
                rw=rw,
                idelay=id_,
                rdelay=rd_
                )
    @classmethod
    def make(cls, hyper: HyperParameters, key: jax.Array):
        assert hyper.ndim == float('inf') or hyper.ndim is None
        keys = jax.random.split(key, 4)
        return DelayLayerNetwork(
            iw = hyper.random_weight(hyper.nhidden, hyper.ninput, keys[0], zero=False, factor=hyper.ifactor),
            rw = hyper.random_weight(hyper.noutput, hyper.nhidden, keys[1], factor=hyper.rfactor),
            idelay = hyper.random_idelay(hyper.nhidden, hyper.ninput, keys[2]),
            rdelay = hyper.random_rdelay(hyper.noutput, hyper.nhidden, keys[3])
            )
    def sim(self, ispikes, **kwargs):
        ninput = self.iw.shape[1]
        nhidden = self.rw.shape[1]
        noutput = self.rw.shape[0]
        return sim.sim(
            ninput=ninput,
            nhidden=nhidden,
            noutput=noutput,
            iweight=self.iw,
            rweight=self.rw,
            idelay=self.idelay,
            rdelay=self.rdelay,
            ispikes=ispikes,
            dt=kwargs.get('dt', 0.05),
            tau_syn=kwargs.get('tau_syn', 2.),
            tau_mem=kwargs.get('tau_mem', 10.),
            max_delay_timesteps=int(1+kwargs.get('max_delay_ms', 20.)/kwargs.get('dt', 0.01)),
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
        numpy.savez_compressed(fn, iw=self.iw, rw=self.rw, ipos=self.ipos, rpos=self.rpos)
    def load(self, fn):
        with open(fn, 'rb') as f:
            f = numpy.load(f)
            iw = jnp.array(f['iw'])
            rw = jnp.array(f['rw'])
            ipos = jnp.array(f['ipos'])
            rpos = jnp.array(f['rpos'])
        return SpatialNetwork(
                iw=iw,
                rw=rw,
                ipos=ipos,
                rpos=rpos
                )

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
