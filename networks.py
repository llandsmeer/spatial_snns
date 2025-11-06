import typing

import sim

import jax
import jax.numpy as jnp

import numpy

NetSpec = \
        typing.Literal['inf'] | \
        typing.Literal['0'] | \
        typing.Literal['1'] | \
        typing.Literal['2'] | \
        typing.Literal['3'] | \
        typing.Literal['4'] | \
        typing.Literal['5'] | \
        typing.Literal['1e0.1'] | \
        typing.Literal['1e0.2'] | \
        typing.Literal['1e0.3'] | \
        typing.Literal['1e0.4'] | \
        typing.Literal['1e0.5'] | \
        typing.Literal['2e0.1'] | \
        typing.Literal['2e0.2'] | \
        typing.Literal['2e0.3'] | \
        typing.Literal['2e0.4'] | \
        typing.Literal['2e0.5']

class NetworkWithReadout(typing.NamedTuple):
    net: 'NoDelayNetwork | DelayNetwork | SpatialNetwork | EpsilonNetwork'
    w: jax.Array 
    def sim(self, iapp, **kwargs):
        s, v = self.net.sim(iapp, **kwargs)
        o = jnp.einsum('oh,th->o', self.w, s)
        # o = jax.tree.map(lambda x: sim.grad_modify(x), o) # not needed
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

class HyperParameters(typing.NamedTuple):
    nhidden: int
    ninput: int = 700
    noutput: int | None = None
    netspec: NetSpec = 'inf'
    ifactor: float = 1.
    rfactor: float = 1.
    delay_mu: float = 8.
    delay_sigma: float = 1.
    pos_sigma: float = 10.
    def build(self, key: jax.Array):
        key, readkey = jax.random.split(key)
        if self.netspec == '0':
            net = NoDelayNetwork.make(self, key)
        elif self.netspec == 'inf':
            net = DelayNetwork.make(self, key)
        elif 'e' in self.netspec:
            net = EpsilonNetwork.make(self, key)
        else:
            net = SpatialNetwork.make(self, key)
        if self.noutput is None:
            return net
        return NetworkWithReadout(net, self.random_weight(self.noutput, self.nhidden, key=readkey))
    @property
    def ndim(self):
        if self.netspec == 'inf':
            return float('inf')
        elif 'e' in self.netspec:
            return int(self.netspec.split('e')[0])
        else:
            return int(self.netspec)

    def random_pos(self, n, key):
        pos = self.pos_sigma * jax.random.normal(key, (n,  self.ndim))
        return pos
    def random_delay(self, a, b, key):
        delays = self.delay_mu + self.delay_sigma*jax.random.normal(key, (a,b)).flatten()
        return delays
    def random_weight(self, a, b, key, zero=False, factor=1.):
        W = jax.random.uniform(key=key, shape=(a,b), minval=-0.2, maxval=0.8)
        if zero:
            assert a == b
            W = zero_diagonal(W)
        weight = factor/(a*b) * W
        return jnp.abs(weight)
    def configdict(self):
        return {
                'nhidden': self.nhidden,
                'ninput': self.ninput,
                'noutput': self.noutput,
                'netspec': self.netspec,
                'ndim': self.ndim,
                'ifactor': self.ifactor,
                'rfactor': self.rfactor,
        }

class NoDelayNetwork(typing.NamedTuple):
    iw: jax.Array
    rw: jax.Array
    @classmethod
    def make(cls, hyper: HyperParameters, key: jax.Array):
        assert hyper.netspec == '0' and hyper.ndim == 0
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
        assert hyper.netspec == 'inf'
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
            rweight=zero_diagonal(self.rw), # avoid self-loops
            idelay=self.idelay,
            rdelay=self.rdelay,
            ispikes=ispikes,
            dt=kwargs.get('dt', 0.05),
            tau_syn=kwargs.get('tau_syn', 2.),
            tau_mem=kwargs.get('tau_mem', 10.),
            max_delay_timesteps=int(1+kwargs.get('max_delay_ms', 20.)/kwargs.get('dt', 0.05)),
            )

class SpatialNetwork(typing.NamedTuple):
    iw: jax.Array
    rw: jax.Array
    ipos: jax.Array
    rpos: jax.Array
    @classmethod
    def make(cls, hyper: HyperParameters, key: jax.Array):
        assert hyper.netspec.isdigit()
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

class EpsilonNetwork(typing.NamedTuple):
    iw: jax.Array
    rw: jax.Array
    ipos: jax.Array
    rpos: jax.Array
    ierr: jax.Array # tanh arg for delay mult
    rerr: jax.Array
    eps:  float
    @classmethod
    def make(cls, hyper: HyperParameters, key: jax.Array):
        assert 'e' in hyper.netspec
        keys = jax.random.split(key, 4)
        return EpsilonNetwork(
            iw = hyper.random_weight(hyper.nhidden, hyper.ninput, keys[0], zero=False, factor=hyper.ifactor),
            rw = hyper.random_weight(hyper.nhidden, hyper.nhidden, keys[1], factor=hyper.rfactor),
            ipos = hyper.random_pos(hyper.ninput, keys[2]),
            rpos = hyper.random_pos(hyper.nhidden, keys[3]),
            ierr = jnp.zeros((hyper.nhidden, hyper.ninput)).flatten(),
            rerr = jnp.zeros((hyper.nhidden, hyper.nhidden)).flatten(),
            # ierr = hyper.random_weight(hyper.nhidden, hyper.ninput, keys[0], zero=False, factor=1).flatten(),
            # rerr = hyper.random_weight(hyper.nhidden, hyper.nhidden, keys[1], factor=1).flatten(),
            eps = float(hyper.netspec.split('e')[1])
            )
    def sim(self, iapp, **kwargs):
        eps = jax.lax.stop_gradient(self.eps)
        constraint_ih = spatial_to_delay(self.rpos, self.ipos)
        constraint_hh = spatial_to_delay(self.rpos)
        return DelayNetwork(
                self.iw,
                self.rw,
                (1 + eps*(0.5 + 0.5*jnp.tanh(self.ierr))) * constraint_ih,
                (1 + eps*(0.5 + 0.5*jnp.tanh(self.rerr))) * constraint_hh).sim(iapp, **kwargs)
    def save(self, fn):
        numpy.savez_compressed(fn, iw=self.iw, rw=self.rw, ipos=self.ipos, rpos=self.rpos, ierr=self.ierr, rerr=self.rerr, eps=self.eps)
    def load(self, fn):
        with open(fn, 'rb') as f:
            f = numpy.load(f)
            iw = jnp.array(f['iw'])
            rw = jnp.array(f['rw'])
            ipos = jnp.array(f['ipos'])
            rpos = jnp.array(f['rpos'])
            ierr = jnp.array(f['ierr'])
            rerr = jnp.array(f['rerr'])
            eps = jnp.array(f['eps']).item()
        return EpsilonNetwork(
                iw=iw,
                rw=rw,
                ipos=ipos,
                rpos=rpos,
                ierr=ierr,
                rerr=rerr,
                eps=eps
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
    params = HyperParameters(netspec='0', ninput=10, nhidden=10)
    net = params.build(jax.random.PRNGKey(0))
    s = jax.random.uniform(jax.random.PRNGKey(0), shape=(2000, 10)) > 0.999
    o = net.sim(s)
    plt.imshow(jnp.hstack([s, o]).T, aspect='auto')
    plt.show()
