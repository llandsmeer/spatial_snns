import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

__file__
lib = os.path.abspath(os.path.join(__file__, '..', 'ml_spike_event_queues'))
sys.path.insert(0, lib)

import synapse
import implementations

def sim(
    *,
    ninput: int,
    nhidden: int,
    iweight: jax.Array,
    rweight: jax.Array,
    idelay: jax.Array,
    rdelay: jax.Array,
    ispikes: jax.Array,
    dt = 0.025,
    tau_syn = 2.,
    tau_mem = 10.,
    vthres = 1.0,
    max_delay_ms = 20.,
    Q = implementations.SortedArray.sized(5),
    checkpoint_every=100
    ):
    inp_syn = synapse.mk_synapses(Q, # type: ignore
          delay_ms=idelay, dt_ms=dt,
          vthres=vthres, tau_syn_ms=tau_syn, n=nhidden*ninput,
          max_delay_ms=max_delay_ms
          )
    rec_syn = synapse.mk_synapses(Q, # type: ignore
          delay_ms=rdelay, dt_ms=dt,
          vthres=vthres, tau_syn_ms=tau_syn, n=nhidden*nhidden,
          max_delay_ms=max_delay_ms
          )
    inp_syn_step = jax.jit(type(inp_syn).timestep_static_spike)
    rec_syn_step = jax.jit(type(rec_syn).timestep_spike_detect_pre)
    v = jnp.zeros(nhidden)
    state = v, inp_syn, rec_syn
    state = jax.tree.map(lambda x: grad_forget(x), state)
    N = ispikes.shape[0]
    def step(state, inp):
        t, ispikes_t = inp
        v, inp_syn, rec_syn = state
        # previous code read @
        # however input seen is different from different viewpoints
        # isyn = (iweight @ inp_syn.isyn.reshape((nhidden,ninput)).sum(0)) + \
        #       (rweight @ rec_syn.isyn.reshape((nhidden,nhidden)).sum(0))
        isyn = (iweight * inp_syn.isyn.reshape((nhidden,ninput))).sum(1) + \
               (rweight * rec_syn.isyn.reshape((nhidden,nhidden))).sum(1)
        vnext, s = lif_step(v, isyn, tau_mem, dt, vthres)
        inp_syn = inp_syn_step(inp_syn, ts=t, s=jnp.tile(ispikes_t, nhidden))
        rec_syn = rec_syn_step(rec_syn, ts=t, v=jnp.tile(v, nhidden),
                       vnext=jnp.tile(vnext, nhidden))
        #vnext = clip_gradient(-100, 100, vnext)
        state = vnext, inp_syn, rec_syn
        state = jax.tree.map(lambda x: grad_forget(x), state)
        return state, (s, v)
    ts = jnp.arange(N) * dt
    if checkpoint_every is None:
        _, (s, v) = jax.lax.scan(step, state, xs=(ts, ispikes))
    else:
        _, (s, v) = checkpointed_scan(step, state, xs=(ts, ispikes), checkpoint_every=checkpoint_every)
    return s, v

@jax.custom_jvp
def superspike(x):
    'doi.dx/10.1162/neco_a_01086'
    return jnp.where(x < 0, 0.0, 1.0)

@superspike.defjvp
def superspike_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    primal_out = jnp.where(x < 0, 0.0, 1.0)
    tangent_out = x_dot / (jnp.abs(x)+1)**2
    return primal_out, tangent_out

@jax.custom_vjp
def clip_gradient(lo, hi, x): return x
def clip_gradient_fwd(lo, hi, x): return x, (lo, hi)
def clip_gradient_bwd(res, g): return (None, None, jnp.clip(g, res[0], res[1]))
clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)

@jax.custom_vjp
def grad_forget(x): return x
def grad_forget_fwd(x): return x, ()
def grad_forget_bwd(res, g): return (g / (1 + jnp.linalg.norm(g)),)
grad_forget.defvjp(grad_forget_fwd, grad_forget_bwd)

# def grad_forget(x): return clip_gradient(-100, 100, x)

def lif_step(U: jax.Array, I: jax.Array, tau_mem: float, dt: float, vth: float =1):
    S = superspike(U - vth)
    beta = jnp.exp(-dt/tau_mem)
    if not isinstance(tau_mem, float):
        beta = jnp.where(tau_mem > 0, beta, jnp.ones_like(tau_mem))
        S = jnp.where(tau_mem > 0, S, jnp.zeros_like(S))
    U_next = (1 - S) * (beta * U + I*dt)
    return U_next, S

def checkpointed_scan(step, state0, xs, checkpoint_every):
    lengths = [x.shape[0] for x in jax.tree.leaves(xs)]
    length = lengths[0]
    assert all(length == lx for lx in lengths)
    ncheckpoints = length // checkpoint_every
    init = ncheckpoints * checkpoint_every
    step_checkpoint = jax.checkpoint(lambda state, xs: jax.lax.scan(step, state, xs=xs), prevent_cse=False) # type: ignore
    xs_init = jax.tree.map(lambda arr: arr[:init].reshape((ncheckpoints, checkpoint_every) + arr.shape[1:]), xs)
    xs_rest = jax.tree.map(lambda arr: arr[init:], xs)
    state, trace_init = jax.lax.scan(step_checkpoint, state0, xs=xs_init)
    state, trace_rest = jax.lax.scan(step, state, xs=xs_rest)
    trace_init = jax.tree.map(lambda arr: arr.reshape((init,) + arr.shape[2:]), trace_init)
    trace = jax.tree.map(lambda init, rest: jnp.concatenate([init, rest], axis=0), trace_init, trace_rest)
    return state, trace
