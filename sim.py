import functools
import typing

import jax
import jax.numpy as jnp
import sys

def sim(
        ninput: int,
        nhidden: int,
        noutput: int,
        iweight: jax.Array,
        rweight: jax.Array,
        idelay: jax.Array,
        rdelay: jax.Array,
        ispikes: jax.Array,
        dt = 0.025,
        tau_syn = 2.,
        tau_mem = 4.,
        checkpoint_every=100,
        max_delay_timesteps = 256
        ):
    # inp_spikes32, inp_delay, inp_weight, rec_delay, rec_weight = ispikes, idelay, iweight, rdelay, rweight
    inp_spikes32 = ispikes
    ninputs = ninput
    vth = 1.
    nneurons = nhidden + noutput

    max_delay = max_delay_timesteps*dt

    reg_idelay = max_delay*jax.lax.logistic((4/max_delay)*(idelay - max_delay/2)) #(max_delay_timesteps*dt)*jax.lax.logistic(idelay)
    reg_rdelay = max_delay*jax.lax.logistic((4/max_delay)*(rdelay - max_delay/2)) #(max_delay_timesteps*dt)*jax.lax.logistic(rdelay)

    # inp_delay = jnp.zeros((nneurons, ninput)).at[0:nhidden, 0:ninput].set(reg_idelay.reshape((nhidden, ninput)))
    inp_weight = jnp.zeros((nneurons, ninput)).at[0:nhidden, 0:ninput].set(iweight) #jnp.abs(jnp.zeros((nneurons, ninput)).at[0:nhidden, 0:ninput].set(iweight))
    # rec_delay = jnp.zeros((nneurons, nneurons)).at[nhidden:nneurons, 0:nhidden].set(reg_rdelay.reshape((noutput, nhidden)))
    rec_weight = jnp.zeros((nneurons, nneurons)).at[nhidden:nneurons, 0:nhidden].set(rweight) #jnp.abs(jnp.zeros((nneurons, nneurons)).at[nhidden:nneurons, 0:nhidden].set(rweight))
    # # jax.debug.print("{}", inp_weight)
    # # jnp.set_printoptions(threshold=sys.maxsize)
    # # jax.debug.print("Rec Delay: {x}", x=rec_delay)

    inp_delay = idelay #reg_idelay
    inp_weight = inp_weight #iweight
    rec_delay = rdelay #reg_rdelay
    rec_weight = rec_weight #rweight

    # assert (delay < max_delay_timesteps/dt).all()
    #
    inp_delay = inp_delay.flatten()
    inp_delay_timesteps = jnp.round(inp_delay/dt).astype(int)
    inp_weight = inp_weight.flatten()
    inp_tgt = jnp.repeat(jnp.arange(nneurons), ninputs)
    #
    rec_delay = rec_delay.flatten()
    rec_delay_timesteps = jnp.round(rec_delay/dt).astype(int)
    rec_weight = rec_weight.flatten()
    rec_tgt = jnp.repeat(jnp.arange(nneurons), nneurons)
    #
    synapse = LTIRingSynapse.init(max_delay_timesteps, nneurons)
    v = jnp.zeros(nneurons)
    isyn = jnp.zeros(nneurons)
    ttfs = jnp.zeros(nneurons) + 1201
    #
    alpha = jnp.exp(-dt/tau_syn)
    beta = jnp.exp(-dt/tau_mem)
    #
    S_hist = jnp.zeros(nneurons)
    def state_next(state, inp):
        (synapse, v, isyn, a, ttfs, S_hist), (t, ispikes_t) = state, inp
        ispikes_t = jnp.unpackbits(ispikes_t.view('uint8')).astype('bool')
        ispikes_t = ispikes_t[:ninputs] 
        dvdt = - v / tau_mem + isyn
        S = heaviside(v - vth)
        ts = annotate_spike_time_gradient(jnp.ones(nneurons, dtype='float64')*t+v*0, v, dvdt)
        cond = (ttfs > ts) & (S == 1)
        ttfs = jnp.where(cond, ts, ttfs)
        Ss = superspike(v - vth)
        synapse = synapse.enqueue_static(
                t + inp_delay_timesteps,
                inp_delay,
                inp_tgt,
                inp_weight * jnp.tile(ispikes_t, nneurons),
                tau_syn
                )
        synapse = synapse.enqueue(
                t + rec_delay_timesteps,
                rec_delay,
                rec_tgt,
                rec_weight * jnp.tile(S, nneurons),
                jnp.tile(v, nneurons),
                jnp.tile(dvdt, nneurons),
                tau_syn)
        synapse, (i_jump, v_jump) = synapse.pop(t)
        vnext_noreset = beta * v + isyn*dt + v_jump
        # vnext_noreset = jnp.where(vnext_noreset <= 0, 0, vnext_noreset)
        dvdt_min_noreset = -tau_mem * v + isyn
        dvdt_plus_reset = isyn + i_jump
        # PREVIOUS (+superspike): v = (1 - S) * (beta * v + isyn*dt + v_jump)
        v = v_reset(S, v, dvdt_min_noreset, dvdt_plus_reset, vnext_noreset)
        v = jnp.where(S_hist, 0.0, v)
        # v = jnp.where(v < 0.0, 0.0, v)
        isyn = isyn * alpha + i_jump
        S_hist = jnp.where(S, 1., S_hist)
        state = (synapse, v, isyn, a, ttfs, S_hist)
        state = jax.tree.map(lambda x: grad_modify(x), state)
        return state, (Ss, v)
    #
    a = jnp.zeros_like(v)
    state = synapse, v, isyn, a, ttfs, S_hist
    if checkpoint_every is None:
        (_, _, _, _, ttfs, S_hist), (s, v) = jax.lax.scan(state_next, state, xs=(jnp.arange(len(inp_spikes32)), inp_spikes32))
    else:
        (_, _, _, _, ttfs, S_hist), (s, v) = checkpointed_scan(state_next, state, xs=(jnp.arange(len(inp_spikes32)), inp_spikes32), checkpoint_every=checkpoint_every)
    return s, v, ttfs

class LTIRingSynapse(typing.NamedTuple):
    ijump: jax.Array
    vjump: jax.Array
    @classmethod
    def init(cls, ndelay, nneurons):
        return cls(
                jnp.full((ndelay+1, nneurons), 0.,),
                jnp.full((ndelay+1, nneurons), 0.,)
                )
    def enqueue_static(self, at, delay, nrn, w, tau_syn):
        max_delay = jnp.int32(self.ijump.shape[0])
        idx = at % max_delay
        ijump, vjump = w_to_isyn_jump_static(tau_syn, w, delay)
        return LTIRingSynapse(
                self.ijump.at[idx, nrn].add(ijump, mode='promise_in_bounds'),
                self.vjump.at[idx, nrn].add(vjump, mode='promise_in_bounds'),
                )
    def enqueue(self, at, delay, nrn, w, vpre, dvpre_dt, tau_syn):
        max_delay = jnp.int32(self.ijump.shape[0])
        idx = at % max_delay
        ijump, vjump = w_to_isyn_jump(tau_syn, w, delay, vpre, dvpre_dt)
        return LTIRingSynapse(
                self.ijump.at[idx, nrn].add(ijump, mode='promise_in_bounds'),
                self.vjump.at[idx, nrn].add(vjump, mode='promise_in_bounds'),
                )
    def pop(self, t):
        delay = jnp.int32(self.ijump.shape[0])
        idx = t % delay
        i_jump = self.ijump.at[idx, :].get(mode='promise_in_bounds')
        v_jump = self.vjump.at[idx, :].get(mode='promise_in_bounds')
        return LTIRingSynapse(
                self.ijump.at[idx, :].set(0, mode='promise_in_bounds'),
                self.vjump.at[idx, :].set(0, mode='promise_in_bounds')
                ), \
                (i_jump, v_jump)

@functools.partial(jax.custom_jvp)
def v_reset(S, v, dvdt_pre, dvdt_post, vnext):
    return jnp.where(S, 0.0, vnext)

@v_reset.defjvp
def v_reset_jvp(primals, tangents):
    'Eq 40.'
    S, v, dvdt_pre, dvdt_post, vnext = primals
    S_t, v_t, dvdt_pre_t, dvdt_post_t, vnext_t = tangents
    reset = 0 * vnext
    dvdt_pre = jax.lax.select(dvdt_pre == 0, jnp.ones(dvdt_pre.shape), dvdt_pre) # prevent nans
    reset_t = dvdt_post / dvdt_pre * v_t
    primals_out = jnp.where(S, reset, vnext)
    tangents_out = jnp.where(S, reset_t, vnext_t)
    return primals_out, tangents_out

@functools.partial(jax.custom_jvp, nondiff_argnames=['tau_syn'])
def w_to_isyn_jump(tau_syn, w, delay, vpre, dvpre_dt):
    'Eq 43. (vjump) & Eq 48 (ijump); modified'
    del delay, dvpre_dt, tau_syn, vpre
    return w, 0 * w

@w_to_isyn_jump.defjvp
def w_to_isyn_jump_jvp(tau_syn, primals, tangents):
    w, delay, vpre, dvpre_dt  = primals
    del delay, vpre
    w_t, delay_t, vpre_t, dvpre_dt_t = tangents
    del dvpre_dt_t
    dvdt = jax.lax.select(dvpre_dt == 0, jnp.ones(dvpre_dt.shape), dvpre_dt) # prevent nans
    tpost_t = -1./dvdt * vpre_t + delay_t # eq 37., p15.
    isyn_jump = w
    isyn_jump_t = w_t*1 + w/tau_syn * tpost_t # eq 48., p16., generalized
    isyn_jump_t = jax.lax.select(w != 0, isyn_jump_t, jnp.zeros_like(w))
    vpost_jump_t = - 1/tau_syn * tpost_t # eq 32
    vpost_jump_t =  vpost_jump_t * 0
    return (isyn_jump, 0 * isyn_jump), (isyn_jump_t, vpost_jump_t)

@functools.partial(jax.custom_jvp, nondiff_argnames=['tau_syn'])
def w_to_isyn_jump_static(tau_syn, w, delay):
    del delay, tau_syn
    return w, 0 * w

@w_to_isyn_jump_static.defjvp
def w_to_isyn_jump_jvp_static(tau_syn, primals, tangents):
    w, delay, = primals
    del delay
    w_t, delay_t = tangents
    tpost_t = delay_t # eq 37., p15.
    isyn_jump = w
    isyn_jump_t = w_t*1 + w/tau_syn * tpost_t # eq 48., p16., generalized
    isyn_jump_t = jax.lax.select(w != 0, isyn_jump_t, jnp.zeros_like(w))
    vpost_jump_t = - 1/tau_syn * tpost_t # eq 32
    vpost_jump_t =  vpost_jump_t * 0
    return (isyn_jump, 0*isyn_jump), (isyn_jump_t, vpost_jump_t)


@jax.custom_jvp
def annotate_spike_time_gradient(t, v, dvdt):
    return t

@annotate_spike_time_gradient.defjvp
def annotate_spike_time_gradient_jvp(primals, tangents):
    t, v, dvdt = primals
    t_t, v_t, dvdt_t = tangents
    #del t_t, dvdt_t, v
    dvdt = jax.lax.select(dvdt == 0, jnp.ones(dvdt.shape), dvdt) # prevent nans
    ts_t = -1./dvdt * v_t 
    return t, ts_t

def heaviside(x):
    return jnp.where(x < 0, 0.0, 1.0)

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
def grad_forget_bwd(res, g): 
    norm = jnp.linalg.norm(g)
    return (g / (0.5 + 0.5 * jnp.maximum(1., norm)),)
    # return (g / (jnp.maximum(1, norm)),)
grad_forget.defvjp(grad_forget_fwd, grad_forget_bwd)

# def grad_forget(x): return clip_gradient(-100, 100, x)
grad_modify = lambda x: grad_forget(clip_gradient(-5, 5, x))
grad_modify = lambda x: grad_forget(x)
# grad_modify = lambda x: x


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
