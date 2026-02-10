import sys
import tqdm
import traceback
import pdb
import glob

import os
import argparse
import jax
import functools
import time
import jax.numpy as jnp

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# jax.config.update("jax_platforms", "cpu")
# print(jax.default_backend())
# print(jax.devices()[0].device_kind)

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import optax

import yy
import networks

import sys
print(time.ctime())


parser = argparse.ArgumentParser()
def none_or_int(value): return None if value == 'None' else int(value)
parser.add_argument('--ndim', type=none_or_int, default=None, help='Dimension (None or int)')
parser.add_argument('--nhidden', type=int, default=100, help='Number of hidden units')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
# parser.add_argument('--downsample', type=int, default=1, help='Downsample factor (int)')
parser.add_argument('--load_limit', type=none_or_int, default=None, help='Load limit no samples')
parser.add_argument('--load_limit_test', type=none_or_int, default=None, help='Load limit no test samples')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--ifactor', type=float, default=1, help='Extra ifactor multiplier')
parser.add_argument('--rfactor', type=float, default=1, help='Extra ifactor multiplier')
parser.add_argument('--force', default=False, action='store_true', help='Overwrite')
parser.add_argument('--skip', default=False, action='store_true', help='Skip metadata loading')
parser.add_argument('--debug', default=False, action='store_true', help='Start pdb on error')
parser.add_argument('--dt', type=float, default=0.01, help='Time step (bigger=faster, smaller=more accurate)')
parser.add_argument('--tmp', dest='save_dir', default='saved', action='store_const', const='/tmp/saved', help='Store in /tmp/saved')
parser.add_argument('--reload', default=False, action='store_true', help='Reload previous')
parser.add_argument('--layer', default=False, action='store_true', help='Create layer-wise network')
parser.add_argument('--margin', type=float, default=0.)
parser.add_argument('--beta', type=float, default=10.)
parser.add_argument('--wseed', type=int, default=42)
parser.add_argument('--bseed', type=int, default=97)
parser.add_argument('--ttrain', type=int, default=10000)
args = parser.parse_args()


test = yy.YY.load('test')

params = networks.HyperParameters(
        ndim=args.ndim,
        ninput=4, #//args.downsample,
        nhidden=args.nhidden,
        ifactor=400 * args.ifactor,
        rfactor=35 * args.rfactor,
        noutput=3,
        layer=args.layer
        )
net = params.build(key=jax.random.PRNGKey(args.wseed))
# fns = sorted([x for x  in glob.glob('saved-snellius/d2_h60_lr0.0001_llNone_dt0.01_beta10.0/*.npz') if not 'read' in x])[::-1]
# fns = sorted([x for x  in glob.glob('saved-snellius/d2_h60_lr7.5e-05_llNone_dt0.01_beta10.0/*.npz') if not 'read' in x])[::-1]
fns = sorted([x for x  in glob.glob('saved-snellius/d2_h45_lr0.00015_llNone_dt0.01_beta10.0/*.npz') if not 'read' in x])[::-1]
net = net.load(fns[0])



tau_mem = jnp.array([0]*3 + [10.] * (args.nhidden-3))
tau_mem = 4. #10.
@functools.partial(jax.jit, static_argnames=['aux'])
def loss(net, in_spikes, label, aux=True):
    o, v, f = net.sim(in_spikes, tau_mem=tau_mem, dt=args.dt, max_delay_ms=3.5)
    logits = - o[-3:] #/ 50 #- 0.5

    # logits = o
    l = optax.softmax_cross_entropy_with_integer_labels(logits, label)


    # spike_times = o[-3:] * args.dt
    # # margin = 10
    # # margin_alpha = 0.1

    # # sorted_logits = jnp.sort(spike_times)[::-1]
    # # margin_l = jnp.maximum(0., margin - (sorted_logits[1] - sorted_logits[0]))

    # correct_time = spike_times[label]
    # idx = jnp.arange(spike_times.shape[0])
    # targets = jnp.full_like(spike_times, correct_time)
    # delta_tau = jnp.where(idx == label, 0.0, args.margin)
    # l = jnp.square(spike_times - targets - delta_tau)
    # l = 0.5 * jnp.sum(l)

    # spike_times = o[-3:]
    # correct_time = spike_times[label]
    # idx = jnp.arange(spike_times.shape[0])
    # mask = idx != label
    # diffs = jnp.where(mask, args.beta * (correct_time - spike_times + args.margin), -100)
    # l = jnp.sum(jax.nn.softplus(diffs))

    # f = f * args.dt

    l = l  #+ margin_alpha * margin_l #(logits < 1) * v[:,:20].mean(0)
    # l = l + ((f - 0.005) ** 2).sum() #* 0.01
    if aux:
        return l, (jax.nn.softmax(logits), v)
    else:
        return l

def batched_loss(net, in_spikes, labels):
    ls = jax.vmap(functools.partial(loss, aux=False), in_axes=(None, 0, 0))(net, jnp.array(in_spikes), jnp.array(labels))
    return ls.mean()


@jax.jit
def performance(net, in_spikes, labels):
    def step(carry, x):
        inp, lbl = x
        ws, v, f = net.sim(inp, tau_mem=tau_mem, dt=args.dt, max_delay_ms=3.5)
        del v
        top3 = jnp.argsort(ws)[-3:]
        top1_hit = (top3[-1] == lbl).astype(jnp.int32)
        top3_hit = jnp.any(top3 == lbl).astype(jnp.int32)
        return (carry[0] + top1_hit, carry[1] + top3_hit, carry[2] + f), None
    (top1_count, top3_count, f), _ = jax.lax.scan(
        step,
        (0, 0, jnp.zeros(args.nhidden)),
        (in_spikes, labels),
        unroll=8
    )
    return 100*top1_count / len(labels), 100*top3_count / len(labels), f / len(labels)

@jax.jit
def performance(net, in_spikes, labels):
    @jax.jit
    def get_logits(x):
        ws, v, f = net.sim(x, tau_mem=tau_mem, dt=args.dt, max_delay_ms=3.5)
        ws = - ws[-3:] #/ 20 #- 0.5
        return ws, f
    # logits, f = jax.lax.map(get_logits, in_spikes, batch_size=64)
    logits, f = jax.vmap(get_logits)(in_spikes)
    f = f.mean(0)
    top3 = jnp.argsort(logits, axis=1)[:,-2:]
    logits = jnp.sort(logits, axis=1)
    top1p = 100 * (top3[:,-1] == labels).mean()
    top3p = 100 * (top3 == labels[:,None]).any(1).mean()
    top2_diff = jnp.where(top3[:,-2] == labels, jnp.abs(logits[:,-1] - logits[:,-2]), jnp.nan)
    top2_diff = jnp.nanmean(top2_diff)
    return top1p, top3p, f, top2_diff
def performance_split(net, in_spikes, labels):
    BATCH_SIZE=64
    n = len(in_spikes)
    top1_list, top3_list, f_list, top2_diff_list = [], [], [], []
    for i in tqdm.tqdm(range(0, n, BATCH_SIZE)):
        batch_inputs = in_spikes[i:i+BATCH_SIZE]
        batch_labels = labels[i:i+BATCH_SIZE]
        top1, top3, f, top2_diff = performance(net, batch_inputs, batch_labels)
        top1_list.append(top1)
        top3_list.append(top3)
        f_list.extend(f)
        top2_diff_list.append(top2_diff)
    avg_top1 = sum(top1_list) / len(top1_list)
    avg_top3 = sum(top3_list) / len(top3_list)
    avg_top2_diff = sum(top2_diff_list) / len(top2_diff_list)
    return avg_top1, avg_top3, jnp.array(f_list), avg_top2_diff

loss_and_grad = jax.jit(jax.value_and_grad(loss, argnums=0, has_aux=True))
batched_loss_and_grad = jax.jit(jax.value_and_grad(batched_loss, argnums=0, has_aux=False))

@jax.jit
def batched_update(opt_state, net, in_spikes, label):
    # log('compiling')
    l, g = batched_loss_and_grad(net, in_spikes, label)
    updates, opt_state = optimizer.update(g, opt_state, net)
    net = optax.apply_updates(net, updates)
    return opt_state, net, l, jax.tree.map(lambda x: x.ptp(), g)

# idx = 0
# inp = train.indicator(idx=idx, pad=True)
# lbl = train.labels[idx]
# l, (lg, s) = loss(net, inp, lbl)

clip_factor = 2.0
weight_decay = 1e-5
warmup_steps = 500
tbptt_len = 50  # truncate sequences for backprop

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=args.lr,
    warmup_steps=warmup_steps,
    decay_steps=10000,
    end_value=args.lr * 0.1
)

# schedule = optax.cosine_decay_schedule(
#     init_value=args.lr,
#     decay_steps=4000,
#     alpha=0.1
# )

def scale_custom(scale: float, check=lambda key: False):
    def init_fn(_): return ()
    def update_fn(updates, state, params=None):
        def f(path, up):
            if check(path[-1].name):
                return up * scale
            return up
        updates = jax.tree_util.tree_map_with_path(f, updates)
        return updates, state
    return optax.GradientTransformation(init_fn, update_fn)

# optimizer = optax.chain(
#     optax.adaptive_grad_clip(clipping=clip_factor, eps=0.001),
#     optax.add_decayed_weights(weight_decay),
#     optax.scale_by_adam(),
#     optax.scale_by_schedule(schedule),
#     scale_custom(10., lambda x: x in ('ipos', 'rpos')),
#     scale_custom(1., lambda x: x in ('idelay', 'rdelay')),
#     optax.scale(-1.0)
# )

optimizer = optax.chain(
    optax.adaptive_grad_clip(clipping=clip_factor, eps=0.001),
    optax.add_decayed_weights(weight_decay),
    optax.scale_by_adam(),
    optax.scale_by_schedule(schedule),
    scale_custom(0., lambda x: x in ('ipos', 'rpos')),
    optax.scale(-1.0)
)


opt_state = optimizer.init(net)

key = jax.random.PRNGKey(args.bseed)

perm = jnp.zeros((args.batch_size,))

ll = []

topii = []
top1p = []
top3p = []
top1p_train = []
top3p_train = []
top2d = []
top2d_train = []
losses = []

inp_test, lbl_test = test.indicators_labels32(idxs=jnp.arange(test.size), dt=args.dt)


# t1p, t3p, f, t2d = performance_split(net, inp_test, lbl_test)

def get_logits(x):
    ws, v, f = net.sim(x, tau_mem=tau_mem, dt=args.dt, max_delay_ms=3.5)
    ws = - ws[-3:] #/ 20 #- 0.5
    return ws, f

t1p = 0


with open("res_debug.txt", 'w') as f:
    f.write("index,prediction,label\n")
    for i in tqdm.tqdm(range(1)): #for i in tqdm.tqdm(range(len(inp_test))):
        logits, _ = get_logits(inp_test[i])
        pred = int(jnp.argsort(logits, axis=0)[-1])
        if pred == lbl_test[i]:
            t1p = t1p + 1
        f.write(f'{i},{pred},{lbl_test[i]}\n')
f.close()
print('TOP1', t1p)