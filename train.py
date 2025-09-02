import sys
import traceback
import pdb

import os
import argparse
import jax
import functools
import time
import jax.numpy as jnp

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import optax


import shd
import networks


import sys
print(time.ctime())
print(sys.argv)

parser = argparse.ArgumentParser()
def none_or_int(value): return None if value == 'None' else int(value)
parser.add_argument('--ndim', type=none_or_int, default=None, help='Dimension (None or int)')
parser.add_argument('--nhidden', type=int, default=100, help='Number of hidden units')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--downsample', type=int, default=1, help='Downsample factor (int)')
parser.add_argument('--load_limit', type=none_or_int, default=None, help='Load limit no samples')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--force', default=False, action='store_true', help='Overwrite')
parser.add_argument('--debug', default=False, action='store_true', help='Start pdb on error')
parser.add_argument('--dt', type=float, default=0.05, help='Time step (bigger=faster, smaller=more accurate)')
args = parser.parse_args()

if args.debug:
    def excepthook(type, value, tb):
        traceback.print_exception(type, value, tb)
        print("\nStarting debugger...")
        pdb.post_mortem(tb)
    sys.excepthook = excepthook

print()
print(' == CONFIG == ')
for k, v in vars(args).items():
    print(k.ljust(20), v)
print()

#train = shd.SHD.load('train', limit=None)
train = shd.SHD.load('train', limit=args.load_limit)
params = networks.HyperParameters(
        ndim=args.ndim,
        ninput=700//args.downsample,
        nhidden=args.nhidden,
        ifactor=400,
        rfactor=35,
        noutput=20,
        )
net = params.build()

tau_mem = jnp.array([0]*20 + [10.] * (args.nhidden-20))
tau_mem = 10.

fndir = f'{params.ndim}_{params.nhidden}_{args.lr}'
os.makedirs(f'saved/{fndir}', exist_ok=args.force)

@functools.partial(jax.jit, static_argnames=['aux'])
def loss(net, in_spikes, label, aux=True):
    out_spikes, v = net.sim(in_spikes, tau_mem=tau_mem, dt=args.dt)
    logits = v[-1,:20]
    l = optax.softmax_cross_entropy_with_integer_labels(
            logits, label)
    l = l # + (logits < 1) * v[:,:20].mean(0)
    if aux:
        return l, (jax.nn.softmax(logits), out_spikes.mean(0))
    else:
        return l

def batched_loss(net, in_spikes, labels):
    ls = jax.vmap(functools.partial(loss, aux=False), in_axes=(None, 0, 0))(net, jnp.array(in_spikes), jnp.array(labels))
    return ls.mean()

loss_and_grad = jax.jit(jax.value_and_grad(loss, argnums=0, has_aux=True))
batched_loss_and_grad = jax.jit(jax.value_and_grad(batched_loss, argnums=0, has_aux=False))

@jax.jit
def batched_update(opt_state, net, in_spikes, label):
    print('compiling')
    l, g = batched_loss_and_grad(net, in_spikes, label)
    updates, opt_state = optimizer.update(g, opt_state)
    net = optax.apply_updates(net, updates)
    return opt_state, net, l, jax.tree.map(lambda x: x.ptp(), g)

# idx = 0
# inp = train.indicator(idx=idx, pad=True)
# lbl = train.labels[idx]
# l, (lg, s) = loss(net, inp, lbl)

optimizer = optax.adam(args.lr)
opt_state = optimizer.init(net)

key = jax.random.PRNGKey(0)

ll = []

try:
    for ii in range(1000000):
        # print('IW', net.iw.min(), net.iw.max(), net.iw.mean(), net.iw.std())
        # print('RW', net.rw.min(), net.rw.max(), net.rw.mean(), net.rw.std())
        ##
        key, nxt = jax.random.split(key)
        idxs = jax.random.randint(nxt, (args.batch_size,), 0, train.size)
        a = time.time()

        inp, lbl = train.indicators_labels(idxs=idxs, dt=args.dt)
        lbl = jnp.array(lbl)
        inp = jnp.array(inp)
        inp = inp[:,:,::args.downsample].block_until_ready()
        ###
        #s, v = net.sim(inp[0], tau_mem=tau_mem, dt=args.dt)
        #logits = v[-1,:20]
        #print(logits.argmax().item(), lbl[0].item()) # , logits)
        # plt.plot(v)
        # plt.show()
        ###
        b = time.time()
        opt_state, net, l, g = batched_update(opt_state, net, inp, lbl)
        l = l.block_until_ready()
        c = time.time()

        print(f'TRAIN {idxs} {ii} L={l:.3f} ttrain={c-b:.2f}s teval={b-a:.2f}s')

        # l2 = batched_loss(net, inp, lbl)
        # d = time.time()
        # print('\t'.join(f'{k}:{v:.2f}' for k, v in g._asdict().items()))
        # print(f'{ii} {l:.3f}->{l2:.3f} {d-c:.2f}s {c-b:.2f}s {b-a:.2f}s')
        if ii % 10 == 0:
            net.save(f'saved/{fndir}/{ii:05d}')
        # ll.append((l, l2))
        ll.append(l)
except KeyboardInterrupt:
    pass

ll = jnp.array(ll)
# plt.plot(ll[:,0], 'o')
# plt.plot(ll[:,1], 'o')
plt.plot(ll, 'o')
plt.show()

breakpoint()






# print(jnp.isnan(g.iw).sum())
# print(jnp.isnan(g.rw).sum())
