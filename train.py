import jax
import functools
import time
import jax.numpy as jnp

jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import optax


import shd
import networks

DOWNSAMPLE = 10 # 1, 2, 4, 5, 7, 10, 14, 20

train = shd.SHD.load('train', limit=100)
params = networks.HyperParameters(
        ndim=None,
        ninput=700//DOWNSAMPLE,
        nhidden=300//DOWNSAMPLE,
        ifactor=100,
        rfactor=0.1
        )
net = params.build()

@functools.partial(jax.jit, static_argnames=['aux'])
def loss(net, in_spikes, label, aux=True):
    out_spikes = net.sim(in_spikes)
    logits = out_spikes[:,:20].sum(0) / 100
    l = optax.softmax_cross_entropy_with_integer_labels(
            logits, label)
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
    l, g = batched_loss_and_grad(net, in_spikes, label)
    updates, opt_state = optimizer.update(g, opt_state)
    net = optax.apply_updates(net, updates)
    return opt_state, net, l, jax.tree.map(lambda x: x.ptp(), g)

# idx = 0
# inp = train.indicator(idx=idx, pad=True)
# lbl = train.labels[idx]
# l, (lg, s) = loss(net, inp, lbl)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(net)

key = jax.random.PRNGKey(0)

ll = []

batch_size = 4
try:
    for ii in range(1000):
        key, nxt = jax.random.split(key)
        idxs = jax.random.randint(nxt, (batch_size,), 0, train.size)
        a = time.time()
        inp, lbl = train.indicators_labels(idxs=idxs)
        inp = jnp.array(inp)
        inp = inp[:,:,::DOWNSAMPLE]
        b = time.time()
        opt_state, net, l, g = batched_update(opt_state, net, inp, lbl)
        c = time.time()
        l2 = batched_loss(net, inp, lbl)
        d = time.time()
        # print('\t'.join(f'{k}:{v:.2f}' for k, v in g._asdict().items()))
        print(f'{ii} {l:.3f}->{l2:.3f} {d-c:.2f}s {c-b:.2f}s {b-a:.2f}s')
        if ii % 10 == 0:
            net.save(f'saved/{ii:04d}')
        ll.append((l, l2))
except:
    pass
ll = jnp.array(ll)
plt.plot(ll[:,0], 'o')
plt.plot(ll[:,1], 'o')
plt.show()

breakpoint()









# print(jnp.isnan(g.iw).sum())
# print(jnp.isnan(g.rw).sum())
