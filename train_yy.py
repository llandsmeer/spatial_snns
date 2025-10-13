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
print(jax.default_backend())
print(jax.devices()[0].device_kind)

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import optax

import yy
import networks

import sys
print(time.ctime())
print(sys.argv)

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
parser.add_argument('--dt', type=float, default=0.05, help='Time step (bigger=faster, smaller=more accurate)')
parser.add_argument('--tmp', dest='save_dir', default='saved', action='store_const', const='/tmp/saved', help='Store in /tmp/saved')
parser.add_argument('--reload', default=False, action='store_true', help='Reload previous')
parser.add_argument('--layer', default=False, action='store_true', help='Create layer-wise network')
args = parser.parse_args()

save_dir = args.save_dir
fndir = f'd{args.ndim}_h{args.nhidden}_lr{args.lr}_ll{args.load_limit}_dt{args.dt}'
done = -1
if not args.reload:
    os.makedirs(f'{save_dir}/{fndir}', exist_ok=args.force)
fn_log = f'{save_dir}/{fndir}/log.txt'

def log(*a, **k):
    print(*a, **k)
    with open(fn_log, 'a') as f:
        print(*a, **k, file=f)


if args.debug:
    def excepthook(type, value, tb):
        traceback.print_exception(type, value, tb)
        log("\nStarting debugger...")
        pdb.post_mortem(tb)
    sys.excepthook = excepthook

log()
log(' == CONFIG == ')
for k, v in vars(args).items():
    log(k.ljust(20), v)
log()

#train = shd.SHD.load('train', limit=None)
train = yy.YY.load('train', limit=args.load_limit, skip=args.skip)
test = yy.YY.load('test', limit=args.load_limit_test, skip=args.skip)

params = networks.HyperParameters(
        ndim=args.ndim,
        ninput=4, #//args.downsample,
        nhidden=args.nhidden,
        ifactor=400 * args.ifactor,
        rfactor=35 * args.rfactor,
        noutput=3,
        layer=True
        )
net = params.build()

if args.reload:
    fns = sorted([x for x  in glob.glob(f'{save_dir}/{fndir}/*.npz') if not 'read' in x])[::-1]
    try:
        net = net.load(fns[0])
        done = int(fns[0].split('/')[-1].split('.')[0])
    except:
        net = net.load(fns[1])
        done = int(fns[0].split('/')[-1].split('.')[0])

tau_mem = jnp.array([0]*3 + [10.] * (args.nhidden-3))
tau_mem = 10.
@functools.partial(jax.jit, static_argnames=['aux'])
def loss(net, in_spikes, label, aux=True):
    o, v, f = net.sim(in_spikes, tau_mem=tau_mem, dt=args.dt)
    logits = - o[-3:] / 50 #- 0.5
    # jax.debug.print("ttfs {x}", x=o)

    l = optax.softmax_cross_entropy_with_integer_labels(
            logits, label)
    l = l # + (logits < 1) * v[:,:20].mean(0)
    # l = l + ((f - 0.05) ** 2).sum() * 0.01
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
        ws, v, f = net.sim(inp, tau_mem=tau_mem, dt=args.dt)
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
        ws, v, f = net.sim(x, tau_mem=tau_mem, dt=args.dt)
        ws = - ws[-3:] / 20 #- 0.5
        return ws, f
    # logits, f = jax.lax.map(get_logits, in_spikes, batch_size=64)
    logits, f = jax.vmap(get_logits)(in_spikes)
    f = f.mean(0)
    top3 = jnp.argsort(logits, axis=1)[:,-3:]
    top1p = 100 * (top3[:,-1] == labels).mean()
    top3p = 100 * (top3 == labels[:,None]).any(1).mean()
    return top1p, top3p, f
def performance_split(net, in_spikes, labels):
    BATCH_SIZE=64
    n = len(in_spikes)
    top1_list, top3_list, f_list = [], [], []
    for i in tqdm.tqdm(range(0, n, BATCH_SIZE)):
        batch_inputs = in_spikes[i:i+BATCH_SIZE]
        batch_labels = labels[i:i+BATCH_SIZE]
        top1, top3, f = performance(net, batch_inputs, batch_labels)
        top1_list.append(top1)
        top3_list.append(top3)
        f_list.extend(f)
    avg_top1 = sum(top1_list) / len(top1_list)
    avg_top3 = sum(top3_list) / len(top3_list)
    return avg_top1, avg_top3, jnp.array(f_list)

loss_and_grad = jax.jit(jax.value_and_grad(loss, argnums=0, has_aux=True))
batched_loss_and_grad = jax.jit(jax.value_and_grad(batched_loss, argnums=0, has_aux=False))

@jax.jit
def batched_update(opt_state, net, in_spikes, label):
    log('compiling')
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

optimizer = optax.chain(
    optax.adaptive_grad_clip(clipping=clip_factor, eps=0.001),
    optax.add_decayed_weights(weight_decay),
    optax.scale_by_adam(),
    optax.scale_by_schedule(schedule),
    scale_custom(10., lambda x: x in ('ipos', 'rpos')),
    optax.scale(-1.0)
)


opt_state = optimizer.init(net)

key = jax.random.PRNGKey(0)

ll = []

topii = []
top1p = []
top3p = []
top1p_train = []
top3p_train = []

inp_test, lbl_test = test.indicators_labels32(idxs=jnp.arange(test.size), dt=args.dt)
#inp = jnp.array(inp)
#inp = inp[:,:,::args.downsample].block_until_ready()

inp_train, lbl_train = train.indicators_labels32(idxs=jnp.arange(train.size), dt=args.dt)
# inp = inp[:,:,::args.downsample].block_until_ready()

try:
    print('start_training')
    for ii in range(10_000):
        if ii < done:
            print('skipping', ii)
            continue
        # log('IW', net.iw.min(), net.iw.max(), net.iw.mean(), net.iw.std())
        # log('RW', net.rw.min(), net.rw.max(), net.rw.mean(), net.rw.std())
        if ii % 100 == 0:
            log('TEST')
            t1p, t3p, f = performance_split(net, inp_test[:100], lbl_test[:100])
            log('FREQ', f.min(), f.max(), f.mean())
            t1p_train, t3p_train, ft = performance_split(net, inp_train[:100], lbl_train[:100])
            log('FREQtrain', ft.min(), ft.max(), ft.mean())
            topii.append(ii)
            top1p.append(t1p)
            top3p.append(t3p)
            top1p_train.append(t1p_train)
            top3p_train.append(t3p_train)
            log('TOP1', t1p, 'TOP3', t3p)
            log('TOP1train', t1p_train, 'TOP3train', t3p_train)
        if ii % 1000 == 999:
            log('TEST')
            t1p, t3p, f = performance_split(net, inp_test, lbl_test)
            log('FREQ', f.min(), f.max(), f.mean())
            t1p_train, t3p_train, ft = performance_split(net, inp_train[:1000], lbl_train[:1000])
            log('FREQtrain', ft.min(), ft.max(), ft.mean())
            topii.append(ii)
            top1p.append(t1p)
            top3p.append(t3p)
            top1p_train.append(t1p_train)
            top3p_train.append(t3p_train)
            log('TOP1', t1p, 'TOP3', t3p)
            log('TOP1train', t1p_train, 'TOP3train', t3p_train)
        key, nxt = jax.random.split(key)
        idxs = jax.random.randint(nxt, (args.batch_size,), 0, train.size)
        a = time.time()

        # inp, lbl = train.indicators_labels(idxs=idxs, dt=args.dt)
        # lbl = jnp.array(lbl)
        # inp = jnp.array(inp)
        # inp = inp[:,:,::args.downsample].block_until_ready()
        inp = inp_train[idxs]
        lbl = lbl_train[idxs]
        ###

        # log(logits.argmax().item(), lbl[0].item()) # , logits)
        if ii % 1000 == 0 and ii > 10:
            o, v, f = net.sim(inp[0], tau_mem=tau_mem, dt=args.dt)
            logits = v[-3:]
            for i, vi in enumerate(v.T):
                plt.plot(vi+i)
                plt.plot(o[i], i, 'o')

            ispikes = jnp.nonzero(inp[0])
            print(ispikes)
            for inpi in ispikes:
                plt.vlines(inpi, ymin= 0, ymax=len(v[0]), color='k')
            plt.show()
        ###
        b = time.time()
        opt_state, net, l, g = batched_update(opt_state, net, inp, lbl)
        # log(g)
        l = l.block_until_ready()
        c = time.time()

        # if ii == 5:
        #     breakpoint()

        if ii % 100 == 0 or True:
            log(f'TRAIN {idxs} {ii} L={l:.3f} ttrain={c-b:.2f}s teval={b-a:.2f}s')
            print(g)

        # l2 = batched_loss(net, inp, lbl)
        # d = time.time()
        # log('\t'.join(f'{k}:{v:.2f}' for k, v in g._asdict().items()))
        # log(f'{ii} {l:.3f}->{l2:.3f} {d-c:.2f}s {c-b:.2f}s {b-a:.2f}s')
        if ii % 10 == 0:
            net.save(f'{save_dir}/{fndir}/{ii:08d}')
        # ll.append((l, l2))
        ll.append(l)
except KeyboardInterrupt:
    pass

ll = jnp.array(ll)
# plt.plot(ll[:,0], 'o')
# plt.plot(ll[:,1], 'o')
plt.plot(topii, top1p, 'o-', label='top1 (test)')
plt.plot(topii, top3p, '-', label='top3 (test)')
plt.plot(topii, top1p_train, 'o--', label='top1 (train)')
plt.plot(topii, top3p_train, '--', label='top3 (train)')
plt.legend()
plt.savefig(f'{save_dir}/{fndir}/score.png')
plt.savefig(f'{save_dir}/{fndir}/score.svg')
plt.plot(ll, 'o', label='train')
plt.savefig(f'{save_dir}/{fndir}/loss.png')
plt.savefig(f'{save_dir}/{fndir}/loss.svg')


# breakpoint()




# f = lambda tree: jax.tree.map(lambda x: (float(jnp.min(x)), float(jnp.max(x))), tree)

# NetworkWithReadout(net=DelayNetwork(
#     iw=(-0.0675046919030234, 0.0758294276957268),
#     rw=(-0.053958577439816746, 0.06031096309966367),
#     idelay=(1.8681221046298968, 6.112585124498017),
#     rdelay=(1.9129557305044265, 5.873335145410393)),
#     w=(-0.05746382569791777, 0.04729931499965209))

# log(jnp.isnan(g.iw).sum())
# log(jnp.isnan(g.rw).sum())
# log(jnp.isnan(g.iw).sum())
# log(jnp.isnan(g.rw).sum())

# log(jnp.isnan(g.iw).sum())
# log(jnp.isnan(g.rw).sum())
