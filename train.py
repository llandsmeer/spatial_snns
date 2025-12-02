import sys
import json
import datetime
import tqdm
import traceback
import pdb
import glob
import uuid

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

import shd
import networks

import sys
print(time.ctime())
print(sys.argv)

parser = argparse.ArgumentParser()
def none_or_int(value): return None if value == 'None' else int(value)
parser.add_argument('--net', type=str, default='inf', help='Dimension (inf or int or <int>e<float>)')
parser.add_argument('--nhidden', type=int, default=100, help='Number of hidden units')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--load_limit', type=none_or_int, default=None, help='Load limit no samples')
parser.add_argument('--load_limit_test', type=none_or_int, default=None, help='Load limit no test samples')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--ifactor', type=float, default=None, help='Extra ifactor multiplier')
parser.add_argument('--rfactor', type=float, default=None, help='Extra ifactor multiplier')
parser.add_argument('--force', default=False, action='store_true', help='Overwrite')
parser.add_argument('--skip', default=False, action='store_true', help='Skip metadata loading')
parser.add_argument('--debug', default=False, action='store_true', help='Start pdb on error')
parser.add_argument('--dt', type=float, default=0.05, help='Time step (bigger=faster, smaller=more accurate)')
parser.add_argument('--tmp', dest='save_dir', default='saved', action='store_const', const='/tmp/saved', help='Store in /tmp/saved')
parser.add_argument('--reload', default=False, action='store_true', help='Reload previous')
parser.add_argument('--seed', type=int, default=0, help='Seed for network generation')
parser.add_argument('--nepochs', type=int, default=100, help='Epochs to trian for')
parser.add_argument('--delaygradscale', type=float, default=10., help='Scale delay gradients (or 0 for none)')
parser.add_argument('--delaymu', type=float, default=8., help='Mu')
parser.add_argument('--delaysigma', type=float, default=1., help='Sigma')
parser.add_argument('--possigma', type=float, default=20., help='Sigma')
parser.add_argument('--tgtfreq', type=float, default=10, help='Target frequency Hz')
parser.add_argument('--population_freq', default=False, action='store_true', help='Target freq after mean')
parser.add_argument('--tag', type=str, default='default', help='Experiment ids')
parser.add_argument('--adex', default=False, action='store_true', help='AdEx model')
parser.add_argument('--adex_a', type=float, default=0.001, help='AdEx a')
parser.add_argument('--adex_b', type=float, default=0.001, help='AdEx b')
# parser.add_argument('--adex_a', type=float, default=0.001, help='AdEx a')
# parser.add_argument('--adex_b', type=float, default=0.0002, help='AdEx b')
parser.add_argument('--adex_tau', type=float, default=150, help='AdEx w tau')
parser.add_argument('--adex_dt', type=float, default=0.1, help='AdEx DeltaT')
parser.add_argument('--vplot', default=False, action='store_true', help='AdEx model')


args = parser.parse_args()

sim_kwargs = { }
if args.adex:
    sim_kwargs['model'] = 'adex'
    sim_kwargs['adex_a'] = args.adex_a
    sim_kwargs['adex_b'] = args.adex_b
    sim_kwargs['adex_tau'] = args.adex_tau
    sim_kwargs['adex_DT'] = args.adex_dt

RUN_ID = str(uuid.uuid4())

now = datetime.datetime.utcnow()

save_dir = args.save_dir
fndir = f'{now.year}{now.month:02d}{now.day:02d}_d{args.net}_h{args.nhidden}_lr{args.lr}_ll{args.load_limit}_dt{args.dt}_{RUN_ID}'

if args.adex:
    fndir = 'adex_' + fndir

done = -1
if not args.reload:
    os.makedirs(f'{save_dir}/{fndir}', exist_ok=args.force)

fn_log = f'{save_dir}/{fndir}/log.txt'
fn_datalog = f'{save_dir}/{fndir}/log.jsons'

def log(*a, **k):
    print(*a, **k)
    with open(fn_log, 'a') as f:
        print(*a, **k, file=f)

def datalog(table, **k):
    d = {}
    d['table'] = table
    d['id'] = RUN_ID
    d.update(k)
    line = json.dumps(d)
    with open(fn_datalog, 'a') as f:
        print(line, file=f)

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

buildkey, key = jax.random.split(jax.random.PRNGKey(args.seed), 2)
datalog('args', **{ k:v for k, v in vars(args).items() })

#train = shd.SHD.load('train', limit=None)

if args.nhidden > 200:
    ifactor = 50
    rfactor = 1
elif args.nhidden > 50:
    ifactor = 10
    rfactor = 1
else:
    ifactor = 3
    rfactor = 1
if args.ifactor is not None:
    ifactor = args.ifactor
if args.rfactor is not None:
    rfactor = args.rfactor

params = networks.HyperParameters(
        netspec=args.net,
        ninput=700, #//args.downsample,
        nhidden=args.nhidden,
        ifactor=400 * ifactor,
        rfactor=35 * rfactor,
        noutput=20,
        pos_sigma=args.possigma,
        delay_sigma=args.delaysigma,
        delay_mu=args.delaymu,
        )

datalog('hyperparams', **params.configdict())


tau_mem = 10.
@functools.partial(jax.jit, static_argnames=['aux'])
def loss(net, in_spikes, label, aux=True):
    ws, v, f = net.sim(in_spikes, tau_mem=tau_mem, dt=args.dt, **sim_kwargs)
    logits = ws
    f = f * args.dt
    l = optax.softmax_cross_entropy_with_integer_labels(logits, label)
    if args.population_freq:
        l = l + ((f.mean() - args.tgtfreq / 1e3) ** 2) * f.shape[0]
    else:
        l = l + ((f - args.tgtfreq / 1e3) ** 2).sum()
    if aux:
        return l, logits.argmax() == label
    else:
        return l

def batched_loss(net, in_spikes, labels):
    ls, ncorrect = jax.vmap(functools.partial(loss, aux=True), in_axes=(None, 0, 0))(net, jnp.array(in_spikes), jnp.array(labels))
    return ls.mean(), ncorrect.sum()

@jax.jit
def performance(net, in_spikes, labels):
    @jax.jit
    def get_logits(x):
        ws, v, f = net.sim(x, tau_mem=tau_mem, dt=args.dt, **sim_kwargs)
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
batched_loss_and_grad = jax.jit(jax.value_and_grad(batched_loss, argnums=0, has_aux=True))

@jax.jit
def batched_update(opt_state, net, in_spikes, label):
    log('compiling')
    (l, ncorrect), g = batched_loss_and_grad(net, in_spikes, label)
    updates, opt_state = optimizer.update(g, opt_state, net)
    net = optax.apply_updates(net, updates)
    return opt_state, net, l, ncorrect, jax.tree.map(lambda x: x.ptp(), g)

# idx = 0
# inp = train.indicator(idx=idx, pad=True)
# lbl = train.labels[idx]
# l, (lg, s) = loss(net, inp, lbl)

clip_factor = 2.0
weight_decay = 1e-5
warmup_steps = 500

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

def make_mask(check=lambda key: False):
    def f(path, up): return check(path[-1].name)
    return lambda updates: jax.tree_util.tree_map_with_path(f, updates)

optimizer = optax.chain(
    optax.adaptive_grad_clip(clipping=clip_factor, eps=0.001),
    optax.add_decayed_weights(weight_decay, mask=make_mask(
        lambda x: x not in ('ipos', 'rpos', 'idelay', 'rdelay', 'eps'))),
    optax.add_decayed_weights(weight_decay/args.delaygradscale, mask=make_mask(
        lambda x: x in ('ipos', 'rpos', 'idelay', 'rdelay'))),
    optax.scale_by_adam(),
    optax.scale_by_schedule(schedule),
    scale_custom(args.delaygradscale, lambda x: x in ('ipos', 'rpos')),
    scale_custom(args.delaygradscale, lambda x: x in ('idelay', 'rdelay')),
    optax.scale(-1.0)
)

net = params.build(buildkey)
if args.reload:
    fns = sorted([x for x  in glob.glob(f'{save_dir}/{fndir}/*.npz') if not 'read' in x])[::-1]
    try:
        net = net.load(fns[0])
        done = int(fns[0].split('/')[-1].split('.')[0])
    except:
        net = net.load(fns[1])
        done = int(fns[0].split('/')[-1].split('.')[0])

opt_state = optimizer.init(net)

train = shd.SHD.load('train', limit=args.load_limit, skip=args.skip)
test  = shd.SHD.load('test', limit=args.load_limit_test, skip=args.skip)
inp_test, lbl_test   = test.indicators_labels32(idxs=jnp.arange(test.size), dt=args.dt)
inp_train, lbl_train = train.indicators_labels32(idxs=jnp.arange(train.size), dt=args.dt)

batch_size = args.batch_size
for epoch_idx in range(args.nepochs):
    print('Epoch', epoch_idx)
    if args.vplot:
        _, v, _f = net.sim(inp_test[0], tau_mem=tau_mem, dt=args.dt, **sim_kwargs)
        for i in range(v.shape[1]):
            plt.plot(v[:,i]+i)
        plt.show()
    key, nxt = jax.random.split(key)
    epoch_perm = jax.random.permutation(nxt, inp_train.shape[0])
    ncorrect_total = 0
    for batch_idx in (bar := tqdm.tqdm(range(0, batch_size*int(inp_train.shape[0]/batch_size), batch_size))):
        batch_idxs = epoch_perm[batch_idx:batch_idx+batch_size]
        inp = inp_train[batch_idxs]
        lbl = lbl_train[batch_idxs]
        net_old = net
        opt_state, net, l, ncorrect_batch, g = batched_update(opt_state, net, inp, lbl)
        d = jax.tree.map(lambda a, b: jnp.mean(jnp.abs(a - b)), net_old, net)
        x = jax.tree.flatten_with_path(d)
        for path, a in jax.tree.flatten_with_path(d)[0]:
            print('.'.join(x.name for x in path).ljust(20), float(a))
        ncorrect_total += ncorrect_batch
        accuracy = ncorrect_total / (batch_idx + batch_size)
        bar.set_postfix_str(f'{accuracy*100:.1f}% {l:.2f}')
        datalog('batch', epoch=epoch_idx, i=batch_idx, l=float(l), ncorrect=float(ncorrect_batch))
    t1p, t3p, f = performance_split(net, inp_test, lbl_test)
    print(f'#TEST# TOP1 {t1p:.1f}%  TOP3 {t3p:.1f}%  FREQ {f.mean():.2e}')
    t1p_train, t3p_train, ft = performance_split(net, inp_train[:500], lbl_train[:500])
    print(f'#TRAIN# TOP1 {t1p_train:.1f}%  TOP3 {t3p_train:.1f}%  FREQ {ft.mean():.2e}')
    datalog('epoch',
            i=epoch_idx,
            t1p=float(t1p),
            t3p=float(t3p),
            t1p_train=float(t1p_train),
            t3p_train=float(t3p_train)
            )
    try:
        net.save(f'{save_dir}/{fndir}/epoch_{epoch_idx:08d}')
    except Exception as ex:
        datalog('error', ex=str(type(ex)), r=repr(ex))
