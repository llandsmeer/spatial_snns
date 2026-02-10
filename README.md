# Spatial SNNs

![Spatial SNNs embed neurons and delays into n-dimensional space](./intro.png?raw=true)

## Installation

```
git clone https://github.com/llandsmeer/spatial_snns/
pip install 'jax[cuda]' tqdm matplotlib h5py numpy optax pandas
```

For efficient execution you will need a GPU with a large amount of memory.
In experiments, RTX PRO 6000 or H100 for larger networks, Quadro 600 for nhidden < 150, and A2000 for small networks. CPU might work for very small networks.

## SHD, Rate Coded

The SHD dataset must be downloaded from https://zenkelab.org/datasets/

To run the training script, please run `python3 train.py`.

Some examples:

 - Train unconstrained network:

    `python3 train.py --net inf --dt 0.5 --nhidden 100 --batch_size 32`

 - Interactive visualization:

    `mkdir -p /tmp/figs/`

    `python3 train.py --net 2 --circle --dt 0.5 --skip --nhidden 100 --lr 1e-3 --vplot`

To run many experiments, ie on a cluster, generate the submit script via `multirun.py`. It provides from functions for creating grid searches over parameters and seeds, and outputs either a SLURM script of a list of command lines to be invoked sequentially on a single-GPU machine.

## YY, TTFS

....

## SHD train.py help

```

usage: train.py [-h] [--net NET] [--nhidden NHIDDEN] [--batch_size BATCH_SIZE]
                [--load_limit LOAD_LIMIT] [--load_limit_test LOAD_LIMIT_TEST] [--lr LR]
                [--ifactor IFACTOR] [--rfactor RFACTOR] [--force] [--skip] [--debug]
                [--dt DT] [--tmp] [--reload] [--seed SEED] [--nepochs NEPOCHS]
                [--delaygradscale DELAYGRADSCALE] [--delaymu DELAYMU]
                [--delaysigma DELAYSIGMA] [--possigma POSSIGMA] [--tgtfreq TGTFREQ]
                [--population_freq] [--tag TAG] [--line] [--circle] [--wstat] [--istat]
                [--sparse SPARSE] [--sparse_iter] [--adex] [--adex_a ADEX_A]
                [--adex_b ADEX_B] [--adex_tau ADEX_TAU] [--adex_dt ADEX_DT] [--vplot]
                [--pos] [--iadapt0 IADAPT0] [--shard]

options:
  -h, --help            show this help message and exit
  --net NET             Dimension (inf or int or <int>e<float> or <int>g<float>)
  --nhidden NHIDDEN     Number of hidden units
  --batch_size BATCH_SIZE
                        Batch size
  --load_limit LOAD_LIMIT
                        Load limit no samples
  --load_limit_test LOAD_LIMIT_TEST
                        Load limit no test samples
  --lr LR               Learning rate
  --ifactor IFACTOR     Extra ifactor multiplier
  --rfactor RFACTOR     Extra ifactor multiplier
  --force               Overwrite
  --skip                Skip metadata loading
  --debug               Start pdb on error
  --dt DT               Time step (bigger=faster, smaller=more accurate)
  --tmp                 Store in /tmp/saved
  --reload              Reload previous
  --seed SEED           Seed for network generation
  --nepochs NEPOCHS     Epochs to trian for
  --delaygradscale DELAYGRADSCALE
                        Scale delay gradients (or 0 for none)
  --delaymu DELAYMU     Mu
  --delaysigma DELAYSIGMA
                        Sigma
  --possigma POSSIGMA   Sigma
  --tgtfreq TGTFREQ     Target frequency Hz
  --population_freq     Target freq after mean
  --tag TAG             Experiment ids
  --line                Initialize inputs on a line
  --circle              Initialize inputs on a circle
  --wstat               Initialize all weights at same value
  --istat               Dont move inputs
  --sparse SPARSE       Sparsify after each epoch
  --sparse_iter         Iteratively increase
  --adex                Use AdEx model instead of LIF
  --adex_a ADEX_A       AdEx a
  --adex_b ADEX_B       AdEx b
  --adex_tau ADEX_TAU   AdEx w tau
  --adex_dt ADEX_DT     AdEx DeltaT
  --vplot               AdEx model
  --pos                 Force pos
  --iadapt0 IADAPT0     Init iadapt
  --shard               Target all devices
```

Please cite as

> Landsmeer, L. P. L., Movahedin, A., Negrello, M., Hamdioui, S., & Strydis, C. (2025). Spatial Spiking Neural Networks Enable Efficient and Robust Temporal Computation. arXiv preprint arXiv:2512.10011.

This work is partially supported by the European-Union Horizon Europe R&I program through projects SEPTON (no. 101094901) and SECURED (no. 101095717) and through the NWO - Gravitation Programme DBI2 (no. 024.005.022). This work used the Dutch national e-infrastructure with the support of the SURF Cooperative using grant no. EINF-10677, EINF-15815, and EINF-16791. The RTX PRO 6000 and Quadro Pro 6000 GPUs used for this research were donated by the NVIDIA Corporation.
