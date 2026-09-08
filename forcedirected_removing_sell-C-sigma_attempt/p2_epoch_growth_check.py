#!/bin/env python3
"""p2_epoch_growth_check.py -- is the P2 divergence reduction-order drift
or a real defect?

Builds ONE `D` (the P2 config: nbr_walk, min_gap, far_bias=0.75,
far_weight=100, fdlinear, k4=0.01, kr=1.0) with `fodiwalk.Fodiwalk`, then
runs the OLD (frozen `forcedirected_old.sell_c_sigma`) kernel and the NEW
(`forcedirected.flat_batch`) kernel against that SAME `D`, same planes,
same degrees, same params, same update rule, for a few epoch counts. If
the gap between the two GROWS with epoch count, the flat kernel's tiny
per-epoch reduction-order noise is being amplified by the dynamics
(expected of a chaotic-ish iterative force system) and not a defect of the
kernel itself. If the gap is already large at low epoch counts and does
not track epoch count, that would point to a real defect instead.
"""
from __future__ import annotations

import os
import sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import functools
import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp

from fodiwalk import Fodiwalk
from fodiwalk.make_graph import load
from forcedirected_old import sell_c_sigma as old_kernel
from forcedirected import flat_batch as new_kernel

GRID = dict(weight="min_gap", far_bias=0.75, far_weight=100.0)
FDL = dict(force="fdlinear", k4=1.0, kr=1.0)

A, n = load("cora")
fw = Fodiwalk(n_dim=64, seed=42, lr=0.999, optim="plain",
              pairs="nbr_walk", **GRID, **FDL)
D = fw.augment_graph(A)
# `nbr_walk` gives a `RowCSR` (no scipy base class); the OLD kernel's own
# `to_csr` cannot coerce it (it only knows `sp.issparse` or dense). Rebuild
# a real `scipy.sparse.csr_matrix` from the same raw arrays -- exactly what
# the OLD `embed/planner.py` did per chunk before handing a chunk to
# `make_plan`. This is a comparison SCRIPT, not a change to `forcedirected_old`.
D = sp.csr_matrix((D.data, D.indices, D.indptr), shape=D.shape)
planes = fw._build_planes(D)
degrees = fw._build_degrees(D, A)
params = fw.params
force_fn = fw.force_fn
print(f"D.nnz={D.nnz}, n={n}")


def run_old(epochs, lr=0.999, seed=42):
    plan_np, inv_ext_np, stats = old_kernel.make_plan(D, planes, degrees=degrees)
    plan = jax.tree_util.tree_map(jax.device_put, plan_np)
    inv_ext = jax.device_put(inv_ext_np)
    kernel = jax.jit(functools.partial(old_kernel.step, n=n, force_fn=force_fn))
    key = jax.random.PRNGKey(seed)
    Z = jax.random.normal(key, (n, 64), dtype=jnp.float32)
    for _ in range(epochs):
        dZ = kernel(Z, plan, inv_ext, params)
        Z = Z + lr * dZ
    dz = float(jnp.linalg.norm(dZ, axis=-1).mean())
    return dz, np.asarray(Z, dtype=np.float64)


def run_new(epochs, lr=0.999, seed=42):
    inv_deg = np.where(degrees == 0, 0.0, 1.0 / np.maximum(degrees, 1.0)).astype(np.float32)
    inv_deg = jax.device_put(inv_deg)
    u_glob, u_loc, v, tiles, R, m_real = new_kernel.build_batch(D, planes, 0, n)
    u_glob, u_loc, v, tiles = jax.tree_util.tree_map(
        jax.device_put, (u_glob, u_loc, v, tiles))
    kernel = new_kernel.make_step(force_fn, R)
    key = jax.random.PRNGKey(seed)
    Z = jax.random.normal(key, (n, 64), dtype=jnp.float32)
    for _ in range(epochs):
        dZ = kernel(Z, u_glob, u_loc, v, tiles, inv_deg, params)
        Z = Z + lr * dZ
    dz = float(jnp.linalg.norm(dZ, axis=-1).mean())
    return dz, np.asarray(Z, dtype=np.float64)


for epochs in (50, 200, 500, 1000, 2000):
    dz_old, Z_old = run_old(epochs)
    dz_new, Z_new = run_new(epochs)
    rel_Z = np.linalg.norm(Z_new - Z_old) / np.linalg.norm(Z_old)
    print(f"epochs={epochs:5d}  dz_old={dz_old:.6f}  dz_new={dz_new:.6f}  "
          f"rel_dz={abs(dz_new - dz_old) / abs(dz_old):.3e}  rel_Z={rel_Z:.3e}")
