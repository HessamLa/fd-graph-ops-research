#!/bin/env python3
"""bench_drop_strategies.py -- is it faster to NOT compute the dropped rows?

The question
============
`fodined/embedding/drop.py` zeroes about half of the rows of `dZ` after the
kernel computed all of them. Half of the arithmetic goes to the waste bin.
The question of this experiment is simple:

    If the engine does not compute the dropped rows, is it faster, and does
    it use less memory?

The answer is not obvious, and the reason is the design of the engine. The
SELL-C-sigma plan has FIXED shapes, and `jax.jit` compiles one kernel for
those shapes. A different set of rows in each epoch gives a different shape,
thus a recompile in each epoch. A subset with a fixed shape needs a GATHER
instead, and a gather moves the same data that the arithmetic reads. This
kernel is limited by memory bandwidth, thus less arithmetic does not
automatically give less time.

The variants
============
Each variant is a different version of the `_step` kernel of fodined. All
of them keep the same force law and the same plan.

  none            No drop. The reference for the cost of a full step.
  rows_after      The CURRENT fodined. Compute every row, then zero about
                  half of the rows of dZ. This is the baseline.
  cells_after     The other strategy of `drop_steady_rate`. It zeroes single
                  cells, thus it changes the DIRECTION of a step. It is a
                  different regularizer, and not a finer row drop.
  mask_in_kernel  Give the keep mask to the kernel, and multiply before the
                  scatter-add. This removes the WRITE of a dropped row, but
                  not its arithmetic. It must give the same numbers as
                  `rows_after`, thus it also tests this harness.
  row_subset      A true subset. In each epoch it takes a fixed fraction of
                  the row slots of every batch with `lax.top_k` on random
                  values, thus the shape stays fixed and one compile is
                  sufficient. This removes the arithmetic AND the memory
                  traffic of the dropped rows, and it pays a gather.
  batch_subset    It drops whole BATCHES of the plan, and not single rows.
                  The gather is over complete batches, thus it is much
                  cheaper. The plan sorts the rows by WIDTH, thus this drops
                  nodes of a similar degree together: the drop is
                  correlated, and it is not equal to a random row drop. It
                  is here because it is the only variant that can be much
                  faster.

Method
======
Each (graph, seed, variant) runs in its OWN process. JAX holds the memory of
a device for the life of a process, thus a shared process gives the peak of
all the variants together, and not of one. The child reports
`peak_bytes_in_use` of the device, the time of the compile, the time of the
epochs, and the quality of the embedding.

`XLA_PYTHON_CLIENT_PREALLOCATE=false` is necessary. The default of JAX takes
75% of the device at the start, thus every measurement of the memory gives
the same number.

Quality
=======
Speed without quality is not a result. Each run also reports the AUC of the
same link-prediction task as `fodined/modular.py`, and the final ||dZ||. A
variant that is fast and gives a bad embedding is not an improvement.

Run from the repo root (fdmap/). `REPORT.md` beside this file holds the
results and the conclusion.

    .venv/bin/python experiments/drop_strategies/bench_drop_strategies.py
    .venv/bin/python experiments/drop_strategies/bench_drop_strategies.py \
        --graphs cora --epochs 200
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

# Must happen BEFORE `import jax`: the default preallocation of 75% of the
# device makes every memory measurement identical and useless.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))     # experiments/drop_strategies
ROOT = os.path.dirname(os.path.dirname(HERE))         # the repo root, fdmap/
sys.path.insert(0, ROOT)
CACHE = os.path.join(HERE, "_cache")                  # the augmented D, per seed

GRAPHS = ("cora", "pubmed")
VARIANTS = ("none", "rows_after", "cells_after", "mask_in_kernel",
            "row_subset", "batch_subset")

# The hyperparameters of `fodined/modular.py`, so the numbers are comparable.
K1, K2, K3, K4 = 0.999, 1.0, 10.0, 0.01
B_CELLS, K_MAX, LADDER_BASE = 16_384, 256, 1.5
H_SHIFT, N_DIM, DROP_RATE = 1.0, 128, 0.5


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------
def load_graph(name):
    """Returns a symmetric CSR of 1.0, and the node count."""
    if name == "cora":
        path = os.path.join(ROOT, "data_cache/cora/cora.cites")
        raw = np.loadtxt(path, dtype=np.int64)
    elif name == "pubmed":
        # `<edge_id>\tpaper:<src>\t|\tpaper:<dst>`, after two header lines.
        path = os.path.join(
            ROOT, "data_cache/pubmed/Pubmed-Diabetes/data/"
                  "Pubmed-Diabetes.DIRECTED.cites.tab")
        src, dst = [], []
        with open(path) as f:
            f.readline(), f.readline()
            for line in f:
                p = line.split("\t")
                src.append(int(p[1].split(":")[1]))
                dst.append(int(p[3].split(":")[1]))
        raw = np.column_stack([src, dst]).astype(np.int64)
    else:
        raise ValueError(f"unknown graph {name!r}")

    _, flat = np.unique(raw.ravel(), return_inverse=True)
    e = flat.reshape(raw.shape)
    e = e[e[:, 0] != e[:, 1]]
    n = int(e.max()) + 1
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    A = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    A.sort_indices()
    return A, n


def augment(name, A, n, seed):
    """The augmentation policy of `modular.py`, with a cache on disk.

    The hop distances come from Pruned Landmark Labeling, and not from
    `scipy.sparse.csgraph.shortest_path`. PubMed has 19,717 nodes, and the
    sample touches nearly all of them, thus the dense matrix of SciPy would
    be 19717 x 19717 x 8 = 3.1 GB. `experiments/bench_shortest_path_gemsec.py`
    measured PLL as the best exact method at this size.
    """
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"{name}_aug_seed{seed}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                             shape=tuple(z["shape"]))

    n_new = int(n * np.log10(n))
    rng = np.random.default_rng(seed)
    Ac = A.tocoo()
    keys = np.sort(Ac.row.astype(np.int64) * n + Ac.col.astype(np.int64))

    u_all = np.empty(0, np.int64)
    v_all = np.empty(0, np.int64)
    while u_all.size < n_new:
        draw = (n_new - u_all.size) * 2 + 1024
        u, v = rng.integers(0, n, draw), rng.integers(0, n, draw)
        ok = u != v
        u, v = u[ok], v[ok]
        k = u * n + v
        pos = np.searchsorted(keys, k)
        pos[pos >= keys.size] = 0
        keep = keys[pos] != k
        u, v = u[keep], v[keep]
        room = n_new - u_all.size
        u_all = np.concatenate([u_all, u[:room]])
        v_all = np.concatenate([v_all, v[:room]])
    pairs = np.column_stack([u_all, v_all])

    import networkit as nk
    up = sp.triu(A, k=1).tocoo()
    g = nk.GraphFromCoo(
        (np.ascontiguousarray(up.row, dtype=np.uint64),
         np.ascontiguousarray(up.col, dtype=np.uint64)),
        n=n, weighted=False, directed=False)
    pll = nk.distance.PrunedLandmarkLabeling(g)
    pll.run()
    q = np.fromiter((pll.query(int(a), int(b)) for a, b in pairs),
                    dtype=np.uint64, count=pairs.shape[0])
    w = q.astype(np.float64)
    w[q == np.uint64(2 ** 64 - 1)] = float(n)          # unreachable sentinel

    D = sp.csr_matrix(
        (np.concatenate([Ac.data, w, w]),
         (np.concatenate([Ac.row, pairs[:, 0], pairs[:, 1]]),
          np.concatenate([Ac.col, pairs[:, 1], pairs[:, 0]]))),
        shape=(n, n))
    np.savez(cache, data=D.data, indices=D.indices, indptr=D.indptr,
             shape=np.array(D.shape))
    return D


# ---------------------------------------------------------------------------
# The kernels. One for each variant.
# ---------------------------------------------------------------------------
def make_kernels():
    """Builds the step functions. Imported late, so the parent stays small."""
    import jax
    import jax.numpy as jnp
    from fodined.embedding.shell_force import shell_force

    def body_factory(Z, inv_deg_ext, params, n, keep_rows=None):
        """The body of one batch. It is the kernel of fodined, and the only
        addition is the optional `keep_rows` mask."""
        def body(dZ, batch):
            rows, nbrs, planes = batch[0], batch[1], batch[2:]
            Zc = Z[jnp.minimum(rows, n - 1)]
            Zj = Z[nbrs]
            diff = Zj - Zc[:, None, :]
            x = jnp.sqrt(jnp.sum(diff * diff, axis=-1))
            x_safe = jnp.where(x == 0, 1.0, x)
            F_mag = shell_force(x, planes, params)
            scale = jnp.where(x == 0, 0.0, F_mag / x_safe)
            F = jnp.sum(diff * scale[..., None], axis=1)
            F = F * inv_deg_ext[rows][:, None]
            if keep_rows is not None:
                # The mask is applied BEFORE the scatter-add, thus a dropped
                # row is never written. The arithmetic above still ran.
                F = F * keep_rows[rows][:, None]
            dZ = dZ.at[rows].add(F, mode="drop")
            return dZ, None
        return body

    def step_plain(Z, plan, inv_deg_ext, params, n):
        dZ = jnp.zeros_like(Z)
        for rung in plan:
            dZ, _ = jax.lax.scan(
                body_factory(Z, inv_deg_ext, params, n), dZ, rung)
        return dZ

    def step_masked(Z, plan, inv_deg_ext, params, n, key):
        """`mask_in_kernel`: no write for a dropped row."""
        keep = (jax.random.uniform(key, (n + 1,)) >= DROP_RATE).astype(Z.dtype)
        dZ = jnp.zeros_like(Z)
        for rung in plan:
            dZ, _ = jax.lax.scan(
                body_factory(Z, inv_deg_ext, params, n, keep), dZ, rung)
        return dZ

    def step_row_subset(Z, plan, inv_deg_ext, params, n, key):
        """`row_subset`: a true subset of the row slots of every batch.

        `lax.top_k` on random values gives a subset WITHOUT repetition. A
        repetition would add the force of one row two times to `dZ`.

        Two differences from a row drop, and both change the physics a
        little:
         * It drops a fixed COUNT in each batch, and not each row with an
           independent probability. The rate is thus exact, and not steady.
         * A hub row that the plan divided into more than one virtual row
           can lose one part and keep another. That node then moves with a
           PART of its force. A row drop gives all or nothing.
        """
        dZ = jnp.zeros_like(Z)
        for i, rung in enumerate(plan):
            nb, R = rung[0].shape
            keep_r = max(1, int(round(R * (1.0 - DROP_RATE))))
            key, sub = jax.random.split(key)
            _, idx = jax.lax.top_k(jax.random.uniform(sub, (nb, R)), keep_r)
            small = (jnp.take_along_axis(rung[0], idx, axis=1),) + tuple(
                jnp.take_along_axis(a, idx[:, :, None], axis=1)
                for a in rung[1:])
            dZ, _ = jax.lax.scan(
                body_factory(Z, inv_deg_ext, params, n), dZ, small)
        return dZ

    def step_batch_subset(Z, plan, inv_deg_ext, params, n, key):
        """`batch_subset`: whole batches are dropped, and not single rows."""
        dZ = jnp.zeros_like(Z)
        for rung in plan:
            nb = rung[0].shape[0]
            keep_b = max(1, int(round(nb * (1.0 - DROP_RATE))))
            key, sub = jax.random.split(key)
            _, idx = jax.lax.top_k(jax.random.uniform(sub, (nb,)), keep_b)
            small = tuple(jnp.take(a, idx, axis=0) for a in rung)
            dZ, _ = jax.lax.scan(
                body_factory(Z, inv_deg_ext, params, n), dZ, small)
        return dZ

    return step_plain, step_masked, step_row_subset, step_batch_subset


# ---------------------------------------------------------------------------
# Child: one (graph, seed, variant)
# ---------------------------------------------------------------------------
def child_main(graph, seed, variant, epochs, out_json):
    import functools
    import jax
    import jax.numpy as jnp
    from fodined.embedding.shell_force import (
        shell_force, shell_coeff_data, degrees_from_D)
    from fodined.embedding.sell_c_sigma import make_plan
    from fodined.embedding.drop import drop_steady_rate

    seed = int(seed)
    epochs = int(epochs)
    rec = {"graph": graph, "seed": seed, "variant": variant, "ok": False}
    try:
        A, n = load_graph(graph)
        D = augment(graph, A, n, seed)

        plan_np, inv_deg_np, stats = make_plan(
            D, (shell_coeff_data(D), D.data), degrees=degrees_from_D(D),
            b_cells=B_CELLS, k_max=K_MAX, ladder_base=LADDER_BASE)
        plan = jax.tree_util.tree_map(jax.device_put, plan_np)
        inv_deg = jax.device_put(inv_deg_np)
        params = dict(k1=K1, k2=K2, k3=K3, k4=K4, h_shift=H_SHIFT)

        step_plain, step_masked, step_row, step_batch = make_kernels()
        key = jax.random.PRNGKey(seed)
        key, sub = jax.random.split(key)
        Z = jax.random.normal(sub, (n, N_DIM), dtype=jnp.float32)

        # Every variant becomes one jitted callable `(Z, key) -> dZ`, thus
        # the epoch loop below is identical for all of them.
        if variant in ("none", "rows_after", "cells_after"):
            f = jax.jit(functools.partial(step_plain, plan=plan,
                                          inv_deg_ext=inv_deg, params=params,
                                          n=n))
            strat = {"rows_after": "random_rows",
                     "cells_after": "random_cells"}.get(variant)
            rate = 0.0 if variant == "none" else DROP_RATE

            def run(Z, k):
                dZ = f(Z)
                return dZ if rate == 0.0 else drop_steady_rate(
                    dZ, k, rate, strategy=strat)
        else:
            g = {"mask_in_kernel": step_masked, "row_subset": step_row,
                 "batch_subset": step_batch}[variant]
            f = jax.jit(functools.partial(g, plan=plan, inv_deg_ext=inv_deg,
                                          params=params, n=n))

            def run(Z, k):
                return f(Z, key=k)

        dev = jax.local_devices()[0]
        t = time.perf_counter()
        key, sub = jax.random.split(key)
        run(Z, sub).block_until_ready()
        rec["compile_s"] = time.perf_counter() - t

        t = time.perf_counter()
        for _ in range(epochs):
            key, sub = jax.random.split(key)
            dZ = run(Z, sub)
            Z = Z + dZ
        Z.block_until_ready()
        rec["epochs_s"] = time.perf_counter() - t
        rec["epochs_per_s"] = epochs / rec["epochs_s"]
        rec["final_dZ"] = float(jnp.linalg.norm(dZ, axis=-1).mean())

        # How much did the variant REALLY drop? A variant can be fast
        # because it does nothing. `batch_subset` on a small graph is
        # exactly that: every rung holds one batch, thus
        # `max(1, round(1 * 0.5))` keeps that batch, and nothing is
        # dropped. Without this number the table below rewards a no-op with
        # the best speed. An isolated node also gives a zero row, thus this
        # is an upper bound of the drop, and not an exact count.
        dZh = np.asarray(dZ)
        rec["zero_rows_frac"] = float(np.mean(np.all(dZh == 0.0, axis=1)))
        rec["zero_cells_frac"] = float(np.mean(dZh == 0.0))
        rec["plan_batches"] = int(sum(r[0].shape[0] for r in plan_np))
        rec["plan_rungs"] = len(plan_np)
        try:
            rec["peak_gpu_mb"] = dev.memory_stats()["peak_bytes_in_use"] / 1e6
        except Exception:                                  # noqa: BLE001
            rec["peak_gpu_mb"] = None

        Zh = np.asarray(Z)
        rec["finite"] = bool(np.isfinite(Zh).all())
        rec["auc"] = link_prediction_auc(A, n, Zh, seed)
        rec["cells"] = stats["cells"]
        rec["ok"] = True
    except Exception as exc:                               # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
    with open(out_json, "w") as f:
        json.dump(rec, f)


def link_prediction_auc(A, n, Z, seed):
    """The same task as `fodined/modular.py`: does the embedding still work?"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    pos = np.column_stack(sp.triu(A, k=1).nonzero())
    Ac = A.tocoo()
    keys = np.sort(Ac.row.astype(np.int64) * n + Ac.col.astype(np.int64))
    neg_u = np.empty(0, np.int64)
    neg_v = np.empty(0, np.int64)
    while neg_u.size < pos.shape[0]:
        draw = (pos.shape[0] - neg_u.size) * 2 + 1024
        u, v = rng.integers(0, n, draw), rng.integers(0, n, draw)
        ok = u != v
        u, v = u[ok], v[ok]
        k = u * n + v
        p = np.searchsorted(keys, k)
        p[p >= keys.size] = 0
        keep = keys[p] != k
        room = pos.shape[0] - neg_u.size
        neg_u = np.concatenate([neg_u, u[keep][:room]])
        neg_v = np.concatenate([neg_v, v[keep][:room]])

    pr = np.vstack([pos, np.column_stack([neg_u, neg_v])])
    X = Z[pr[:, 0]] * Z[pr[:, 1]]
    y = np.concatenate([np.ones(pos.shape[0]), np.zeros(neg_u.size)])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", nargs=5, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--graphs", default=",".join(GRAPHS))
    ap.add_argument("--seeds", default="42,56,88")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--timeout", type=float, default=1800)
    args = ap.parse_args()

    if args.child:
        g, s, v, e, oj = args.child
        child_main(g, s, v, e, oj)
        return

    graphs = [g.strip() for g in args.graphs.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    print(f"[drop-bench] graphs={graphs} seeds={seeds} epochs={args.epochs}\n"
          f"             variants={variants}", flush=True)

    rows = []
    with tempfile.TemporaryDirectory(prefix="dropbench_") as tmp:
        for g in graphs:
            print(f"\n=== {g} ===", flush=True)
            print(f"  {'variant':>15s} {'seed':>5s} {'ep/s':>9s} "
                  f"{'compile':>8s} {'peakGPU':>9s} {'0rows':>6s} "
                  f"{'0cells':>7s} {'final|dZ|':>10s} {'AUC':>7s}", flush=True)
            for v in variants:
                for s in seeds:
                    oj = os.path.join(tmp, f"{g}_{v}_{s}.json")
                    cmd = [sys.executable, os.path.abspath(__file__),
                           "--child", g, str(s), v, str(args.epochs), oj]
                    try:
                        p = subprocess.run(cmd, timeout=args.timeout,
                                           capture_output=True, text=True)
                    except subprocess.TimeoutExpired:
                        print(f"  {v:>15s} {s:>5d}  TIMEOUT", flush=True)
                        continue
                    if not os.path.exists(oj):
                        tail = (p.stderr or "").strip().splitlines()[-1:] or ["?"]
                        print(f"  {v:>15s} {s:>5d}  CRASH: {tail[0][:70]}",
                              flush=True)
                        continue
                    r = json.load(open(oj))
                    if not r["ok"]:
                        print(f"  {v:>15s} {s:>5d}  FAILED: {r['error'][:70]}",
                              flush=True)
                        continue
                    rows.append(r)
                    print(f"  {v:>15s} {s:>5d} {r['epochs_per_s']:>9.1f} "
                          f"{r['compile_s']:>7.1f}s "
                          f"{(r['peak_gpu_mb'] or 0):>8.0f}M "
                          f"{r['zero_rows_frac']:>6.0%} "
                          f"{r['zero_cells_frac']:>7.0%} "
                          f"{r['final_dZ']:>10.4f} {r['auc']:>7.4f}",
                          flush=True)

    # ---- mean over the seeds ---------------------------------------------
    print("\n=== mean over seeds (speedup is against rows_after) ===",
          flush=True)
    print(f"{'graph':>8s} {'variant':>15s} {'ep/s':>9s} {'speedup':>8s} "
          f"{'peakGPU':>9s} {'0rows':>6s} {'AUC':>8s}", flush=True)
    for g in graphs:
        base = [r["epochs_per_s"] for r in rows
                if r["graph"] == g and r["variant"] == "rows_after"]
        base = float(np.mean(base)) if base else float("nan")
        for v in variants:
            sel = [r for r in rows if r["graph"] == g and r["variant"] == v]
            if not sel:
                continue
            eps = float(np.mean([r["epochs_per_s"] for r in sel]))
            mem = float(np.mean([r["peak_gpu_mb"] or 0 for r in sel]))
            auc = float(np.mean([r["auc"] for r in sel]))
            aucsd = float(np.std([r["auc"] for r in sel]))
            zr = float(np.mean([r["zero_rows_frac"] for r in sel]))
            print(f"{g:>8s} {v:>15s} {eps:>9.1f} {eps/base:>7.2f}x "
                  f"{mem:>8.0f}M {zr:>6.0%} {auc:>6.4f}+-{aucsd:<.3f}",
                  flush=True)
        nb = [r["plan_batches"] for r in rows if r["graph"] == g]
        if nb:
            print(f"{'':>8s} (plan has {nb[0]} batches in total; "
                  f"`batch_subset` cannot drop anything when a rung holds "
                  f"one batch)", flush=True)


if __name__ == "__main__":
    main()
