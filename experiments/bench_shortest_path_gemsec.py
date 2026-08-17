#!/bin/env python3
"""bench_shortest_path_gemsec.py -- which shortest-path method can feed the
force-directed pipeline at scale?

The pipeline is: build a graph, augment the graph, then embed the nodes with
the SELL-C-sigma engine (`fodined/`). This benchmark examines the AUGMENT
step, because that step became the memory wall.

The augmentation samples node pairs that have no edge. Then it gives each new
edge a weight. The weight is the hop distance between the two nodes in the
original graph. Thus the augmentation must know the shortest-path distance of
many arbitrary pairs. The first implementation asked SciPy for one distance
matrix of all the sources. That matrix is DENSE: `len(sources) x n` float64.
The sample touches almost all the nodes, thus the matrix is the full `(n, n)`
matrix. This is 20 GB for the largest graph here, and it gives only 100k
values that the pipeline keeps.

This script measures five methods that get the same values. It reports the
time, the CPU time, the peak memory, and the accuracy of each one. Then it
profiles the winner with `cProfile`, and it profiles the augmented graph that
the winner produced: the size, the hop distribution, and the SELL-C-sigma
batch plan that the embedding engine builds from it.

The dataset is GEMSEC Facebook (https://snap.stanford.edu/data/gemsec-Facebook.html).
The 8 graphs are mutual-like networks of verified Facebook pages. They have
3.9k to 50.5k nodes, and they have a power-law degree distribution. This is
the graph class that the SELL-C-sigma engine is for.

Run from the repo root (fdmap/):

    .venv/bin/python bench_shortest_path_gemsec.py                  # all 8 graphs
    .venv/bin/python bench_shortest_path_gemsec.py --datasets tvshow,politician
    .venv/bin/python bench_shortest_path_gemsec.py --pairs 20000 --epochs 0

Options:
    --datasets   comma-separated graph names (default: all, small to large)
    --pairs      number of new edges to sample per graph (default: 50000)
    --methods    comma-separated method names (default: all)
    --budget     seconds. A SciPy method is skipped if the probe projects a
                 longer time than this. (default: 300)
    --max-dense  bytes. The one-shot dense method is skipped above this
                 projected size. (default: 2e9)
    --epochs     embedding epochs for the output profile. 0 turns it off.
                 (default: 20)
    --profile    dataset to profile with cProfile (default: the largest one
                 that completed)
"""
from __future__ import annotations

import argparse
import cProfile
import gc
import io
import os
import pstats
import sys
import threading
import time
import tracemalloc

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

# This file is in `experiments/`, but it imports `fodined` from the repo root
# and it reads `data_cache/` there. Python puts only the directory of the
# script on sys.path, thus the root goes on sys.path here, and the data path
# is absolute. The script then runs from any working directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

DATA_DIR = os.path.join(_ROOT, "data_cache", "gemsec_facebook_dataset")

# Small to large. The order matters: the script prints the result of each
# graph when it has it, thus a long run still gives the small graphs first.
DATASETS = ["tvshow", "politician", "government", "public_figure",
            "athletes", "company", "new_sites", "artist"]

UNREACHABLE = np.inf     # internal marker. The pipeline maps it to `n` later.


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------
def _rss() -> int:
    """Resident set size of this process, in bytes."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) * 1024
    return 0


class PeakRSS(threading.Thread):
    """Samples the resident set size in the background.

    `tracemalloc` sees only the allocations of Python. NetworKit and SciPy
    allocate in C++. Thus this thread samples `/proc/self/status`, and it
    gives the peak that includes the native allocations.
    """

    def __init__(self, interval: float = 0.005):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        # NOT `_stop`: `threading.Thread` already has a private `_stop()`
        # method, and the thread machinery calls it when the thread ends.
        # An attribute of that name shadows it, and the thread then fails
        # with "'Event' object is not callable".
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            self.peak = max(self.peak, _rss())
            time.sleep(self.interval)

    def stop(self) -> int:
        self._halt.set()
        self.join(timeout=1.0)
        return self.peak


class Measured:
    """Context manager. It measures the wall time, the CPU time, and the memory.

    `cpu / wall` shows the parallelism. A value near 1.0 shows one thread. A
    larger value shows that the library used more than one core.
    """

    def __init__(self):
        self.wall = self.cpu = 0.0
        self.peak_rss = self.peak_py = 0

    def __enter__(self):
        gc.collect()
        self._base_rss = _rss()
        self._sampler = PeakRSS()
        self._sampler.start()
        tracemalloc.start()
        self._t0, self._c0 = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self._t0
        self.cpu = time.process_time() - self._c0
        self.peak_py = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        self.peak_rss = max(0, self._sampler.stop() - self._base_rss)
        return False


# ---------------------------------------------------------------------------
# Load the graph and sample the new pairs
# ---------------------------------------------------------------------------
def load_graph(name: str):
    """Reads one GEMSEC edge list. Returns a symmetric CSR matrix of 1.0.

    The files have a `node_1,node_2` header, and the node ids are already
    0-based and contiguous.
    """
    path = os.path.join(DATA_DIR, f"{name}_edges.csv")
    e = np.loadtxt(path, dtype=np.int64, delimiter=",", skiprows=1)
    e = e[e[:, 0] != e[:, 1]]
    n = int(e.max()) + 1
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    A = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0                    # a duplicate edge must stay weight 1.0
    A.sort_indices()
    return A, n


def sample_new_pairs(A, n: int, count: int, seed: int = 42):
    """Samples `count` node pairs that have no edge, and no duplicate pair.

    This is vectorized, and it does not build a Python set of the edges. A
    set of 1.6M tuples costs more than 150 MB on the largest graph here. The
    test uses one sorted array of the composite keys `u * n + v` instead, and
    `np.searchsorted` finds a pair in it.
    """
    rng = np.random.default_rng(seed)
    Ac = A.tocoo()
    edge_keys = np.sort(Ac.row.astype(np.int64) * n + Ac.col.astype(np.int64))

    keep_u = np.empty(0, dtype=np.int64)
    keep_v = np.empty(0, dtype=np.int64)
    taken = np.empty(0, dtype=np.int64)          # sorted keys already used
    while keep_u.size < count:
        draw = max(count - keep_u.size, 1024) * 2
        u = rng.integers(0, n, draw)
        v = rng.integers(0, n, draw)
        ok = u != v
        u, v = u[ok], v[ok]
        key = u * n + v

        pos = np.searchsorted(edge_keys, key)
        pos[pos >= edge_keys.size] = 0
        is_edge = edge_keys[pos] == key
        u, v, key = u[~is_edge], v[~is_edge], key[~is_edge]

        # No pair twice, and no pair that is the reverse of a kept pair.
        key_r = v * n + u
        pos = np.searchsorted(taken, key)
        pos[pos >= max(taken.size, 1)] = 0
        dup = taken[pos] == key if taken.size else np.zeros(key.size, bool)
        u, v, key, key_r = u[~dup], v[~dup], key[~dup], key_r[~dup]
        key, idx = np.unique(key, return_index=True)
        u, v, key_r = u[idx], v[idx], key_r[idx]

        room = count - keep_u.size
        u, v, key, key_r = u[:room], v[:room], key[:room], key_r[:room]
        keep_u = np.concatenate([keep_u, u])
        keep_v = np.concatenate([keep_v, v])
        taken = np.sort(np.concatenate([taken, key, key_r]))
    return np.column_stack([keep_u, keep_v])


# ---------------------------------------------------------------------------
# The methods under test. Each one returns the hop distance of every pair.
# ---------------------------------------------------------------------------
def m_scipy_dense(A, n, pairs, srcs, **kw):
    """One SciPy call for every source. This is the method to replace.

    `shortest_path` always returns a DENSE ndarray of `(len(srcs), n)`
    float64. The `indices` argument limits the number of ROWS only. It cannot
    make the result sparse, because the value of an unreachable pair is `inf`,
    thus there is no zero to omit.
    """
    hop = shortest_path(A, method="D", unweighted=True, indices=srcs)
    return hop[np.searchsorted(srcs, pairs[:, 0]), pairs[:, 1]]


def m_scipy_chunked(A, n, pairs, srcs, chunk=128, **kw):
    """The same SciPy call, but over blocks of sources.

    Only the needed cells of each block are kept, thus the peak memory is
    `chunk * n * 8` bytes and it does not grow with the number of pairs.
    """
    src_pos = np.searchsorted(srcs, pairs[:, 0])
    out = np.empty(pairs.shape[0])
    for s in range(0, srcs.size, chunk):
        blk = srcs[s:s + chunk]
        d = shortest_path(A, method="D", unweighted=True, indices=blk)
        m = (src_pos >= s) & (src_pos < s + blk.size)
        out[m] = d[src_pos[m] - s, pairs[m, 1]]
    return out


def m_networkit_pll(A, n, pairs, srcs, nk_graph=None, **kw):
    """Pruned Landmark Labeling (Akiba et al., SIGMOD 2013), from NetworKit.

    The index holds a small label set for each node. A query intersects two
    label sets, thus it is exact and it needs microseconds. No distance
    matrix exists at any time.

    An unreachable pair gives 2**64-1. The documentation does not state this,
    thus the code tests for it explicitly.
    """
    import networkit as nk
    pll = nk.distance.PrunedLandmarkLabeling(nk_graph)
    pll.run()
    q = np.fromiter((pll.query(int(u), int(v)) for u, v in pairs),
                    dtype=np.uint64, count=pairs.shape[0])
    out = q.astype(np.float64)
    out[q == np.uint64(2 ** 64 - 1)] = UNREACHABLE
    return out


def m_networkit_multitarget(A, n, pairs, srcs, nk_graph=None, **kw):
    """One NetworKit BFS per source, to that source's targets only.

    The output is sparse: it holds one distance for each target that was
    asked for. The loop is in Python, thus it does not use more than one core.
    """
    import networkit as nk
    order = np.argsort(pairs[:, 0], kind="stable")
    su, sv = pairs[order, 0], pairs[order, 1]
    bounds = np.searchsorted(su, srcs)
    out = np.empty(pairs.shape[0])
    for i, u in enumerate(srcs):
        lo = bounds[i]
        hi = bounds[i + 1] if i + 1 < srcs.size else su.size
        tg = sv[lo:hi].tolist()
        b = nk.distance.MultiTargetBFS(nk_graph, int(u), tg)
        b.run()
        out[order[lo:hi]] = np.asarray(b.getDistances())
    out[out > n] = UNREACHABLE
    return out


def m_landmark_approx(A, n, pairs, srcs, landmarks=64, seed=42, **kw):
    """Approximation. It gives an upper bound, not the true distance.

    It runs a BFS from `landmarks` nodes only, thus it needs
    `landmarks * n * 8` bytes. Then `d(u,v) <= d(l,u) + d(l,v)` for every
    landmark `l`, and the smallest of these sums is the estimate. This is the
    only inexact method here, thus the table reports its error.
    """
    rng = np.random.default_rng(seed)
    deg = np.diff(A.indptr)
    # High-degree nodes are better landmarks: they sit on many shortest paths.
    top = np.argsort(deg)[-landmarks * 4:]
    lm = rng.choice(top, size=min(landmarks, top.size), replace=False)
    dl = shortest_path(A, method="D", unweighted=True, indices=lm)

    out = np.full(pairs.shape[0], np.inf)
    step = 20000                                 # keeps the (L, P) tile small
    for s in range(0, pairs.shape[0], step):
        p = pairs[s:s + step]
        out[s:s + step] = (dl[:, p[:, 0]] + dl[:, p[:, 1]]).min(axis=0)
    return out


METHODS = {
    "scipy_dense": m_scipy_dense,
    "scipy_chunked": m_scipy_chunked,
    "networkit_pll": m_networkit_pll,
    "networkit_multitarget": m_networkit_multitarget,
    "landmark_approx": m_landmark_approx,
}
EXACT = {"scipy_dense", "scipy_chunked", "networkit_pll", "networkit_multitarget"}


def build_nk_graph(A, n):
    """Makes a NetworKit graph from the CSR matrix.

    It gives the upper triangle only. `GraphFromCoo` inserts each pair twice
    when the graph is undirected, thus a symmetric input makes a multi-graph.
    A symmetric input with a `data` array also caused a segmentation fault
    during the tests of this script. The `(i, j)` form with `uint64` is safe.
    """
    import networkit as nk
    up = sp.triu(A, k=1).tocoo()
    return nk.GraphFromCoo(
        (np.ascontiguousarray(up.row, dtype=np.uint64),
         np.ascontiguousarray(up.col, dtype=np.uint64)),
        n=n, weighted=False, directed=False)


# ---------------------------------------------------------------------------
# Profile of the output: the augmented graph, and the engine's batch plan
# ---------------------------------------------------------------------------
def build_augmented_D(A, n, pairs, w):
    """Makes the augmented CSR that the embedding engine consumes.

    The original edges keep weight 1.0. Each new pair gets its hop distance.
    An unreachable pair gets `n`, which is the sentinel of the origin package
    (`graph_augmenting.hopfill`).
    """
    w = np.where(np.isfinite(w), w, float(n))
    Ac = A.tocoo()
    return sp.csr_matrix(
        (np.concatenate([Ac.data, w, w]),
         (np.concatenate([Ac.row, pairs[:, 0], pairs[:, 1]]),
          np.concatenate([Ac.col, pairs[:, 1], pairs[:, 0]]))),
        shape=(n, n))


def profile_output(D, n, epochs: int):
    """Reports what the augmented graph costs, and how the engine batches it."""
    nbytes = D.data.nbytes + D.indices.nbytes + D.indptr.nbytes
    dense_gb = n * n * 8 / 1e9
    finite = D.data[D.data < n]
    print(f"    D: {D.shape[0]}x{D.shape[1]}, nnz={D.nnz:,}, "
          f"{nbytes/1e6:.1f} MB sparse vs {dense_gb:.1f} GB dense "
          f"({dense_gb*1e9/max(nbytes,1):.0f}x)")
    hist = np.bincount(finite.astype(int), minlength=8)[:8]
    print(f"    hop weights: 1..{int(finite.max()) if finite.size else 0}, "
          f"counts h=1..7 {hist[1:8].tolist()}, "
          f"unreachable(={n}) {int((D.data >= n).sum()):,}")

    # The engine's own view: the SELL-C-sigma batch plan.
    from fodined.embedding.shell_force import shell_coeff_data, degrees_from_D
    from fodined.embedding.sell_c_sigma import make_plan
    t = time.perf_counter()
    plan, inv_deg, stats = make_plan(
        D, (shell_coeff_data(D), D.data), degrees=degrees_from_D(D),
        b_cells=16_384, k_max=256, ladder_base=1.5)
    plan_bytes = sum(a.nbytes for rung in plan for a in rung)
    print(f"    SELL-C-sigma plan: {stats} in {time.perf_counter()-t:.2f}s, "
          f"tiles {plan_bytes/1e6:.1f} MB")

    if epochs > 0:
        run_epochs(D, n, epochs)


def run_epochs(D, n, epochs: int):
    """Runs a few real embedding epochs, to show the augmented D works."""
    import functools
    import jax
    import jax.numpy as jnp
    from fodined.embedding.shell_force import (
        shell_force, shell_coeff_data, degrees_from_D)
    from fodined.embedding.sell_c_sigma import make_plan, _step
    from fodined.embedding.drop import drop_steady_rate

    plan, inv_deg, stats = make_plan(
        D, (shell_coeff_data(D), D.data), degrees=degrees_from_D(D),
        b_cells=16_384, k_max=256, ladder_base=1.5)
    plan = jax.tree_util.tree_map(jax.device_put, plan)
    inv_deg = jax.device_put(inv_deg)
    step = jax.jit(functools.partial(_step, n=n, force_fn=shell_force))
    params = dict(k1=0.999, k2=1.0, k3=10.0, k4=0.01, h_shift=1.0)

    key = jax.random.PRNGKey(42)
    Z = jax.random.normal(key, (n, 128), dtype=jnp.float32)
    t = time.perf_counter()
    dZ = step(Z, plan, inv_deg, params)
    dZ.block_until_ready()
    compile_s = time.perf_counter() - t

    t = time.perf_counter()
    for e in range(epochs):
        key, sub = jax.random.split(key)
        dZ = drop_steady_rate(step(Z, plan, inv_deg, params), sub, 0.5)
        Z = Z + 1.0 * dZ
    Z.block_until_ready()
    el = time.perf_counter() - t
    print(f"    embedding: compile {compile_s:.1f}s, {epochs} epochs in "
          f"{el:.2f}s = {epochs/el:.1f} epochs/s, "
          f"{stats['cells']*epochs/el/1e6:.1f} Mcells/s, "
          f"finite={bool(np.isfinite(np.asarray(Z)).all())}")


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------
def run_dataset(name, args):
    print(f"\n=== {name} ===", flush=True)
    with Measured() as m:
        A, n = load_graph(name)
    print(f"  loaded: n={n:,} nodes, {A.nnz//2:,} undirected edges, "
          f"avg degree {A.nnz/n:.1f}, CSR {(A.data.nbytes+A.indices.nbytes+A.indptr.nbytes)/1e6:.1f} MB "
          f"({m.wall:.2f}s)", flush=True)

    pairs = sample_new_pairs(A, n, args.pairs)
    srcs = np.unique(pairs[:, 0])
    print(f"  sampled {pairs.shape[0]:,} unconnected pairs, "
          f"{srcs.size:,} distinct sources ({srcs.size/n:.0%} of nodes)", flush=True)

    nk_graph = None
    if any(x.startswith("networkit") for x in args.methods):
        try:
            nk_graph = build_nk_graph(A, n)
        except Exception as exc:                       # noqa: BLE001
            print(f"  networkit graph build failed: {exc}", flush=True)

    # A probe of the SciPy cost. It runs two blocks of sources, and it
    # projects the total. A method above the budget is skipped, thus the
    # benchmark cannot stall for hours on the largest graph.
    probe_chunk, probe_blocks = 64, 2
    t = time.perf_counter()
    for s in range(0, min(srcs.size, probe_chunk * probe_blocks), probe_chunk):
        shortest_path(A, method="D", unweighted=True, indices=srcs[s:s + probe_chunk])
    per_src = (time.perf_counter() - t) / min(srcs.size, probe_chunk * probe_blocks)
    projected = per_src * srcs.size
    dense_bytes = srcs.size * n * 8
    print(f"  probe: {per_src*1e3:.2f} ms per source -> SciPy full sweep "
          f"~{projected:.0f}s; one-shot dense would be {dense_bytes/1e9:.1f} GB",
          flush=True)

    results, values = {}, {}
    for mname in args.methods:
        fn = METHODS[mname]
        if mname == "scipy_dense" and dense_bytes > args.max_dense:
            print(f"  {mname:22s} SKIPPED: needs {dense_bytes/1e9:.1f} GB "
                  f"> --max-dense {args.max_dense/1e9:.1f} GB", flush=True)
            continue
        # `networkit_multitarget` also runs one BFS per source, thus the
        # SciPy probe projects its cost well enough to guard it too.
        if mname in ("scipy_dense", "scipy_chunked", "networkit_multitarget") \
                and projected > args.budget:
            print(f"  {mname:22s} SKIPPED: projected {projected:.0f}s "
                  f"> --budget {args.budget}s", flush=True)
            continue
        if mname.startswith("networkit") and nk_graph is None:
            continue
        try:
            gc.collect()
            with Measured() as m:
                w = fn(A, n, pairs, srcs, nk_graph=nk_graph,
                       landmarks=args.landmarks)
            results[mname] = m
            values[mname] = w
            print(f"  {mname:22s} wall {m.wall:8.2f}s  cpu {m.cpu:8.2f}s "
                  f"({m.cpu/max(m.wall,1e-9):.1f}x)  peakRSS {m.peak_rss/1e6:8.1f} MB "
                  f"  pyPeak {m.peak_py/1e6:7.1f} MB", flush=True)
        except MemoryError:
            print(f"  {mname:22s} FAILED: MemoryError", flush=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {mname:22s} FAILED: {type(exc).__name__}: {exc}", flush=True)

    # Accuracy. The reference is any exact method that ran.
    ref_name = next((k for k in ("scipy_chunked", "scipy_dense",
                                 "networkit_pll", "networkit_multitarget")
                     if k in values), None)
    if ref_name:
        ref = values[ref_name]
        print(f"  accuracy vs {ref_name} (unreachable pairs: "
              f"{int((~np.isfinite(ref)).sum()):,}):", flush=True)
        fin = np.isfinite(ref)
        for k, w in values.items():
            same_inf = np.array_equal(np.isfinite(w), fin)
            d = w[fin] - ref[fin]
            print(f"    {k:22s} exact={bool((d == 0).all()) and same_inf}  "
                  f"max_err={np.abs(d).max() if d.size else 0:.0f}  "
                  f"mean_err={np.abs(d).mean() if d.size else 0:.3f}  "
                  f"overestimated={float((d > 0).mean()):.1%}", flush=True)

    if ref_name:
        print("  output profile:", flush=True)
        D = build_augmented_D(A, n, pairs, values[ref_name])
        profile_output(D, n, args.epochs)
    return results, values, (A, n, pairs, srcs, nk_graph)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--pairs", type=int, default=50_000)
    ap.add_argument("--landmarks", type=int, default=64)
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--max-dense", type=float, default=2e9)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()
    args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in args.methods:
        if m not in METHODS:
            ap.error(f"unknown method {m!r}; choose from {list(METHODS)}")

    print(f"[bench] {len(args.datasets)} graphs, {args.pairs:,} new pairs each, "
          f"methods: {', '.join(args.methods)}", flush=True)

    summary = {}
    last_ok = None
    for name in args.datasets:
        try:
            res, vals, ctx = run_dataset(name, args)
            summary[name] = res
            if res:
                last_ok = (name, ctx)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {name} FAILED: {type(exc).__name__}: {exc}", flush=True)

    # ---- summary table ----------------------------------------------------
    print("\n=== summary: wall seconds (peak RSS MB) ===", flush=True)
    head = f"{'dataset':16s}" + "".join(f"{m[:20]:>22s}" for m in args.methods)
    print(head)
    for name, res in summary.items():
        row = f"{name:16s}"
        for m in args.methods:
            row += (f"{res[m].wall:>13.2f}s({res[m].peak_rss/1e6:6.0f})"
                    if m in res else f"{'--':>22s}")
        print(row, flush=True)

    # ---- cProfile of the winner ------------------------------------------
    target = args.profile or (last_ok[0] if last_ok else None)
    if target and last_ok:
        name, (A, n, pairs, srcs, nk_graph) = last_ok
        if args.profile and args.profile != name:
            print(f"\n[bench] --profile {args.profile} is not the last graph "
                  f"that ran; profiling {name} instead.", flush=True)
        winner = ("networkit_pll" if nk_graph is not None
                  and "networkit_pll" in args.methods else "scipy_chunked")
        print(f"\n=== cProfile: {winner} on {name} ===", flush=True)
        pr = cProfile.Profile()
        pr.enable()
        METHODS[winner](A, n, pairs, srcs, nk_graph=nk_graph,
                        landmarks=args.landmarks)
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(12)
        print(s.getvalue(), flush=True)


if __name__ == "__main__":
    main()
