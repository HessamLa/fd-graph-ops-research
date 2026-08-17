#!/bin/env python3
"""bench_large_graphs.py -- augmentation methods on graphs of a million nodes.

This experiment continues `bench_shortest_path_gemsec.py`. That experiment
used graphs of 3.9k to 50.5k nodes, and it found that Pruned Landmark
Labeling (PLL) is the best exact method for the hop distances that the
augmentation step needs. This experiment asks a different question, because a
graph of a million nodes changes the question:

    Which augmentation method is POSSIBLE at all at this size, and what does
    it cost in time and in memory?

At 50k nodes every method completed, thus the comparison was about speed. At
1.9M nodes most methods cannot complete:

  * A dense `(n, n)` distance matrix is 30 TB. It is not an option.
  * One BFS for each source is not an option either. The sample has ~100k
    distinct sources, and one BFS on a graph of 11M edges needs a large
    fraction of a second. This is many hours.
  * PLL builds one index, and then each query is free. Thus its cost does not
    grow with the number of pairs. This is the method that can win here.
  * The landmark method is an approximation, and it is the fallback if PLL
    needs too much memory.

The three graphs come from SNAP (https://snap.stanford.edu/data/). They are
different on purpose, because the PLL literature says that the structure of a
graph, and not only its size, controls the size of the index:

    com_youtube   1.13M nodes,  2.99M edges  social, power-law: the good case
    as_skitter    1.70M nodes, 11.10M edges  internet topology. The PLL paper
                                             (Akiba et al., SIGMOD 2013)
                                             gives 359 s and 2.7 GB for this
                                             same graph, thus it validates
                                             this setup against published
                                             numbers.
    roadnet_ca    1.97M nodes,  5.53M edges  a road network: high diameter and
                                             no hubs. This is the documented
                                             FAILURE case of PLL. It is here
                                             to find the limit, not to confirm
                                             the recommendation.

SAFETY. This machine has approximately 3 GB of free RAM and a full swap file.
An index of several GB can stop the desktop. Thus each method runs in its own
process, and that process has a hard address-space limit (`RLIMIT_AS`) and a
timeout. A method that asks for too much memory gets a `MemoryError` and dies
alone. It cannot take the machine with it. The parent process stays small.

Run from the repo root (fdmap/):

    .venv/bin/python experiments/bench_large_graphs.py
    .venv/bin/python experiments/bench_large_graphs.py --graphs com_youtube
    .venv/bin/python experiments/bench_large_graphs.py --mem-cap-gb 4 --timeout 1800
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
CACHE = os.path.join(HERE, "_cache")

# name -> (edge file below data_cache/, nodes, edges) for the report header.
GRAPHS = {
    "com_youtube": ("com_youtube/com-youtube.ungraph.txt", 1_134_890, 2_987_624),
    "as_skitter":  ("as_skitter/as-skitter.txt",           1_696_415, 11_095_298),
    "roadnet_ca":  ("roadnet_ca/roadNet-CA.txt",           1_965_206, 5_533_214),
}
UNREACHABLE = np.inf


# ---------------------------------------------------------------------------
# Load, with a cache
# ---------------------------------------------------------------------------
def load_csr(graph_name: str):
    """Reads a SNAP edge list, and returns a symmetric CSR matrix of 1.0.

    `pandas.read_csv` does the parse, because `np.loadtxt` needs minutes on a
    file of 11M lines. The node ids of SNAP are not contiguous, thus
    `np.unique` maps them to 0..n-1. The result goes into a cache file,
    because the parse is the slowest part of a repeated experiment.
    """
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"{graph_name}_csr.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        A = sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                          shape=tuple(z["shape"]))
        return A, A.shape[0]

    import pandas as pd
    path = os.path.join(ROOT, "data_cache", GRAPHS[graph_name][0])
    df = pd.read_csv(path, sep="\t", comment="#", header=None,
                     names=["u", "v"], dtype=np.int64)
    e = df.to_numpy()
    del df
    e = e[e[:, 0] != e[:, 1]]
    _, flat = np.unique(e.ravel(), return_inverse=True)
    e = flat.reshape(-1, 2).astype(np.int64)
    n = int(e.max()) + 1

    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    del e
    A = sp.csr_matrix((np.ones(rows.size, dtype=np.float64), (rows, cols)),
                      shape=(n, n))
    A.data[:] = 1.0
    A.sort_indices()
    np.savez(cache, data=A.data, indices=A.indices, indptr=A.indptr,
             shape=np.array(A.shape))
    return A, n


def sample_pairs(A, n, count, max_sources=0, seed=42):
    """Samples `count` pairs of nodes that have no edge.

    `max_sources` limits how many DISTINCT source nodes the sample uses. This
    matters, because the cost of every BFS method is proportional to the
    number of distinct sources. A limit of 0 means no limit.
    """
    rng = np.random.default_rng(seed)
    Ac = A.tocoo()
    keys = np.sort(Ac.row.astype(np.int64) * n + Ac.col.astype(np.int64))
    del Ac
    pool = (rng.choice(n, size=max_sources, replace=False)
            if max_sources else None)

    u_all = np.empty(0, np.int64)
    v_all = np.empty(0, np.int64)
    while u_all.size < count:
        draw = (count - u_all.size) * 2 + 1024
        u = rng.choice(pool, size=draw) if pool is not None else rng.integers(0, n, draw)
        v = rng.integers(0, n, draw)
        ok = u != v
        u, v = u[ok], v[ok]
        k = u * n + v
        pos = np.searchsorted(keys, k)
        pos[pos >= keys.size] = 0
        keep = keys[pos] != k
        u, v = u[keep], v[keep]
        room = count - u_all.size
        u_all = np.concatenate([u_all, u[:room]])
        v_all = np.concatenate([v_all, v[:room]])
    return np.column_stack([u_all, v_all])


# ---------------------------------------------------------------------------
# The methods. Each one returns one hop distance for each pair.
# ---------------------------------------------------------------------------
def m_scipy_chunked(A, n, pairs, chunk=0, **kw):
    """One SciPy BFS for each distinct source, over blocks of sources.

    The memory is bounded, but the time is proportional to the number of
    distinct sources. At a million nodes this is the method that runs out of
    time, and not out of memory.

    The block size adapts to `n`. A fixed block of 64 rows is 973 MB at
    n = 1.9M, because each row is `n` float64 values. The rule below keeps one
    block near 200 MB at any size.
    """
    from scipy.sparse.csgraph import shortest_path
    if not chunk:
        chunk = max(4, int(200e6 / (n * 8)))
    srcs = np.unique(pairs[:, 0])
    pos = np.searchsorted(srcs, pairs[:, 0])
    out = np.empty(pairs.shape[0])
    for s in range(0, srcs.size, chunk):
        blk = srcs[s:s + chunk]
        d = shortest_path(A, method="D", unweighted=True, indices=blk)
        m = (pos >= s) & (pos < s + blk.size)
        out[m] = d[pos[m] - s, pairs[m, 1]]
    return out


def m_networkit_pll(A, n, pairs, **kw):
    """Pruned Landmark Labeling. One index, then each query is free.

    The build is the whole cost. The index of a road network can be very
    large, thus this is the method that runs out of MEMORY, and not out of
    time, on `roadnet_ca`.
    """
    import networkit as nk
    up = sp.triu(A, k=1).tocoo()
    g = nk.GraphFromCoo(
        (np.ascontiguousarray(up.row, dtype=np.uint64),
         np.ascontiguousarray(up.col, dtype=np.uint64)),
        n=n, weighted=False, directed=False)
    del up
    t = time.perf_counter()
    pll = nk.distance.PrunedLandmarkLabeling(g)
    pll.run()
    build = time.perf_counter() - t

    t = time.perf_counter()
    q = np.fromiter((pll.query(int(u), int(v)) for u, v in pairs),
                    dtype=np.uint64, count=pairs.shape[0])
    query = time.perf_counter() - t
    out = q.astype(np.float64)
    out[q == np.uint64(2 ** 64 - 1)] = UNREACHABLE      # undocumented sentinel
    return out, {"build_s": build, "query_s": query}


def m_landmark_approx(A, n, pairs, landmarks=32, seed=42, **kw):
    """Approximation: `d(u,v) <= min over landmarks of d(l,u) + d(l,v)`.

    The distance table is `uint16`, and not the `float64` that SciPy returns.
    A hop distance is a small integer, thus `float64` wastes 4x the memory. At
    n = 2M and 32 landmarks this is the difference between 512 MB and 128 MB.
    """
    from scipy.sparse.csgraph import shortest_path
    rng = np.random.default_rng(seed)
    deg = np.diff(A.indptr)
    top = np.argsort(deg)[-landmarks * 8:]
    lm = rng.choice(top, size=min(landmarks, top.size), replace=False)

    big = np.iinfo(np.uint16).max
    dl = np.empty((lm.size, n), dtype=np.uint16)
    for i in range(0, lm.size, 4):                  # 4 rows of float64 at a time
        d = shortest_path(A, method="D", unweighted=True, indices=lm[i:i + 4])
        d[~np.isfinite(d)] = big
        dl[i:i + 4] = d.astype(np.uint16)
        del d

    out = np.full(pairs.shape[0], np.inf)
    step = 20_000
    for s in range(0, pairs.shape[0], step):
        p = pairs[s:s + step]
        est = (dl[:, p[:, 0]].astype(np.int32)
               + dl[:, p[:, 1]].astype(np.int32)).min(axis=0)
        est = est.astype(np.float64)
        est[est >= big] = np.inf
        out[s:s + step] = est
    return out


METHODS = {
    "scipy_chunked": m_scipy_chunked,
    # "networkit_pll": m_networkit_pll,
    "landmark_approx": m_landmark_approx,
}


# ---------------------------------------------------------------------------
# Child process: run one method under a hard memory limit
# ---------------------------------------------------------------------------
def rss_watchdog(limit_bytes: int):
    """Kills this process if the RESIDENT memory passes `limit_bytes`.

    This replaced an `RLIMIT_AS` cap, and the reason is important. `RLIMIT_AS`
    limits the VIRTUAL address space. NetworKit uses OpenMP, and its threads
    reserve much more address space than they ever touch: a measurement here
    showed 1.52 GB of virtual space at only 0.47 GB of resident memory. Thus
    an `RLIMIT_AS` cap stops a method that is still far below the cap in real
    memory, and it reports a peak that is much too low. A watchdog on `VmRSS`
    measures what the machine actually feels.

    The exit code 42 tells the parent that the limit stopped this process.
    """
    def loop():
        while True:
            if _vm("VmRSS") > limit_bytes:
                sys.stderr.write(f"RSS cap {limit_bytes/1e9:.2f} GB exceeded\n")
                sys.stderr.flush()
                os._exit(42)
            time.sleep(0.05)
    t = threading.Thread(target=loop, daemon=True)
    t.start()


def child_main(graph, method, out_json, out_npy, pairs_npy, mem_cap, kwargs):
    """Runs one method, and writes the result and the cost to disk."""
    if mem_cap > 0:
        rss_watchdog(mem_cap)

    rec = {"graph": graph, "method": method, "ok": False}
    try:
        A, n = load_csr(graph)
        pairs = np.load(pairs_npy)
        base = peak_rss()
        t, c = time.perf_counter(), time.process_time()
        res = METHODS[method](A, n, pairs, **kwargs)
        extra = {}
        if isinstance(res, tuple):
            res, extra = res
        rec.update(ok=True, wall=time.perf_counter() - t,
                   cpu=time.process_time() - c, peak_rss=peak_rss(),
                   peak_vm=_vm("VmPeak"), base_rss=base, **extra)
        np.save(out_npy, res)
    except MemoryError:
        rec["error"] = "MemoryError"
        rec["peak_rss"], rec["peak_vm"] = peak_rss(), _vm("VmPeak")
    except Exception as exc:                                   # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["peak_rss"], rec["peak_vm"] = peak_rss(), _vm("VmPeak")
    with open(out_json, "w") as f:
        json.dump(rec, f)


def _vm(field: str) -> int:
    """Reads one `VmXxx` field of /proc/self/status, in bytes."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field):
                return int(line.split()[1]) * 1024
    return 0


def peak_rss() -> int:
    """The high-water RESIDENT set size of this process, in bytes.

    `VmHWM` is the true peak over the whole life of the process. A sampling
    thread can miss a short peak, thus this is the better measurement when
    each method has its own process.
    """
    return _vm("VmHWM")


# ---------------------------------------------------------------------------
# Output profile
# ---------------------------------------------------------------------------
def profile_output(A, n, pairs, w, args):
    """Reports the augmented graph, and the plan that the engine builds."""
    w = np.where(np.isfinite(w), w, float(n))
    Ac = A.tocoo()
    D = sp.csr_matrix(
        (np.concatenate([Ac.data, w, w]),
         (np.concatenate([Ac.row, pairs[:, 0], pairs[:, 1]]),
          np.concatenate([Ac.col, pairs[:, 1], pairs[:, 0]]))),
        shape=(n, n))
    del Ac
    nb = D.data.nbytes + D.indices.nbytes + D.indptr.nbytes
    print(f"    D: nnz={D.nnz:,}, {nb/1e6:.0f} MB sparse vs "
          f"{n*n*8/1e12:.1f} TB dense", flush=True)

    from fodined.embedding.shell_force import shell_coeff_data, degrees_from_D
    from fodined.embedding.sell_c_sigma import make_plan
    t = time.perf_counter()
    plan, inv_deg, stats = make_plan(
        D, (shell_coeff_data(D), D.data), degrees=degrees_from_D(D),
        b_cells=16_384, k_max=256, ladder_base=1.5)
    tiles = sum(a.nbytes for rung in plan for a in rung)
    print(f"    plan: {stats} in {time.perf_counter()-t:.1f}s, "
          f"tiles {tiles/1e6:.0f} MB", flush=True)

    # The GPU is the limit at this size, and not the augmentation. Z and dZ
    # are each n * n_dim * 4 bytes. Report the budget before any allocation.
    free_vram = gpu_free_bytes()
    need = n * args.n_dim * 4 * 2 + tiles
    print(f"    embedding at n_dim={args.n_dim} needs ~{need/1e9:.2f} GB VRAM "
          f"(Z + dZ + tiles); free VRAM {free_vram/1e9:.2f} GB -> "
          f"{'fits' if need < free_vram*0.9 else 'DOES NOT FIT'}", flush=True)
    if need < free_vram * 0.9 and args.epochs > 0:
        run_epochs(D, n, args)


def gpu_free_bytes() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        return int(out.stdout.split()[0]) * 1024 * 1024
    except Exception:                                          # noqa: BLE001
        return 0


def run_epochs(D, n, args):
    import functools
    import jax
    import jax.numpy as jnp
    from fodined.embedding.shell_force import (
        shell_force, shell_coeff_data, degrees_from_D)
    from fodined.embedding.sell_c_sigma import make_plan, _step
    from fodined.embedding.drop import drop_steady_rate

    plan, inv_deg, stats = make_plan(
        D, (shell_coeff_data(D), D.data), degrees=degrees_from_D(D))
    plan = jax.tree_util.tree_map(jax.device_put, plan)
    inv_deg = jax.device_put(inv_deg)
    step = jax.jit(functools.partial(_step, n=n, force_fn=shell_force))
    params = dict(k1=0.999, k2=1.0, k3=10.0, k4=0.01, h_shift=1.0)
    key = jax.random.PRNGKey(42)
    Z = jax.random.normal(key, (n, args.n_dim), dtype=jnp.float32)
    step(Z, plan, inv_deg, params).block_until_ready()
    t = time.perf_counter()
    for _ in range(args.epochs):
        key, sub = jax.random.split(key)
        Z = Z + drop_steady_rate(step(Z, plan, inv_deg, params), sub, 0.5)
    Z.block_until_ready()
    el = time.perf_counter() - t
    print(f"    embedding: {args.epochs} epochs in {el:.1f}s = "
          f"{args.epochs/el:.2f} epochs/s, "
          f"{stats['cells']*args.epochs/el/1e6:.0f} Mcells/s", flush=True)


# ---------------------------------------------------------------------------
def run_graph(graph_name, args, tmp):
    print(f"\n=== {graph_name} ===", flush=True)
    t = time.perf_counter()
    A, n = load_csr(graph_name)
    print(f"  n={n:,} nodes, {A.nnz//2:,} undirected edges, "
          f"avg degree {A.nnz/n:.1f}, CSR "
          f"{(A.data.nbytes+A.indices.nbytes+A.indptr.nbytes)/1e6:.0f} MB "
          f"(load {time.perf_counter()-t:.1f}s)", flush=True)

    args.pairs = int(n*(np.log10(n)-1))

    pairs = sample_pairs(A, n, args.pairs, args.max_sources)
    srcs = np.unique(pairs[:, 0])
    pairs_npy = os.path.join(tmp, f"{graph_name}_pairs.npy")
    np.save(pairs_npy, pairs)
    print(f"  {pairs.shape[0]:,} pairs, {srcs.size:,} distinct sources "
          f"({srcs.size/n:.1%} of nodes); a dense (n,n) matrix would be "
          f"{n*n*8/1e12:.1f} TB", flush=True)
    del A

    results, values = {}, {}
    for m in args.methods:
        oj = os.path.join(tmp, f"{graph_name}_{m}.json")
        on = os.path.join(tmp, f"{graph_name}_{m}.npy")
        cmd = [sys.executable, os.path.abspath(__file__), "--child", graph_name, m,
               oj, on, pairs_npy, str(int(args.mem_cap_gb * 1e9)),
               json.dumps({"landmarks": args.landmarks})]
        t = time.perf_counter()
        try:
            p = subprocess.run(cmd, timeout=args.timeout,
                               capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            print(f"  {m:18s} TIMEOUT after {args.timeout}s", flush=True)
            continue
        if p.returncode == 42:
            print(f"  {m:18s} EXCEEDED RSS CAP of {args.mem_cap_gb} GB "
                  f"after {time.perf_counter()-t:.0f}s -- needs more resident "
                  f"memory than this machine has free", flush=True)
            continue
        if not os.path.exists(oj):
            tail = (p.stderr or p.stdout or "").strip().splitlines()[-1:] or ["no output"]
            print(f"  {m:18s} CRASHED (rc={p.returncode}): {tail[0][:90]}", flush=True)
            continue
        rec = json.load(open(oj))
        if not rec["ok"]:
            print(f"  {m:18s} FAILED: {rec['error']}  "
                  f"(peak RSS {rec.get('peak_rss',0)/1e9:.2f} GB, "
                  f"peak VM {rec.get('peak_vm',0)/1e9:.2f} GB)", flush=True)
            continue
        results[m] = rec
        values[m] = np.load(on)
        extra = ""
        if "build_s" in rec:
            extra = f"  [build {rec['build_s']:.1f}s + query {rec['query_s']:.2f}s]"
        print(f"  {m:18s} wall {rec['wall']:8.1f}s  cpu {rec['cpu']:8.1f}s "
              f"({rec['cpu']/max(rec['wall'],1e-9):.1f}x)  peakRSS "
              f"{rec['peak_rss']/1e9:5.2f} GB  peakVM {rec.get('peak_vm',0)/1e9:5.2f} GB"
              f"{extra}", flush=True)

    ref = next((k for k in ("scipy_chunked", "networkit_pll") if k in values), None)
    if ref:
        r = values[ref]
        fin = np.isfinite(r)
        print(f"  accuracy vs {ref} ({int((~fin).sum()):,} unreachable):", flush=True)
        for k, v in values.items():
            d = v[fin] - r[fin]
            print(f"    {k:18s} exact={bool((d==0).all())}  "
                  f"max_err={np.abs(d).max() if d.size else 0:.0f}  "
                  f"mean_err={np.abs(d).mean() if d.size else 0:.3f}", flush=True)
        print("  output profile:", flush=True)
        A, n = load_csr(graph_name)
        # The parent cannot use RLIMIT_AS, because JAX and CUDA reserve a very
        # large virtual address space and the limit would stop them. Thus the
        # profile has a `try` instead, and the experiment continues with the
        # next graph if the memory is not sufficient.
        try:
            profile_output(A, n, pairs, values[ref], args)
        except MemoryError:
            print("    profile SKIPPED: not enough memory in the parent",
                  flush=True)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", nargs=7, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--graphs", default=",".join(GRAPHS))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--pairs", type=int, default=100_000) # we will use n*(log10(n)-1)
    ap.add_argument("--max-sources", type=int, default=0,
                    help="limit distinct source nodes (0 = no limit)")
    ap.add_argument("--landmarks", type=int, default=32)
    ap.add_argument("--mem-cap-gb", type=float, default=2.5)
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-dim", type=int, default=8)
    args = ap.parse_args()

    if args.child:
        g, m, oj, on, pn, cap, kw = args.child
        child_main(g, m, oj, on, pn, int(cap), json.loads(kw))
        return

    args.graphs = [g.strip() for g in args.graphs.split(",") if g.strip()]
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"[bench] graphs {args.graphs}, methods {args.methods}, "
          f"{args.pairs:,} pairs, memory cap {args.mem_cap_gb} GB, "
          f"timeout {args.timeout}s", flush=True)

    summary = {}
    with tempfile.TemporaryDirectory(prefix="bench_large_") as tmp:
        for g in args.graphs:
            try:
                summary[g] = run_graph(g, args, tmp)
            except Exception as exc:                           # noqa: BLE001
                print(f"  {g} FAILED: {type(exc).__name__}: {exc}", flush=True)

    print("\n=== summary: wall seconds (peak RSS GB) ===", flush=True)
    print(f"{'graph':14s}" + "".join(f"{m[:17]:>20s}" for m in args.methods))
    for g, res in summary.items():
        row = f"{g:14s}"
        for m in args.methods:
            row += (f"{res[m]['wall']:>12.1f}s({res[m]['peak_rss']/1e9:4.2f})"
                    if m in res else f"{'--':>20s}")
        print(row, flush=True)


if __name__ == "__main__":
    main()
