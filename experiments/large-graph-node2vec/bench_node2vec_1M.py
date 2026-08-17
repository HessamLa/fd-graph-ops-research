#!/bin/env python3
"""bench_node2vec_1M.py -- node2vec on a graph of more than a million nodes.

The companion of the `fodined/modular.py` run on the same graph. The question
is the same for both: does the method reach a graph of this size on this
machine, and what does it cost in time and in memory?

Graph: com_youtube from SNAP. 1,134,890 nodes and 2,987,624 edges. It is the
smallest of the three SNAP graphs in `data_cache/`, and it has a power-law
degree distribution.

Three things are different from `experiments/other-ge/bench_other_ge.py`, and
all three are necessary at this size:

1. The walks go to a FILE, and not to a list. 1.13M nodes with 5 walks of
   length 20 give 113M tokens. A list of Python strings for those tokens
   needs more than 10 GB. `Word2Vec(corpus_file=...)` reads the file, thus
   the memory holds only the model.
2. The walks are shorter and fewer (5 x 20, and not 10 x 40), and the
   training uses 3 epochs, and not 5. The full setting needs more than an
   hour of CPU.
3. The hop distances come from a LIMITED set of sources. An exact hop
   distance for 20,000 random pairs needs about 20,000 BFS runs, and one BFS
   on this graph needs 0.36 s: that is two hours. The sample therefore takes
   its pairs from `--sp-sources` source nodes only.

The report gives the time of each stage, the peak memory, the link
prediction, and the hop distance approximation.

Run from the repo root (fdmap/):

    .venv/bin/python experiments/large-graph-node2vec/bench_node2vec_1M.py
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

SNAP = {"com_youtube": "com_youtube/com-youtube.ungraph.txt",
        "as_skitter": "as_skitter/as-skitter.txt",
        "roadnet_ca": "roadnet_ca/roadNet-CA.txt"}


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) / 1024
    return 0.0


def load(graph):
    import pandas as pd
    t = time.perf_counter()
    e = pd.read_csv(os.path.join(ROOT, "data_cache", SNAP[graph]), sep="\t",
                    comment="#", header=None, names=["u", "v"],
                    dtype=np.int64).to_numpy()
    _, flat = np.unique(e.ravel(), return_inverse=True)
    e = flat.reshape(-1, 2)
    e = e[e[:, 0] != e[:, 1]]
    n = int(e.max()) + 1
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    A = sp.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)),
                      shape=(n, n))
    A.data[:] = 1.0
    A.sort_indices()
    return A, n, time.perf_counter() - t


def write_walks(A, n, path, n_walks, walk_len, seed, batch=200_000):
    """Uniform random walks, written to `path`, one walk per line.

    The nodes are processed in batches, thus the memory holds one batch of
    walks and not all of them.
    """
    rng = np.random.default_rng(seed)
    deg = np.diff(A.indptr)
    t = time.perf_counter()
    with open(path, "w") as f:
        for _ in range(n_walks):
            for s in range(0, n, batch):
                cur = np.arange(s, min(s + batch, n))
                walk = np.empty((cur.size, walk_len), dtype=np.int64)
                walk[:, 0] = cur
                for j in range(1, walk_len):
                    d = deg[cur]
                    off = (rng.random(cur.size) * np.maximum(d, 1)).astype(np.int64)
                    nxt = A.indices[A.indptr[cur] + off]
                    cur = np.where(d > 0, nxt, cur)
                    walk[:, j] = cur
                f.write("\n".join(" ".join(map(str, w)) for w in walk))
                f.write("\n")
    return time.perf_counter() - t


def sample_non_edges(A, n, count, rng, sources=None):
    Ac = A.tocoo()
    keys = np.sort(Ac.row.astype(np.int64) * n + Ac.col.astype(np.int64))
    del Ac
    u_all = np.empty(0, np.int64)
    v_all = np.empty(0, np.int64)
    while u_all.size < count:
        draw = (count - u_all.size) * 2 + 1024
        u = (rng.choice(sources, size=draw) if sources is not None
             else rng.integers(0, n, draw))
        v = rng.integers(0, n, draw)
        ok = u != v
        u, v = u[ok], v[ok]
        k = u * n + v
        pos = np.searchsorted(keys, k)
        pos[pos >= keys.size] = 0
        keep = keys[pos] != k
        room = count - u_all.size
        u_all = np.concatenate([u_all, u[keep][:room]])
        v_all = np.concatenate([v_all, v[keep][:room]])
    return np.column_stack([u_all, v_all])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="com_youtube", choices=list(SNAP))
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--walks", type=int, default=5)
    ap.add_argument("--walk-len", type=int, default=20)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lp-pairs", type=int, default=40_000)
    ap.add_argument("--sp-sources", type=int, default=200)
    ap.add_argument("--sp-pairs", type=int, default=20_000)
    args = ap.parse_args()

    from gensim.models import Word2Vec
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, mean_absolute_error,
                                 mean_squared_error, r2_score,
                                 mean_absolute_percentage_error)
    from scipy.sparse.csgraph import shortest_path

    A, n, t_load = load(args.graph)
    print(f"[n2v-1M] {args.graph}: n={n:,} nodes, {A.nnz//2:,} undirected "
          f"edges, avg degree {A.nnz/n:.2f} (load {t_load:.1f}s)", flush=True)
    print(f"[n2v-1M] settings: dim={args.dim}, {args.walks} walks x "
          f"{args.walk_len} steps, window={args.window}, "
          f"{args.epochs} epochs, {args.workers} workers", flush=True)

    rng = np.random.default_rng(args.seed)
    with tempfile.TemporaryDirectory(prefix="walks_") as tmp:
        wp = os.path.join(tmp, "walks.txt")
        t_walk = write_walks(A, n, wp, args.walks, args.walk_len, args.seed)
        size_gb = os.path.getsize(wp) / 1e9
        print(f"[n2v-1M] walks: {n*args.walks:,} walks x {args.walk_len} "
              f"steps = {n*args.walks*args.walk_len/1e6:.0f}M tokens, "
              f"{size_gb:.2f} GB on disk ({t_walk:.1f}s)", flush=True)

        t = time.perf_counter()
        model = Word2Vec(corpus_file=wp, vector_size=args.dim,
                         window=args.window, min_count=0, sg=1, negative=5,
                         workers=args.workers, epochs=args.epochs,
                         seed=args.seed)
        t_train = time.perf_counter() - t
    print(f"[n2v-1M] word2vec: {t_train:.1f}s, vocabulary {len(model.wv):,}, "
          f"peak RSS {rss_mb():.0f} MB", flush=True)

    t = time.perf_counter()
    Z = np.zeros((n, args.dim), dtype=np.float32)
    for i in range(n):
        k = str(i)
        if k in model.wv:
            Z[i] = model.wv[k]
    del model
    t_gather = time.perf_counter() - t
    total = t_load + t_walk + t_train + t_gather
    print(f"[n2v-1M] TOTAL embedding time {total:.1f}s "
          f"(load {t_load:.1f} + walks {t_walk:.1f} + train {t_train:.1f} "
          f"+ gather {t_gather:.1f}), peak RSS {rss_mb():.0f} MB", flush=True)

    # ---- link prediction --------------------------------------------------
    t = time.perf_counter()
    up = sp.triu(A, k=1).tocoo()
    take = rng.choice(up.row.size, min(args.lp_pairs, up.row.size),
                      replace=False)
    pos = np.column_stack([up.row[take], up.col[take]])
    del up
    neg = sample_non_edges(A, n, pos.shape[0], rng)
    pairs = np.vstack([pos, neg])
    X = Z[pairs[:, 0]] * Z[pairs[:, 1]]
    y = np.concatenate([np.ones(pos.shape[0]), np.zeros(neg.shape[0])])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=args.seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=args.seed,
                                 n_jobs=-1).fit(Xtr, ytr)
    pred, prob = clf.predict(Xte), clf.predict_proba(Xte)[:, 1]
    print(f"[n2v-1M] link prediction on {X.shape[0]:,} pairs "
          f"({time.perf_counter()-t:.1f}s)", flush=True)
    print(f"  accuracy : {accuracy_score(yte, pred):.4f}")
    print(f"  precision: {precision_score(yte, pred, zero_division=0):.4f}")
    print(f"  recall   : {recall_score(yte, pred, zero_division=0):.4f}")
    print(f"  f1-score : {f1_score(yte, pred, zero_division=0):.4f}")
    print(f"  auc      : {roc_auc_score(yte, prob):.4f}", flush=True)

    # ---- hop distance approximation --------------------------------------
    t = time.perf_counter()
    sources = rng.choice(n, size=args.sp_sources, replace=False)
    pairs = sample_non_edges(A, n, args.sp_pairs, rng, sources=sources)
    srcs = np.unique(pairs[:, 0])
    chunk = max(4, int(200e6 / (n * 8)))
    pos_of = np.searchsorted(srcs, pairs[:, 0])
    hop = np.empty(pairs.shape[0])
    for s in range(0, srcs.size, chunk):
        blk = srcs[s:s + chunk]
        d = shortest_path(A, method="D", unweighted=True, indices=blk)
        m = (pos_of >= s) & (pos_of < s + blk.size)
        hop[m] = d[pos_of[m] - s, pairs[m, 1]]
    t_hop = time.perf_counter() - t
    ok = np.isfinite(hop) & (hop > 1)
    pairs, hop = pairs[ok], hop[ok]
    print(f"[n2v-1M] hop distances for {hop.size:,} pairs from "
          f"{srcs.size} sources ({t_hop:.1f}s), hops "
          f"{hop.min():.0f}..{hop.max():.0f}", flush=True)

    X = np.linalg.norm(Z[pairs[:, 0]] - Z[pairs[:, 1]], axis=1)[:, None]
    Xtr, Xte, ytr, yte = train_test_split(X, hop, test_size=0.2,
                                          random_state=args.seed)
    sc = StandardScaler().fit(Xtr)

    def report(name, p, secs=None):
        print(f"  {name:>14s} {mean_absolute_error(yte, p):>8.3f} "
              f"{mean_absolute_percentage_error(yte, p):>8.3f} "
              f"{np.sqrt(mean_squared_error(yte, p)):>8.3f} "
              f"{r2_score(yte, p):>8.3f} "
              f"{np.mean(np.rint(p) == yte):>8.1%}"
              f"{'' if secs is None else f'  ({secs:.1f}s)'}", flush=True)

    print(f"  {'model':>14s} {'MAE':>8s} {'MRE':>8s} {'RMSE':>8s} "
          f"{'R2':>8s} {'exact':>8s}", flush=True)
    report("mean baseline", np.full(yte.shape, ytr.mean()))
    t = time.perf_counter()
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=25,
                               random_state=args.seed, n_jobs=-1).fit(Xtr, ytr)
    report("random forest", rf.predict(Xte), time.perf_counter() - t)
    t = time.perf_counter()
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                       early_stopping=True, random_state=args.seed)
    mlp.fit(sc.transform(Xtr), ytr)
    report("MLP", mlp.predict(sc.transform(Xte)), time.perf_counter() - t)

    print(f"[n2v-1M] peak RSS for the whole run: {rss_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main()
