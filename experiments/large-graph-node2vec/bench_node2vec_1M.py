#!/bin/env python3
"""bench_node2vec_1M.py -- node2vec on a graph of more than a million nodes.

The companion of the `fodined/modular.py` run on the same graph. The question
is the same for both: does the method reach a graph of this size on this
machine, and what does it cost in time and in memory?

Graph: com_youtube from SNAP. 1,134,890 nodes and 2,987,624 edges. It is the
smallest of the three SNAP graphs in `data_cache/`, and it has a power-law
degree distribution.

Two things are different from `experiments/other-ge/bench_other_ge.py`, and
both are necessary at this size:

1. The walks go to a FILE, and not to a list. 1.13M nodes with 5 walks of
   length 20 give 113M tokens. A list of Python strings for those tokens
   needs more than 10 GB. `Word2Vec(corpus_file=...)` reads the file, thus
   the memory holds only the model.
2. The walks are shorter and fewer (5 x 20, and not 10 x 40), and the
   training uses 3 epochs, and not 5. The full setting needs more than an
   hour of CPU.

The link prediction and the hop-distance approximation come from
`evaluator`, under the protocol name `n2v1m`. That protocol is a frozen
record of what this script did: it caps the hop-distance sources at 200
(`--sp-sources`) and reads their distances with a blocked, source-limited
scipy BFS -- an exact hop distance for 20,000 random pairs needs about
20,000 BFS runs, and one BFS on this graph needs 0.36 s: that is two hours.
`evaluator.hops` blocks that BFS internally; this script never reimplements
it. `node2vec_com_youtube_1M.log` is the reference log.

The report gives the time of each stage, the peak memory, the link
prediction, and the hop distance approximation.

SMALL-GRAPH SMOKE PATH. `--graph cora` and `--graph pubmed` load a
citation graph (2,708 and 19,717 nodes) through `evaluator.load_graph`
instead of a SNAP file, thus the whole pipeline -- walks, training,
gathering, and both tasks -- runs end to end in seconds on a machine that
cannot hold a SNAP graph. The correctness of this script does not depend
on the SIZE of the graph: the same `evaluator` code path serves every
size, and `evaluator.hops` blocks its BFS internally. Thus a correct small
run is the verification, and pubmed is the largest of them.

`--graph` is REQUIRED and it has no default. It named `com_youtube` until
2026-08-22, thus a bare run loaded 1.13M nodes with nothing to bound it.

Run from the repo root (fdmap/):

    .venv/bin/python experiments/large-graph-node2vec/bench_node2vec_1M.py \
        --graph pubmed
    .venv/bin/python experiments/large-graph-node2vec/bench_node2vec_1M.py \
        --graph com_youtube          # the sized run. 1.13M nodes, ~1 hour
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

import evaluator as ev

SNAP = {"com_youtube": "com_youtube/com-youtube.ungraph.txt",
        "as_skitter": "as_skitter/as-skitter.txt",
        "roadnet_ca": "roadnet_ca/roadNet-CA.txt"}
PROTOCOL = "n2v1m"

# The smoke graphs. They come from `evaluator.load_graph` and its dataset
# registry, and not from a SNAP file. They are for development and for
# verification; they are not the sized graphs this script measures.
SMALL = ("cora", "pubmed")


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


def load_small(graph, seed):
    """The `--graph cora` smoke path. Returns `(A, n, t_load)`, the same
    shape as `load()`, via `evaluator.load_graph` and its dataset registry.
    """
    t = time.perf_counter()
    A, n, _ = ev.load_graph(graph, seed=seed)
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


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # REQUIRED, and it has NO default. The default was `com_youtube` until
    # 2026-08-22. That graph has 1,134,890 nodes, thus a bare run of this
    # script loaded it, and there is no `--max-nodes` here to bound it. A
    # machine without the memory then swaps or dies, and the user asked for
    # no such run. An explicit name costs one word and it removes the trap.
    ap.add_argument("--graph", required=True,
                    choices=list(SNAP) + list(SMALL),
                    help="the graph. The SNAP names are LARGE "
                         "(com_youtube is 1.13M nodes); cora and pubmed "
                         "are the small verification graphs")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--walks", type=int, default=5)
    ap.add_argument("--walk-len", type=int, default=20)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lp-pairs", type=int, default=None,
                    help="link-prediction positive pairs (protocol "
                         "n2v1m default: 40,000; unset uses the protocol)")
    ap.add_argument("--sp-sources", type=int, default=None,
                    help="hop-distance source nodes (protocol n2v1m "
                         "default: 200; unset uses the protocol)")
    ap.add_argument("--sp-pairs", type=int, default=None,
                    help="hop-distance pairs (protocol n2v1m default: "
                         "20,000; unset uses the protocol)")
    args = ap.parse_args()

    from gensim.models import Word2Vec

    if args.graph in SNAP:
        A, n, t_load = load(args.graph)
    else:
        A, n, t_load = load_small(args.graph, args.seed)
    print(f"[n2v-1M] {args.graph}: n={n:,} nodes, {A.nnz//2:,} undirected "
          f"edges, avg degree {A.nnz/n:.2f} (load {t_load:.1f}s)", flush=True)
    print(f"[n2v-1M] settings: dim={args.dim}, {args.walks} walks x "
          f"{args.walk_len} steps, window={args.window}, "
          f"{args.epochs} epochs, {args.workers} workers", flush=True)

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

    # ---- evaluation (evaluator, protocol n2v1m) ---------------------------
    # A keyword left unset takes the protocol's value; passing one explicitly
    # would set `protocol_modified` and break comparability to the baseline.
    lp_kw = {} if args.lp_pairs is None else {"max_pairs": 2 * args.lp_pairs}
    da_kw = {}
    if args.sp_sources is not None:
        da_kw["n_sources"] = args.sp_sources
    if args.sp_pairs is not None:
        da_kw["n_pairs"] = args.sp_pairs

    t = time.perf_counter()
    lp = ev.link_prediction(A, Z, protocol=PROTOCOL, seed=args.seed, **lp_kw)
    print(f"[n2v-1M] link prediction on {lp.sizes['pairs']:,} pairs "
          f"({time.perf_counter()-t:.1f}s)", flush=True)
    print(f"  accuracy : {lp.scores['accuracy']:.4f}")
    print(f"  precision: {lp.scores['precision']:.4f}")
    print(f"  recall   : {lp.scores['recall']:.4f}")
    print(f"  f1_score : {lp.scores['f1_score']:.4f}")
    print(f"  auc      : {lp.scores['auc']:.4f}", flush=True)

    t = time.perf_counter()
    da = ev.dist_approx(A, Z, protocol=PROTOCOL, seed=args.seed, **da_kw)
    print(f"[n2v-1M] hop distances for {da.sizes['n_pairs']:,} pairs from "
          f"{da.cfg['n_sources']} sources ({time.perf_counter()-t:.1f}s), "
          f"hops {da.sizes['hop_min']:.0f}..{da.sizes['hop_max']:.0f}",
          flush=True)
    print(f"  {'model':>14s} {'MAE':>8s} {'MRE':>8s} {'RMSE':>8s} "
          f"{'R2':>8s} {'exact':>8s}", flush=True)
    for name in da.scores:
        s = da.scores[name]
        print(f"  {name:>14s} {s['mae']:>8.3f} {s['mre']:>8.3f} "
              f"{s['rmse']:>8.3f} {s['r2']:>8.3f} {s['exact']:>8.1%}",
              flush=True)

    modified = lp.protocol_modified or da.protocol_modified
    print(f"[n2v-1M] protocol={PROTOCOL} protocol_modified={modified}",
          flush=True)
    print(f"[n2v-1M] peak RSS for the whole run: {rss_mb():.0f} MB",
          flush=True)


if __name__ == "__main__":
    main()
