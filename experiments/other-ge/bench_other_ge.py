#!/bin/env python3
"""bench_other_ge.py -- node2vec and Poincare, on the same tasks as fodined.

This is the baseline experiment. `fodined` is a force-directed embedding.
This script runs two other methods on the same graphs and the same tasks, so
that the numbers of fodined have something to stand against. The tasks
themselves live in `evaluator`, under the protocol name `otherge`: that
protocol is a frozen record of what this script did, thus every number below
reproduces `results_2026-08-14.log` for the same seed.

The two methods
===============
node2vec  A random walk on the graph gives a sequence of nodes. Word2Vec
          (skip-gram with negative sampling) then treats that sequence as a
          sentence. Two nodes that appear near each other in many walks get
          embeddings that are near each other. The space is Euclidean.

          The walks here use p = q = 1. With those values the second-order
          walk of node2vec becomes a first-order uniform walk, thus this is
          node2vec in its DeepWalk configuration. The reason is speed: a
          uniform walk is one vectorized NumPy operation for all the walks
          together, and a second-order walk needs a Python loop over every
          step. Set `--p` and `--q` away from 1.0 to get the slow, true
          second-order walk.

Poincare  The embedding goes into hyperbolic space (the Poincare ball), and
          not into Euclidean space. The volume of hyperbolic space grows
          exponentially with the radius, and the number of nodes of a tree
          also grows exponentially with the depth. Thus a tree fits into few
          hyperbolic dimensions. This is the method to beat on a tree, and
          it is the reason this experiment includes two trees.

          The Poincare ball needs its OWN distance: `dist_approx` below gets
          `metric="poincare"` for this method. A Euclidean distance in the
          Poincare ball has no meaning.

The graphs
==========
    wordnet         82,144 nouns, the hypernym hierarchy of WordNet 3.0.
                    A tree, and the classic benchmark of the Poincare paper.
    ncbi_taxonomy   2,937,016 nodes, the NCBI taxonomy. A true tree: the
                    file gives one parent for each node.
    cora            2,708 nodes. A citation graph, not a tree.
    pubmed          19,717 nodes. A citation graph, not a tree.

The two trees are large, and `gensim` runs on the CPU. Thus `--max-nodes`
takes ONE subtree of about that size, and the whole tree stays on the disk.
`evaluator.load_graph` (via `fodiwalk.make_graph.datasets`) picks the subtree
for a tree and a BFS ball for anything else, and both stay connected, thus
the hop distances stay meaningful.

The tasks (`evaluator`, protocol `otherge`)
============================================
1. Link prediction (`evaluator.link_prediction`). A random forest predicts
   if two nodes have an edge. The feature of a pair is the Hadamard product
   of the two embeddings. The report gives accuracy, precision, recall, F1,
   and AUC.
2. Shortest path distance approximation (`evaluator.dist_approx`). A random
   forest and an MLP predict the hop distance from the distance of the two
   embeddings. Each method uses its OWN distance: Euclidean for node2vec,
   Poincare for Poincare (the `metric` keyword).
3. Speed. The time to build the walks, and the time to train.

Run from the repo root (fdmap/):

    .venv/bin/python experiments/other-ge/bench_other_ge.py
    .venv/bin/python experiments/other-ge/bench_other_ge.py \
        --graphs cora,wordnet --methods node2vec --max-nodes 5000
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import evaluator as ev

GRAPHS = ("cora", "pubmed", "wordnet", "ncbi_taxonomy")
METHODS = ("node2vec", "poincare")
PROTOCOL = "otherge"


# ---------------------------------------------------------------------------
# The embedding methods
# ---------------------------------------------------------------------------
def uniform_walks(A, n, n_walks, walk_len, seed):
    """`n_walks` uniform random walks from every node, all at the same time.

    Every walk takes one step in each iteration, thus the loop runs
    `walk_len` times and not `n * n_walks * walk_len` times. A node with no
    neighbour stays where it is.
    """
    rng = np.random.default_rng(seed)
    deg = np.diff(A.indptr)
    cur = np.tile(np.arange(n), n_walks)
    walks = np.empty((cur.size, walk_len), dtype=np.int64)
    walks[:, 0] = cur
    for t in range(1, walk_len):
        d = deg[cur]
        off = (rng.random(cur.size) * np.maximum(d, 1)).astype(np.int64)
        nxt = A.indices[A.indptr[cur] + off]
        cur = np.where(d > 0, nxt, cur)
        walks[:, t] = cur
    return walks


def second_order_walks(A, n, n_walks, walk_len, p, q, seed):
    """The true node2vec walk. It is slow, thus it is not the default.

    The probability of the next node depends on the PREVIOUS node: 1/p to
    return, 1 for a common neighbour, and 1/q for a node further away.
    """
    rng = np.random.default_rng(seed)
    walks = []
    nbr = [A.indices[A.indptr[i]:A.indptr[i + 1]] for i in range(n)]
    for _ in range(n_walks):
        for start in range(n):
            w = [start]
            while len(w) < walk_len:
                cur = w[-1]
                cands = nbr[cur]
                if cands.size == 0:
                    break
                if len(w) == 1:
                    w.append(int(rng.choice(cands)))
                    continue
                prev = w[-2]
                prev_nbr = nbr[prev]
                wgt = np.ones(cands.size)
                wgt[cands == prev] = 1.0 / p
                far = ~np.isin(cands, prev_nbr) & (cands != prev)
                wgt[far] = 1.0 / q
                wgt /= wgt.sum()
                w.append(int(rng.choice(cands, p=wgt)))
            walks.append(w)
    return walks


def embed_node2vec(A, n, args):
    """Random walks + Word2Vec. Returns `(Z, timings)`."""
    from gensim.models import Word2Vec

    t = time.perf_counter()
    if args.p == 1.0 and args.q == 1.0:
        walks = uniform_walks(A, n, args.walks, args.walk_len, args.seed)
        sents = [[str(x) for x in w] for w in walks]
    else:
        sents = [[str(x) for x in w] for w in second_order_walks(
            A, n, args.walks, args.walk_len, args.p, args.q, args.seed)]
    t_walk = time.perf_counter() - t

    t = time.perf_counter()
    model = Word2Vec(sents, vector_size=args.dim, window=args.window,
                     min_count=0, sg=1, negative=5, workers=args.workers,
                     epochs=args.epochs_n2v, seed=args.seed)
    t_train = time.perf_counter() - t

    Z = np.zeros((n, args.dim), dtype=np.float32)
    for i in range(n):
        k = str(i)
        if k in model.wv:
            Z[i] = model.wv[k]
    return Z, {"walk_s": t_walk, "train_s": t_train}


def embed_poincare(A, n, args):
    """gensim's PoincareModel on the edges. Returns `(Z, timings)`.

    The model wants relations, thus each edge becomes one `(u, v)` pair. The
    result lives in the Poincare ball: every vector has a norm below 1, and
    the distance is NOT Euclidean. `evaluator.dist_approx(..., metric=
    "poincare")` is the correct distance for this `Z`.
    """
    from gensim.models.poincare import PoincareModel

    up = sp.triu(A, k=1).tocoo()
    relations = [(str(u), str(v)) for u, v in zip(up.row, up.col)]
    t = time.perf_counter()
    model = PoincareModel(relations, size=args.dim, negative=args.negative,
                          seed=args.seed)
    model.train(epochs=args.epochs_poincare, print_every=10 ** 9)
    t_train = time.perf_counter() - t

    Z = np.zeros((n, args.dim), dtype=np.float64)
    for i in range(n):
        k = str(i)
        if k in model.kv:
            Z[i] = model.kv[k]
    return Z, {"walk_s": 0.0, "train_s": t_train}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", default=",".join(GRAPHS))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--max-nodes", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    # node2vec
    ap.add_argument("--walks", type=int, default=10)
    ap.add_argument("--walk-len", type=int, default=40)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--epochs-n2v", type=int, default=5)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=1.0)
    # poincare
    ap.add_argument("--epochs-poincare", type=int, default=50)
    ap.add_argument("--negative", type=int, default=10)
    args = ap.parse_args()

    graphs = [g.strip() for g in args.graphs.split(",") if g.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"[other-ge] graphs={graphs} methods={methods} dim={args.dim} "
          f"max_nodes={args.max_nodes} seed={args.seed}", flush=True)
    if args.p != 1.0 or args.q != 1.0:
        print(f"[other-ge] node2vec uses the SLOW second-order walk "
              f"(p={args.p}, q={args.q})", flush=True)

    rows = []
    for gname in graphs:
        t = time.perf_counter()
        A, n, ginfo = ev.load_graph(gname, max_nodes=args.max_nodes,
                                    seed=args.seed)
        print(f"\n=== {gname} === n={n:,}, {ginfo['n_edges']:,} undirected "
              f"edges, avg degree {ginfo['avg_degree']:.2f}, load "
              f"{time.perf_counter()-t:.1f}s", flush=True)

        for m in methods:
            try:
                t = time.perf_counter()
                Z, tm = (embed_node2vec if m == "node2vec"
                         else embed_poincare)(A, n, args)
                total = time.perf_counter() - t

                lp = ev.link_prediction(A, Z, protocol=PROTOCOL,
                                        seed=args.seed)
                da_kw = {"metric": "poincare"} if m == "poincare" else {}
                da = ev.dist_approx(A, Z, protocol=PROTOCOL, seed=args.seed,
                                    **da_kw)

                rows.append(dict(graph=gname, method=m, n=n, total_s=total,
                                 lp=lp, da=da))
                print(f"  {m:9s} embed {total:7.1f}s "
                      f"(walk {tm['walk_s']:.1f}s + train {tm['train_s']:.1f}s), "
                      f"{n/total:8.0f} nodes/s", flush=True)
                print(f"            link pred: acc {lp.scores['accuracy']:.4f}  "
                      f"prec {lp.scores['precision']:.4f}  "
                      f"rec {lp.scores['recall']:.4f}  "
                      f"f1_score {lp.scores['f1_score']:.4f}  auc {lp.scores['auc']:.4f}",
                      flush=True)
                print(f"            hop regr ({da.sizes['n_pairs']} pairs, "
                      f"hops {da.sizes['hop_min']:.0f}.."
                      f"{da.sizes['hop_max']:.0f}): "
                      f"base MAE {da.scores['baseline']['mae']:.3f} | "
                      f"rf MAE {da.scores['rf']['mae']:.3f} "
                      f"R2 {da.scores['rf']['r2']:.3f} "
                      f"exact {da.scores['rf']['exact']:.1%} | "
                      f"mlp MAE {da.scores['mlp']['mae']:.3f} "
                      f"R2 {da.scores['mlp']['r2']:.3f} "
                      f"exact {da.scores['mlp']['exact']:.1%}", flush=True)
            except Exception as exc:                    # noqa: BLE001
                print(f"  {m:9s} FAILED: {type(exc).__name__}: {exc}",
                      flush=True)

    # ---- summary ---------------------------------------------------------
    print("\n=== summary ===", flush=True)
    print(f"{'graph':>14s} {'method':>9s} {'n':>8s} {'embed_s':>8s} "
          f"{'acc':>6s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'auc':>6s} "
          f"{'hopMAE':>7s} {'hopR2':>6s}", flush=True)
    for r in rows:
        lp, da = r["lp"], r["da"]
        print(f"{r['graph']:>14s} {r['method']:>9s} {r['n']:>8,} "
              f"{r['total_s']:>8.1f} {lp.scores['accuracy']:>6.3f} "
              f"{lp.scores['precision']:>6.3f} {lp.scores['recall']:>6.3f} "
              f"{lp.scores['f1_score']:>6.3f} {lp.scores['auc']:>6.3f} "
              f"{da.scores['mlp']['mae']:>7.3f} "
              f"{da.scores['mlp']['r2']:>6.3f}", flush=True)


if __name__ == "__main__":
    main()
