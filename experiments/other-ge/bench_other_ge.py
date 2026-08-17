#!/bin/env python3
"""bench_other_ge.py -- node2vec and Poincare, on the same tasks as fodined.

This is the baseline experiment. `fodined` is a force-directed embedding.
This script runs two other methods on the same graphs and the same tasks, so
that the numbers of fodined have something to stand against.

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
The subtree keeps the depth of the original tree. `--truncate bfs` gives the
older behaviour, but read the warning in `_pick_subtree` first: on NCBI a BFS
returns a star, and every method then reaches a perfect score.

The tasks (the same as `fodined/modular.py`)
============================================
1. Link prediction. A random forest predicts if two nodes have an edge. The
   feature of a pair is the Hadamard product of the two embeddings. The
   report gives accuracy, precision, recall, F1, and AUC.
2. Shortest path distance approximation. A random forest and an MLP predict
   the hop distance from the distance of the two embeddings. Each method
   uses its OWN distance: Euclidean for node2vec, and the Poincare distance
   for Poincare. A Euclidean distance in the Poincare ball has no meaning.
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

GRAPHS = ("cora", "pubmed", "wordnet", "ncbi_taxonomy")
METHODS = ("node2vec", "poincare")


# ---------------------------------------------------------------------------
# Loading. Every loader returns one edge array of shape (m, 2), 0-based.
# ---------------------------------------------------------------------------
def _edges_cora():
    raw = np.loadtxt(os.path.join(ROOT, "data_cache/cora/cora.cites"),
                     dtype=np.int64)
    return raw


def _edges_pubmed():
    path = os.path.join(ROOT, "data_cache/pubmed/Pubmed-Diabetes/data/"
                              "Pubmed-Diabetes.DIRECTED.cites.tab")
    src, dst = [], []
    with open(path) as f:
        f.readline(), f.readline()
        for line in f:
            p = line.split("\t")
            src.append(int(p[1].split(":")[1]))
            dst.append(int(p[3].split(":")[1]))
    return np.column_stack([src, dst]).astype(np.int64)


def _edges_wordnet():
    """The hypernym edges of the WordNet 3.0 nouns.

    Each line of `data.noun` is one synset. The pointer list holds the
    symbol `@` for a hypernym, and `@i` for an instance hypernym. The
    offset of the target follows the symbol. The offsets become the node
    ids, thus no name table is necessary.
    """
    path = os.path.join(ROOT, "data_cache/wordnet/dict/data.noun")
    src, dst = [], []
    with open(path, encoding="latin-1") as f:
        for line in f:
            if line.startswith("  "):                 # the licence header
                continue
            head, _, _ = line.partition("|")
            parts = head.split()
            offset = int(parts[0])
            # ... w_cnt words ... p_cnt then the pointers.
            i = 3
            w_cnt = int(parts[i], 16)
            i += 1 + 2 * w_cnt                        # word + lex_id, each
            p_cnt = int(parts[i])
            i += 1
            for _ in range(p_cnt):
                sym, target = parts[i], int(parts[i + 1])
                if sym in ("@", "@i"):
                    src.append(offset)
                    dst.append(target)
                i += 4                                # sym, off, pos, st
    return np.column_stack([src, dst]).astype(np.int64)


def _edges_ncbi():
    """The NCBI taxonomy tree, from `nodes.dmp`: `tax_id | parent_tax_id`."""
    path = os.path.join(ROOT, "data_cache/ncbi_taxonomy/nodes.dmp")
    src, dst = [], []
    with open(path) as f:
        for line in f:
            p = line.split("\t|\t", 2)
            a, b = int(p[0]), int(p[1])
            if a != b:                                # the root is its parent
                src.append(a)
                dst.append(b)
    return np.column_stack([src, dst]).astype(np.int64)


LOADERS = {"cora": _edges_cora, "pubmed": _edges_pubmed,
           "wordnet": _edges_wordnet, "ncbi_taxonomy": _edges_ncbi}


def _pick_subtree(e, n, max_nodes, seed):
    """Node ids of ONE subtree of about `max_nodes` nodes.

    Why not a BFS from the biggest node: NCBI has nodes with an enormous
    number of children, thus a BFS spends the whole budget on the children
    of one hub. The first version of this function did that, and it returned
    a STAR: one node of degree 19,999, 19,999 leaves, and a distance of
    exactly 2 between every pair that has no edge. Every method then reaches
    a perfect score, and the measurement says nothing about a tree.

    A subtree keeps the depth and the branching of the original tree. The
    search starts at a random node and it goes UP to the parent until the
    subtree is large enough.

    `e` holds the directed edges as `(child, parent)`.
    """
    rng = np.random.default_rng(seed)
    parent = np.full(n, -1, dtype=np.int64)
    parent[e[:, 0]] = e[:, 1]                  # the first parent wins (DAG)
    order = np.argsort(e[:, 1], kind="stable")
    ch_of = e[order, 0]
    starts = np.searchsorted(e[order, 1], np.arange(n))
    ends = np.searchsorted(e[order, 1], np.arange(n), side="right")

    def collect(root, cap):
        out, frontier = [root], [root]
        while frontier and len(out) < cap:
            nxt = []
            for u in frontier:
                kids = ch_of[starts[u]:ends[u]]
                for k in kids:
                    if len(out) >= cap:
                        break
                    out.append(int(k))
                    nxt.append(int(k))
            frontier = nxt
        return out

    node = int(rng.integers(0, n))
    for _ in range(200):                       # a guard against a long climb
        got = collect(node, max_nodes + 1)
        if len(got) >= max_nodes // 2 or parent[node] < 0:
            return got[:max_nodes]
        node = int(parent[node])
    return collect(node, max_nodes)


def load_csr(name, max_nodes=0, seed=42, truncate="subtree"):
    """Edge list -> a symmetric CSR of 1.0, with contiguous ids 0..n-1.

    `truncate` chooses how to make a large graph smaller:
      'subtree' -- one subtree of about `max_nodes` nodes. It keeps the
                   depth. Use this for a tree.
      'bfs'     -- a BFS from the node of the highest degree. It keeps the
                   nodes near one hub, and it can return a star on a graph
                   with a large fan-out. See `_pick_subtree`.
    Both keep the result connected, thus the hop distances stay meaningful.
    """
    e = LOADERS[name]()
    _, flat = np.unique(e.ravel(), return_inverse=True)
    e = flat.reshape(-1, 2)
    e = e[e[:, 0] != e[:, 1]]
    n = int(e.max()) + 1
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    A = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    A.sort_indices()

    if max_nodes and n > max_nodes:
        if truncate == "subtree":
            order = _pick_subtree(e, n, max_nodes, seed)
        else:
            start = int(np.argmax(np.diff(A.indptr)))
            seen = np.zeros(n, dtype=bool)
            seen[start] = True
            order = [start]
            frontier = np.array([start])
            while len(order) < max_nodes and frontier.size:
                nxt = np.unique(np.concatenate(
                    [A.indices[A.indptr[u]:A.indptr[u + 1]] for u in frontier]))
                nxt = nxt[~seen[nxt]]
                nxt = nxt[:max_nodes - len(order)]
                seen[nxt] = True
                order.extend(nxt.tolist())
                frontier = nxt
        keep = np.sort(np.asarray(order))
        A = A[keep][:, keep].tocsr()
        A.eliminate_zeros()
        n = A.shape[0]
    return A, n


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
    the distance is NOT Euclidean. `poincare_distance` below is the correct
    distance.
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


def poincare_distance(u, v):
    """The distance of two points of the Poincare ball, for whole arrays."""
    du = np.sum(u * u, axis=1)
    dv = np.sum(v * v, axis=1)
    duv = np.sum((u - v) ** 2, axis=1)
    x = 1.0 + 2.0 * duv / np.maximum((1.0 - du) * (1.0 - dv), 1e-12)
    return np.arccosh(np.maximum(x, 1.0 + 1e-12))


# ---------------------------------------------------------------------------
# The tasks
# ---------------------------------------------------------------------------
def sample_non_edges(A, n, count, rng):
    Ac = A.tocoo()
    keys = np.sort(Ac.row.astype(np.int64) * n + Ac.col.astype(np.int64))
    u_all = np.empty(0, np.int64)
    v_all = np.empty(0, np.int64)
    while u_all.size < count:
        draw = (count - u_all.size) * 2 + 1024
        u, v = rng.integers(0, n, draw), rng.integers(0, n, draw)
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


def task_link_prediction(A, n, Z, seed):
    """The protocol of `fodined/modular.py`: Hadamard feature, balanced."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)

    rng = np.random.default_rng(seed)
    pos = np.column_stack(sp.triu(A, k=1).nonzero())
    if pos.shape[0] > 40000:                 # a cap, to hold the time down
        pos = pos[rng.choice(pos.shape[0], 40000, replace=False)]
    neg = sample_non_edges(A, n, pos.shape[0], rng)

    pairs = np.vstack([pos, neg])
    X = Z[pairs[:, 0]] * Z[pairs[:, 1]]
    y = np.concatenate([np.ones(pos.shape[0]), np.zeros(neg.shape[0])])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=seed,
                                 n_jobs=-1).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    prob = clf.predict_proba(Xte)[:, 1]
    return dict(accuracy=accuracy_score(yte, pred),
                precision=precision_score(yte, pred, zero_division=0),
                recall=recall_score(yte, pred, zero_division=0),
                f1=f1_score(yte, pred, zero_division=0),
                auc=roc_auc_score(yte, prob))


def task_sp_regression(A, n, Z, method, seed, n_pairs=20000):
    """Predict the hop distance from the distance of the two embeddings.

    The feature is ONE number: the distance of the pair in the space of the
    method. This is the same feature that `fodined/modular.py` uses now, thus
    the numbers are comparable. Pairs at hop 1 are removed, because a
    neighbour is the easy case and it hides the rest.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score,
                                 mean_absolute_percentage_error)
    import networkit as nk

    rng = np.random.default_rng(seed)
    pairs = sample_non_edges(A, n, n_pairs, rng)

    up = sp.triu(A, k=1).tocoo()
    g = nk.GraphFromCoo(
        (np.ascontiguousarray(up.row, dtype=np.uint64),
         np.ascontiguousarray(up.col, dtype=np.uint64)),
        n=n, weighted=False, directed=False)
    pll = nk.distance.PrunedLandmarkLabeling(g)
    pll.run()
    q = np.fromiter((pll.query(int(a), int(b)) for a, b in pairs),
                    dtype=np.uint64, count=pairs.shape[0])
    ok = q != np.uint64(2 ** 64 - 1)                    # keep reachable pairs
    pairs, y = pairs[ok], q[ok].astype(np.float64)
    keep = y > 1
    pairs, y = pairs[keep], y[keep]
    if y.size < 200:
        return None

    if method == "poincare":
        X = poincare_distance(Z[pairs[:, 0]], Z[pairs[:, 1]])[:, None]
    else:
        X = np.linalg.norm(Z[pairs[:, 0]] - Z[pairs[:, 1]], axis=1)[:, None]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    out = {"n_pairs": int(y.size), "hop_min": float(y.min()),
           "hop_max": float(y.max())}

    def score(tag, pred):
        out[f"{tag}_mae"] = mean_absolute_error(yte, pred)
        out[f"{tag}_mre"] = mean_absolute_percentage_error(yte, pred)
        out[f"{tag}_rmse"] = float(np.sqrt(mean_squared_error(yte, pred)))
        out[f"{tag}_r2"] = r2_score(yte, pred)
        out[f"{tag}_exact"] = float(np.mean(np.rint(pred) == yte))

    score("base", np.full(yte.shape, ytr.mean()))
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=25,
                               random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    score("rf", rf.predict(Xte))
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300,
                       early_stopping=True, random_state=seed)
    mlp.fit(sc.transform(Xtr), ytr)
    score("mlp", mlp.predict(sc.transform(Xte)))
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", default=",".join(GRAPHS))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--max-nodes", type=int, default=20000)
    ap.add_argument("--truncate", default="subtree", choices=("subtree", "bfs"))
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
        A, n = load_csr(gname, args.max_nodes, args.seed, args.truncate)
        print(f"\n=== {gname} === n={n:,}, {A.nnz//2:,} undirected edges, "
              f"avg degree {A.nnz/n:.2f}, load {time.perf_counter()-t:.1f}s",
              flush=True)

        for m in methods:
            try:
                t = time.perf_counter()
                Z, tm = (embed_node2vec if m == "node2vec"
                         else embed_poincare)(A, n, args)
                total = time.perf_counter() - t
                lp = task_link_prediction(A, n, Z, args.seed)
                spr = task_sp_regression(A, n, Z, m, args.seed)
                rows.append(dict(graph=gname, method=m, n=n, total_s=total,
                                 **tm, **{f"lp_{k}": v for k, v in lp.items()},
                                 **({f"sp_{k}": v for k, v in spr.items()}
                                    if spr else {})))
                print(f"  {m:9s} embed {total:7.1f}s "
                      f"(walk {tm['walk_s']:.1f}s + train {tm['train_s']:.1f}s), "
                      f"{n/total:8.0f} nodes/s", flush=True)
                print(f"            link pred: acc {lp['accuracy']:.4f}  "
                      f"prec {lp['precision']:.4f}  rec {lp['recall']:.4f}  "
                      f"f1 {lp['f1']:.4f}  auc {lp['auc']:.4f}", flush=True)
                if spr:
                    print(f"            hop regr ({spr['n_pairs']} pairs, "
                          f"hops {spr['hop_min']:.0f}..{spr['hop_max']:.0f}): "
                          f"base MAE {spr['base_mae']:.3f} | "
                          f"rf MAE {spr['rf_mae']:.3f} R2 {spr['rf_r2']:.3f} "
                          f"exact {spr['rf_exact']:.1%} | "
                          f"mlp MAE {spr['mlp_mae']:.3f} R2 {spr['mlp_r2']:.3f} "
                          f"exact {spr['mlp_exact']:.1%}", flush=True)
            except Exception as exc:                    # noqa: BLE001
                print(f"  {m:9s} FAILED: {type(exc).__name__}: {exc}",
                      flush=True)

    # ---- summary ---------------------------------------------------------
    print("\n=== summary ===", flush=True)
    print(f"{'graph':>14s} {'method':>9s} {'n':>8s} {'embed_s':>8s} "
          f"{'acc':>6s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'auc':>6s} "
          f"{'hopMAE':>7s} {'hopR2':>6s}", flush=True)
    for r in rows:
        print(f"{r['graph']:>14s} {r['method']:>9s} {r['n']:>8,} "
              f"{r['total_s']:>8.1f} {r['lp_accuracy']:>6.3f} "
              f"{r['lp_precision']:>6.3f} {r['lp_recall']:>6.3f} "
              f"{r['lp_f1']:>6.3f} {r['lp_auc']:>6.3f} "
              f"{r.get('sp_mlp_mae', float('nan')):>7.3f} "
              f"{r.get('sp_mlp_r2', float('nan')):>6.3f}", flush=True)


if __name__ == "__main__":
    main()
