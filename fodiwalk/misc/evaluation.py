#!/bin/env python3
"""misc.evaluation -- the two measurements of an embedding.

Measurement, and not engine. The force law, the plan and the loop do not
import this file, and it does not import them.

  `link_prediction`  does the geometry hold the adjacency? A classifier
                     reads the Hadamard product of the two vectors of a
                     pair and it must say whether the pair has an edge.
  `hop_sample` +     does the geometry hold the DISTANCE? A regressor
  `task_hop`         reads the two vectors of a pair and it must say how
                     many hops apart the two nodes are. The pairs come
                     from a BFS, thus the measurement is OUT OF SAMPLE --
                     most of them are pairs that `D` never held.

Both are a SAMPLE and not a census: a graph of a million nodes has
millions of edges, and a balanced sample measures the same property at a
constant cost.

The random generator arrives as an argument, and this module never makes
one. That is what keeps a run reproducible.

Provenance, verbatim moves:
  * `sample_positives`, `sample_negatives`, `edge_features`, `classify`,
    `link_prediction` from `fodined/link_prediction.py`;
  * `hop_sample`, `task_hop` from `experiments/fdwalk/bench_fdwalk.py`.
Only the imports changed.

Import note: `sample_negatives` needs `augment_graph.far_pairs`, thus this
module imports one stage sideways. There is no cycle -- `augment_graph`
imports nothing of `misc` -- and the alternative is a second copy of the
rejection sampler.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, mean_squared_error,
                             r2_score, mean_absolute_error,
                             mean_absolute_percentage_error)

from ..augment_graph.far_pairs import sample_far_pairs


# ---------------------------------------------------------------------------
# Link prediction
# ---------------------------------------------------------------------------
def sample_positives(A, max_count: int, rng):
    """At most `max_count` of the ORIGINAL undirected edges, once each.

    The upper triangle gives each edge one time. A graph with more edges than
    `max_count` gets a sample without replacement, thus no edge arrives two
    times.
    """
    pos = np.column_stack(sp.triu(A, k=1).nonzero())
    if pos.shape[0] <= max_count:
        return pos
    return pos[rng.choice(pos.shape[0], max_count, replace=False)]


def sample_negatives(A, n: int, count: int, rng):
    """`count` random pairs that have NO edge.

    `sample_far_pairs` rejects every pair that the matrix it receives holds.
    With `A` that means "connected", thus the result is exactly the set of
    unconnected pairs, and no pair arrives two times.

    It works on sorted composite keys and it needs no Python set. The older
    loop used a set, which cost 1.8 GB at a million nodes. See
    `experiments/large-graph-node2vec/REPORT.md`.
    """
    return sample_far_pairs(n, count, A, rng)


def edge_features(Z, pos, neg):
    """`(X, y)`: the Hadamard product of the two endpoints, and the label.

    The elementwise product is the standard edge feature of an embedding, and
    it is symmetric in `(u, v)`, as an undirected edge feature must be.
    """
    pairs = np.vstack([pos, neg])
    X = Z[pairs[:, 0]] * Z[pairs[:, 1]]
    y = np.concatenate([np.ones(pos.shape[0]), np.zeros(neg.shape[0])])
    return X, y


def classify(X, y, seed: int, test_size: float = 0.2, n_estimators: int = 200):
    """Fit a random forest on a part of the pairs, score it on the rest.

    Returns `(scores, sizes)`. `stratify` holds the balance of the classes in
    both parts, thus the accuracy of the test part stays comparable.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return ({"accuracy": accuracy_score(y_test, y_pred),
             "precision": precision_score(y_test, y_pred),
             "recall": recall_score(y_test, y_pred),
             "f1_score": f1_score(y_test, y_pred),
             "auc": roc_auc_score(y_test, y_prob)},
            {"train": X_train.shape[0], "test": X_test.shape[0]})


def link_prediction(Z, A, n: int, max_pairs: int, rng, seed: int,
                    test_size: float = 0.2, n_estimators: int = 200):
    """The whole test. Returns `(scores, info)`.

    `max_pairs` counts the positives AND the negatives together, thus each
    class gets one half of it.

    `info` holds the numbers that a caller prints: `positives`, `negatives`,
    `edges` (how many the graph has), `pairs`, `train`, and `test`.
    """
    pos = sample_positives(A, max_pairs // 2, rng)
    neg = sample_negatives(A, n, pos.shape[0], rng)
    X, y = edge_features(Z, pos, neg)
    scores, sizes = classify(X, y, seed, test_size, n_estimators)
    return scores, {"positives": pos.shape[0], "negatives": neg.shape[0],
                    "edges": A.nnz // 2, "pairs": X.shape[0], **sizes}


# ---------------------------------------------------------------------------
# The hop-distance regression
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# The evaluation
# ---------------------------------------------------------------------------
def hop_sample(A, n, rng, n_sources, n_pairs, gap_csr=None):
    """Exact hop distances for a sample of pairs. Returns `(u, v, d, h2)`.

    `shortest_path` gives a DENSE `(len(indices), n)` array, thus the
    sources go in blocks: one block costs `block * n * 8` bytes. A full
    matrix at 1.13M nodes is 10 TB, and one block near 200 MB is the rule
    that `../bench_shortest_path_gemsec.py` measured.

    `gap_csr` holds the walk gap of every stored pair. While a row of the
    BFS is in the memory, this function reads EVERY stored partner of that
    row and it compares the gap to the true distance. That is the
    measurement of H2, and it gives thousands of pairs. A comparison
    against the random sample above gives only the pairs that both sets
    hold, which was 140 of 17,769 in the first run.

    THIS FUNCTION DOES NOT FILTER HOP-1 PAIRS. The one guard is `d > 0`
    (self-pairs and unreachable pairs only); a true edge (`d == 1`, the
    easy case for `task_hop`) is returned like any other pair. Every
    CALLER in this repository filters it out afterward -- `tests/harness.py`
    keeps `d >= hop_min` (2 by default) before scoring. A caller that skips
    that filter silently scores an easier problem, in silence, the same
    way an unfiltered NCBI star silently scores a trivial one.
    """
    src = rng.choice(n, size=min(n_sources, n), replace=False)
    per = max(1, n_pairs // src.size)
    block = max(1, int(200e6 / (n * 8)))
    us, vs, ds = [], [], []
    gaps, trues = [], []
    for s in range(0, src.size, block):
        blk = src[s:s + block]
        hop = shortest_path(A, method="D", unweighted=True, indices=blk)
        for i, u in enumerate(blk):
            tgt = rng.choice(n, size=min(per * 2, n), replace=False)
            d = hop[i, tgt]
            ok = np.isfinite(d) & (d > 0)
            tgt, d = tgt[ok][:per], d[ok][:per]
            us.append(np.full(tgt.size, u))
            vs.append(tgt)
            ds.append(d)
            if gap_csr is not None:
                lo, hi = gap_csr.indptr[u], gap_csr.indptr[u + 1]
                part = gap_csr.indices[lo:hi]
                g = gap_csr.data[lo:hi]
                t = hop[i, part]
                fin = np.isfinite(t)
                gaps.append(g[fin])
                trues.append(t[fin])
        del hop
    h2 = None
    if gaps and sum(x.size for x in gaps):
        g = np.concatenate(gaps)
        t = np.concatenate(trues)
        h2 = {"pairs": int(g.size), "exact": float(np.mean(g == t)),
              "over": float(np.mean(g > t)), "under": float(np.mean(g < t)),
              "mae": float(np.mean(np.abs(g - t)))}
    return (np.concatenate(us), np.concatenate(vs),
            np.concatenate(ds).astype(np.float64), h2)


def task_hop(Z, u, v, d, seed, feature="distance", n_estimators: int = 100,
             hidden=(256, 128), early_stopping: bool = False):
    """Can a model read the hop distance out of two embeddings?

    Two feature sets, because this repository holds two protocols and a
    number must be comparable to the baseline that it claims to beat:

      'distance' the Euclidean distance alone, thus ONE feature. This is
                 the protocol of `fodined/modular.py` -- CORRECTED
                 2026-08-21, see `dev-docs/CATALOG.md` -- and of
                 `../other-ge/bench_other_ge.py`, whose own docstring
                 states the two are the same feature on purpose. The
                 node2vec and Poincaré numbers use it too.
      'vector'   `|Z[u] - Z[v]|`, thus `n_dim` features. More informative,
                 and NOT the feature of any baseline recorded in this
                 repository -- a number under this mode is comparable only
                 to another number under this mode, never to a published
                 fodined or other-ge figure.

    A model with 128 features can win only because it has more of them, thus
    the two rows must not be mixed.

    The three model arguments carry the same rule. The defaults are the
    campaign's: `RandomForestRegressor(100)` and `MLPRegressor((256, 128))`
    with no early stopping. `fodined/modular.py` uses `200`, `(128, 64)`
    and `early_stopping=True`. A number is comparable to the baseline that
    used the SAME settings, and to no other.
    """
    if feature == "distance":
        X = np.linalg.norm(Z[u] - Z[v], axis=1)[:, None]
    else:
        X = np.abs(Z[u] - Z[v])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, d, test_size=0.2, random_state=seed)
    rows = []

    def score(name, pred, secs=None):
        rows.append((name, mean_absolute_error(y_te, pred),
                     mean_absolute_percentage_error(y_te, pred),
                     np.sqrt(mean_squared_error(y_te, pred)),
                     r2_score(y_te, pred),
                     float(np.mean(np.rint(pred) == y_te)), secs))

    score("mean baseline", np.full(y_te.size, y_tr.mean()))
    t = time.time()
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed,
                               n_jobs=-1)
    rf.fit(X_tr, y_tr)
    score("random forest", rf.predict(X_te), time.time() - t)
    t = time.time()
    sc = StandardScaler().fit(X_tr)
    mlp = MLPRegressor(hidden_layer_sizes=tuple(hidden), max_iter=300,
                       early_stopping=early_stopping, random_state=seed)
    mlp.fit(sc.transform(X_tr), y_tr)
    score("MLP", mlp.predict(sc.transform(X_te)), time.time() - t)
    return rows


# H2 is measured inside `hop_sample`, where a row of the BFS is already in
# the memory. The gap of a walk is an UPPER BOUND of the hop distance: a
# walk that goes from `u` to `v` in `t` steps proves a path of `t` steps.
# Thus "under" must be 0.0, and any other value is a defect of this code.
