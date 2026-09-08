#!/bin/env python3
"""reference/fodiwalk_eval.py -- the FROZEN reference of the `fodiwalk`
protocols.

Verbatim copies of `hop_sample` and `task_hop` of
`fodiwalk/misc/evaluation.py`, taken on 2026-08-22. Parity tests P6a and
P6b score an embedding with these and with `evaluator`, and the two must
agree to 1e-12 (PRD section 9).

WHY A COPY, when the original is not deleted by this campaign: the file it
comes from belongs to ANOTHER active session, which edited it on
2026-08-22 (it changed the default of `task_hop(feature=...)` from
"vector" to "distance"). A reference that a second party can edit is not a
reference. This copy pins the behaviour that the parity numbers describe.

The default of `feature` here is "vector", as it was when the `fodiwalk`
protocol cells were read. That default is NOT the protocol: every caller
of the original -- `fodiwalk/tests/harness.py:63`,
`experiments/fodiwalk/bench_fodiwalk.py:245`,
`experiments/fdwalk/bench_fdwalk.py:856` -- loops over BOTH
`("distance", "vector")` and records both. Thus the baseline is a SWEEP of
two rows, and `evaluator` splits it into two protocol names,
`fodiwalk_dist` and `fodiwalk_vec` (PRD block B1).

DO NOT TIDY THIS FILE. See `other_ge.py`.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.sparse.csgraph import shortest_path
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score,
                             mean_absolute_error,
                             mean_absolute_percentage_error)


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


def task_hop(Z, u, v, d, seed, feature="vector", n_estimators: int = 100,
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
