#!/bin/env python3
"""reference/other_ge.py -- the FROZEN reference of the `otherge` protocol.

Verbatim copies of four functions of
`experiments/other-ge/bench_other_ge.py`, taken on 2026-08-22, BEFORE the
rewrite of that script removed them. `tests/test_parity.py` scores an
embedding with these functions and with `evaluator`, and the two must
agree to 1e-12 (PRD section 9, tests P1 to P4).

DO NOT TIDY THIS FILE. A reference that is improved is not a reference.
A change here invalidates every parity number of the campaign. The only
edit this file ever accepts is a NEW function, copied the same way, for a
new parity test.

Provenance, one for each function:
    sample_non_edges       bench_other_ge.py, the pair sampler
    task_link_prediction   bench_other_ge.py, the LP protocol
    task_sp_regression     bench_other_ge.py, the hop protocol
    poincare_distance      bench_other_ge.py, the hyperbolic distance

The imports moved to the top of this file. Nothing else changed.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


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


def poincare_distance(u, v):
    """The distance of two points of the Poincare ball, for whole arrays."""
    du = np.sum(u * u, axis=1)
    dv = np.sum(v * v, axis=1)
    duv = np.sum((u - v) ** 2, axis=1)
    x = 1.0 + 2.0 * duv / np.maximum((1.0 - du) * (1.0 - dv), 1e-12)
    return np.arccosh(np.maximum(x, 1.0 + 1e-12))
