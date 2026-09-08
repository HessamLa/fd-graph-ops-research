#!/bin/env python3
"""evaluator.metrics -- the distance registry, and the pair feature registry.

The KEYS are the enum: `cli.py` prints them and `guards.check_metric`
reads them. An `if/elif` chain would hide the names, thus the dicts stay.

`DISTANCES` maps a name to `fn(A, B) -> (k,)` and `FEATURES` maps a name
to `fn(A, B) -> (k, ?)`, where `A` and `B` hold the gathered endpoint
rows. Every function works on whole arrays. There is NO Python loop over
the pairs, at any size.

The Poincare distance is a copy of
`experiments/other-ge/bench_other_ge.py:poincare_distance` (line 354),
the function that made the recorded `gensim` PoincareModel numbers. It
keeps the two `np.maximum` clamps EXACTLY, the `1e-12` epsilons included.
The clamp has one visible effect: two points at the origin give
`arccosh(1 + 1e-12)`, which is `1.41e-6` and not `0.0`. Parity with the
reference wins over the clean zero.

NO normalization happens here, and no knob turns it on. `hadamard` is
`Z[u] * Z[v]` exactly. Checked on 2026-08-22 against all four baselines
(`fodined/link_prediction.edge_features`,
`fodiwalk/misc/evaluation.edge_features`, `bench_other_ge`,
`bench_node2vec_1M`) -- NONE of them normalizes `Z`. A `normalize` field
stood in the first draft of `config.LPCfg` and it was deleted for that
reason.
"""
from __future__ import annotations

import numpy as np

from .config import RECON_MAX_N

__all__ = ["DISTANCES", "FEATURES", "distance", "feature", "knn"]

_EPS = 1e-12


# ---------------------------------------------------------------------------
# The distances
# ---------------------------------------------------------------------------
def _euclidean(A, B):
    """The L2 distance of the two endpoints."""
    return np.linalg.norm(A - B, axis=1)


def _poincare(A, B):
    """The geodesic distance of the Poincare ball,
    `arccosh(1 + 2*|u-v|^2 / ((1-|u|^2)(1-|v|^2)))`. A copy of
    `bench_other_ge.poincare_distance`, clamps included.
    """
    du = np.sum(A * A, axis=1)
    dv = np.sum(B * B, axis=1)
    duv = np.sum((A - B) ** 2, axis=1)
    x = 1.0 + 2.0 * duv / np.maximum((1.0 - du) * (1.0 - dv), 1e-12)
    return np.arccosh(np.maximum(x, 1.0 + 1e-12))


def _cosine(A, B):
    """The cosine distance, `1 - cos(A, B)`. The range is `[0, 2]`.

    A zero row gives a zero denominator, thus the clamp. The benchmark
    scripts leave a zero row for a node that the vocabulary misses; such a
    pair gets 1.0 and it does not give a NaN.
    """
    num = np.sum(A * B, axis=1)
    den = np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1),
                     _EPS)
    return 1.0 - num / den


def _dot(A, B):
    """The inner product. A SIMILARITY and not a distance: a large value
    means a near pair. It is symmetric, thus it satisfies the pair rule, but
    a caller that wants an ordering by nearness must change the sign."""
    return np.sum(A * B, axis=1)


DISTANCES = {
    "euclidean": _euclidean,
    "poincare": _poincare,
    "cosine": _cosine,
    "dot": _dot,
}


# ---------------------------------------------------------------------------
# The pair features
# ---------------------------------------------------------------------------
def _hadamard(A, B):
    """`Z[u] * Z[v]`, the standard edge feature. Symmetric in `(u, v)`."""
    return A * B


def _l1(A, B):
    """`|Z[u] - Z[v]|`, one column for each dimension. Symmetric in
    `(u, v)`. This is the `vector` feature of
    `fodiwalk/misc/evaluation.task_hop`."""
    return np.abs(A - B)


def _concat(A, B):
    """`[Z[u], Z[v]]`, thus `2*d` columns.

    NOT symmetric: `concat(u, v)` and `concat(v, u)` are different rows. An
    undirected pair has no first endpoint, thus the model learns the order
    of the sampler and not the structure of the graph. Choose it for a
    DIRECTED task only.
    """
    return np.concatenate([A, B], axis=1)


def _avg(A, B):
    """`(Z[u] + Z[v]) / 2`, the midpoint. Symmetric in `(u, v)`."""
    return (A + B) / 2.0


FEATURES = {
    "hadamard": _hadamard,
    "l1": _l1,
    "concat": _concat,
    "avg": _avg,
}


# ---------------------------------------------------------------------------
# The accessors
# ---------------------------------------------------------------------------
def _pick(table, name, what):
    try:
        return table[name]
    except KeyError:
        raise ValueError(
            f"unknown {what} {name!r}. Use one of: "
            + ", ".join(sorted(table))) from None


def distance(Z, u, v, metric="euclidean"):
    """The distance of each pair `(u[i], v[i])`. Returns `(k,)`. `u` and
    `v` index into `Z`; `metric` is a key of `DISTANCES`.
    """
    fn = _pick(DISTANCES, metric, "metric")
    return fn(Z[np.asarray(u)], Z[np.asarray(v)])


def feature(Z, u, v, kind="hadamard"):
    """The pair feature of each pair `(u[i], v[i])`. Returns `(k, ?)`:
    `d` columns for `hadamard`, `l1` and `avg`, `2*d` for `concat`. NO
    normalization happens here; see the module docstring.
    """
    fn = _pick(FEATURES, kind, "feature")
    return fn(Z[np.asarray(u)], Z[np.asarray(v)])


# ---------------------------------------------------------------------------
# knn -- the one path through which every all-pairs operation runs
# ---------------------------------------------------------------------------
def _poincare_pair(u, v):
    """The Poincare distance of two single rows. `sklearn`'s callable
    metric calls this once per pair, not once per batch; it reuses
    `_poincare` at batch size 1 so the formula stays one copy."""
    return float(_poincare(u[None, :], v[None, :])[0])


def _self_exclude(idx, dist, self_idx):
    """Drop each query's own row from its `k + 1` neighbours. Returns
    `(n, k)` arrays. If a row's self match is missing (the approximate
    index can miss it), the farthest of the `k + 1` is dropped instead, so
    the output width never changes."""
    n_rows, kp1 = idx.shape
    cols = np.arange(kp1)
    self_mask = idx == self_idx[:, None]
    found = self_mask.any(axis=1)
    drop_col = np.where(found, np.argmax(self_mask, axis=1), kp1 - 1)
    keep = cols[None, :] != drop_col[:, None]
    return (idx[keep].reshape(n_rows, kp1 - 1),
            dist[keep].reshape(n_rows, kp1 - 1))


def _knn_brute(Z, self_idx, k, metric):
    import sklearn
    from sklearn.neighbors import NearestNeighbors

    metric_arg = _poincare_pair if metric == "poincare" else metric
    # working_memory=512 caps the row block sklearn's brute search holds
    # live at once, so it never allocates `(len(queries), n)`. The
    # context form applies the budget for this call only and restores the
    # global default after, per PRD-v2 B5.
    with sklearn.config_context(working_memory=512):
        model = NearestNeighbors(n_neighbors=k + 1, algorithm="brute",
                                  metric=metric_arg, n_jobs=-1)
        model.fit(Z)
        dist, idx = model.kneighbors(Z[self_idx])
    idx, dist = _self_exclude(idx, dist, self_idx)
    return idx, dist, "brute"


def _knn_approx(Z, self_idx, k, metric):
    from pynndescent import NNDescent

    # `n_neighbors` at construction sets the approximate graph's own
    # connectivity, not the count returned per query; pynndescent's
    # documented default is 30, and a graph built at `k + 1` when `k` is
    # small starves the search of recall regardless of query-time `k`.
    build_k = max(k + 1, 30)
    index = NNDescent(Z, n_neighbors=build_k, metric=metric)
    idx, dist = index.query(Z[self_idx], k=k + 1)
    idx, dist = _self_exclude(idx, dist, self_idx)
    return idx, dist, "pynndescent"


def knn(Z, queries, k, metric="euclidean", exact=None):
    """The `k` nearest OTHER rows of `Z` to each row `queries` indexes
    into `Z`. Returns `(idx, dist, path)`: `idx` and `dist` are `(len(
    queries), k)`; `path` is `"brute"` or `"pynndescent"`, the search that
    ran -- a record that drops this cannot say whether an index was
    approximate.

    Every all-pairs search in this package goes through here (PRD-v2
    invariant 4). Nothing here allocates `(len(queries), n)`: the exact
    path is `sklearn.neighbors.NearestNeighbors(algorithm="brute")` under
    `working_memory=512`, which chunks its own search; the approximate
    path is `pynndescent.NNDescent`, an index, not a full distance table.

    `exact=None` (default) picks the exact path at `len(Z) <=
    config.RECON_MAX_N` rows and the approximate path above it.
    `exact=True` or `exact=False` forces the path. Above `RECON_MAX_N`
    rows the exact path still runs correctly if forced; it is slow there,
    which is why the default switches over.

    `metric="poincare"` runs on the exact path only, as a Python callable
    passed to `sklearn` -- slow, because no C loop backs it, and exact.
    Use it on small and medium graphs (T1, T2); `pynndescent` has no
    matching path, so `metric="poincare"` with the approximate path
    raises.

    `queries` indexes into `Z`; self is excluded by asking for `k + 1`
    neighbours and dropping the query's own row.
    """
    Z = np.asarray(Z)
    self_idx = np.asarray(queries)
    n = Z.shape[0]
    use_exact = (n <= RECON_MAX_N) if exact is None else bool(exact)

    if metric == "poincare" and not use_exact:
        raise ValueError(
            "knn: metric='poincare' has no pynndescent path; it runs "
            "exact only (pass exact=True), and only fits small and "
            "medium graphs there.")

    if use_exact:
        return _knn_brute(Z, self_idx, k, metric)
    return _knn_approx(Z, self_idx, k, metric)
