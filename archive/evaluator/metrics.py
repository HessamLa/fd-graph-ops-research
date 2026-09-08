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

__all__ = ["DISTANCES", "FEATURES", "distance", "feature"]

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
