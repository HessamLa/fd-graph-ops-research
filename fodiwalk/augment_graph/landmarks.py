#!/bin/env python3
"""landmarks.py -- a real distance for the pairs that a walk cannot reach.

The problem that this module solves:

A walk gives a distance only inside its window. Every pair further away
takes ONE constant, 100. Cora has a diameter near 19 and a window of 5,
thus every pair at 6 hops or more looks the same to the force law. The
force law cannot separate a pair at 6 hops from a pair at 19, because the
augmentation did not separate them. That is where the hop R2 is lost.

The repair is the landmark distance oracle. Take `count` nodes, run one
exact BFS from each, and keep the distance of every node to every landmark.
Then, for any pair:

    d(u, v) <= min over L of ( d(u, L) + d(L, v) )

by the triangle inequality. The estimate is an UPPER bound, thus it has the
same character as the walk gap, which is also an upper bound. The two
therefore do not fight: the walk is tight near, and the landmarks give a
number far away.

The cost is small. One BFS on com_youtube needs 0.36 s (measured in
`../bench_large_graphs.py`), thus 50 landmarks need 18 s. The table is
`count * n` int16, which is 113 MB at 1.13M nodes and 50 landmarks.

References: the landmark oracle of Potamias et al. (2009) and the
comparison of the selection rules in Akiba et al. [15]. A mix of the
highest degree and of random nodes is the usual choice, and it is what
`pick` does.

Every function is pure, it prints nothing, and the generator arrives as an
argument.
"""
# Provenance: moved from `experiments/fdwalk/landmarks.py` on 2026-08-19.
# Verbatim: not one line of a body changed.
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import shortest_path

# The value of "no path through any landmark". It stays inside int16.
UNREACHED = np.int16(32767)


def pick(A, n: int, count: int, rng):
    """`count` landmark nodes: one half by degree, one half at random.

    A node of a high degree covers many paths, thus it gives a tight bound
    for many pairs. Only high-degree nodes is a bad rule, because the hubs
    of one region give the same information many times. The random half
    repairs that.
    """
    count = min(count, n)
    deg = np.diff(A.indptr)
    n_hub = count // 2
    hubs = np.argsort(-deg)[:n_hub]
    rest = np.setdiff1d(np.arange(n), hubs)
    extra = rng.choice(rest, size=min(count - n_hub, rest.size),
                       replace=False)
    return np.sort(np.concatenate([hubs, extra]))


def distances(A, n: int, lm, budget_bytes: float = 200e6):
    """`(count, n)` int16 of the distance of every node to every landmark.

    `shortest_path` gives a DENSE `(len(indices), n)` float64 array, thus
    one call for 50 landmarks at 1.13M nodes is 454 MB. The landmarks
    therefore go in blocks, and each block becomes int16 immediately: the
    table is then 113 MB, and not 454 MB.
    """
    block = max(1, int(budget_bytes / (n * 8)))
    out = np.empty((lm.size, n), dtype=np.int16)
    for s in range(0, lm.size, block):
        blk = lm[s:s + block]
        d = shortest_path(A, method="D", unweighted=True, indices=blk)
        d[~np.isfinite(d)] = float(UNREACHED)
        out[s:s + blk.size] = np.minimum(d, float(UNREACHED)).astype(np.int16)
        del d
    return out


def pair_distance(table, u, v, w_max: int, fallback: float,
                  block: int = 200_000):
    """The landmark estimate of `d(u, v)`, for each pair.

    `min over L of table[L, u] + table[L, v]`, in blocks of pairs. One block
    holds `count * block` int32, which is 40 MB at 50 landmarks and 200,000
    pairs.

    A pair that no landmark reaches takes `fallback`, which is the constant
    of the old policy. That case is a pair in another component, and it has
    no distance at all.

    `w_max` caps the result. The stored weight goes into the repulsion
    directly: `fdlinear` gives a pair at `h >= 2` the coefficient
    `h / freq`, thus the coefficient increases LINEARLY with the weight.
    Without a cap, a graph with a diameter of 850 (roadNet-CA) gives one
    far pair a coefficient hundreds of times larger than a near pair, and
    that pair alone controls the row sum. The cap also says "further than
    `w_max` is only far", which is true for a force.
    """
    out = np.empty(u.size, dtype=np.float64)
    lim = int(UNREACHED)
    for s in range(0, u.size, block):
        e = min(s + block, u.size)
        du = table[:, u[s:e]].astype(np.int32)
        dv = table[:, v[s:e]].astype(np.int32)
        est = (du + dv).min(axis=0)
        del du, dv
        bad = est >= lim
        est = np.clip(est, 1, w_max).astype(np.float64)
        est[bad] = fallback
        out[s:e] = est
    return out
