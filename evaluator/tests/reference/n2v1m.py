#!/bin/env python3
"""reference/n2v1m.py -- the FROZEN reference of the `n2v1m` protocol.

Verbatim copy of `sample_non_edges` of
`experiments/large-graph-node2vec/bench_node2vec_1M.py`, taken on
2026-08-22, BEFORE the rewrite of that script removed it.

It differs from the `other_ge.py` copy in ONE way, and that way is the
whole reason both copies exist: this one takes `sources`, thus the first
node of a pair comes from a small set. The draw is
`rng.choice(sources, size=draw)` and not `rng.integers(0, n, draw)`, thus
the generator advances differently and every following pair differs.

The scoring of that script has no copy here. It lives INSIDE `main` and
it is not a function. Its reference is the recorded log
`node2vec_com_youtube_1M.log` (PRD section 9, tests P7 and P8).

DO NOT TIDY THIS FILE. See `other_ge.py`.
"""
from __future__ import annotations

import numpy as np


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
