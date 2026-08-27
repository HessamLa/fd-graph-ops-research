#!/bin/env python3
"""policy_buckets.py -- `policy = buckets` on the UNDIRECTED walk pairs.

Every original edge at `h = 1`, plus a stratified budget of the walk pairs
at `h >= 2`. The pairs at `h = 1` are not candidates: the policy keeps all
of them, thus only the walk pairs at `h >= 2` enter the sampler.

`freq` is built HERE, on the same sparsity as `D`. Until 2026-08-18 this
branch returned none, thus an `fdlinear` run took the planes of another law
from a fallback and read a coefficient plane as `h`. The physics was wrong,
nothing raised, and the run went to NaN at 2000 epochs.

This module continues `policy_walk`: it takes the capped statistics and the
weights that the walk stage already made. Provenance:
`Fodiwalk._build_D_buckets`, moved line for line.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from . import merge
from .buckets import bucket_sample, budget
from .far_pairs import degree_table, sample_far_pairs
from .pairs import to_csr
from .result import Augmentation, take


def finish(A, n: int, spec, rng, stats, h, info, t0) -> Augmentation:
    """The bucket sample, the edges of `A`, and the optional far pairs."""
    candidates = h >= 2
    idx, info["buckets"] = bucket_sample(
        h[candidates], spec.bucket_total or budget(n), rng)
    sel = np.flatnonzero(candidates)[idx]
    stats = take(stats, sel)
    h = h[sel]

    edges = sp.triu(A, k=1).tocoo()
    edge_key = (edges.row.astype(np.int64) * n
                + edges.col.astype(np.int64))
    added = edge_key.size
    stats["key"] = np.concatenate([stats["key"], edge_key])
    stats["mn"] = np.concatenate([stats["mn"],
                                  np.ones(added, dtype=np.int32)])
    stats["sm"] = np.concatenate([stats["sm"],
                                  np.ones(added, dtype=np.int32)])
    stats["cnt"] = np.concatenate([stats["cnt"],
                                   np.ones(added, dtype=np.int32)])
    h = np.concatenate([h, np.ones(added)])
    info["edges_added"] = int(added)

    D = to_csr(stats["key"], h, n)
    info["near_nnz"] = int(D.nnz)
    freq = to_csr(stats["key"], stats["cnt"].astype(np.float64), n)
    info["far_pairs"] = 0
    info["far_asked"] = 0

    if spec.far_with_buckets:
        # `buckets` has NO far pairs by construction, and every measurement
        # says the long-range term carries the hop R2. This adds them ON TOP
        # of the bucket budget, thus it separates two causes that the policy
        # confounds.
        asked = spec.far or int(n * np.log10(max(n, 10)))
        cum = degree_table(A, spec.far_bias)
        far = sample_far_pairs(n, asked, D, rng, cum=cum)
        info["far_pairs"] = int(far.shape[0])
        info["far_asked"] = asked
        if far.shape[0]:
            weights = np.full(far.shape[0], spec.far_weight)
            D, freq = merge.add_far_pairs(D, freq, far, weights)
            info["near_nnz"] = int(D.nnz)

    info["t_aug"] = time.time() - t0
    return Augmentation(D=D, freq=freq.data, stats=dict(stats, freq=freq),
                        info=info)
