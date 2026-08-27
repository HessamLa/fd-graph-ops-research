#!/bin/env python3
"""policy_walk.py -- the UNDIRECTED policies `walk` and `walk_edges`.

Every pair inside the window of a walk, undirected, with a cap and a prune.
`weights.RULES` turns the walk statistics into `h`, `cap_per_node` keeps the
best pairs of each node, and the long-range pairs come last.

`walk_edges` adds the first-order edges of `A` at `h = 1`: a walk can miss
the edge of a low-degree node, and that edge is the pair that we trust
most. EVERY array of the statistics grows there, and not only the key.

`policy = buckets` continues in `policy_buckets.finish`, which is the same
branch the old class took.

Provenance: `Fodiwalk._build_D_walk`, moved line for line.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from . import landmarks as LM
from . import merge
from . import policy_buckets
from .far_pairs import degree_table, sample_far_pairs
from .pairs import cap_per_node, to_csr
from .result import Augmentation, take
from .walks import make_walker, walk_pair_stats
from .weights import RULES as WEIGHT_RULES


def pair_stats(A, n: int, spec, rng):
    """The walks of `walk` and `walk_edges`: windowed, undirected, capped.

    `make_walker` gives `uniform_walks` ITSELF at `p = q = 1`, and the two
    paths must never be unified: the rejection sampler draws one more number
    for each accept test, thus the same seed gives other walks.
    """
    walker = make_walker(A, n, spec.p, spec.q)
    return walk_pair_stats(A, n, spec.walks, spec.walk_len, spec.window, rng,
                           cap=spec.cap, prune_max=spec.prune_max,
                           prune_factor=spec.prune_factor, walker=walker)


# -- the first-order edges, the `walk_edges` axis ---------------------------
def keep_walk_pairs(A, n: int, stats, h, info):
    """`pairs = walk`: the walk pairs alone."""
    return stats, h


def add_edges(A, n: int, stats, h, info):
    """`pairs = walk_edges`: every edge of `A` that the walks missed, at `h = 1`."""
    edges = sp.triu(A, k=1).tocoo()
    edge_key = (edges.row.astype(np.int64) * n
                + edges.col.astype(np.int64))
    missing = ~np.isin(edge_key, stats["key"])
    added = int(missing.sum())
    stats["key"] = np.concatenate([stats["key"], edge_key[missing]])
    stats["mn"] = np.concatenate([stats["mn"],
                                  np.ones(added, dtype=np.int32)])
    stats["sm"] = np.concatenate([stats["sm"],
                                  np.ones(added, dtype=np.int64)])
    stats["cnt"] = np.concatenate([stats["cnt"],
                                   np.ones(added, dtype=np.int64)])
    info["edges_added"] = added
    return stats, np.concatenate([h, np.ones(added)])


EDGE_RULES = {"walk": keep_walk_pairs, "walk_edges": add_edges}


# -- the weight of a far pair -----------------------------------------------
def constant_far_weight(A, n: int, spec, rng, far, info):
    """One constant for every far pair."""
    return np.full(far.shape[0], spec.far_weight)


def landmark_far_weight(A, n: int, spec, rng, far, info):
    """A real distance for the far pairs, from a landmark table."""
    t_start = time.time()
    seeds = LM.pick(A, n, spec.landmarks, rng)
    table = LM.distances(A, n, seeds)
    info["t_landmark"] = time.time() - t_start
    info["landmark_mb"] = table.nbytes / 1e6
    weights = LM.pair_distance(table, far[:, 0], far[:, 1], spec.far_max,
                               spec.far_weight / spec.far_scale)
    weights = weights * spec.far_scale
    info["far_unreached"] = int((weights == spec.far_weight).sum())
    del table
    return weights


# -- the second axis: how the kept pairs become `D` -------------------------
def finish_cap(A, n: int, spec, rng, stats, h, info, t0) -> Augmentation:
    """`policy = cap`: the capped pairs, plus the long-range pairs."""
    stats, h = EDGE_RULES.get(spec.pairs, keep_walk_pairs)(A, n, stats, h,
                                                           info)
    D = to_csr(stats["key"], h, n)
    info["near_nnz"] = int(D.nnz)
    # `freq` on the SAME sparsity as `D`, thus the two `.data` arrays line
    # up entry by entry after the CSR build.
    freq = to_csr(stats["key"], stats["cnt"].astype(np.float64), n)

    asked = spec.far or int(n * np.log10(max(n, 10)))
    cum = degree_table(A, spec.far_bias)
    far = sample_far_pairs(n, asked, D, rng, cum=cum)
    if cum is not None:
        deg = np.diff(A.indptr)
        info["far_mean_deg"] = (float(deg[far.ravel()].mean())
                                if far.size else 0.0)
    info["far_pairs"] = int(far.shape[0])
    info["far_asked"] = asked
    if far.shape[0]:
        weight_fn = (landmark_far_weight if spec.landmarks
                     else constant_far_weight)
        weights = weight_fn(A, n, spec, rng, far, info)
        D, freq = merge.add_far_pairs(D, freq, far, weights)
    info["t_aug"] = time.time() - t0
    return Augmentation(D=D, freq=freq.data, stats=dict(stats, freq=freq),
                        info=info)


# `cap` is the default of every other name, as the old branch was.
POLICY_RULES = {"cap": finish_cap, "buckets": policy_buckets.finish}


def build(A, n: int, spec, rng) -> Augmentation:
    """Stage 2 by the `walk` and `walk_edges` policies."""
    t0 = time.time()
    info = {"walk_order": 1 if (spec.p == 1.0 and spec.q == 1.0) else 2}

    stats = pair_stats(A, n, spec, rng)
    info["prunes"] = int(stats.get("prunes", 0))
    info["raw_pairs"] = int(stats["raw"])
    info["unique_pairs"] = int(stats["key"].size)
    info["t_pairs"] = time.time() - t0

    keep = cap_per_node(stats["key"], stats["cnt"], n, spec.cap)
    stats = take(stats, keep)
    info["capped_pairs"] = int(stats["key"].size)

    h = WEIGHT_RULES[spec.weight](stats, spec.window, n)
    finish = POLICY_RULES.get(spec.policy, finish_cap)
    return finish(A, n, spec, rng, stats, h, info, t0)
