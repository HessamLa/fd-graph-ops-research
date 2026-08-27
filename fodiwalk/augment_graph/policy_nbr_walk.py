#!/bin/env python3
"""policy_nbr_walk.py -- the DIRECTED policy of 2026-08-17.

Row `u` holds every neighbour of `u`, plus every node that a walk FROM `u`
reached, at `h` = the first step that reached it. `D` is not symmetric, and
that is legal: the force law reads a ROW, thus node `u` moves with respect
to what `u` found.

THE ORDER IS THE CONTRACT. `row_cap` bounds the WALK partners only and it
runs BEFORE the neighbours are added, thus a neighbour is never dropped. A
cap after would break that contract in silence.

Provenance: `Fodiwalk._build_D_nbr_walk`, moved line for line.
"""
from __future__ import annotations

import time

import numpy as np

from . import merge
from .buckets import budget, row_buckets
from .far_pairs import degree_table, sample_far_pairs
from .pairs import row_cap, to_csr_directed
from .result import Augmentation
from .walks import (make_walker, walk_rows, with_all_neighbours,
                    with_neighbours_low_deg)


def row_stats(A, n: int, spec, rng):
    """The walks of `nbr_walk`: directed rows, no window, no cap.

    The budget IS `walks * walk_len` for each row, thus `walk_rows` needs no
    prune. `make_walker` gives `uniform_walks` ITSELF at `p = q = 1`, and
    the two paths must never be unified: the rejection sampler draws one
    more number for each accept test, thus the same seed gives other walks.
    """
    walker = make_walker(A, n, spec.p, spec.q)
    return walk_rows(A, n, spec.walks, spec.walk_len, rng, walker=walker)


# -- the neighbour rule -----------------------------------------------------
# `both` is the default of every other name, as the old branch was.
NEIGHBOUR_RULES = {"low_deg": with_neighbours_low_deg,
                   "both": with_all_neighbours}


# -- the second axis: which walk partners survive ---------------------------
def select_all(stats, A, n: int, spec, rng):
    """`policy = cap`: every partner stays. `row_cap` already bounded the row."""
    return stats, {}


def select_buckets(stats, A, n: int, spec, rng):
    """`policy = buckets`: every `h = 1` entry, plus a stratified `h >= 2` budget."""
    stats, report = row_buckets(stats, A, n,
                                spec.bucket_total or budget(n), rng)
    return stats, {"buckets": report}


SELECT = {"cap": select_all, "buckets": select_buckets}


# -- the frequency plane ----------------------------------------------------
def freq_pair(stats, n: int):
    """How many times the row reached that node."""
    return stats["cnt"].astype(np.float64)


def freq_node(stats, n: int):
    """How many times the node was reached, over every row."""
    visits = np.bincount(stats["key"] % n, weights=stats["cnt"], minlength=n)
    return visits[stats["key"] % n]


FREQ_MODES = {"pair": freq_pair, "node": freq_node}


def build(A, n: int, spec, rng) -> Augmentation:
    """Stage 2 by the `nbr_walk` policy."""
    t0 = time.time()
    info = {"walk_order": 1 if (spec.p == 1.0 and spec.q == 1.0) else 2}

    stats = row_stats(A, n, spec, rng)
    if spec.row_cap:
        info["walk_pairs_before_rowcap"] = int(stats["key"].size)
        stats = row_cap(stats, n, spec.row_cap)
        info["walk_pairs_after_rowcap"] = int(stats["key"].size)
    stats = NEIGHBOUR_RULES.get(spec.edge_rule, with_all_neighbours)(
        stats, A, n)
    stats, report = SELECT.get(spec.policy, select_all)(
        stats, A, n, spec, rng)
    info.update(report)
    info["prunes"] = 0
    info["raw_pairs"] = int(stats["raw"])
    info["unique_pairs"] = int(stats["key"].size)
    info["capped_pairs"] = int(stats["key"].size)
    info["t_pairs"] = time.time() - t0

    h = stats["mn"].astype(np.float64)
    D = to_csr_directed(stats["key"], h, n)
    info["near_nnz"] = int(D.nnz)
    info["h1_entries"] = int((D.data == 1).sum())
    info["edges_of_A"] = int(A.nnz)
    # `freq`, on the SAME sparsity, thus the two `.data` arrays match entry
    # by entry after the CSR build.
    counts = FREQ_MODES.get(spec.freq_mode, freq_pair)(stats, n)
    freq = to_csr_directed(stats["key"], counts.astype(np.float64), n)
    info["far_pairs"] = 0
    info["far_asked"] = 0

    if spec.far > 0:
        cum = degree_table(A, spec.far_bias)
        far = sample_far_pairs(n, spec.far, D, rng, cum=cum)
        if far.shape[0]:
            far = merge.drop_pairs_of(far, D, n)
        if far.shape[0]:
            weights = np.full(far.shape[0], spec.far_weight)
            D, freq = merge.add_far_pairs(D, freq, far, weights)
            info["far_pairs"] = int(far.shape[0])
            info["far_asked"] = int(spec.far)
            info["near_nnz"] = int(D.nnz)

    info["t_aug"] = time.time() - t0
    return Augmentation(D=D, freq=freq.data, stats=dict(stats, freq=freq),
                        info=info)
