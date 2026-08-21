#!/bin/env python3
"""pairs.py -- the pair key, the caps, and the CSR build.

A pair of nodes travels through the augmentation as ONE integer key, and
not as two columns. `split_key` gives the two node ids back.

Two key forms, and the difference decides the shape of `D`:

  UNDIRECTED  `min(u, v) * n + max(u, v)`. The two directions of a pair
              collect into one group, thus `to_csr` can store both and `D`
              stays symmetric. `walk_pair_stats` makes this form.
  DIRECTED    `u * n + v`, where `u` is the row that found `v`. `D` is then
              not symmetric: row `u` holds only what `u` reached.
              `walk_rows` makes this form, and `to_csr_directed` stores it.

Two caps, and they are not interchangeable:

  `cap_per_node`  a pair stays when EITHER of its two nodes keeps it, thus
                  a row can hold more than `m` entries.
  `row_cap`       row `u` keeps its own best `m`, thus EVERY row has the
                  same width. It runs on a directed key only.

Provenance: every function is a verbatim move from
`experiments/fdwalk/walks.py`. Only the location changed.

Every function is pure, and it prints nothing.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def split_key(key, n: int):
    """A key array back into `(u, v)`, with `u < v`."""
    return key // n, key % n


def to_csr(key, h, n: int):
    """A symmetric CSR from the pair keys and their weights.

    Both directions go in, thus `D` is symmetric, which the force law needs.
    """
    u, v = split_key(key, n)
    return sp.csr_matrix(
        (np.concatenate([h, h]),
         (np.concatenate([u, v]), np.concatenate([v, u]))), shape=(n, n))


def to_csr_directed(key, val, n: int):
    """A CSR from directed keys. Row `u` holds only what `u` reached.

    `D` is NOT symmetric here. The force law reads a ROW, thus an
    asymmetric `D` is legal: node `u` moves with respect to what `u` found.
    `degrees_from_D` is per row, thus it also stays correct. See PLAN.md
    axis E.
    """
    return sp.csr_matrix((val, (key // n, key % n)), shape=(n, n))


def cap_per_node(key, cnt, n: int, m: int):
    """Keep at most `m` pairs for each node, the ones with the most visits.

    A pair has two nodes, thus the rule looks at it two times. The pair
    stays if EITHER of its two nodes keeps it. This is the symmetrisation
    that a neighbour graph of LargeVis or UMAP uses [8, 9]: a low-degree
    node keeps its partners, although a hub does not choose it.

    Two pairs with the same count take the order that `np.lexsort` gives.
    That order is deterministic, thus a run repeats.

    Returns the indices of the pairs to keep.
    """
    if m <= 0 or key.size == 0:
        return np.arange(key.size)
    # int32 everywhere that it is safe. A node id is below 2^31 for every
    # graph here, and so is the index of a pair. The int64 form of this
    # function held six arrays of 16M entries at the same time, about
    # 1.4 GB, and it stopped the first two runs of G3 at 1.13M nodes.
    u, v = split_key(key, n)
    node = np.concatenate([u, v]).astype(np.int32)
    count = np.concatenate([cnt, cnt]).astype(np.int32)
    idx = np.arange(key.size, dtype=np.int32)
    pair = np.concatenate([idx, idx])
    del u, v, idx

    order = np.lexsort((-count, node))           # by node, then most visits
    del count
    node_s = node[order]
    del node
    pair_s = pair[order]
    del pair, order
    first = np.ones(node_s.size, dtype=bool)
    first[1:] = node_s[1:] != node_s[:-1]
    del node_s
    starts = np.flatnonzero(first)
    group_len = np.diff(np.append(starts, first.size))
    del first
    rank = np.arange(pair_s.size, dtype=np.int64) - np.repeat(starts,
                                                              group_len)
    return np.unique(pair_s[rank < m])


# ---------------------------------------------------------------------------
# 2026-08-18 -- the directed cap. It arrived with the row-wise bucket rule
# (`buckets.row_buckets`) and the real node2vec walk (`walks.node2vec_walks`),
# which the split of 2026-08-19 moved to their own files. See
# `experiments/fdwalk/TODOs.md` and `dev-docs/CATALOG.md`.
# ---------------------------------------------------------------------------
def row_cap(stats, n: int, m: int):
    """Keep the `m` most-visited partners of EACH ROW. Directed.

    This is the directed form of `cap_per_node`, and the two differ in a
    way that decides the layout of the plan:

        cap_per_node  a pair survives if EITHER endpoint keeps it, thus a
                      row can hold MORE than `m` and a hub collects many.
        row_cap       row `u` holds exactly its own best `m`, thus EVERY
                      row has the same width.

    The second is axis E-B of PLAN.md, and the payoff is in the plan and
    not only in `D`: no hub split (`n_split` -> 0), little padding, balanced
    batches, and an exact bound of `n * m` cells.

    It runs on the output of `walk_rows`, which is ALREADY directed and
    row-disjoint, thus no direction has to be recovered and no accumulator
    is doubled. `_pairs_of` cannot be used for this: it collapses the
    direction at the source with the key `min*n + max`.
    """
    if m <= 0 or stats["key"].size == 0:
        return stats
    row = (stats["key"] // n).astype(np.int64)
    order = np.lexsort((-stats["cnt"].astype(np.int64), row))
    row_s = row[order]
    first = np.ones(row_s.size, dtype=bool)
    first[1:] = row_s[1:] != row_s[:-1]
    starts = np.flatnonzero(first)
    glen = np.diff(np.append(starts, first.size))
    rank = np.arange(row_s.size, dtype=np.int64) - np.repeat(starts, glen)
    keep = np.sort(order[rank < m])
    return {k: (v[keep] if isinstance(v, np.ndarray) else v)
            for k, v in stats.items()}
