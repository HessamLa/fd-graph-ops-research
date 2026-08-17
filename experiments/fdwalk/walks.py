#!/bin/env python3
"""walks.py -- random walks that make the pairs of the augmented graph.

The augmentation of fodined must give a matrix `D` whose stored entry
`D[u, v] = h >= 1` is a distance. This module makes the PAIRS with random
walks, and it collects three statistics for each pair:

  * `mn`  -- the smallest step gap between the two nodes, over all the walks
  * `sm`  -- the sum of the gaps, thus `sm / cnt` is the mean gap
  * `cnt` -- how many times the two nodes appear together inside the window

`weights.py` turns those three numbers into the weight `h`.

Why a walk, and not the ball of k hops: the ball follows the degree of the
hubs, thus it holds 2,200 pairs for each node of com_youtube at k=2 and
46,874 at k=3. A walk has a BUDGET, `n_walks * walk_len` steps for each
node, and a cap of `m` pairs for each node. The size of `D` is therefore a
number that we choose. See `PLAN.md`, sections 1 and 2, and [3, 4, 6].

Every function is pure, it prints nothing, and the random generator arrives
as an argument.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def uniform_walks(A, starts, walk_len: int, rng):
    """One uniform random walk from each entry of `starts`.

    All the walks take one step at the same time, thus the loop runs
    `walk_len` times, and not one time for each walk. A node with no
    neighbour stays where it is.

    This is the walk of DeepWalk [3], and the walk of node2vec at
    `p = q = 1` [4]. `../other-ge/bench_other_ge.py` has the same function
    for the baseline, thus the two methods read the same walks.
    """
    deg = np.diff(A.indptr)
    cur = np.asarray(starts, dtype=np.int64)
    walks = np.empty((cur.size, walk_len), dtype=np.int64)
    walks[:, 0] = cur
    for t in range(1, walk_len):
        d = deg[cur]
        off = (rng.random(cur.size) * np.maximum(d, 1)).astype(np.int64)
        nxt = A.indices[A.indptr[cur] + off]
        cur = np.where(d > 0, nxt, cur)
        walks[:, t] = cur
    return walks


def _pairs_of(walks, window: int, n: int):
    """`(key, gap)` for every pair inside the window of these walks.

    The key of the pair `(u, v)` is `min * n + max`. One key for a pair,
    thus the two directions collect into one group, and the matrix stays
    symmetric later.
    """
    keys, gaps = [], []
    for t in range(1, window + 1):
        if t >= walks.shape[1]:
            break
        u = walks[:, :-t].ravel()
        v = walks[:, t:].ravel()
        ok = u != v                              # a walk can stay in place
        u, v = u[ok], v[ok]
        keys.append(np.minimum(u, v) * n + np.maximum(u, v))
        gaps.append(np.full(u.size, t, dtype=np.int32))
    if not keys:
        return np.empty(0, np.int64), np.empty(0, np.int32)
    return np.concatenate(keys), np.concatenate(gaps)


def _reduce(key, mn, sm, cnt):
    """Group by `key`: the minimum of `mn`, and the sums of `sm` and `cnt`.

    It also merges two results that were already reduced, because a reduced
    group is only a group of one entry to this function. The block loop
    below uses it both ways.
    """
    if key.size == 0:
        return key, mn, sm, cnt
    o = np.argsort(key, kind="stable")
    key, mn, sm, cnt = key[o], mn[o], sm[o], cnt[o]
    first = np.ones(key.size, dtype=bool)
    first[1:] = key[1:] != key[:-1]
    idx = np.flatnonzero(first)
    return (key[idx], np.minimum.reduceat(mn, idx),
            np.add.reduceat(sm, idx), np.add.reduceat(cnt, idx))


def walk_pair_stats(A, n: int, n_walks: int, walk_len: int, window: int, rng,
                    block: int = 50_000, cap: int = 0,
                    prune_factor: int = 4, prune_max: int = 4_000_000):
    """The statistics of every pair that a walk gives. Returns a dict.

    The keys of the dict are `key`, `mn`, `sm`, `cnt`, `raw` (how many pairs
    the walks gave before the grouping), and `prunes`.

    The work goes in blocks of start nodes, and each block reduces into the
    result immediately. Thus the peak memory does not hold
    `n * n_walks * walk_len * window` pairs at the same time.

    The blocks alone are NOT sufficient, and the first run at 1.13M nodes
    showed it. The walks gave 306M raw pairs there, and the accumulator
    keeps every DISTINCT pair, at 28 bytes each. The process passed 2.8 GB
    in 20 seconds. The cap of `m` pairs for each node bounds the RESULT, and
    it did not bound the work before it.

    `cap` repairs that. When the accumulator passes `prune_max` entries,
    this function cuts it to `prune_factor * cap` pairs for each node, and
    it continues. The memory is then bounded at any size of the graph, which
    is the whole promise of the design (H1).

    The prune is an APPROXIMATION, and the report must say so: a pair that
    the prune removes cannot come back, although a later block might have
    raised its count above the threshold. `prune_factor` holds that risk
    small, because a pair must fall out of the best `4 * cap` of BOTH of its
    nodes to be lost, and the final cap keeps only the best `cap` anyway.
    The count `prunes` in the result says how many times it happened, thus a
    run with `prunes = 0` is exact.
    """
    # Only the key needs int64: it is `u * n + v`, thus 1.3e12 on a graph of
    # a million nodes. A gap is below `window`, a sum of gaps is below
    # `count * window`, and a count is small. All three are int32, thus one
    # pair costs 20 bytes and not 28.
    acc = (np.empty(0, np.int64), np.empty(0, np.int32),
           np.empty(0, np.int32), np.empty(0, np.int32))
    raw, prunes = 0, 0
    starts_all = np.tile(np.arange(n, dtype=np.int64), n_walks)
    for s in range(0, starts_all.size, block):
        w = uniform_walks(A, starts_all[s:s + block], walk_len, rng)
        key, gap = _pairs_of(w, window, n)
        raw += key.size
        acc = _reduce(
            np.concatenate([acc[0], key]),
            np.concatenate([acc[1], gap]),
            np.concatenate([acc[2], gap]),
            np.concatenate([acc[3], np.ones(key.size, dtype=np.int32)]))
        del w, key, gap
        if cap and acc[0].size > prune_max:
            keep = cap_per_node(acc[0], acc[3], n, cap * prune_factor)
            acc = tuple(a[keep] for a in acc)
            prunes += 1
    return {"key": acc[0], "mn": acc[1], "sm": acc[2], "cnt": acc[3],
            "raw": raw, "prunes": prunes}


def walk_rows(A, n: int, n_walks: int, walk_len: int, rng,
              block: int = 20_000):
    """Row `u` holds EVERY node that a walk FROM `u` reached.

    This is the augmentation that the specification of 2026-08-17 asks for:
    "the embedding of each node is updated with respect to its neighbouring
    nodes and all the nodes found in its walk".

    It differs from `walk_pair_stats` in three ways, and each one matters:

    1. **The walk is anchored at the row.** A pair is `(start, visited)`,
       and not any two nodes inside a window. Thus a block of start nodes
       gives pairs that belong ONLY to those rows, and no block has to be
       merged with another. The accumulator that stopped G3 three times
       cannot appear, and no prune approximates anything.
    2. **There is no window.** Every position of the walk is a partner, thus
       `h` runs from 1 to `walk_len - 1`.
    3. **There is no cap.** The budget IS `n_walks * walk_len`, thus a row
       holds at most that many partners, and `D.nnz <= n * n_walks *
       walk_len` by construction.

    `h` is the FIRST step at which the walk reached the node, over all the
    walks of that row, thus it is the same minimum-gap estimator as before,
    and it is still an upper bound of the hop distance.

    Returns `key`, `mn`, `cnt`, and `raw`. `cnt` is how many times the row
    reached that node, which `weights.py` and the `fdlinear` force read as
    the frequency of the node in the walks of that row.
    """
    keys, mns, cnts = [], [], []
    raw = 0
    for s in range(0, n, block):
        e = min(s + block, n)
        starts = np.repeat(np.arange(s, e, dtype=np.int64), n_walks)
        w = uniform_walks(A, starts, walk_len, rng)
        # column t of the walk is t steps from the start
        src = np.repeat(w[:, 0], walk_len - 1)
        dst = w[:, 1:].ravel()
        gap = np.tile(np.arange(1, walk_len, dtype=np.int32), w.shape[0])
        ok = src != dst
        src, dst, gap = src[ok], dst[ok], gap[ok]
        raw += src.size
        key = src * n + dst                  # DIRECTED: the row is the start
        k, mn, sm, cnt = _reduce(key, gap, gap, np.ones(key.size, np.int32))
        keys.append(k); mns.append(mn); cnts.append(cnt)
        del w, src, dst, gap, key
    if not keys:
        return {"key": np.empty(0, np.int64), "mn": np.empty(0, np.int32),
                "cnt": np.empty(0, np.int32), "raw": 0}
    return {"key": np.concatenate(keys), "mn": np.concatenate(mns),
            "cnt": np.concatenate(cnts), "raw": raw}


def with_neighbours_low_deg(stats, A, n: int):
    """Every edge, stored ONE time, in the row of the lower-degree node.

    The rule of 2026-08-17: for an edge `(u, v)`, `v` enters the row of `u`
    only when `deg(v) >= deg(u)`. Thus a leaf keeps its edge to a hub, and
    the hub does not keep the same edge in its own row. An edge between two
    nodes of the same degree enters both rows.

    The gain is the memory: an edge costs one entry and not two. At 1.13M
    nodes that is 5,975,248 entries against 2,987,624.

    The reason it is safe to drop the hub side: the row of a hub already
    holds hundreds of partners, thus one more says little about where the
    hub belongs. The row of a leaf holds few, thus every one of them
    matters. The force stays reciprocal in the SUM over the graph, because
    the leaf still pulls the hub through its own row.

    A WARNING, and it is the reason `deg_source` exists in the caller: a
    hub can now hold NO entry at `h = 1`. `degrees_from_D` counts the
    entries at `h = 1`, thus it would return 0 for that row, and
    `inv_deg_ext` turns a 0 into 0.0, which zeroes EVERY force of the row,
    the repulsion too. The node would never move. The caller must therefore
    give `make_plan` the true degree of `A`, and not the count of `D`.
    """
    c = sp.triu(A, k=1).tocoo()
    deg = np.diff(A.indptr)
    du, dv = deg[c.row], deg[c.col]
    # (row, col) when deg(col) >= deg(row), and the other way when it is <=
    keep_fwd = dv >= du
    keep_bwd = du >= dv
    src = np.concatenate([c.row[keep_fwd], c.col[keep_bwd]])
    dst = np.concatenate([c.col[keep_fwd], c.row[keep_bwd]])
    ekey = src.astype(np.int64) * n + dst.astype(np.int64)
    key = np.concatenate([stats["key"], ekey])
    mn = np.concatenate([stats["mn"], np.ones(ekey.size, np.int32)])
    cnt = np.concatenate([stats["cnt"], np.ones(ekey.size, np.int32)])
    k, mn, _sm, cnt = _reduce(key, mn, mn, cnt)
    return {"key": k, "mn": mn, "cnt": cnt, "raw": stats.get("raw", 0),
            "edges_kept": int(ekey.size)}


def with_all_neighbours(stats, A, n: int):
    """Add EVERY edge of `A` at `h = 1`, and keep the walk pairs.

    The specification says that no neighbour may be missing. A walk finds
    most of them and not all: at 1.13M nodes the walks of the earlier
    policy held 68% of the directed edges. This function closes that gap by
    construction, thus the count of `h = 1` entries EQUALS `A.nnz`.

    An edge that a walk also found keeps `h = 1`, which is the minimum, thus
    the merge cannot raise a distance.
    """
    c = A.tocoo()
    ekey = c.row.astype(np.int64) * n + c.col.astype(np.int64)
    key = np.concatenate([stats["key"], ekey])
    mn = np.concatenate([stats["mn"], np.ones(ekey.size, np.int32)])
    cnt = np.concatenate([stats["cnt"], np.ones(ekey.size, np.int32)])
    k, mn, _sm, cnt = _reduce(key, mn, mn, cnt)
    return {"key": k, "mn": mn, "cnt": cnt, "raw": stats.get("raw", 0)}


def to_csr_directed(key, val, n: int):
    """A CSR from directed keys. Row `u` holds only what `u` reached.

    `D` is NOT symmetric here. The force law reads a ROW, thus an
    asymmetric `D` is legal: node `u` moves with respect to what `u` found.
    `shell_coeff_data` and `degrees_from_D` are per row, thus they also
    stay correct. See PLAN.md axis E.
    """
    return sp.csr_matrix((val, (key // n, key % n)), shape=(n, n))


def split_key(key, n: int):
    """A key array back into `(u, v)`, with `u < v`."""
    return key // n, key % n


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


def to_csr(key, h, n: int):
    """A symmetric CSR from the pair keys and their weights.

    Both directions go in, thus `D` is symmetric, which the force law needs.
    """
    u, v = split_key(key, n)
    return sp.csr_matrix(
        (np.concatenate([h, h]),
         (np.concatenate([u, v]), np.concatenate([v, u]))), shape=(n, n))
