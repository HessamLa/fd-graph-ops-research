#!/bin/env python3
"""evaluator.pairs -- the positive and the negative pairs of a task.

THE GENERATOR ORDER IS PART OF THE PARITY. A `numpy` Generator is a
stream: each draw moves it. `rng.integers` where the source called
`rng.choice`, a `v` drawn before a `u`, or another batch size gives other
pairs from the same seed. Nothing fails. The numbers only differ, and
they match no recorded baseline (`fodiwalk/tests/harness.py`).

Thus these functions do not sample. They TRANSCRIBE a recorded draw
sequence, inelegance included.

PRD invariant I3: no `(n, n)` array, at any time.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path


# ---------------------------------------------------------------------------
# The positive pairs
# ---------------------------------------------------------------------------
def positives(A, max_count, rng, draw="over_cap"):
    """The undirected edges of `A`, at most `max_count` of them.

    Reads the UPPER TRIANGLE, thus each edge comes one time, never both
    directions.

    `draw` picks WHEN the generator moves, and the two baselines disagree:

      'over_cap'  `rng.choice` runs only OVER the cap; under it the
                  generator is UNTOUCHED (`reference/other_ge.py:58`).
      'always'    `rng.choice` runs every time
                  (`bench_node2vec_1M.py:198`).

    They leave the generator in other states, thus the NEGATIVES drawn
    next differ. Exact parity needs both.
    """
    up = sp.triu(A, k=1).tocoo()
    pos = np.column_stack([up.row, up.col])
    m = pos.shape[0]
    if draw == "over_cap":
        if m > max_count:                    # a cap, to hold the time down
            pos = pos[rng.choice(m, max_count, replace=False)]
    elif draw == "always":
        pos = pos[rng.choice(m, min(max_count, m), replace=False)]
    else:
        raise ValueError(
            f"positives: unknown draw {draw!r}; the names are "
            f"'over_cap', 'always'.")
    return pos


# ---------------------------------------------------------------------------
# The negative pairs -- the rejection sampler
# ---------------------------------------------------------------------------
def negatives(A, n, count, rng, sources=None, draw="reject"):
    """`count` node pairs that have NO edge in `A` and no self-pair.

    TWO draw sequences, and they give other pairs from the same `rng`:

      'reject'     the loop below, of `bench_other_ge.sample_non_edges`.
      'far_pairs'  `_far_pairs`, of `fodined.graph_augmentation
                   .sample_far_pairs`, which `fodined` and `fodiwalk` both
                   call for their negatives.

    THE DRAW ORDER of 'reject', and it is the contract:

        keys = sort(row * n + col) of every edge of A
        while short of `count`:
            draw = (count - have) * 2 + 1024
            u = rng.choice(sources, size=draw)   if sources is not None
                rng.integers(0, n, draw)         otherwise
            v = rng.integers(0, n, draw)
            drop u == v, drop a pair that `keys` holds,
            keep the first `room` of what is left

    `u` comes BEFORE `v`. The `* 2` and the `+ 1024` are the batch rule of
    the benchmark scripts. A swap, another factor or another constant
    changes every pair from the same seed. Do not tidy this loop.

    Rejection is one sorted key array plus one `np.searchsorted` for a
    whole batch. A Python set of the edges cost 1.8 GB at 1.13M nodes and
    is forbidden here (`experiments/large-graph-node2vec/REPORT.md`).

    `sources` restricts the FIRST node to a small set, because a BFS costs
    one full sweep for each distinct source. The CALLER draws it.

    Transcribed from `reference/other_ge.py` (no `sources`) and
    `reference/n2v1m.py` (with).
    """
    if draw == "far_pairs":
        if sources is not None:
            raise ValueError(
                "negatives: draw='far_pairs' takes no `sources`. "
                "`fodined.graph_augmentation.sample_far_pairs` draws both "
                "endpoints over the whole graph.")
        return _far_pairs(A, n, count, rng)
    if draw != "reject":
        raise ValueError(
            f"negatives: unknown draw {draw!r}; the names are 'reject', "
            f"'far_pairs'.")
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
        pos[pos >= keys.size] = 0            # a key over the end: probe 0
        keep = keys[pos] != k
        room = count - u_all.size
        u_all = np.concatenate([u_all, u[keep][:room]])
        v_all = np.concatenate([v_all, v[keep][:room]])
    return np.column_stack([u_all, v_all])


def _far_pairs(A, n, count, rng, batch=2_000_000, max_tries=40):
    """`count` pairs with no edge, in the draw of
    `fodined.graph_augmentation.sample_far_pairs` at `cum=None`.

    THE DRAW ORDER, and it differs from `negatives` in TWO places that both
    move the generator:

        while short of `count`:
            draw = min((count - have) * 2 + 1024, batch)
            u = rng.integers(0, n, draw);  v = rng.integers(0, n, draw)
            drop u == v, drop a pair that `A` holds
            k = unique(min*n + max)          <- direction collapsed, SORTED
            drop a key already accepted
            if k is longer than the room:  k = rng.choice(k, room)   <- a
                                                        SECOND draw
            acc = sort(concat(acc, k))

    `negatives` keeps its batch in DRAW order and takes the first `room`.
    This one sorts by key and, when the batch overfills, spends one more
    `rng.choice`. Thus the two return different pairs AND leave the
    generator in different states, and everything drawn afterwards differs.

    `fodined/link_prediction.py:sample_negatives` and
    `fodiwalk/misc/evaluation.py:sample_negatives` both call the original,
    thus every protocol of those two files needs this sequence.
    """
    b = A.tocoo()
    keys = np.sort(b.row.astype(np.int64) * n + b.col.astype(np.int64))
    del b
    acc = np.empty(0, np.int64)
    limit = max_tries * max(1, -(-count // batch))
    for _ in range(limit):
        if acc.size >= count:
            break
        draw = min((count - acc.size) * 2 + 1024, batch)
        u = rng.integers(0, n, draw)
        v = rng.integers(0, n, draw)
        ok = u != v
        u, v = u[ok], v[ok]
        k = u * n + v
        pos = np.searchsorted(keys, k)
        pos[pos >= keys.size] = 0            # a key over the end: probe 0
        near = keys[pos] == k
        u, v = u[~near], v[~near]
        k = np.unique(np.minimum(u, v) * n + np.maximum(u, v))
        if acc.size:                         # not accepted before
            pos = np.searchsorted(acc, k)
            pos[pos >= acc.size] = 0
            k = k[acc[pos] != k]
        room = count - acc.size
        if k.size > room:
            # `np.unique` sorted the batch, thus a plain `k[:room]` would
            # keep the smallest node ids and the sample would be biased.
            k = rng.choice(k, room, replace=False)
        acc = np.sort(np.concatenate([acc, k]))
    return np.column_stack([acc // n, acc % n])


# ---------------------------------------------------------------------------
# The pairs of the hop task
# ---------------------------------------------------------------------------
def pairs_for_hops(A, n, count, rng, sources=None, draw="reject"):
    """The pairs of the hop-distance task. TWO algorithms, not two
    options: they give other pairs from the same `(count, sources, rng)`.

      'reject'    `negatives(...)`. Returns the pairs alone; the caller
                  asks `hops.hop_distance` for the ground truth.
                  Protocols `otherge`, `n2v1m`, `fodined`.
      'bfs_rows'  the loop of `fodiwalk/misc/evaluation.py:hop_sample`.
                  Returns `(pairs, hops)`, because the BFS row is already
                  in the memory and a second pass costs another full
                  sweep. The caller SKIPS `hops.hop_distance`.
                  Protocols `fodiwalk_dist`, `fodiwalk_vec`.

    `bfs_rows` needs `sources`, and the CALLER draws it: that is the FIRST
    draw of the task, and one place must hold the whole task's order.

    NO FILTER OF THE HOPS. `bfs_rows` keeps every pair at `d > 0`, thus a
    true neighbour (`d == 1`) stays in the sample. The CALLER applies
    `min_hop`. A filter here changes the pair count and breaks parity.
    """
    if draw == "reject":
        return negatives(A, n, count, rng, sources)
    if draw == "bfs_rows":
        if sources is None:
            raise ValueError(
                "pairs_for_hops: draw='bfs_rows' needs `sources`. Draw them "
                "in the caller with "
                "rng.choice(n, size=min(n_sources, n), replace=False), "
                "which is the first draw of the fodiwalk protocols.")
        return _bfs_rows(A, n, count, rng, sources)
    raise ValueError(
        f"pairs_for_hops: unknown draw {draw!r}; the names are "
        f"'reject', 'bfs_rows'.")


def _bfs_rows(A, n, count, rng, sources, budget=200e6):
    """`(pairs, hops)` from one BFS row at a time.

    THE DRAW ORDER, from `fodiwalk/misc/evaluation.py:hop_sample`:

        per = max(1, count // src.size)
        for each block of sources:
            hop = shortest_path(A, indices=block)     # one dense block
            for each row of the block:
                tgt = rng.choice(n, size=min(per * 2, n), replace=False)
                keep the first `per` targets at a finite d > 0

    One `rng.choice` for each SOURCE, inside the block loop, thus the draw
    of row `i` depends on how many rows came before it.

    `shortest_path` always returns a DENSE `(len(indices), n)` array (trap
    written down in `hops._bfs`), thus sources go in blocks of
    `budget / (n * 8)`, holding the peak near 200 MB.
    """
    src = np.asarray(sources)
    per = max(1, count // src.size)
    block = max(1, int(budget / (n * 8)))
    us, vs, ds = [], [], []
    for s in range(0, src.size, block):
        blk = src[s:s + block]
        hop = shortest_path(A, method="D", unweighted=True, indices=blk)
        for i, u in enumerate(blk):
            tgt = rng.choice(n, size=min(per * 2, n), replace=False)
            d = hop[i, tgt]
            ok = np.isfinite(d) & (d > 0)    # no self-pair, no other part
            tgt, d = tgt[ok][:per], d[ok][:per]
            us.append(np.full(tgt.size, u))
            vs.append(tgt)
            ds.append(d)
        del hop
    pairs = np.column_stack([np.concatenate(us), np.concatenate(vs)])
    return pairs, np.concatenate(ds).astype(np.float64)
