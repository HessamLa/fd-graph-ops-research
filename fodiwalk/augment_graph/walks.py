#!/bin/env python3
"""walks.py -- random walks that make the pairs of the augmented graph.

The augmentation must give a matrix `D` whose stored entry `D[u, v] = h >= 1`
is a distance. This module makes the PAIRS with random walks, and it
collects three statistics for each pair:

  * `mn`  -- the smallest step gap between the two nodes, over all the walks
  * `sm`  -- the sum of the gaps, thus `sm / cnt` is the mean gap
  * `cnt` -- how many times the two nodes appear together inside the window

`weights.py` turns those three numbers into the weight `h`.

Why a walk, and not the ball of k hops: the ball follows the degree of the
hubs, thus it holds 2,200 pairs for each node of com_youtube at k=2 and
46,874 at k=3. A walk has a BUDGET, `n_walks * walk_len` steps for each
node, and a cap of `m` pairs for each node. The size of `D` is therefore a
number that we choose.

THREE INVARIANTS OF THIS FILE. Each one is a recorded defect.

1. `make_walker(A, n, 1.0, 1.0)` gives `uniform_walks` ITSELF, and not the
   rejection sampler with trivial weights. The two have the same
   DISTRIBUTION and not the same array, because the rejection loop draws
   one more number for each accept test. Every result before 2026-08-18
   used the uniform walk, thus only this rule keeps a default run
   bit-exact.
2. `walk_rows` has NO prune, BY DESIGN: its bound IS `n_walks * walk_len`
   for each row. `walk_pair_stats` DOES prune, and its prune is an
   approximation. Do not unify the two.
3. `_pairs_of` collapses the direction at the source, with the key
   `min * n + max`. A directed form doubles the accumulator, which stopped
   three runs at 1.13M nodes.

A FOURTH, added 2026-08-28: `walk_rows` now emits `rows.RowStats`, a
ROW-BLOCKED carrier (`indptr`/`col`/`mn`/`cnt`), and NOT the one-array
`key` (`row * n + col`) it used until then; `with_all_neighbours` and
`with_neighbours_low_deg`, which read and extend that carrier, moved to
`row_merge.py` the same day (to keep this file under
`tests/test_structure.py`'s 422-line cap). Measured at 150,000 nodes of
com_youtube, default `nbr_walk` (`walks=10, walk_len=20`):

    peak RSS through the augmentation      3067 MB
    resting result (`D` + `freq.data`)      495 MB   (6.2x the peak/rest gap)

Stage by stage, the OLD code: `walk_rows`'s final `np.concatenate` of its
per-block lists cost +414 MB (rss 1166 -> peak 1580); `with_all_neighbours`
alone cost +1707 MB more (peak 3067), almost all of it ONE `np.argsort`
over 26M already-sorted keys, plus the four-array gather its permutation
forced. Both walk output and `A`'s own CSR are ALREADY sorted by
`(row, col)` -- `walk_rows` reduces block by block in row order, and a
CSR keeps `indices` ascending inside a row -- so neither the concatenate
nor the sort was buying anything. `walk_rows` now pre-sizes its output
once the walk is done and copies each block in, freeing the block as it
goes, instead of holding the whole list alive through one `concatenate`;
`row_merge.py`'s functions merge `stats` with `A`'s edges ROW BLOCK BY ROW
BLOCK, with `np.searchsorted` finding where an edge falls and where every
kept entry lands in the merged order -- no `argsort` at any size.

Provenance: a verbatim move from `experiments/fdwalk/walks.py`. The pair
keys, the caps and the CSR build moved to `pairs.py`; the row-wise bucket
rule moved to `buckets.py`. Only `make_walker` has a changed body, and the
change is `functools.partial(uniform_walks, A)` in place of a lambda of
the same call. See its docstring.

Every function is pure, it prints nothing, and the random generator arrives
as an argument.
"""
from __future__ import annotations

import functools

import numpy as np

from .pairs import cap_per_node
from .rows import RowStats


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
                    prune_factor: int = 4, prune_max: int = 4_000_000,
                    walker=None):
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
    walker = walker or (lambda st, L, r: uniform_walks(A, st, L, r))
    acc = (np.empty(0, np.int64), np.empty(0, np.int32),
           np.empty(0, np.int32), np.empty(0, np.int32))
    raw, prunes = 0, 0
    starts_all = np.tile(np.arange(n, dtype=np.int64), n_walks)
    for s in range(0, starts_all.size, block):
        w = walker(starts_all[s:s + block], walk_len, rng)
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
              block: int = 20_000, walker=None):
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

    Returns a `RowStats` (`indptr`/`col`/`mn`/`cnt`, plus `raw`); see its
    docstring. `cnt` is how many times the row reached that node, which
    `weights.py` and the `fdlinear` force read as the frequency of the
    node in the walks of that row.

    2026-08-28: `col` is `int32`, `mn` is `int16` (`h` fits `1 .. walk_len -
    1`), `cnt` is `int32` -- explicitly, since `np.add.reduceat` widens an
    int32 input to int64 and `_reduce` (below) still does that for
    `walk_pair_stats`, which this function must not disturb. The block
    loop below therefore casts its OWN output right after `_reduce`
    returns, block by block, never touching `_reduce` itself.

    The blocks are collected in a list, as before -- each block already
    holds only ITS OWN rows' pairs after `_reduce`, thus the list's total
    size is the same across the whole loop either way. What changed is the
    step after the loop: the old code called `np.concatenate(keys)`, which
    holds the WHOLE list alive while it also builds the new array (the
    +414 MB spike this docstring's numbers name). This function instead
    pre-sizes the output from the row counts, then copies each block in
    and drops it immediately, so the block list and the output array are
    never both complete at the same time.
    """
    walker = walker or (lambda st, L, r: uniform_walks(A, st, L, r))
    col_blocks, mn_blocks, cnt_blocks = [], [], []
    counts = np.zeros(n, dtype=np.int64)
    raw = 0
    for s in range(0, n, block):
        e = min(s + block, n)
        starts = np.repeat(np.arange(s, e, dtype=np.int64), n_walks)
        w = walker(starts, walk_len, rng)
        # column t of the walk is t steps from the start
        src = np.repeat(w[:, 0], walk_len - 1)
        dst = w[:, 1:].ravel()
        gap = np.tile(np.arange(1, walk_len, dtype=np.int32), w.shape[0])
        ok = src != dst
        src, dst, gap = src[ok], dst[ok], gap[ok]
        raw += src.size
        key = src * n + dst                  # DIRECTED: the row is the start
        k, mn, sm, cnt = _reduce(key, gap, gap, np.ones(key.size, np.int32))
        row_local = k // n - s
        counts[s:e] = np.bincount(row_local, minlength=e - s)
        col_blocks.append((k - (k // n) * n).astype(np.int32))
        mn_blocks.append(mn.astype(np.int16))
        cnt_blocks.append(cnt.astype(np.int32))
        del w, src, dst, gap, key, k, mn, sm, cnt, row_local
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    total = int(indptr[-1])
    col = np.empty(total, dtype=np.int32)
    mn = np.empty(total, dtype=np.int16)
    cnt = np.empty(total, dtype=np.int32)
    pos = 0
    while col_blocks:
        c, m, k = col_blocks.pop(0), mn_blocks.pop(0), cnt_blocks.pop(0)
        end = pos + c.size
        col[pos:end], mn[pos:end], cnt[pos:end] = c, m, k
        pos = end
        del c, m, k
    return RowStats(indptr, col, mn, cnt, n, raw=raw)


def _edge_keys(A, n: int):
    """`row * n + col` for every stored edge, GLOBALLY sorted.

    A scipy CSR with sorted indices gives rows in order and columns sorted
    inside a row, thus this array is sorted as a whole and one vectorized
    `searchsorted` answers "is `x` a neighbour of `t`" for a whole batch.
    That is what replaces node2vec's per-edge alias tables.
    """
    A = A.tocsr()
    A.sort_indices()
    rows = np.repeat(np.arange(A.shape[0], dtype=np.int64),
                     np.diff(A.indptr))
    return rows * n + A.indices.astype(np.int64)


def node2vec_walks(A, starts, walk_len: int, rng, p: float = 1.0,
                   q: float = 1.0, ekeys=None, max_tries: int = 20):
    """The SECOND-ORDER walk of node2vec [4], by rejection sampling.

    The transition from `v`, having come from `t`, to a candidate `x` has
    the unnormalised weight

        1/p   if x == t          (return to where the walk came from)
        1     if x is adjacent to t
        1/q   otherwise          (move away)

    `p` is the return parameter and `q` is the in-out parameter. `q > 1`
    keeps the walk near `t`, which is BFS-like; `q < 1` sends it away,
    which is DFS-like.

    **The correctness gate, and it is DISTRIBUTIONAL and not bit-exact.**
    At `p = q = 1` every weight is 1, thus `w_max` is 1, thus the accept
    test `rng.random() * w_max < w` never fails and the candidate is a
    plain uniform draw from the neighbour list. The walk therefore has the
    SAME DISTRIBUTION as `uniform_walks`. It is NOT the same array: the
    rejection loop draws one extra random number for each accept test, thus
    the two functions consume the random stream differently and the same
    seed gives different walks. A first version of this test asserted array
    equality and failed for exactly that reason; the gate is a
    distributional comparison over many walks.

    **Why rejection sampling, and not the alias tables of the reference
    implementation.** An alias table for each EDGE costs `sum(deg^2)`.
    com_youtube holds a node of degree 28,754, thus that node alone
    contributes 827 million entries, and the tables are not an option at
    1.13M nodes. Rejection sampling needs NO preprocessing: draw a
    neighbour uniformly, accept it with probability `w(x) / w_max`, repeat.
    The expected number of draws is `w_max / w_mean`, which is a small
    constant for any reasonable `p` and `q`.

    The one test the method needs -- "is `x` adjacent to `t`" -- is one
    vectorized `searchsorted` into the globally sorted edge keys, thus it
    is `O(log nnz)` for a whole batch and it holds one int64 array of
    `nnz`.

    A walk that does not get an acceptance inside `max_tries` keeps its
    last uniform draw. That is a bias, it is bounded by
    `(1 - w_min/w_max)^max_tries`, and at the default it is below 1e-6 for
    every `p, q` in [0.25, 4].
    """
    deg = np.diff(A.indptr)
    if ekeys is None:
        ekeys = _edge_keys(A, A.shape[0])
    n = A.shape[0]
    cur = np.asarray(starts, dtype=np.int64)
    walks = np.empty((cur.size, walk_len), dtype=np.int64)
    walks[:, 0] = cur
    if walk_len < 2:
        return walks
    # step 1 is first-order: there is no previous node yet
    d = deg[cur]
    off = (rng.random(cur.size) * np.maximum(d, 1)).astype(np.int64)
    nxt = A.indices[A.indptr[cur] + off]
    prev, cur = cur, np.where(d > 0, nxt, cur)
    walks[:, 1] = cur

    w_max = max(1.0 / p, 1.0, 1.0 / q)
    for t in range(2, walk_len):
        d = deg[cur]
        live = d > 0
        acc = np.zeros(cur.size, dtype=bool)
        pick = cur.copy()
        for _ in range(max_tries):
            todo = live & ~acc
            if not todo.any():
                break
            idx = np.flatnonzero(todo)
            off = (rng.random(idx.size) * d[idx]).astype(np.int64)
            x = A.indices[A.indptr[cur[idx]] + off]
            # the three cases of the node2vec weight
            w = np.full(idx.size, 1.0 / q)
            back = x == prev[idx]
            w[back] = 1.0 / p
            if q != 1.0:
                key = prev[idx].astype(np.int64) * n + x.astype(np.int64)
                pos = np.searchsorted(ekeys, key)
                hit = (pos < ekeys.size) & (ekeys[np.minimum(
                    pos, ekeys.size - 1)] == key)
                w[hit & ~back] = 1.0
            ok = rng.random(idx.size) * w_max < w
            pick[idx[ok]] = x[ok]
            pick[idx[~ok]] = x[~ok]        # keep the last draw as fallback
            acc[idx[ok]] = True
        prev, cur = cur, np.where(live, pick, cur)
        walks[:, t] = cur
    return walks


def make_walker(A, n: int, p: float = 1.0, q: float = 1.0):
    """A walk function `(starts, walk_len, rng) -> walks`, first or second order.

    At `p = q = 1` it returns `uniform_walks` ITSELF, and not the rejection
    sampler with trivial weights. Two reasons, and both matter: the uniform
    walk is faster, and every result recorded before 2026-08-18 used it,
    thus a run at the default reproduces those numbers exactly rather than
    only distributionally.

    Above `p = q = 1` the edge keys are built ONE time here and reused for
    every block, thus the `O(nnz)` int64 array is paid once for the whole
    augmentation and not once for each block.

    The binding is `functools.partial` and not a lambda, thus the identity
    of the walk function stays visible: `make_walker(A, n).func is
    uniform_walks`. The call and its numbers do not change --
    `partial(uniform_walks, A)(starts, L, rng)` IS
    `uniform_walks(A, starts, L, rng)`.
    """
    if p == 1.0 and q == 1.0:
        return functools.partial(uniform_walks, A)
    ek = _edge_keys(A, n)
    return functools.partial(node2vec_walks, A, p=p, q=q, ekeys=ek)
