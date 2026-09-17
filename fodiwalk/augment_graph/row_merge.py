#!/bin/env python3
"""row_merge.py -- `A`'s edges into `walk_rows`' result, with NO sort.

`with_all_neighbours` and `with_neighbours_low_deg` add edges of `A` to the
walk statistics `walks.walk_rows` built. Until 2026-08-28 both did it with
one `A.tocoo()` (or `sp.triu(A, k=1).tocoo()`) and one global
sort-and-reduce over the concatenation of the walk pairs and the edges --
the `np.argsort` over 26M already-sorted keys that cost +1707 MB at
150,000 nodes of com_youtube (`walks.py`'s module docstring has the full
stage-by-stage number).

**The insight this rests on.** `walk_rows` reduces block by block in row
order, thus its result is ALREADY sorted by `(row, col)`. `A` is a CSR
with sorted indices, thus its own rows are ALREADY sorted too. Two sorted
sequences, merged per row, need no sort at all: `np.searchsorted` finds
where an edge already sits in the walk statistics (a hit takes the `min`
of `h` and the sum of `cnt`), and two more `searchsorted` calls place
every kept entry at its final rank in the merged order -- the standard
rank-merge of two DISJOINT sorted arrays. `_merge_row_edges` is that
merge, run in ROW BLOCKS so the transient it costs is bounded by one
block and not by the whole graph.

Provenance: new 2026-08-28 (`agentic-log/10.mem-agent/`), split out of
`walks.py` on the same day to keep both files under
`tests/test_structure.py`'s line cap. `with_all_neighbours` and
`with_neighbours_low_deg` carry forward the docstrings they had in
`walks.py`, with the 2026-08-28 note on what changed and why.

Every function here is pure, and none of them print.
"""
from __future__ import annotations

import numpy as np

from .rows import RowStats


def _merge_row_edges(stats: RowStats, n: int, edge_source, block: int):
    """The per-row union of `stats` with extra edges, with NO sort.

    `edge_source(s, e)` gives `(row_local, col)` for the extra edges of
    rows `[s, e)`: `row_local` in `[0, e - s)`, and inside a row `col`
    climbs and holds no repeat -- `with_all_neighbours` and
    `with_neighbours_low_deg` both read this straight from `A`'s own CSR,
    which already keeps that rule.

    `stats` and the extra edges are therefore BOTH already sorted by
    `(row, col)`, one block at a time, and their union needs no `argsort`:
    `np.searchsorted` finds where an extra edge already sits in `stats`
    (a hit takes the `min` of `h` and the sum of `cnt`), and two more
    `searchsorted` calls give the RANK of every kept entry in the merged
    order -- the standard rank-merge of two DISJOINT sorted arrays, which
    places every value at its final index with two scatters and never
    moves anything through a sort. This is the `argsort` over 26M
    already-sorted keys that this module's docstring measures at +203 MB
    for the permutation and +609 MB for the four-array gather it forced;
    row-blocking plus this merge removes both.

    A partner both sides hold takes `h = min(stats, edge)` and
    `cnt = stats + edge`. An added edge always carries `h = 1`, the
    smallest legal value, so the `min` always keeps it -- but this computes
    it and does not special-case it, because a caller with a different
    edge weight must not silently break.

    Returns `(indptr, col, mn, cnt)`, block-copied into pre-sized output
    arrays exactly as `walks.walk_rows` does, for the same reason: the
    per-block list and the final array are never both complete at once.
    """
    col_blocks, mn_blocks, cnt_blocks = [], [], []
    counts = np.zeros(n, dtype=np.int64)
    for s in range(0, n, block):
        e = min(s + block, n)
        lo_s, hi_s = int(stats.indptr[s]), int(stats.indptr[e])
        row_s = np.repeat(np.arange(e - s, dtype=np.int64),
                          np.diff(stats.indptr[s:e + 1]))
        col_s = stats.col[lo_s:hi_s]
        mn_s = stats.mn[lo_s:hi_s].copy()
        cnt_s = stats.cnt[lo_s:hi_s].copy()
        key_s = row_s * n + col_s.astype(np.int64)

        row_a, col_a = edge_source(s, e)
        key_a = row_a.astype(np.int64) * n + col_a.astype(np.int64)

        hit = np.zeros(key_a.size, dtype=bool)
        if key_a.size and key_s.size:
            pos = np.searchsorted(key_s, key_a)
            safe = np.minimum(pos, key_s.size - 1)
            hit = (pos < key_s.size) & (key_s[safe] == key_a)
            where = pos[hit]
            mn_s[where] = np.minimum(mn_s[where], np.int16(1))
            cnt_s[where] = cnt_s[where] + 1
        new_key = key_a[~hit]

        if new_key.size:
            # Two DISJOINT sorted arrays, merged by rank (no argsort): row
            # `i` of `key_s` lands at `i + (new keys below it)`; row `i` of
            # `new_key` lands at `i + (stats keys at or below it)`.
            rank_s = np.arange(key_s.size) + np.searchsorted(
                new_key, key_s, side="left")
            rank_new = np.arange(new_key.size) + np.searchsorted(
                key_s, new_key, side="right")
            total = key_s.size + new_key.size
            m_col = np.empty(total, dtype=np.int32)
            m_mn = np.empty(total, dtype=np.int16)
            m_cnt = np.empty(total, dtype=np.int32)
            m_row = np.empty(total, dtype=np.int64)
            new_row = new_key // n
            m_col[rank_s], m_col[rank_new] = col_s, new_key - new_row * n
            m_mn[rank_s], m_mn[rank_new] = mn_s, 1
            m_cnt[rank_s], m_cnt[rank_new] = cnt_s, 1
            m_row[rank_s], m_row[rank_new] = row_s, new_row
        else:
            m_col, m_mn, m_cnt, m_row = col_s, mn_s, cnt_s, row_s

        counts[s:e] = np.bincount(m_row, minlength=e - s)
        col_blocks.append(np.ascontiguousarray(m_col, dtype=np.int32))
        mn_blocks.append(np.ascontiguousarray(m_mn, dtype=np.int16))
        cnt_blocks.append(np.ascontiguousarray(m_cnt, dtype=np.int32))
        del (row_s, col_s, mn_s, cnt_s, key_s, row_a, col_a, key_a, hit,
             new_key, m_col, m_mn, m_cnt, m_row)

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
    return indptr, col, mn, cnt


def with_neighbours_low_deg(stats: RowStats, A, n: int, block: int = 20_000):
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
    Every force law divides a 0 by 1, thus the row is never averaged,
    the repulsion too. The node would never move. The caller must therefore
    give `make_plan` the true degree of `A`, and not the count of `D`.

    2026-08-28: reads `A`'s CSR row by row instead of `sp.triu(A, k=1)`.
    The old code built the undirected edge list once (`triu`) and derived
    BOTH directions' keep decisions from it; scanning `A`'s own directed
    entries and keeping `A[u, v]` in row `u` iff `deg(v) >= deg(u)` (and
    never `u == v`, matching `triu(k=1)`'s exclusion of the diagonal) asks
    the SAME question of the SAME edges, one direction at a time, and it is
    already row-major and column-ascending -- `A.sort_indices()` keeps
    that -- so no COO and no sort are needed to reach it.
    """
    deg = np.diff(A.indptr)
    edges_kept = 0

    def source(s, e):
        nonlocal edges_kept
        lo, hi = int(A.indptr[s]), int(A.indptr[e])
        col = A.indices[lo:hi]
        row = np.repeat(np.arange(e - s, dtype=np.int64),
                        np.diff(A.indptr[s:e + 1]))
        keep = (deg[col] >= deg[row + s]) & (col != row + s)
        row, col = row[keep], col[keep]
        edges_kept += col.size
        return row, col

    indptr, col, mn, cnt = _merge_row_edges(stats, n, source, block)
    return RowStats(indptr, col, mn, cnt, n, raw=stats.extra.get("raw", 0),
                    edges_kept=int(edges_kept))


def with_all_neighbours(stats: RowStats, A, n: int, block: int = 20_000):
    """Add EVERY edge of `A` at `h = 1`, and keep the walk pairs.

    The specification says that no neighbour may be missing. A walk finds
    most of them and not all: at 1.13M nodes the walks of the earlier
    policy held 68% of the directed edges. This function closes that gap by
    construction, thus the count of `h = 1` entries EQUALS `A.nnz`.

    An edge that a walk also found keeps `h = 1`, which is the minimum, thus
    the merge cannot raise a distance.

    2026-08-28: reads `A`'s CSR row by row and merges it into `stats` in
    ROW BLOCKS, in place of one `A.tocoo()` and one global sort-and-reduce
    over the concatenation. See `_merge_row_edges` and this module's
    docstring for the memory this removes.
    """
    def source(s, e):
        lo, hi = int(A.indptr[s]), int(A.indptr[e])
        col = A.indices[lo:hi]
        row = np.repeat(np.arange(e - s, dtype=np.int64),
                        np.diff(A.indptr[s:e + 1]))
        return row, col

    indptr, col, mn, cnt = _merge_row_edges(stats, n, source, block)
    return RowStats(indptr, col, mn, cnt, n, raw=stats.extra.get("raw", 0))
