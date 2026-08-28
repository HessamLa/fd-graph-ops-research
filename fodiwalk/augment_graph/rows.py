#!/bin/env python3
"""rows.py -- the row-blocked carrier of `nbr_walk`, and its plain CSR.

`walk_rows` and its neighbour merges (`row_merge.py`) used to carry a pair
as ONE integer key, `row * n + col` (`pairs.py`'s form). `RowStats` carries
the SAME information split into `indptr`/`col`, the layout a CSR already
keeps in `indptr`/`indices` -- because building `key` for the whole graph
cost +198 MB at 150,000 nodes, on top of an already-large peak, for a
number `col // n` gives back for free. `to_csr_directed_rows` builds a CSR
from it directly, with no COO step and no sort; it returns a `RowCSR`, a
plain object and not a `scipy.sparse.csr_matrix` -- see its docstring for
why that is safe for `D` and `freq`.

Provenance: new 2026-08-28 (`agentic-log/10.mem-agent/`), split out of
`pairs.py` on the same day to keep both files under
`tests/test_structure.py`'s 300-line cap -- `pairs.py`'s own docstring
still names the flat `key` form these replace.

Every function and method here is pure, and none of them print.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class RowStats:
    """The row-blocked carrier of a directed walk result. Replaces `key`.

    Row `u`'s partners are `col[indptr[u]:indptr[u + 1]]`, ascending and
    free of a repeat -- the same rule a CSR's `indices` keeps. `mn[i]` is
    the first step that reached `col[i]` (int16, `1 .. walk_len - 1`), and
    `cnt[i]` is how many times the row reached it (int32).

    `key`, the OLD one-array carrier (`row * n + col`), is NOT stored: for
    the whole graph it cost +198 MB at 150,000 nodes, on top of an already
    large peak, for a number the row-blocked form already implies. A caller
    that still wants it -- `golden.py`'s digest, a debug script -- reads it
    through `stats["key"]`, built ONLY on that ask and never kept. `n`
    therefore travels with the object.

    `["mn"]` and `["cnt"]` also go through `__getitem__`, and they widen
    back to the OLD dtype there (`int32`, `int64`). That keeps
    `golden_baseline.json` byte-exact: its digest hashes the dtype, and
    `cnt` was `int64` by the accident `np.add.reduceat` makes of an int32
    input (see `walks.py`'s `_reduce`). The narrow dtype is what the STORED
    object holds; the wide one is a compatibility view for a reader outside
    the hot path -- `golden.py`, `check_api.py`,
    `experiments/fodiwalk/bench_fodiwalk.py`. Internal code reads `.mn` /
    `.cnt` directly and never pays for the widening.

    Iteration (`__iter__`, thus `set(stats)`, `dict(stats)`) gives
    `indptr`/`col`/`mn`/`cnt` and whatever is in `extra` -- NOT `key`,
    which stays reachable only by asking for it BY NAME through
    `__getitem__`. `test_smoke.py::test_b8_graph_walk_reproduces_
    walk_rows_for_the_same_seed` compares two `RowStats` this way and does
    not need `key`; a caller that does still gets it through `stats["key"]`.

    Provenance: new 2026-08-28, `agentic-log/10.mem-agent/`.
    """

    __slots__ = ("indptr", "col", "mn", "cnt", "n", "extra")

    _ARRAY_KEYS = ("indptr", "col", "mn", "cnt")

    def __init__(self, indptr, col, mn, cnt, n: int, **extra):
        self.indptr, self.col, self.mn, self.cnt = indptr, col, mn, cnt
        self.n = n
        self.extra = extra          # small, non-array passthrough: `raw`, ...

    @property
    def row(self):
        """`row[i]`: which row `col[i]` belongs to. Built on the ask, and
        never cached -- the whole point of this class is that nothing on
        the hot path holds an `(nnz,)` row array."""
        return np.repeat(np.arange(self.n, dtype=np.int64),
                         np.diff(self.indptr))

    def rows_of(self, idx):
        """The row of each position in the SORTED array `idx`, without
        building `row` for the whole object. One `searchsorted`, thus it
        costs `O(idx.size log n)` and not `O(nnz)`."""
        return (np.searchsorted(self.indptr, idx, side="right") - 1).astype(
            np.int64)

    def __getitem__(self, k):
        if k == "key":
            return self.row * self.n + self.col.astype(np.int64)
        if k == "col":
            return self.col
        if k == "indptr":
            return self.indptr
        if k == "mn":
            return self.mn.astype(np.int32)
        if k == "cnt":
            return self.cnt.astype(np.int64)
        if k in self.extra:
            return self.extra[k]
        raise KeyError(k)

    def __contains__(self, k):
        return k == "key" or k in self._ARRAY_KEYS or k in self.extra

    def __iter__(self):
        # "key" is NOT here: it is the array this class exists to avoid
        # building. A caller that wants it must ask for it BY NAME
        # (`__getitem__`, above).
        yield from self._ARRAY_KEYS
        yield from self.extra

    def __len__(self):
        return len(self._ARRAY_KEYS) + len(self.extra)

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default

    def with_extra(self, **kw):
        """The same rows, with an extra field attached (`freq`, e.g.). The
        arrays are NOT copied -- only the small `extra` dict grows."""
        extra = dict(self.extra)
        extra.update(kw)
        return RowStats(self.indptr, self.col, self.mn, self.cnt, self.n,
                        **extra)


def assert_row_sorted(indptr, col, block: int = 20_000):
    """Trap 5: `sp.csr_matrix((data, indices, indptr))` does NOT validate.
    A `col` that is not ascending inside a row gives a SILENTLY wrong
    matrix, not an error.

    Checked in ROW BLOCKS, like every other pass this task added: a row
    cannot span two blocks (a block is a range of whole rows), thus a
    block checks its own rows completely on its own, and the transient
    this costs is bounded by `block` rows and not by the whole of `col`.
    The first version of this function checked in one pass and cost
    another ~100 MB of live boolean arrays at 150,000 nodes -- measured
    building this memory gate, `agentic-log/10.mem-agent/` -- enough to
    matter next to the peak it guards.
    """
    n = indptr.size - 1
    for s in range(0, n, block):
        e = min(s + block, n)
        lo, hi = int(indptr[s]), int(indptr[e])
        if hi <= lo:
            continue
        c = col[lo:hi]
        row_start = np.zeros(c.size, dtype=bool)
        # every row's first entry in this block but row `s`'s own. A
        # trailing empty row's "start" equals `c.size` -- out of bounds,
        # and there is no entry there to flag anyway -- thus it is
        # dropped before the assignment.
        starts = indptr[s + 1:e] - lo
        row_start[starts[(starts >= 0) & (starts < c.size)]] = True
        not_start = row_start[1:]
        np.logical_not(not_start, out=not_start)     # in place: row_start -> not-a-start
        bad = c[1:] <= c[:-1]
        np.logical_and(bad, not_start, out=bad)
        if bad.any():
            i = int(np.flatnonzero(bad)[0]) + 1 + lo
            raise ValueError(
                f"col is not strictly ascending inside a row at position "
                f"{i} ({col[i - 1]} -> {col[i]}). RowStats/RowCSR promise "
                f"ascending, duplicate-free columns per row; a caller "
                f"broke it.")


class RowCSR:
    """A directed CSR, held as plain arrays and not `scipy.sparse.csr_matrix`.

    `nbr_walk`'s `D` and `freq` never need a scipy METHOD: no `.dot`, no
    `@`, no transpose, no `.getrow()`. Every real reader --
    `forcedirected.sell_c_sigma.make_plan` (through a chunk-local rebuild in
    `embed/planner.py`), `embed.degrees.resolve_degrees`,
    `embed.plan_contract.check` -- touches only `.indptr`, `.indices`,
    `.data`, `.shape`, `.nnz`. A `scipy.sparse.csr_matrix` validates and can
    copy on construction; this class does neither, and
    `to_csr_directed_rows` builds it straight from the arrays the row-blocked
    merge already holds.

    `.tocoo()` exists for the ONE reader that needs real matrix machinery:
    `merge.add_far_pairs`, which depends on `scipy.sparse` SUMMING a
    duplicate COO coordinate (`far > 0` only, `merge.py`'s docstring). It
    wraps into a real `sp.csr_matrix` for that one call; `add_far_pairs`
    itself already returns a real `sp.csr_matrix` from there on, so the
    far-pair path needs no other change.

    Provenance: new 2026-08-28, `agentic-log/10.mem-agent/`, after the
    project owner asked what `D` is a `scipy.sparse.csr_matrix` FOR. The
    answer, checked against every consumer: nothing that a plain object
    with these five attributes cannot give.
    """

    __slots__ = ("indptr", "indices", "data", "shape")

    def __init__(self, indptr, indices, data, shape):
        self.indptr, self.indices, self.data = indptr, indices, data
        self.shape = shape

    @property
    def nnz(self):
        return self.data.size

    def tocoo(self):
        return sp.csr_matrix((self.data, self.indices, self.indptr),
                             shape=self.shape).tocoo()


_IDX32_MAX = np.iinfo(np.int32).max


def to_csr_directed_rows(indptr, col, val, n: int):
    """A directed CSR straight from a row-blocked carrier. No COO, no sort.

    `indptr`/`col` already hold the CSR structure the walk built: row `u`'s
    slice is `col[indptr[u]:indptr[u + 1]]`, ascending, no duplicate.
    `assert_row_sorted` must have checked that already -- this function
    trusts it and does not check again, to keep the O(nnz) cost of the
    check to ONE pass for `D` and `freq` together (they share `indptr` and
    `col`; see `policy_nbr_walk.build`).

    `indptr` narrows to `int32` when `nnz` and `n` both fit -- true at
    every size this project runs, up to com_youtube's 1.13M nodes and
    ~23M edges. `RowStats.indptr` stays `int64` throughout the merge (the
    row-blocked design's own choice); this is the ONE place a copy narrows
    it, to match what `sp.csr_matrix((data, (row, col)))` already chose
    for `indptr` at this project's sizes -- `golden.py`'s digest hashes
    the dtype, and a wider `indptr` here would fail it for no value
    that moved. `col` is `int32` already (`RowStats`'s own contract), so
    the two stay the SAME dtype, as a CSR expects.

    `np.asarray(indptr, dtype=...)` is a no-op, no copy, when `indptr` is
    ALREADY that dtype -- which is true the second time this function
    runs on `D.indptr` to build `freq`, thus `freq.indptr is D.indptr`
    (see `policy_nbr_walk.build`).

    Returns a `RowCSR`, not a `scipy.sparse.csr_matrix` -- see its
    docstring for why that is safe here.
    """
    idx_max = max(int(indptr[-1]) if indptr.size else 0, n)
    idx_dtype = np.int32 if idx_max <= _IDX32_MAX else np.int64
    indptr = np.asarray(indptr, dtype=idx_dtype)
    return RowCSR(indptr, col, val, (n, n))
