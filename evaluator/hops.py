#!/bin/env python3
"""evaluator.hops -- the TRUE hop distance of a pair, from the graph.

The hop target must be the graph's real distance, not a weight an
augmentation stored earlier. `fodined/modular.py` reads `D.data`, the walk
gap of the pairs its policy sampled: an upper bound, not the hop distance.
This module reads the graph.

Three backends, one meaning: a `(k,)` float64 array, `np.inf` for a pair
with no path.

    _pll        exact.  NetworKit PrunedLandmarkLabeling. No matrix at any
                time; a query intersects two small label sets.
    _bfs        exact.  SciPy `shortest_path`, in BLOCKS of sources.
    _landmark   approximate. An UPPER bound from a few full BFS sweeps.

PRD invariant I3: no function here builds an `(n, n)` array, at any time.
The block rule of `_bfs` and the tile rule of `_landmark` enforce it.

`networkit` is imported INSIDE `_pll` (PRD invariant I6), thus
`import evaluator` works when NetworKit is absent.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

# The label of "no path". Every backend of this module returns it.
UNREACHABLE = np.inf

# The return of a PLL query for a pair with no path. NetworKit does not
# document this value; the benchmark found it. See `_pll`.
PLL_UNREACHABLE = np.uint64(2 ** 64 - 1)


# ---------------------------------------------------------------------------
# The choice of the backend
# ---------------------------------------------------------------------------
def choose_backend(n, n_sources):
    """The backend name for a graph of `n` nodes and `n_sources` sources.

    The rule comes from the measurements of
    `experiments/bench_shortest_path_gemsec.py`:

        n_sources > 0 and n_sources <= 512 and n > 200_000  ->  'bfs'
        networkit importable                                ->  'pll'
        otherwise                                           ->  'bfs'

    PLL pays one fixed cost for each GRAPH, and that cost grows with the
    graph. SciPy pays one BFS sweep for each distinct SOURCE. Thus few
    sources on a big graph is the one case where SciPy wins, and the PLL
    index for that graph would cost more than the whole query set.

    `n_sources` is the count of DISTINCT sources, and not the pair count.
    `landmark` is never returned: it is approximate, thus a user asks for
    it by name.
    """
    if n_sources > 0 and n_sources <= 512 and n > 200_000:
        return "bfs"
    return "pll" if _has_networkit() else "bfs"


def _has_networkit():
    """True when `networkit` can be imported. It does NOT import it."""
    try:
        return importlib.util.find_spec("networkit") is not None
    except (ImportError, ValueError):        # a broken or partial install
        return False


# ---------------------------------------------------------------------------
# The public function
# ---------------------------------------------------------------------------
def hop_distance(A, n, pairs, backend="auto", rng=None, budget=200e6):
    """The hop distance of every pair. `(k,)` float64, `np.inf` when no path.

    `backend` is 'auto', 'pll', 'bfs' or 'landmark'. 'auto' counts the
    DISTINCT sources of `pairs` and it asks `choose_backend`.

    'landmark' is approximate and it needs `rng`: PRD invariant I1 gives
    one generator to one task, thus a missing `rng` raises and it does not
    take a silent seed.
    """
    pairs = np.asarray(pairs)
    if pairs.size == 0:
        return np.empty(0, dtype=np.float64)
    if backend == "auto":
        backend = choose_backend(n, int(np.unique(pairs[:, 0]).size))
    if backend == "pll":
        return _pll(A, n, pairs)
    if backend == "bfs":
        return _bfs(A, n, pairs, budget)
    if backend == "landmark":
        if rng is None:
            raise ValueError(
                "hop_distance: backend='landmark' needs `rng`. The landmarks "
                "are sampled, thus the result is not reproducible without "
                "the generator of the task.")
        return _landmark(A, n, pairs, rng)
    raise ValueError(
        f"hop_distance: unknown backend {backend!r}; the names are "
        f"'auto', 'pll', 'bfs', 'landmark'.")


# ---------------------------------------------------------------------------
# Exact: Pruned Landmark Labeling
# ---------------------------------------------------------------------------
def _pll(A, n, pairs):
    """Exact hop distance from NetworKit PrunedLandmarkLabeling.

    Akiba et al., SIGMOD 2013. A query intersects two small label sets,
    thus NO distance matrix exists at any time.

    TWO TRAPS, both found by `experiments/bench_shortest_path_gemsec.py`:

    1. A query of a pair with NO PATH returns `2**64 - 1`. NetworKit does
       not document this. Untested, it becomes a distance of 18
       quintillion and it poisons every score. This function tests for it
       and it writes `np.inf`.
    2. `nk.GraphFromCoo` SEGFAULTS on a symmetric matrix that carries a
       `data` array -- a segmentation fault, and not an exception, thus
       nothing is caught. The safe form is the UPPER TRIANGLE as two
       contiguous `uint64` `(i, j)` arrays, which is what this function
       passes. `GraphFromCoo` inserts each undirected pair twice anyway.
    """
    import networkit as nk                   # optional, thus inside (I6)

    up = sp.triu(A, k=1).tocoo()
    g = nk.GraphFromCoo(
        (np.ascontiguousarray(up.row, dtype=np.uint64),
         np.ascontiguousarray(up.col, dtype=np.uint64)),
        n=n, weighted=False, directed=False)
    del up
    pll = nk.distance.PrunedLandmarkLabeling(g)
    pll.run()
    q = np.fromiter((pll.query(int(u), int(v)) for u, v in pairs),
                    dtype=np.uint64, count=pairs.shape[0])
    out = q.astype(np.float64)
    out[q == PLL_UNREACHABLE] = UNREACHABLE
    return out


# ---------------------------------------------------------------------------
# Exact: SciPy, in blocks of sources
# ---------------------------------------------------------------------------
def _bfs(A, n, pairs, budget=200e6):
    """Exact hop distance from SciPy `shortest_path`, in source blocks.

    THE TRAP: `shortest_path` ALWAYS returns a DENSE `(len(indices), n)`
    float64 array. `indices=` limits the ROWS and not the density, because
    the value of an unreachable cell is `inf` and `inf` is not a zero to
    omit. A call for every source of a large sample is therefore the full
    `(n, n)` matrix: 10 TB at 1.13M nodes.

    Thus the DISTINCT sources go in blocks of `budget / (n * 8)` rows. One
    block holds the peak near 200 MB and it does not grow with the pair
    count.

    Cost: one BFS sweep for each DISTINCT source.
    """
    srcs = np.unique(pairs[:, 0])
    block = max(1, int(budget / (n * 8)))
    src_pos = np.searchsorted(srcs, pairs[:, 0])
    out = np.empty(pairs.shape[0], dtype=np.float64)
    for s in range(0, srcs.size, block):
        blk = srcs[s:s + block]
        d = shortest_path(A, method="D", unweighted=True, indices=blk)
        m = (src_pos >= s) & (src_pos < s + blk.size)
        out[m] = d[src_pos[m] - s, pairs[m, 1]]
        del d                                # free the block before the next
    return out


# ---------------------------------------------------------------------------
# Approximate: the landmark oracle
# ---------------------------------------------------------------------------
def _landmark(A, n, pairs, rng, count=64):
    """An UPPER bound of the hop distance, from `count` full BFS sweeps.

    `d(u, v) <= min over L of ( d(u, L) + d(L, v) )`, by the triangle
    inequality. The estimate is never below the true distance, thus the
    caller must report the mean error with the score.

    It reuses `fodiwalk.augment_graph.landmarks`, which is validated at
    1.13M nodes: `pick` takes one half of the landmarks by degree and one
    half at random, and `distances` builds the `(count, n)` table as int16
    in BLOCKS, thus 113 MB and not 454 MB at that size.

    The table is `(count, n)` and never `(n, n)`. The min-sum runs over
    blocks of PAIRS, thus the temporary tile stays small.

    A pair that no landmark reaches gets `np.inf`, the same label that the
    exact backends give.
    """
    lm, table, unreached = _landmark_table(A, n, count, rng)
    out = np.empty(pairs.shape[0], dtype=np.float64)
    step = 200_000                           # keeps the (count, step) tile small
    for s in range(0, pairs.shape[0], step):
        p = pairs[s:s + step]
        du = table[:, p[:, 0]].astype(np.int32)
        dv = table[:, p[:, 1]].astype(np.int32)
        est = (du + dv).min(axis=0)
        del du, dv
        val = est.astype(np.float64)
        val[est >= unreached] = UNREACHABLE  # one leg has no path
        out[s:s + step] = val
    return out


def _landmark_table(A, n, count, rng):
    """`(lm, table, unreached)`. The `(count, n)` int16 distance table.

    `fodiwalk.augment_graph.landmarks` owns this code and it is validated
    at scale, thus it is the first choice. The import is INSIDE the
    function (PRD invariant I6) and the local form below runs when
    `fodiwalk` is absent.
    """
    try:
        from fodiwalk.augment_graph import landmarks as lmod
        lm = lmod.pick(A, n, count, rng)
        return lm, lmod.distances(A, n, lm), int(lmod.UNREACHED)
    except ImportError:
        pass

    # The local form. Same rule: one half by degree, one half at random,
    # and the table goes to int16 one block at a time.
    unreached = 32767
    count = min(count, n)
    deg = np.diff(A.indptr)
    n_hub = count // 2
    hubs = np.argsort(-deg)[:n_hub]
    rest = np.setdiff1d(np.arange(n), hubs)
    extra = rng.choice(rest, size=min(count - n_hub, rest.size), replace=False)
    lm = np.sort(np.concatenate([hubs, extra]))
    block = max(1, int(200e6 / (n * 8)))
    table = np.empty((lm.size, n), dtype=np.int16)
    for s in range(0, lm.size, block):
        d = shortest_path(A, method="D", unweighted=True,
                          indices=lm[s:s + block])
        d[~np.isfinite(d)] = float(unreached)
        table[s:s + d.shape[0]] = np.minimum(d, float(unreached)).astype(np.int16)
        del d
    return lm, table, unreached
