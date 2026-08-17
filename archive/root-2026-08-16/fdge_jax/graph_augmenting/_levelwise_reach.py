"""graph_augmenting._levelwise_reach -- shared level-synchronous BFS core.

Internal helper (leading underscore -- not part of the package's public
surface) shared by ``hopfill.py`` and ``sparse_hops.py``. Both policies
need "hop distance from every node to every other node it can reach,"
differing only in *when they stop* (hopfill: convergence; sparse_hops:
a fixed radius) and *what they do with unvisited pairs at the end*
(hopfill: fill an ``unreachable`` sentinel; sparse_hops: leave absent).
Factoring the loop here means neither module reimplements the
matrix-power BFS (docs/DESIGN.md "graph_augmenting/ -- plain
NumPy/SciPy, no Numba, no JAX").

Why sparse boolean matrix powers, not a per-source BFS loop
-------------------------------------------------------------
fdge2's Numba kernels parallelized "one BFS per source node" across
threads -- fast, but that's an ``n``-way Python-shaped loop under the
hood (even if it runs in parallel C). Without Numba, an explicit
per-source loop in plain Python would be catastrophically slow. Instead
this treats the *whole node set* as simultaneous BFS sources at once,
using the adjacency matrix's own sparsity: at level ``k`` the reachable
set is (boolean) ``A^k``, computed incrementally as
``frontier_{k} = new_{k-1} @ A`` (expand only from nodes newly visited
at the previous level, not the whole visited set -- that's what keeps
each step's cost proportional to the *frontier*, not to ``n``). Total
work is a handful of sparse matmuls (one per hop level, up to the graph
diameter or a caller-supplied radius), each a compiled SciPy/BLAS call
-- independent of ``n`` in the number of Python-level steps, unlike a
per-source loop which is ``O(n)`` Python-level steps no matter how fast
each one is in C.

Import discipline (docs/DESIGN.md): numpy + scipy.sparse only.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_adjacency(indptr, indices, n) -> sp.csr_matrix:
    """Symmetric boolean adjacency from a CSR triple (as ``graph_to_csr`` gives).

    ``graph_to_csr`` already emits both ``u->v`` and ``v->u`` for every
    edge and never emits self loops, so the raw CSR triple is already
    exactly what a symmetric, loop-free boolean adjacency needs -- no
    extra symmetrization step here.
    """
    data = np.ones(indices.shape[0], dtype=bool)
    return sp.csr_matrix((data, indices, indptr), shape=(n, n), dtype=bool)


def levelwise_reach(A: sp.csr_matrix, max_level: int | None = None):
    """Level-synchronous BFS via boolean matrix powers, every source at once.

    Parameters
    ----------
    A         : symmetric boolean adjacency (``scipy.sparse.csr_matrix``,
                no self loops -- see ``build_adjacency``).
    max_level : stop after this many hop levels even if the frontier is
                still nonempty (``sparse_hops``'s ``radius``). ``None``
                (default, used by ``hopfill``) runs to convergence --
                i.e. until the frontier empties, which happens after
                exactly ``diameter`` levels.

    Returns
    -------
    rows, cols, levels : 1-D int arrays, one entry per ``(u, v)`` pair
        discovered with ``u != v``, ``levels[i]`` = hop distance
        ``d(rows[i], cols[i])``. Self pairs (distance 0) are never
        included -- callers that want a diagonal add it themselves
        (neither ``hopfill`` nor ``sparse_hops`` wants a stored
        ``D[u,u]``, see their docstrings).
    visited : ``scipy.sparse.csr_matrix`` (bool), the union of the
        identity and every ``new`` matrix recorded -- i.e. ``visited[u,
        v]`` is True iff ``v`` was reached from ``u`` at or before the
        level the loop stopped at. ``hopfill`` uses this to find the
        never-visited pairs it must stamp with the ``unreachable``
        sentinel; ``sparse_hops`` ignores it (absence is the contract).
    """
    n = A.shape[0]
    visited = sp.identity(n, dtype=bool, format="csr")
    frontier = A
    level = 1

    collected_rows = []
    collected_cols = []
    collected_levels = []

    while frontier.nnz > 0 and (max_level is None or level <= max_level):
        # new = frontier AND NOT visited -- boolean set-difference. scipy
        # sparse bool arithmetic wraps like numpy bool (True+True==True,
        # True-True==False), so plain +/- on bool csr matrices already
        # gives correct boolean OR / AND-NOT here -- no int8 detour needed.
        # frontier.multiply(visited) == frontier AND visited (elementwise
        # multiply on 0/1 is AND); subtracting that from frontier leaves
        # exactly the positions in frontier but not yet visited.
        new = frontier - frontier.multiply(visited)
        new.eliminate_zeros()

        if new.nnz > 0:
            new_coo = new.tocoo()
            collected_rows.append(new_coo.row)
            collected_cols.append(new_coo.col)
            collected_levels.append(np.full(new_coo.row.shape[0], level, dtype=np.int32))

        visited = (visited + new).astype(bool)

        if max_level is not None and level >= max_level:
            break
        # expand only from nodes newly visited this level -- keeps each
        # matmul's cost tied to the frontier size, not to n or to the full
        # visited set.
        frontier = (new @ A)
        frontier.eliminate_zeros()
        if frontier.nnz == 0:
            break
        level += 1

    if collected_rows:
        rows = np.concatenate(collected_rows)
        cols = np.concatenate(collected_cols)
        levels = np.concatenate(collected_levels)
    else:
        rows = np.empty(0, dtype=np.int64)
        cols = np.empty(0, dtype=np.int64)
        levels = np.empty(0, dtype=np.int32)

    return rows, cols, levels, visited
