"""graph_augmenting.hopfill -- dense hop-fill ``augment_graph`` (T2.1).

Reference implementation of the "augmenting graph" stage
(ARCHITECTURE.md): given a graph ``G``, connect every reachable pair
``(u, v)`` with an edge weighted by hop distance ``d(u, v)``. This is a
faithful port of ``get_hops_csr`` (parallel BFS, one source node per
thread) from ``fdge_numba/forcedirected_numba/model_204_shell.py``, built
on top of ``core.csr.graph_to_csr`` instead of reimplementing the
NetworkX -> CSR conversion.

Contract (API_DESIGN.md, "New augmentation policy"):

    augment_graph(G, is_sparse=True, unreachable=None, **kwargs) -> D

``D`` is the **hop-distance matrix**, not a 0/1 adjacency indicator and
not a physical/Euclidean weight:

* ``D[u, u] == 0``
* ``D[u, v] == 1`` for a direct edge ``(u, v)`` in ``G``
* ``D[u, v] == k`` for a pair whose shortest path in ``G`` has ``k`` hops
* ``D[u, v] == n`` for node pairs from disconnected components (no path 
  between them)

This is exactly the "hops" contract the shell-averaging force law
(``embedding/``) needs: it derives shell counts ``|S_h(u)|`` from ``D``
itself (see ``get_shell_counts`` below, kept here as an optional helper,
*not* part of ``augment_graph``'s return value -- ARCHITECTURE.md
"Derived quantities live between stages, not inside a stage's output").

``is_sparse`` -- dense-by-nature, not a real toggle
----------------------------------------------------
Per ARCHITECTURE.md's "Known limitation": hop-fill connects *every*
reachable pair, so the result is inherently `O(n^2)` dense information
regardless of what container holds it -- there is no genuinely sparse
representation of "distance to every other node" the way there is for an
adjacency list. Rather than silently ignoring the ``is_sparse`` flag
(surprising) or raising on ``is_sparse=False`` (needlessly rigid for a
free function meant to be called either way), this implementation honors
the flag **at the container level only**:

* ``is_sparse=False`` -> plain ``(n, n)`` ``int32`` ``np.ndarray``.
* ``is_sparse=True`` (default) -> the same dense int32 matrix wrapped in
  ``scipy.sparse.csr_matrix``. Because the matrix has essentially no
  structural zeros (every reachable pair is "nonzero" in the sense of
  carrying a real hop distance, and even same-component zero-hop
  self-pairs are explicit diagonal entries), this buys none of sparse's
  usual memory/compute win -- it exists only so callers that always ask
  for ``is_sparse=True`` (the ``ForceDirected.augment_graph`` default)
  get a `scipy.sparse` type back, satisfying the *type* half of the
  indexability contract (API_DESIGN.md) rather than silently downgrading
  them to dense. Both branches satisfy ``D[i, j]`` indexing
  (API_DESIGN.md "indexability contract") -- a `csr_matrix` returns a
  ``(1, 1)`` sparse matrix for scalar ``[i, j]`` access, which is why
  tests below coerce with ``int(D[i, j])`` before comparing.

A genuinely sparse hop-fill (bounded hop radius, kNN-based) is future
work (T2.2 / RPD.md Q5), not this task.

Import discipline (RPD.md Sec5): this module imports ``numpy``,
``scipy.sparse``, ``numba``, and ``fdge2.core`` only. It does not import
``fdge2.graph_building`` or ``fdge2.embedding``, and it does not assume
``G`` came from a specific builder -- any object satisfying
``core.csr.graph_to_csr``'s duck-typed interface (``.nodes()`` /
``.degree()`` / ``.edges()``) works.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numba import njit, prange

from fdge2.core import graph_to_csr


# ---------------------------------------------------------------------------
# Parallel BFS -> all-pairs hop distances (ported from get_hops_csr)
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _get_hops_csr(indptr, indices, n, unreachable):
    """All-pairs hop distances by parallel BFS, one source per thread.

    Returns int32 (n, n); hops[u, u] = 0; unreachable pairs get the
    sentinel value ``unreachable``. Verbatim port of
    ``model_204_shell.get_hops_csr`` -- do not change the algorithm here,
    only where it's called from.
    """
    hops = np.full((n, n), unreachable, dtype=np.int32)
    for src in prange(n):
        dist = hops[src]
        dist[src] = 0
        frontier = np.empty(n, dtype=np.int64)
        nxt = np.empty(n, dtype=np.int64)
        frontier[0] = src
        f_len = 1
        level = 0
        while f_len > 0:
            level += 1
            nxt_len = 0
            for fi in range(f_len):
                u = frontier[fi]
                for p in range(indptr[u], indptr[u + 1]):
                    v = indices[p]
                    if dist[v] == unreachable:
                        dist[v] = level
                        nxt[nxt_len] = v
                        nxt_len += 1
            frontier, nxt = nxt, frontier
            f_len = nxt_len
    return hops


def get_shell_counts(D, n_bins):
    """counts[u, h] = |S_h(u)| = number of nodes exactly h hops from u.

    Optional helper, ported from ``model_204_shell.get_shell_counts``.
    NOT part of ``augment_graph``'s contract (ARCHITECTURE.md "Derived
    quantities live between stages, not inside a stage's output") --
    kept here because it's broadly useful and cheap to derive from the
    same hop matrix, but the embedding stage is expected to call this
    itself (or an equivalent) rather than have ``augment_graph`` hand it
    a ``(D, shell_counts)`` tuple.

    ``D`` may be dense or a ``scipy.sparse`` matrix produced by
    ``augment_graph``; this densifies internally (``np.bincount`` needs a
    dense row) -- callers with a huge ``D`` may prefer to write a
    row-streamed version against the CSR triple instead.

    ``n_bins`` must be > the max hop value present in ``D``.
    """
    Dd = D.toarray() if sp.issparse(D) else np.asarray(D)
    n = Dd.shape[0]
    counts = np.zeros((n, n_bins), dtype=np.int64)
    for u in range(n):
        binc = np.bincount(Dd[u], minlength=n_bins)
        counts[u, :] = binc[:n_bins]
    return counts


# ---------------------------------------------------------------------------
# augment_graph
# ---------------------------------------------------------------------------
def augment_graph(G, is_sparse: bool = True, unreachable: int | None = None, **kwargs):
    """Stage 2 reference implementation: dense hop-fill.

    Connects every reachable pair ``(u, v)`` in ``G`` with an edge
    weighted by hop distance. See module docstring for the exact ``D``
    semantics and the ``is_sparse`` decision.

    Parameters
    ----------
    G           : an (unweighted) graph satisfying ``core.csr.graph_to_csr``'s
                  duck-typed interface (``.nodes()`` / ``.degree()`` /
                  ``.edges()``), e.g. any ``networkx.Graph``. Edge weights on
                  ``G``, if any, are ignored -- hop-fill only uses graph
                  topology (matches the reference ``get_hops_csr``, which
                  never looks at edge weights either).
    is_sparse   : True (default) -> wrap the dense result in
                  ``scipy.sparse.csr_matrix``. False -> plain ``(n, n)``
                  int32 ndarray. See module docstring: the underlying
                  computation is dense either way (known limitation, not
                  a bug -- ARCHITECTURE.md).
    unreachable : sentinel hop value for disconnected pairs. Defaults to
                  ``n`` (node count), matching the legacy convention used
                  by ``model_204_shell.py`` and assumed by the shell
                  force law's shell-counting logic. Pass an explicit
                  value only if you have a concrete reason to deviate.

    Returns
    -------
    D : int32, shape (n, n) -- either a plain ``np.ndarray`` (is_sparse=False)
        or a ``scipy.sparse.csr_matrix`` wrapping the same values
        (is_sparse=True). ``D[u, u] == 0``; ``D[u, v] == 1`` for direct
        edges; higher for indirect; ``unreachable`` for disconnected pairs.
        Node index ``i`` corresponds to ``list(G.nodes())[i]`` (same
        ordering ``core.csr.graph_to_csr`` uses).
    """
    indptr, indices, degrees = graph_to_csr(G)
    n = indptr.shape[0] - 1
    sentinel = n if unreachable is None else int(unreachable)

    hops = _get_hops_csr(indptr, indices, n, np.int32(sentinel))
    D = np.asarray(hops, dtype=np.int32)

    if is_sparse:
        return sp.csr_matrix(D)
    return D
