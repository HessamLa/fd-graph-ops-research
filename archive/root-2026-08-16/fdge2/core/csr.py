"""core.csr -- shared CSR / indexable utilities for fdge2.

This module holds the small, stable primitives that both ``graph_augmenting/``
(which *produces* the augmented matrix ``D``) and ``embedding/`` (which
*walks* ``D``'s rows) need, so neither has to re-derive them and so the
CSR layout they agree on lives in exactly one place.

Import discipline (RPD.md §5): ``core/`` imports **numpy only**. These
helpers deliberately take a NetworkX graph *duck-typed* (via ``.nodes()``
/ ``.degree()`` / ``.edges()``) rather than importing ``networkx``, so
``core`` stays dependency-light and never imports a pipeline stage.

The indexability contract (API_DESIGN.md "indexability contract"):

* The *outward* contract on any ``G`` / ``D`` is that ``D[i, j]`` works
  for testing/debugging. A dense ``(n, n)`` ndarray satisfies this
  natively; a ``scipy.sparse.csr_matrix`` satisfies it at the Python
  level.
* The *hot loop* never uses ``D[i, j]`` random access on sparse data.
  Instead ``njit`` kernels iterate a row's actual entries via the CSR
  triple ``indptr`` / ``indices`` / ``data``:

      row i's neighbors : indices[indptr[i] : indptr[i+1]]
      row i's values    : data[indptr[i] : indptr[i+1]]   # same positions

  Neighbor ``v`` and its value sit at the same position ``p``, so no
  separate lookup is needed to go from a neighbor to its weight.
"""
from __future__ import annotations

import numpy as np


def graph_to_csr(Gx):
    """Convert an (undirected) NetworkX-style graph to CSR adjacency arrays.

    Ported verbatim from ``model_204_shell.graph_to_csr`` -- this is the
    proven conversion the reference augmentation (BFS hop-fill) is built
    on, promoted into ``core`` so ``graph_augmenting/`` and ``embedding/`` share
    one definition.

    Nodes are indexed ``0..n-1`` in the order of ``Gx.nodes()``. The graph
    is treated as undirected: each edge ``(u, v)`` contributes both
    ``u->v`` and ``v->u`` entries.

    Returns
    -------
    indptr  : int64 (n + 1,)   row pointer; row i spans [indptr[i], indptr[i+1])
    indices : int64 (nnz,)     neighbor node ids, grouped by source row
    degrees : int64 (n,)       degree of each node (== np.diff(indptr))
    """
    nodes = list(Gx.nodes())
    index = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    degrees = np.zeros(n, dtype=np.int64)
    for u, deg in Gx.degree():
        degrees[index[u]] = deg
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(degrees)
    indices = np.empty(indptr[-1], dtype=np.int64)
    cursor = indptr[:-1].copy()
    for u, v in Gx.edges():
        iu, iv = index[u], index[v]
        indices[cursor[iu]] = iv
        cursor[iu] += 1
        indices[cursor[iv]] = iu
        cursor[iv] += 1
    return indptr, indices, degrees


def row_of(indptr):
    """Source-node id for every edge position in a CSR layout.

    ``row_of(indptr)[p]`` is the row (source node) that owns edge position
    ``p`` in ``indices`` / ``data``. This is the ``np.repeat`` trick from
    IMPLEMENTATION.md -- it turns a per-row loop into a single vectorized
    NumPy gather, e.g. ``Z[indices] - Z[row_of(indptr)]`` computes every
    edge's ``Zdiff`` in one call (useful for the vectorized-NumPy
    correctness reference; the ``njit`` kernels iterate ``indptr`` rows
    directly and don't need it).

    Parameters
    ----------
    indptr : int (n + 1,) CSR row pointer.

    Returns
    -------
    int64 (nnz,) : source node id per edge position.
    """
    indptr = np.asarray(indptr)
    n = indptr.shape[0] - 1
    return np.repeat(np.arange(n), np.diff(indptr))


def n_rows(D):
    """Number of rows (nodes) in an indexable ``D``, dense or sparse.

    Both a dense ``(n, n)`` ndarray and a ``scipy.sparse`` matrix expose
    ``.shape``, so this works without ``core`` importing scipy. Used by
    ``ForceDirected.embed`` to size ``Z`` / ``dZ`` and the batch bounds
    from the augmented matrix alone.
    """
    return D.shape[0]
