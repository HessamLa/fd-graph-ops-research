"""graph_augmenting.hopfill -- dense hop-fill ``augment_graph`` (fdge_jax port).

Reference implementation of the "augmenting graph" stage. Given a graph
``G``, connects every reachable pair ``(u, v)`` with an edge weighted by
hop distance ``d(u, v)``. Port of ``fdge2.graph_augmenting.hopfill``, but
built on ``_levelwise_reach``'s sparse boolean-matrix-power BFS instead
of fdge2's ``numba``-parallel per-source BFS (``get_hops_csr``) -- see
``_levelwise_reach.py`` for why matrix powers replace the per-source
loop now that there's no ``numba`` to parallelize it with, and
``docs/DESIGN.md`` for the overall port rationale.

Contract (unchanged from fdge2):

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
*not* part of ``augment_graph``'s return value).

``D[u, u] == 0`` without a stored entry
----------------------------------------
Unlike fdge2's dense/scipy.sparse ``D`` (which stores an explicit
diagonal), ``HopMatrix``'s absence-means-0 semantics (``core/csr.py``)
already give ``D[u, u] == 0`` for free with *no* stored entry -- so this
implementation never adds self pairs to the CSR triple. Every other
pair distinct from ``u`` genuinely needs a decision (reachable: store
the real hop count; unreachable: store the sentinel), which is why
those, unlike the diagonal, are always stored explicitly.

``is_sparse`` -- dense-by-nature, not a real toggle
----------------------------------------------------
Hop-fill connects *every* reachable pair, so the result is inherently
``O(n^2)`` information regardless of what container holds it (including
the disconnected-pair sentinel, which must be stored for every single
disconnected pair -- there's no way to represent "everything not
otherwise listed is unreachable" as an absence, because absence in
``HopMatrix`` already means something else: "no stored distance",
which per the module contract must mean 0, not ``n``). This
implementation honors ``is_sparse`` at the *container* level only:

* ``is_sparse=False`` -> plain ``(n, n)`` ``int32`` ``np.ndarray``.
* ``is_sparse=True`` (default) -> a ``core.csr.HopMatrix`` with (for a
  fully connected graph) essentially no structural zeros -- this buys
  none of the usual sparse memory/compute win, it exists only so
  ``ForceDirected.augment_graph``'s default gets the shared ``HopMatrix``
  type back rather than being silently downgraded to dense.

A genuinely sparse hop-fill is ``sparse_hops.py`` (bounded hop radius) --
not this module.

Import discipline (docs/DESIGN.md): this module imports ``numpy``,
``scipy.sparse``, and ``fdge_jax.core`` only. It does not import
``fdge_jax.graph_building`` or ``fdge_jax.embedding``, and it does not
assume ``G`` came from a specific builder -- any object satisfying
``core.csr.graph_to_csr``'s duck-typed interface (``.nodes()`` /
``.degree()`` / ``.edges()``) works.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from fdge_jax.core import graph_to_csr, HopMatrix

from ._levelwise_reach import build_adjacency, levelwise_reach


def get_shell_counts(D, n_bins):
    """counts[u, h] = |S_h(u)| = number of nodes exactly h hops from u.

    Optional helper, ported from fdge2's ``model_204_shell.get_shell_counts``.
    NOT part of ``augment_graph``'s contract -- kept here because it's
    broadly useful and cheap to derive from the same hop matrix, but the
    embedding stage is expected to call this itself (or an equivalent)
    rather than have ``augment_graph`` hand it a ``(D, shell_counts)``
    tuple.

    ``D`` may be a ``HopMatrix`` or a plain ``np.ndarray`` (either return
    shape of ``augment_graph``); this densifies internally (``np.bincount``
    needs a dense row) -- callers with a huge ``D`` may prefer to write a
    row-streamed version against the CSR triple instead.

    ``n_bins`` must be > the max hop value present in ``D``.
    """
    Dd = D.toarray() if isinstance(D, HopMatrix) else np.asarray(D)
    n = Dd.shape[0]
    counts = np.zeros((n, n_bins), dtype=np.int64)
    for u in range(n):
        binc = np.bincount(Dd[u], minlength=n_bins)
        counts[u, :] = binc[:n_bins]
    return counts


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
                  topology.
    is_sparse   : True (default) -> return a ``core.csr.HopMatrix``.
                  False -> plain ``(n, n)`` int32 ndarray. See module
                  docstring: the underlying computation is dense either
                  way (known limitation, not a bug).
    unreachable : sentinel hop value for disconnected pairs. Defaults to
                  ``n`` (node count), matching the legacy convention
                  assumed by the shell force law's shell-counting logic.
                  Pass an explicit value only if you have a concrete
                  reason to deviate.

    Returns
    -------
    D : int32, shape (n, n) -- either a plain ``np.ndarray`` (is_sparse=False)
        or a ``core.csr.HopMatrix`` (is_sparse=True). ``D[u, u] == 0``;
        ``D[u, v] == 1`` for direct edges; higher for indirect;
        ``unreachable`` for disconnected pairs. Node index ``i``
        corresponds to ``list(G.nodes())[i]`` (same ordering
        ``core.csr.graph_to_csr`` uses).
    """
    indptr, indices, degrees = graph_to_csr(G)
    n = indptr.shape[0] - 1
    sentinel = n if unreachable is None else int(unreachable)

    A = build_adjacency(indptr, indices, n)
    rows, cols, levels, visited = levelwise_reach(A, max_level=None)

    # Fill in the unreachable sentinel for every off-diagonal pair that the
    # BFS never touched (both directions -- symmetric, since G is
    # undirected). This is the one genuinely O(n^2) step, unavoidable
    # because hop-fill's whole point is to make disconnection an explicit,
    # stored value rather than an absence (see module docstring).
    visited_dense = visited.toarray()
    np.fill_diagonal(visited_dense, True)  # never mark self as "unreachable"
    missing_u, missing_v = np.nonzero(~visited_dense)

    if missing_u.size:
        rows = np.concatenate([rows, missing_u])
        cols = np.concatenate([cols, missing_v])
        levels = np.concatenate(
            [levels, np.full(missing_u.shape[0], sentinel, dtype=np.int32)])

    coo = sp.coo_matrix((levels, (rows, cols)), shape=(n, n))
    csr = coo.tocsr()

    if is_sparse:
        return HopMatrix(indptr=csr.indptr, indices=csr.indices,
                          data=csr.data, n=n)
    return csr.toarray().astype(np.int32)
