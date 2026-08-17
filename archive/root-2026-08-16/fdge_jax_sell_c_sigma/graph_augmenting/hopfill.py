"""graph_augmenting.hopfill -- dense hop-fill ``augment_graph`` (fdge_jax_sell_c_sigma port).

Reference implementation of the "augmenting graph" stage. Given a graph
``G``, connects every reachable pair ``(u, v)`` with an edge weighted by
hop distance ``d(u, v)``. Port of ``fdge2.graph_augmenting.hopfill``, but
built on ``scipy.sparse.csgraph.shortest_path`` (all-pairs Dijkstra,
``unweighted=True``) instead of fdge2's ``numba``-parallel per-source
BFS (``get_hops_csr``) -- a compiled all-pairs shortest-path call is the
simplest correct thing here since hop-fill's result is dense anyway (see
"``is_sparse`` -- dense-by-nature" below), and ``docs/DESIGN.md`` has the
overall port rationale. (``sparse_hops.py`` is the genuinely sparse
policy; it stays on a hand-rolled boolean-matrix-power BFS instead, for
reasons documented there.)

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
itself.

``D[u, u] == 0`` without a stored entry
----------------------------------------
A ``scipy.sparse.csr_matrix``'s absence-means-0 semantics already give
``D[u, u] == 0`` for free with *no* stored entry -- so this
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
otherwise listed is unreachable" as an absence, because absence in a
``csr_matrix`` already means something else: "no stored distance",
which per the module contract must mean 0, not ``n``). This
implementation honors ``is_sparse`` at the *container* level only:

* ``is_sparse=False`` -> plain ``(n, n)`` ``int32`` ``np.ndarray``.
* ``is_sparse=True`` (default) -> a ``scipy.sparse.csr_matrix`` with (for
  a fully connected graph) essentially no structural zeros -- this buys
  none of the usual sparse memory/compute win, it exists only so
  ``ForceDirected.augment_graph``'s default gets a CSR container back
  rather than being silently downgraded to dense.

A genuinely sparse hop-fill is ``sparse_hops.py`` (bounded hop radius) --
not this module.

Import discipline (docs/DESIGN.md): this module imports ``numpy``,
``scipy.sparse``, ``networkx``, and ``fdge_jax_sell_c_sigma.core`` only. It
does not import ``fdge_jax_sell_c_sigma.graph_building`` or
``fdge_jax_sell_c_sigma.embedding``, and it does not assume ``G`` came from
a specific builder -- any ``networkx``-compatible graph (accepted by
``nx.to_scipy_sparse_array``) works.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import networkx as nx


def augment_graph(G, is_sparse: bool = True, unreachable: int | None = None, **kwargs):
    """Stage 2 reference implementation: dense hop-fill.

    Connects every reachable pair ``(u, v)`` in ``G`` with an edge
    weighted by hop distance. See module docstring for the exact ``D``
    semantics and the ``is_sparse`` decision.

    Parameters
    ----------
    G           : any graph accepted by ``nx.to_scipy_sparse_array``, e.g. a
                  ``networkx.Graph``. Edge weights on ``G``, if any, are
                  ignored -- hop-fill only uses graph topology.
    is_sparse   : True (default) -> return a ``scipy.sparse.csr_matrix``.
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
        or a ``scipy.sparse.csr_matrix`` (is_sparse=True). ``D[u, u] == 0``;
        ``D[u, v] == 1`` for direct edges; higher for indirect;
        ``unreachable`` for disconnected pairs. Node index ``i``
        corresponds to ``list(G.nodes())[i]``.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    sentinel = n if unreachable is None else int(unreachable)

    # weight=None -> pure 0/1 topology (hop-fill ignores edge weights).
    # scipy's all-pairs Dijkstra ('D') on an unweighted graph is exactly
    # BFS from every source at once, and returns a dense (n, n) float
    # matrix -- which costs nothing extra here since hop-fill's result is
    # O(n^2) anyway (see module docstring), unlike sparse_hops where a
    # forced-dense return would defeat the whole point.
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None, format="csr")
    D_sp = sp.csgraph.shortest_path(A, method="D", unweighted=True, directed=False)

    # inf (no path) -> sentinel. The diagonal comes back 0 from
    # shortest_path itself, matching D[u,u] == 0 -- no special-casing
    # needed there; only unreachable off-diagonal pairs need the sentinel.
    D_filled = np.where(np.isinf(D_sp), sentinel, D_sp).astype(np.int32)

    if is_sparse:
        # Diagonal must NOT be a stored entry (see module docstring on
        # D[u,u] == 0 without a stored diagonal) -- csr_matrix() would
        # otherwise store the 0s that shortest_path put on the diagonal.
        csr = sp.csr_matrix(D_filled)
        csr.setdiag(0)
        csr.eliminate_zeros()
        return csr
    return D_filled
