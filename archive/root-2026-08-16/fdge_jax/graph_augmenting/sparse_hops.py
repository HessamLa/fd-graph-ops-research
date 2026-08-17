"""graph_augmenting.sparse_hops -- bounded-hop-radius ``augment_graph`` (fdge_jax port).

A *genuinely sparse* alternative to ``hopfill.augment_graph``. Where
hop-fill connects every reachable pair (dense, ``O(n^2)`` no matter how
sparse ``G`` is), this policy connects each node ``u`` only to the nodes
within ``radius`` hops of it, and leaves every farther pair
**structurally absent** from ``D``. For a graph of bounded average
degree that is ``O(n * avg_degree^radius)`` stored entries -- i.e.
linear in ``n`` at fixed ``radius`` and degree, not quadratic. Port of
``fdge2.graph_augmenting.sparse_hops``, but built on
``_levelwise_reach``'s sparse boolean-matrix-power BFS instead of
fdge2's ``numba``-njit two-pass CSR builder (``_bounded_hops_csr``) --
see ``_levelwise_reach.py`` for the matrix-power approach and
``docs/DESIGN.md`` for the overall port rationale.

Contract (same shape as ``hopfill.augment_graph``):

    augment_graph(G, is_sparse=True, radius=2, **kwargs) -> D

``D`` carries the same hop-distance semantics as hop-fill *for the entries
it stores*:

* ``D[u, u] == 0``                (self -- absent in the CSR triple, but
  reads back as 0 for free, same as ``hopfill``; see that module's
  docstring)
* ``D[u, v] == 1`` for a direct edge ``(u, v)`` in ``G``
* ``D[u, v] == k`` for a pair whose shortest path is ``k <= radius`` hops
* ``D[u, v]`` **absent** (structural zero, not stored) for every pair more
  than ``radius`` hops apart -- INCLUDING pairs in different components.

That last point is the whole difference from hop-fill: there is **no
``unreachable`` sentinel**. "Absent" uniformly means "no stored
hop-distance for this pair" -- whether the pair is beyond the radius or in
a disconnected component is not distinguished, because the embedding stage
treats both identically (no attraction edge). A caller that needs to tell
"far" from "disconnected" apart should use the dense ``hopfill`` policy;
that distinction is exactly the ``O(n^2)`` information a sparse policy
declines to materialize.

Why bounded-hop-radius BFS, and not kNN
=======================================
This force law is **shell-averaged**: for each node ``u`` it groups the
other nodes into shells ``S_h(u)`` by hop distance ``h`` and averages the
per-shell contribution. Bounded-radius BFS keeps every shell it keeps
*complete*: all of ``S_1(u) .. S_radius(u)`` are present in full, so the
shell counts ``|S_h(u)|`` for ``h <= radius`` derivable from ``D`` are
**exact** -- the retained physics is unchanged, only the tail (``h >
radius``) is dropped.

A kNN policy ("keep each node's ``k`` nearest neighbours by hop distance")
would instead cut *through* a shell: hop distance is heavily tied (a node
can have dozens of neighbours all at hop 2), so "the ``k`` nearest" keeps
an arbitrary subset of a shell and discards the rest. That corrupts the
denominator the shell-averaging divides by -- ``|S_h(u)|`` would be
undercounted by an amount that depends on an arbitrary tie-break. Hence
bounded-radius BFS, not kNN. (See fdge2's
``archive/T2.2_sparse_hops_worklog.md`` for the full original design
writeup; unchanged reasoning, just re-summarized here.)

Import discipline (docs/DESIGN.md): imports ``numpy``, ``scipy.sparse``,
and ``fdge_jax.core`` only. Never ``fdge_jax.graph_building`` or
``fdge_jax.embedding``. Reuses ``core.csr.graph_to_csr`` for the
graph->CSR conversion (does not reimplement it), and
``_levelwise_reach`` for the BFS core (shared with ``hopfill.py``).

Q4 (repulsion under a sparse ``D``) is out of scope here, same as in
fdge2 -- see fdge2's ``sparse_hops.py`` module docstring for the full
mitigation writeup (sampled negatives), which is an ``embedding/``-stage
concern, not this module's.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from fdge_jax.core import graph_to_csr, HopMatrix

from ._levelwise_reach import build_adjacency, levelwise_reach


def augment_graph(G, is_sparse: bool = True, radius: int = 2, **kwargs):
    """Stage 2 (sparse policy): bounded-hop-radius hop-fill.

    Connects each node only to nodes within ``radius`` hops; every farther
    pair is structurally absent from ``D``. See the module docstring for
    the exact ``D`` semantics, the no-sentinel decision, and why
    bounded-radius (not kNN) is the right first sparse policy for this
    shell-averaged force law.

    Parameters
    ----------
    G         : any graph satisfying ``core.csr.graph_to_csr``'s duck-typed
                interface (``.nodes()`` / ``.degree()`` / ``.edges()``),
                e.g. any ``networkx.Graph``. Edge weights are ignored --
                only topology is used (matches hop-fill).
    is_sparse : True (default) -> return a ``core.csr.HopMatrix`` whose
                stored pattern IS the bounded neighbourhood (genuinely
                sparse; ``nnz`` reflects the policy, not ``n^2``). False ->
                densify to a plain ``(n, n)`` int32 ndarray (convenience
                for tiny graphs / debugging; the absent pairs become plain
                ``0`` and are then indistinguishable from the diagonal, so
                the sparse form is the one that actually carries the
                "absent vs stored" distinction).
    radius    : max hop distance to fill in (inclusive). ``radius=1`` keeps
                only direct neighbours; ``radius=2`` (default) keeps two
                shells. Larger ``radius`` -> denser ``D`` -> back toward
                ``O(n^2)`` in the limit ``radius >= diameter``.

    Returns
    -------
    D : int32 ``(n, n)``. ``core.csr.HopMatrix`` (is_sparse=True) with
        only within-radius entries stored, or a dense ndarray
        (is_sparse=False). Node index ``i`` == ``list(G.nodes())[i]`` (same
        ordering as ``core.csr.graph_to_csr``).
    """
    indptr, indices, degrees = graph_to_csr(G)
    n = indptr.shape[0] - 1

    A = build_adjacency(indptr, indices, n)
    rows, cols, levels, _visited = levelwise_reach(A, max_level=int(radius))

    if rows.size:
        coo = sp.coo_matrix((levels, (rows, cols)), shape=(n, n))
        csr = coo.tocsr()
        csr_indptr, csr_indices, csr_data = csr.indptr, csr.indices, csr.data
    else:
        csr_indptr = np.zeros(n + 1, dtype=np.int64)
        csr_indices = np.empty(0, dtype=np.int32)
        csr_data = np.empty(0, dtype=np.int32)

    if is_sparse:
        return HopMatrix(indptr=csr_indptr, indices=csr_indices,
                          data=csr_data, n=n)
    D = np.zeros((n, n), dtype=np.int32)
    if rows.size:
        D[rows, cols] = levels
    return D
