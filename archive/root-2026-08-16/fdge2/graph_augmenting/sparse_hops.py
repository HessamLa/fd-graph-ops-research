"""graph_augmenting.sparse_hops -- bounded-hop-radius ``augment_graph`` (T2.2).

A *genuinely sparse* alternative to ``hopfill.augment_graph``. Where
hop-fill connects every reachable pair (dense, ``O(n^2)`` no matter how
sparse ``G`` is -- see ARCHITECTURE.md "Known limitation"), this policy
connects each node ``u`` only to the nodes within ``radius`` hops of it,
and leaves every farther pair **structurally absent** from ``D``. For a
graph of bounded average degree that is ``O(n * avg_degree^radius)``
stored entries -- i.e. linear in ``n`` at fixed ``radius`` and degree, not
quadratic. Confirmed empirically in ``test_sparse_hops_smoke.py`` /
``_bench`` (see the report accompanying this task).

Contract (same shape as ``hopfill.augment_graph``, RPD.md §3 / API_DESIGN.md):

    augment_graph(G, is_sparse=True, radius=2, **kwargs) -> D

``D`` carries the same hop-distance semantics as hop-fill *for the entries
it stores*:

* ``D[u, u] == 0``                (self, stored explicitly)
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
per-shell contribution (the reference ``get_shell_counts`` / shell physics
in ``model_204_shell.py``). Bounded-radius BFS keeps every shell it keeps
*complete*: all of ``S_1(u) .. S_radius(u)`` are present in full, so the
shell counts ``|S_h(u)|`` for ``h <= radius`` derivable from ``D`` are
**exact** -- the retained physics is unchanged, only the tail (``h >
radius``) is dropped.

A kNN policy ("keep each node's ``k`` nearest neighbours by hop distance")
would instead cut *through* a shell: hop distance is heavily tied (a node
can have dozens of neighbours all at hop 2), so "the ``k`` nearest" keeps
an arbitrary subset of a shell and discards the rest. That corrupts the
denominator the shell-averaging divides by -- ``|S_h(u)|`` would be
undercounted by an amount that depends on an arbitrary tie-break. For a
force law whose whole structure is "average over complete shells,"
truncating shells is the wrong first sparse policy; truncating *radius*
(keeping whole shells, dropping far ones) is the coherent one. Hence
bounded-radius BFS. (kNN would be defensible for a force law defined
directly on a feature/Euclidean distance with no shell structure -- not
this one.)

Import discipline (RPD.md §5): imports ``numpy``, ``scipy.sparse``,
``numba``, and ``fdge2.core`` only. Never ``fdge2.graph_building`` or
``fdge2.embedding``. Reuses ``core.csr.graph_to_csr`` for the graph->CSR
conversion (does not reimplement it).


================================================================================
Q4 mitigation proposal -- repulsion under a sparse ``D`` (ARCHITECTURE.md Q4)
================================================================================
STATUS: risk is REAL and DEMONSTRATED here; a concrete, actionable
mitigation is proposed below for the embedding-stage owner. Building it is
out of scope for T2.2 (it lives in ``fdge2/embedding/``, owned by another
agent) -- this is the "written decision / tracked" note RPD.md success
metric #7 requires for Q4.

The risk, concretely
--------------------
The current force law derives *repulsion* from the same support as
attraction: the pairs present in ``D``. Under dense hop-fill every
reachable pair is present, so repulsion has plenty of pairs to push apart.
Under THIS policy, any two nodes more than ``radius`` hops apart are absent
from ``D`` -> zero attraction (intended) but ALSO zero repulsion (the bug).
Two nodes that are far in the graph but happen to drift close together in
embedding space feel **no force keeping them apart** -- classic embedding
collapse. ``test_sparse_hops_smoke.py::test_collapse_risk_is_real`` makes
this concrete: it exhibits a pair that is absent from ``D`` (hence zero
repulsion today) yet is a candidate to sit arbitrarily close in ``Z``.

Proposed mitigation: sampled negatives (primary)
------------------------------------------------
Add repulsion between each row node ``u`` and a small number ``m`` of
uniformly random *other* nodes ("negative samples"), re-drawn each epoch,
scaled by ``1 / m`` so the sampled term is an unbiased estimate of the
mean repulsion ``u`` would feel from the whole graph. This restores a
global repulsive pressure without restoring the ``O(n^2)`` pair set.

How it plugs into the existing ``forces(self, Z, D, row_start, row_end,
**kwargs)`` contract (``core/force_directed.py``) WITHOUT changing that
signature:

* ``embed`` already threads ``**kwargs`` -> ``updateGradient`` -> ``forces``,
  so a hyperparameter arrives for free: call
  ``model.embed(G, neg_samples=5)`` and ``forces`` sees ``neg_samples=5``
  in ``**kwargs``. No new positional argument, no contract break.
* Everything else the mitigation needs is derivable from arguments already
  present -- it does NOT need the missing pairs passed in as data (which
  would defeat the point):
    - ``n`` (to draw random node ids from): ``n = D.shape[0]``.
    - the embedding to push apart: ``Z`` (already passed; reading all of
      ``Z`` is fine -- it is ``(n, d)``, within the ``O(n*d)`` budget).
    - randomness: ``self.rng`` (the model already owns a seeded
      ``np.random.default_rng`` -- ``core.ForceDirected.__init__``).
* Row-streamed, memory-safe (RPD.md §3 guardrail): for each ``u`` in
  ``[row_start, row_end)`` draw ``m`` ids, accumulate ``m`` repulsion
  vectors into that row's output. Peak extra memory is
  ``O((row_end-row_start)*d + m*d)`` -- no ``(n,n,d)`` or ``(nnz,d)``
  temporary. Sketch of the added block inside ``forces``::

      n = D.shape[0]
      for u in range(row_start, row_end):
          neg = self.rng.integers(0, n, size=neg_samples)   # m random nodes
          for j in neg:
              if j == u:
                  continue
              diff = Z[u] - Z[j]
              dZ[u - row_start] += repulse(diff) / neg_samples  # same _scalar_force
      # ... existing attraction loop over D's row u unchanged ...

  ``repulse`` is the *existing* repulsive branch of ``_scalar_force`` --
  the mitigation reuses the force law, it does not invent a new one.

Batching note: negatives are drawn per row, so this stays correct under
any ``batch_count`` (unlike a whole-array momentum override). It only
perturbs the ``updateGradient`` default-passthrough path by adding a
batch-local term, so the batching-invariance guarantee (API_DESIGN.md Q1)
is preserved in expectation.

Secondary / complementary option: a wider repulsion radius. Return TWO
patterns from a variant augmenter -- ``radius_att`` (small, for
attraction) and ``radius_rep > radius_att`` (a bit larger, for repulsion)
-- and let ``forces`` read repulsion off the wider one via a second
``D_rep`` passed through ``**kwargs``. This is cheaper to reason about but
still bounded, so it does NOT cover the truly-far / cross-component tail;
sampled negatives do. Recommendation: sampled negatives as the primary
fix, a modest ``radius_rep`` bump only if a purely-local stiffening is
also wanted. Either way, the attraction support (this module's ``D``) is
unchanged -- the mitigation is purely additive on the repulsion side, so
it does not touch this augmenter's contract.
================================================================================
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numba import njit

from fdge2.core import graph_to_csr


# ---------------------------------------------------------------------------
# Bounded BFS -> sparse CSR of within-radius hop distances
# ---------------------------------------------------------------------------
@njit(cache=True)
def _bounded_hops_csr(indptr, indices, n, radius):
    """Bounded-radius BFS from every source -> CSR of within-radius hops.

    Two passes over the nodes (count, then fill) so the output CSR arrays
    can be sized exactly without ever materializing an ``(n, n)`` matrix.
    A single length-``n`` ``dist`` scratch buffer is reused across sources
    and reset touching **only the visited entries** after each source, so
    total work is ``O(n + nnz)`` -- linear in the number of stored entries,
    NOT ``O(n^2)`` (that reset-only-what-you-touched detail is what keeps
    the whole thing sub-quadratic; a naive per-source ``dist.fill(-1)``
    would silently reintroduce an ``O(n^2)`` term).

    Returns CSR triple ``(indptr_out, indices_out, data_out)`` where row
    ``u`` lists every node within ``radius`` hops of ``u`` (including ``u``
    itself at distance 0) and ``data_out`` holds the corresponding hop
    distance.
    """
    dist = np.full(n, np.int32(-1), dtype=np.int32)
    frontier = np.empty(n, dtype=np.int64)
    nxt = np.empty(n, dtype=np.int64)
    visited = np.empty(n, dtype=np.int64)  # nodes touched this source, to reset

    # ---- pass 1: per-row counts -------------------------------------------
    row_counts = np.zeros(n, dtype=np.int64)
    for src in range(n):
        dist[src] = 0
        frontier[0] = src
        f_len = 1
        visited[0] = src
        vis_len = 1
        level = 0
        while f_len > 0 and level < radius:
            level += 1
            nxt_len = 0
            for fi in range(f_len):
                u = frontier[fi]
                for p in range(indptr[u], indptr[u + 1]):
                    v = indices[p]
                    if dist[v] == -1:
                        dist[v] = level
                        nxt[nxt_len] = v
                        nxt_len += 1
                        visited[vis_len] = v
                        vis_len += 1
            frontier, nxt = nxt, frontier
            f_len = nxt_len
        row_counts[src] = vis_len
        for i in range(vis_len):          # reset only touched entries
            dist[visited[i]] = -1

    indptr_out = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        indptr_out[i + 1] = indptr_out[i] + row_counts[i]
    nnz = indptr_out[n]
    indices_out = np.empty(nnz, dtype=np.int32)
    data_out = np.empty(nnz, dtype=np.int32)

    # ---- pass 2: fill -----------------------------------------------------
    for src in range(n):
        dist[src] = 0
        frontier[0] = src
        f_len = 1
        visited[0] = src
        vis_len = 1
        level = 0
        while f_len > 0 and level < radius:
            level += 1
            nxt_len = 0
            for fi in range(f_len):
                u = frontier[fi]
                for p in range(indptr[u], indptr[u + 1]):
                    v = indices[p]
                    if dist[v] == -1:
                        dist[v] = level
                        nxt[nxt_len] = v
                        nxt_len += 1
                        visited[vis_len] = v
                        vis_len += 1
            frontier, nxt = nxt, frontier
            f_len = nxt_len
        base = indptr_out[src]
        for i in range(vis_len):
            node = visited[i]
            indices_out[base + i] = np.int32(node)
            data_out[base + i] = dist[node]
            dist[node] = -1               # reset while we walk visited
    return indptr_out, indices_out, data_out


# ---------------------------------------------------------------------------
# augment_graph
# ---------------------------------------------------------------------------
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
    is_sparse : True (default) -> return a ``scipy.sparse.csr_matrix`` whose
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
    D : int32 ``(n, n)``. ``scipy.sparse.csr_matrix`` (is_sparse=True) with
        only within-radius entries stored, or a dense ndarray
        (is_sparse=False). Node index ``i`` == ``list(G.nodes())[i]`` (same
        ordering as ``core.csr.graph_to_csr``).
    """
    indptr, indices, degrees = graph_to_csr(G)
    n = indptr.shape[0] - 1

    indptr_out, indices_out, data_out = _bounded_hops_csr(
        indptr, indices, n, np.int64(radius))

    D = sp.csr_matrix((data_out, indices_out, indptr_out),
                      shape=(n, n), dtype=np.int32)
    D.sort_indices()  # canonical column order per row (cheap, tidy for [i,j])

    if is_sparse:
        return D
    return D.toarray()
