"""graph_augmenting.sparse_hops -- bounded-hop-radius ``augment_graph`` (fdge_jax_sell_c_sigma port).

A *genuinely sparse* alternative to ``hopfill.augment_graph``. Where
hop-fill connects every reachable pair (dense, ``O(n^2)`` no matter how
sparse ``G`` is), this policy connects each node ``u`` only to the nodes
within ``radius`` hops of it, and leaves every farther pair
**structurally absent** from ``D``. For a graph of bounded average
degree that is ``O(n * avg_degree^radius)`` stored entries -- i.e.
linear in ``n`` at fixed ``radius`` and degree, not quadratic. Port of
``fdge2.graph_augmenting.sparse_hops``, but built on a sparse
boolean-matrix-power BFS (``levelwise_reach`` below) instead of fdge2's
``numba``-njit two-pass CSR builder (``_bounded_hops_csr``) -- see
``levelwise_reach``'s docstring for the matrix-power approach and
``docs/DESIGN.md`` for the overall port rationale.

Why not ``scipy.sparse.csgraph.dijkstra(..., limit=radius)``
==============================================================
``hopfill.py`` uses ``scipy.sparse.csgraph.shortest_path`` -- and it is
tempting to reach for the same call here with ``method="D"`` and
``limit=radius`` to cut it off early. Verified equivalence: it gives the
same hop values. Rejected anyway, on memory, not correctness:
``dijkstra``'s ``limit`` only stops *expanding* early, it does not
change the *output shape* -- the return is still a dense ``(n, n)``
array with ``inf`` standing in for pairs beyond ``limit``. At this
policy's target scale (500k nodes) that dense return alone is ~1TB
(``500_000^2 * 8`` bytes for float64), which defeats the entire reason
this module exists. ``levelwise_reach`` below only ever materializes the
sparse ``(rows, cols, levels)`` triple actually within radius, so its
memory scales with what ``D`` will store, not with ``n^2``. Do not
"simplify" this back to scipy without re-deriving that memory bound.

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
``networkx``, and ``fdge_jax_sell_c_sigma.core`` only. Never
``fdge_jax_sell_c_sigma.graph_building`` or ``fdge_jax_sell_c_sigma.embedding``. Uses
``nx.to_scipy_sparse_array`` for the graph->CSR conversion, and its own
``levelwise_reach`` for the BFS core.

Q4 (repulsion under a sparse ``D``) is out of scope here, same as in
fdge2 -- see fdge2's ``sparse_hops.py`` module docstring for the full
mitigation writeup (sampled negatives), which is an ``embedding/``-stage
concern, not this module's.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import networkx as nx


def levelwise_reach(A: sp.csr_matrix, max_level: int | None = None):
    """Level-synchronous BFS via boolean matrix powers, every source at once.

    ``hopfill.py`` used to share this helper too (via a now-deleted
    ``_levelwise_reach`` module) since both policies need "hop distance
    from every node to every other node it can reach," differing only in
    *when they stop* (hopfill: convergence; sparse_hops: a fixed radius)
    and *what they do with unvisited pairs at the end* (hopfill: fill an
    ``unreachable`` sentinel; sparse_hops: leave absent). ``hopfill`` has
    since moved to ``scipy.sparse.csgraph.shortest_path`` (its result is
    dense anyway, so a compiled all-pairs call is simplest there), which
    leaves ``sparse_hops`` as the only caller -- see the module docstring
    above for why sparse_hops itself does NOT make the same switch.

    Why sparse boolean matrix powers, not a per-source BFS loop
    -------------------------------------------------------------
    fdge2's Numba kernels parallelized "one BFS per source node" across
    threads -- fast, but that's an ``n``-way Python-shaped loop under the
    hood (even if it runs in parallel C). Without Numba, an explicit
    per-source loop in plain Python would be catastrophically slow.
    Instead this treats the *whole node set* as simultaneous BFS sources
    at once, using the adjacency matrix's own sparsity: at level ``k``
    the reachable set is (boolean) ``A^k``, computed incrementally as
    ``frontier_{k} = new_{k-1} @ A`` (expand only from nodes newly
    visited at the previous level, not the whole visited set -- that's
    what keeps each step's cost proportional to the *frontier*, not to
    ``n``). Total work is a handful of sparse matmuls (one per hop level,
    up to the graph diameter or a caller-supplied radius), each a
    compiled SciPy/BLAS call -- independent of ``n`` in the number of
    Python-level steps, unlike a per-source loop which is ``O(n)``
    Python-level steps no matter how fast each one is in C.

    Parameters
    ----------
    A         : symmetric boolean adjacency (``scipy.sparse.csr_matrix``, no
                self loops -- callers build this with
                ``nx.to_scipy_sparse_array(G, nodelist=..., weight=None,
                format="csr").astype(bool)``, which already gives both
                ``u->v`` and ``v->u`` for every edge with no extra
                symmetrization step needed).
    max_level : stop after this many hop levels even if the frontier is
                still nonempty (``sparse_hops``'s ``radius``). ``None``
                runs to convergence -- i.e. until the frontier empties,
                which happens after exactly ``diameter`` levels.
                ``sparse_hops.augment_graph`` always passes an int here;
                ``max_level=None`` is supported for completeness (e.g.
                standalone reachability queries) but has no caller in
                this package.

    Returns
    -------
    rows, cols, levels : 1-D int arrays, one entry per ``(u, v)`` pair
        discovered with ``u != v``, ``levels[i]`` = hop distance
        ``d(rows[i], cols[i])``. Self pairs (distance 0) are never
        included -- ``sparse_hops`` doesn't want a stored ``D[u,u]``,
        see its module docstring.
    visited : ``scipy.sparse.csr_matrix`` (bool), the union of the
        identity and every ``new`` matrix recorded -- i.e. ``visited[u,
        v]`` is True iff ``v`` was reached from ``u`` at or before the
        level the loop stopped at. ``sparse_hops`` ignores this (absence
        is the contract); kept in the return value because dropping it
        would only save the caller from unpacking one extra name.
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


def augment_graph(G, is_sparse: bool = True, radius: int = 2, **kwargs):
    """Stage 2 (sparse policy): bounded-hop-radius hop-fill.

    Connects each node only to nodes within ``radius`` hops; every farther
    pair is structurally absent from ``D``. See the module docstring for
    the exact ``D`` semantics, the no-sentinel decision, and why
    bounded-radius (not kNN) is the right first sparse policy for this
    shell-averaged force law.

    Parameters
    ----------
    G         : any graph accepted by ``nx.to_scipy_sparse_array``, e.g. a
                ``networkx.Graph``. Edge weights are ignored -- only
                topology is used (matches hop-fill).
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
        (is_sparse=False). Node index ``i`` == ``list(G.nodes())[i]``.
    """
    nodes = list(G.nodes())
    n = len(nodes)

    # weight=None -> pure 0/1 topology; .astype(bool) gives levelwise_reach
    # the symmetric, loop-free boolean adjacency it expects.
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None, format="csr").astype(bool)
    rows, cols, levels, _visited = levelwise_reach(A, max_level=int(radius))

    # coo_matrix accepts empty (rows, cols, levels) fine (an all-isolated
    # graph at radius=0, say) -- no separate empty-case branch needed.
    coo = sp.coo_matrix((levels, (rows, cols)), shape=(n, n))
    csr = coo.tocsr()

    if is_sparse:
        return csr
    return csr.toarray().astype(np.int32)
