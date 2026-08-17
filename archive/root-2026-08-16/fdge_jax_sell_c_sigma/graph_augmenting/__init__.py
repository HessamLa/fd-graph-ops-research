"""fdge_jax_sell_c_sigma.graph_augmenting -- the "augment graph" stage (see docs/DESIGN.md).

Owns ``augment_graph(G, is_sparse=True, **kwargs) -> D``: turns an
(unweighted) graph ``G`` into a weighted, indexable matrix ``D`` (a
``scipy.sparse.csr_matrix`` or a dense ndarray, per ``is_sparse``). The
reference policy is dense hop-fill -- see ``hopfill.py`` for the exact
contract and the ``is_sparse`` handling. ``hopfill`` computes it via
``scipy.sparse.csgraph.shortest_path`` (all-pairs, unweighted); its
result is dense by nature, so a compiled all-pairs call is the simplest
correct approach -- no shared BFS helper needed for this policy.

Per the fdge_jax_sell_c_sigma import discipline (docs/DESIGN.md): this package may
depend on ``numpy``, ``scipy.sparse``, ``networkx``, and
``fdge_jax_sell_c_sigma.core`` only -- never on ``fdge_jax_sell_c_sigma.graph_building`` or
``fdge_jax_sell_c_sigma.embedding``. It accepts any graph ``nx.to_scipy_sparse_array``
accepts, not specifically a ``graph_building``-produced one.

Sparse augmentation policies live in sibling modules, not in
``hopfill.py``. The first one is ``sparse_hops`` -- bounded-hop-radius
BFS via a sparse boolean-matrix-power ``levelwise_reach`` (its own
helper, private to that module -- see ``sparse_hops.py`` for why it
can't just reuse ``scipy.sparse.csgraph.dijkstra`` the way ``hopfill``
reuses ``shortest_path``: ``dijkstra``'s ``limit=`` still forces a dense
``(n, n)`` return, which defeats the point of a sparse policy at scale).
It is exposed as ``graph_augmenting.sparse_hops.augment_graph`` (kept
namespaced rather than re-exported as bare ``augment_graph``, so the
dense reference stays the unqualified default and the two policies never
shadow each other).
"""
from __future__ import annotations

from . import hopfill
from . import sparse_hops
from .hopfill import augment_graph

__all__ = ["hopfill", "sparse_hops", "augment_graph"]
