"""fdge_jax.graph_augmenting -- the "augment graph" stage (see docs/DESIGN.md).

Owns ``augment_graph(G, is_sparse=True, **kwargs) -> D``: turns an
(unweighted) graph ``G`` into a weighted, indexable matrix ``D`` (a
``core.csr.HopMatrix`` or a dense ndarray, per ``is_sparse``). The
reference policy is dense hop-fill -- see ``hopfill.py`` for the exact
contract and the ``is_sparse`` handling.

Per the fdge_jax import discipline (docs/DESIGN.md): this package may
depend on ``numpy``, ``scipy.sparse``, and ``fdge_jax.core`` only --
never on ``fdge_jax.graph_building`` or ``fdge_jax.embedding``. It
accepts any graph satisfying ``core.csr.graph_to_csr``'s duck-typed
interface, not specifically a ``graph_building``-produced one.

Sparse augmentation policies live in sibling modules, not in
``hopfill.py``. The first one is ``sparse_hops`` -- bounded-hop-radius
BFS, genuinely sub-``O(n^2)``. It is exposed as
``graph_augmenting.sparse_hops.augment_graph`` (kept namespaced rather
than re-exported as bare ``augment_graph``, so the dense reference stays
the unqualified default and the two policies never shadow each other).

Both policies share their level-synchronous BFS core -- see
``_levelwise_reach.py`` (internal, not re-exported here).
"""
from __future__ import annotations

from . import hopfill
from . import sparse_hops
from .hopfill import augment_graph, get_shell_counts

__all__ = ["hopfill", "sparse_hops", "augment_graph", "get_shell_counts"]
