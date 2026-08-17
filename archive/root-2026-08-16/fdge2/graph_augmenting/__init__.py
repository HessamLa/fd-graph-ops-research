"""fdge2.graph_augmenting -- the "augment graph" stage (see ../docs/ARCHITECTURE.md).

Owns ``augment_graph(G, is_sparse=True, **kwargs) -> D``: turns an
(unweighted) graph ``G`` into a weighted, indexable matrix ``D``. The
reference policy (T2.1, this package's only implementation so far) is
dense hop-fill -- see ``hopfill.py`` for the exact contract and the
``is_sparse`` handling.

Per the fdge2 import discipline (RPD.md Sec 5): this package may depend on
``numpy``, ``scipy.sparse``, ``numba``, and ``fdge2.core`` only -- never
on ``fdge2.graph_building`` or ``fdge2.embedding``. It accepts any graph
satisfying ``core.csr.graph_to_csr``'s duck-typed interface, not
specifically a ``graph_building``-produced one.

Sparse augmentation policies live in sibling modules, not in
``hopfill.py``. The first one (T2.2, RPD.md Q5) is ``sparse_hops`` --
bounded-hop-radius BFS, genuinely sub-``O(n^2)``. It is exposed as
``graph_augmenting.sparse_hops.augment_graph`` (kept namespaced rather than
re-exported as bare ``augment_graph``, so the dense reference stays the
unqualified default and the two policies never shadow each other).
"""
from __future__ import annotations

from . import hopfill
from . import sparse_hops
from .hopfill import augment_graph, get_shell_counts

__all__ = ["hopfill", "sparse_hops", "augment_graph", "get_shell_counts"]
