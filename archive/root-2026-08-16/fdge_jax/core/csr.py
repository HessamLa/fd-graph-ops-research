"""core.csr -- shared CSR utilities + the HopMatrix ``D`` container.

Import discipline: this module imports **numpy only** (see docs/DESIGN.md).

``graph_to_csr`` / ``row_of`` / ``n_rows`` are ported verbatim from fdge2 --
plain CPU/NumPy bookkeeping, nothing here needs JAX.

``HopMatrix`` is new: it's the one representation both graph_augmenting
policies (dense hop-fill, bounded-radius hop-fill) return, replacing
fdge2's "dense ndarray or scipy.sparse.csr_matrix" duck-typed ``D`` (see
docs/DESIGN.md for the reasoning). It's a plain CSR triple with a
``__getitem__``/``toarray()`` for the indexability contract (debugging,
tests, small graphs), while giving ``embedding/`` direct CSR access with
no densify-then-reparse step.
"""
from __future__ import annotations

import numpy as np


def graph_to_csr(Gx):
    """Convert an (undirected) NetworkX-style graph to CSR adjacency arrays.

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
    ``p`` in ``indices`` / ``data``. The ``np.repeat`` trick: turns a
    per-row loop into a single vectorized gather, e.g.
    ``Z[indices] - Z[row_of(indptr)]`` computes every edge's ``Zdiff`` at
    once.
    """
    indptr = np.asarray(indptr)
    n = indptr.shape[0] - 1
    return np.repeat(np.arange(n), np.diff(indptr))


def n_rows(D):
    """Number of rows (nodes) in an indexable ``D`` -- ndarray or HopMatrix."""
    return D.shape[0]


class HopMatrix:
    """A sparse ``(n, n)`` hop-distance matrix, stored as a CSR triple.

    Row ``u``'s stored pairs are ``indices[indptr[u]:indptr[u+1]]`` with
    hop distances ``data[indptr[u]:indptr[u+1]]`` at the same positions. A
    pair absent from row ``u`` means "no stored distance" (self, or beyond
    whatever the augmentation policy chose to fill) -- not the same as a
    distance of 0.

    Satisfies the indexability contract (``D[i, j]`` works, dense-array
    semantics: 0 for an absent pair) via ``__getitem__`` / ``toarray()``,
    for debugging, tests, and small graphs. The hot loop
    (``embedding/shell_force.py``) never uses ``__getitem__`` -- it reads
    ``.indptr`` / ``.indices`` / ``.data`` directly.
    """

    __slots__ = ("indptr", "indices", "data", "n")

    def __init__(self, indptr, indices, data, n: int):
        self.indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(indices, dtype=np.int32)
        self.data = np.ascontiguousarray(data, dtype=np.int32)
        self.n = int(n)

    @property
    def shape(self):
        return (self.n, self.n)

    @property
    def nnz(self):
        return self.indices.shape[0]

    def __getitem__(self, key):
        i, j = key
        lo, hi = self.indptr[i], self.indptr[i + 1]
        row_cols = self.indices[lo:hi]
        pos = np.nonzero(row_cols == j)[0]
        return int(self.data[lo + pos[0]]) if pos.size else 0

    def toarray(self) -> np.ndarray:
        """Densify to a plain ``(n, n)`` int32 ndarray (0 where absent)."""
        out = np.zeros((self.n, self.n), dtype=np.int32)
        out[row_of(self.indptr), self.indices] = self.data
        return out

    def __repr__(self) -> str:
        return f"HopMatrix(n={self.n}, nnz={self.nnz})"
