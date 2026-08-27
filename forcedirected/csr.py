"""forcedirected.csr -- shared CSR utilities.

Import discipline: this module imports **numpy only** (see docs/DESIGN.md).

``row_of`` / ``n_rows`` are small, generic helpers for working with any
CSR-like triple (``.indptr`` / ``.indices``) -- built by whatever produced
it, e.g. ``networkx.to_scipy_sparse_array`` -- so they belong here rather
than in a specific stage package.

The hop-distance / weighted-distance matrix ``D`` that flows through this
package (``graph_augmenting`` -> ``embedding``) is a plain
``scipy.sparse.csr_matrix`` -- no bespoke container, and no bespoke
graph->CSR conversion either: ``graph_augmenting`` builds it directly with
``networkx.to_scipy_sparse_array`` (NetworkX is already a hard requirement
of this package). ``scipy.sparse`` already gives everything this package
needs from ``D``: the indexability contract (``D[i, j]``, dense semantics,
0 for an absent pair) via its own ``__getitem__``, ``.toarray()`` for
densifying, and direct ``.indptr`` / ``.indices`` / ``.data`` access for
``embedding/``'s bucketed plan builder -- with no bespoke class needed to
provide any of that.
"""
# Provenance: moved VERBATIM from `fodiwalk/core/csr.py` on 2026-08-26,
# with the engine that reads it. `fodiwalk/core/csr.py` is a forwarder now.
# Provenance: moved VERBATIM from `fodined/core/csr.py` on 2026-08-19.
# Not one line of a body changed; the file has no import to change either.
# Provenance: this file is a copy of `fdge_jax_sell_c_sigma/core/csr.py`.
# The code is identical. Some docstrings refer to documents that are not in
# this directory, for example `docs/DESIGN.md`. Refer to the origin package
# for these documents.
from __future__ import annotations

import numpy as np


def row_of(indptr):
    """Source-node id for every edge position in a CSR layout.

    ``row_of(indptr)[p]`` is the row (source node) that owns edge position
    ``p`` in ``indices`` / ``data``. The ``np.repeat`` trick: turns a
    per-row loop into a single vectorized gather, e.g.
    ``Z[indices] - Z[row_of(indptr)]`` computes every edge's ``Zdiff`` at
    once. Cheaper than ``D.tocoo().row`` -- no format-conversion pass, just
    ``indptr`` you already have.
    """
    indptr = np.asarray(indptr)
    n = indptr.shape[0] - 1
    return np.repeat(np.arange(n), np.diff(indptr))


def n_rows(D):
    """Number of rows (nodes) in an indexable ``D`` -- ndarray or csr_matrix."""
    return D.shape[0]
