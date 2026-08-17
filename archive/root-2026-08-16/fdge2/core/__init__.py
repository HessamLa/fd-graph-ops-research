"""fdge2.core -- base classes + shared CSR/indexable contract.

Import discipline (RPD.md §5): this package imports **numpy only**. It
defines the contracts that ``graph_building/``, ``graph_augmenting/`` and
``embedding/`` depend on, and must never import any of them back (no
circular dependency).

Public surface:

    from fdge2.core import ForceDirected, Callback_Base
    from fdge2.core import graph_to_csr, row_of, n_rows
"""
from __future__ import annotations

from .force_directed import ForceDirected, Callback_Base
from .csr import graph_to_csr, row_of, n_rows

__all__ = [
    "ForceDirected",
    "Callback_Base",
    "graph_to_csr",
    "row_of",
    "n_rows",
]
