"""fdge_jax.core -- base classes + shared CSR/indexable contract.

Import discipline (docs/DESIGN.md): this package imports **numpy + jax**
only. It defines the contracts that ``graph_building/``,
``graph_augmenting/`` and ``embedding/`` depend on, and must never import
any of them back (no circular dependency).

Public surface:

    from fdge_jax.core import ForceDirected, Callback_Base
    from fdge_jax.core import graph_to_csr, row_of, n_rows, HopMatrix
"""
from __future__ import annotations

from .force_directed import ForceDirected, Callback_Base
from .csr import graph_to_csr, row_of, n_rows, HopMatrix

__all__ = [
    "ForceDirected",
    "Callback_Base",
    "graph_to_csr",
    "row_of",
    "n_rows",
    "HopMatrix",
]
