"""fdge_jax_sell_c_sigma.core -- base classes + shared CSR/indexable contract.

Import discipline (docs/DESIGN.md): this package imports **numpy + jax**
only. It defines the contracts that ``graph_building/``,
``graph_augmenting/`` and ``embedding/`` depend on, and must never import
any of them back (no circular dependency).

Public surface:

    from fdge_jax_sell_c_sigma.core import ForceDirected, Callback_Base
    from fdge_jax_sell_c_sigma.core import row_of, n_rows
"""
from __future__ import annotations

from .force_directed import ForceDirected, Callback_Base
from .csr import row_of, n_rows

__all__ = [
    "ForceDirected",
    "Callback_Base",
    "row_of",
    "n_rows",
]
