"""fodined.core -- base classes + shared CSR/indexable contract.

Copy of `fdge_jax_sell_c_sigma/core/__init__.py`. Only the module names in
the example below are different.

Import discipline (docs/DESIGN.md): this package imports **numpy + jax**
only. It defines the contracts that ``graph_building/``,
``graph_augmenting/`` and ``embedding/`` depend on, and must never import
any of them back (no circular dependency).

Public surface:

    from fodined.core import Fodined, Callback_Base
    from fodined.core import row_of, n_rows
"""
from __future__ import annotations

from .fodined import Fodined, Callback_Base
from .csr import row_of, n_rows

__all__ = [
    "Fodined",
    "Callback_Base",
    "row_of",
    "n_rows",
]
