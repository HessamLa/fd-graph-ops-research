"""
Importing this package registers every built-in strategy.

Registration is a side effect of import (@register_strategy runs at import
time), so each strategy module must be imported here or it silently won't
be reachable through make_graph. Adding a new strategy = new file in this
directory + one import line below.
"""

from . import mst, union_mst_nndescent, mst_k_minimum  # noqa: F401
