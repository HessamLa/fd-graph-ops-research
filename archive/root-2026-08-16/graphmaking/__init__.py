"""
graphmaking: graph construction strategies for force-directed embedding
pipelines.

Public API:

    from graphmaking import make_graph, register_strategy, available_strategies

    G = make_graph(data, "mst")
    G = make_graph(data, "union_mst_nndescent", k_neighbors=10)
    G = make_graph(data, "mst_k_minimum", k_neighbors=10)
"""

from .registry import make_graph, register_strategy, available_strategies
from . import strategies  # noqa: F401  (side effect: registers all strategies)

__all__ = ["make_graph", "register_strategy", "available_strategies"]
