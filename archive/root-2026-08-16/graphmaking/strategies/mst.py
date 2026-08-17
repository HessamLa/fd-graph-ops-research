"""Strategy: exact Euclidean minimum spanning tree."""

from __future__ import annotations

from ..registry import register_strategy
from ..building_blocks import mst_edges, base_graph, add_edges


@register_strategy("mst")
def _strategy_mst(data):
    """Exact Euclidean minimum spanning tree: n-1 edges, connected, acyclic."""
    G = base_graph(data.shape[0])
    edges, _ = mst_edges(data)
    add_edges(G, edges, source="mst")
    return G
