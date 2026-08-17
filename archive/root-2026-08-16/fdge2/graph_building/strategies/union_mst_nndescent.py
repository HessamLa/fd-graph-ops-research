"""Strategy: union of MST edges and PyNNDescent kNN edges."""

from __future__ import annotations

from ..registry import register_strategy
from ..building_blocks import mst_edges, knn_edges, base_graph, add_edges


@register_strategy("union_mst_nndescent")
def _strategy_union_mst_nndescent(
    data, k_neighbors, metric="euclidean", random_state=42
):
    """Union of MST edges and PyNNDescent kNN edges (undirected union).

    Connected by construction since the MST is a subset of the edge set.
    """
    G = base_graph(data.shape[0])
    mst_e, _ = mst_edges(data)
    add_edges(G, mst_e, source="mst")
    knn_e = knn_edges(data, k_neighbors, metric, random_state)
    add_edges(G, knn_e, source="knn")   # duplicates of MST edges skipped
    return G
