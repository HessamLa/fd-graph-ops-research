"""Strategy: MST, then raise every node's degree to at least min(k, n-1)."""

from __future__ import annotations

from ..make_graph import register_strategy
from ..building_blocks import mst_edges, nearest_neighbors, base_graph, add_edges


@register_strategy("mst_k_minimum")
def _strategy_mst_k_minimum(data, k_neighbors):
    """MST, then raise every node's degree to at least min(k, n-1).

    For each node below the floor, connect it to its nearest points that are
    not already neighbors, in ascending distance order. Fill edges may push
    *other* endpoints above the floor -- k is a minimum, not a maximum.
    Candidates come from an exact k-nearest-neighbour query (no
    approximation) -- a list of exactly k nearest points is always enough
    to raise a node from its current degree d to k, since at most d of
    those k candidates can already be connected.
    """
    n = data.shape[0]
    k = min(k_neighbors, n - 1)
    if k < 1:
        raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}.")

    G = base_graph(n)
    mst_e = mst_edges(data)
    add_edges(G, mst_e, source="mst")

    nn_idx = nearest_neighbors(data, k)     # (n, k), ascending distance, self excluded
    for i in range(n):
        if G.degree(i) >= k:
            continue
        for j in map(int, nn_idx[i]):
            if G.degree(i) >= k:
                break
            if not G.has_edge(i, j):
                G.add_edge(i, j, source="fill")
    return G
