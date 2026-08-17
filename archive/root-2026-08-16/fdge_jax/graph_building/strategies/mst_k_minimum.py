"""Strategy: MST, then raise every node's degree to at least min(k, n-1)."""

from __future__ import annotations

import numpy as np

from ..registry import register_strategy
from ..building_blocks import mst_edges, base_graph, add_edges


@register_strategy("mst_k_minimum")
def _strategy_mst_k_minimum(data, k_neighbors):
    """MST, then raise every node's degree to at least min(k, n-1).

    For each node below the floor, connect it to its nearest points that are
    not already neighbors, in ascending distance order. Fill edges may push
    *other* endpoints above the floor -- k is a minimum, not a maximum.
    Distances come from the dense matrix already computed for the MST, so
    this stage is exact (no approximation).
    """
    n = data.shape[0]
    k = min(k_neighbors, n - 1)
    if k < 1:
        raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}.")

    G = base_graph(n)
    mst_e, dist = mst_edges(data)
    add_edges(G, mst_e, source="mst")

    for i in range(n):
        if G.degree(i) >= k:
            continue
        # Nearest candidates first; position 0 is i itself (distance 0).
        for j in map(int, np.argsort(dist[i])[1:]):
            if G.degree(i) >= k:
                break
            if not G.has_edge(i, j):
                G.add_edge(i, j, source="fill")
    return G
