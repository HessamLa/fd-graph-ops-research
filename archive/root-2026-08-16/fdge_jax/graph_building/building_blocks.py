"""
Shared helpers used across graph-construction strategies.

These are a cross-file internal API (not underscored) so strategy modules
can import them cleanly: `from ..building_blocks import mst_edges, ...`.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist
from pynndescent import NNDescent


def mst_edges(data):
    """Exact Euclidean MST edges as a list of (i, j) tuples.

    Dense O(n^2) distance matrix -> scipy MST. Exact and simple; fine for
    MVP scales (up to ~10-20k points). Swap for an approximate MST later
    if needed. Returns the distance matrix too, since callers reuse it.
    """
    dist = squareform(pdist(data))          # (n, n) symmetric, zeros on diag
    mst = minimum_spanning_tree(dist)       # sparse, upper-triangular
    rows, cols = mst.nonzero()
    edges = [(int(i), int(j)) for i, j in zip(rows, cols)]
    return edges, dist


def knn_edges(data, k_neighbors, metric="euclidean", random_state=42):
    """Directed kNN relations from PyNNDescent as (i, j) pairs, self excluded.

    Undirectedness/union semantics come for free when added to nx.Graph.
    """
    n = data.shape[0]
    if not 1 <= k_neighbors <= n - 1:
        raise ValueError(
            f"k_neighbors must be in [1, n-1] = [1, {n - 1}], got {k_neighbors}."
        )
    index = NNDescent(
        data,
        n_neighbors=k_neighbors + 1,        # +1: self comes back first
        metric=metric,
        random_state=random_state,
    )
    idx, _ = index.neighbor_graph           # (n, k+1), ascending distance
    return [
        (i, int(j))
        for i in range(n)
        for j in idx[i]
        if j != i                            # drop self by index, not distance
    ]


def base_graph(n):
    """Empty simple graph with all n nodes (keeps isolated nodes explicit)."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    return G


def add_edges(G, edges, source):
    """Add edges with a `source` tag, never overwriting an existing tag."""
    for i, j in edges:
        if not G.has_edge(i, j):
            G.add_edge(i, j, source=source)
