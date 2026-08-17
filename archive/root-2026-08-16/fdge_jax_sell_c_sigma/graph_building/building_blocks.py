"""
Shared helpers used across graph-construction strategies.

These are a cross-file internal API (not underscored) so strategy modules
can import them cleanly: `from ..building_blocks import mst_edges, ...`.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from pynndescent import NNDescent

# Default neighbourhood size for the kNN graph MST builds its spanning tree
# over. Benchmarked: k=5 already recovers the exact Euclidean MST edge-for-
# edge on well-behaved data; 10 gives headroom on real (non-toy) data at
# negligible extra cost (still O(n*k), not O(n^2)).
_DEFAULT_MST_K = 10


def mst_edges(data, k_neighbors=None):
    """Euclidean MST edges as a list of (i, j) tuples.

    Built over a kNN graph (sklearn.kneighbors_graph -> scipy MST) instead
    of the dense (n, n) distance matrix: measured 80GB+ RAM for the dense
    matrix at n=100k vs ~230MB for this, with identical MST weight on
    non-adversarial data (benchmarked in logs/subagent_graph-building.log).
    A plain kNN graph can be disconnected -- e.g. well-separated clusters
    stay disconnected at ANY k, since no cross-cluster point is ever a
    k-nearest-neighbour -- so we detect that and bridge components with
    their true nearest cross-component edge before running MST. This keeps
    the "always connected" contract without ever raising k on the (false)
    hope that a bigger k will bridge the gap.
    """
    n = data.shape[0]
    k = min(k_neighbors or _DEFAULT_MST_K, n - 1)
    graph = kneighbors_graph(data, n_neighbors=k, mode="distance")
    graph = graph.maximum(graph.T)          # kNN is directed; MST needs undirected
    n_components, labels = connected_components(graph, directed=False)
    if n_components > 1:
        graph = _connect_components(data, graph, n_components, labels)
    mst = minimum_spanning_tree(graph)      # sparse, upper-triangular
    rows, cols = mst.nonzero()
    return [(int(i), int(j)) for i, j in zip(rows, cols)]


def _connect_components(data, graph, n_components, labels):
    """Bridge a disconnected kNN graph's components with real edges.

    Rare path (only fires on disconnection, e.g. well-separated clusters),
    so it optimizes for correctness over micro-perf, while still avoiding
    an O(n^2) all-pairs search: pick WHICH components to bridge via a
    (cheap, C x C) MST over centroid distances, then for each chosen pair
    find the true nearest cross-component point pair with one exact kNN
    query (O(|A| log|B|), not O(|A|*|B|)).
    """
    centroids = np.array([data[labels == c].mean(axis=0) for c in range(n_components)])
    bridge_topology = minimum_spanning_tree(squareform(pdist(centroids)))
    ca, cb = bridge_topology.nonzero()

    graph = graph.tolil()
    for a, b in zip(ca, cb):
        pts_a = np.where(labels == a)[0]
        pts_b = np.where(labels == b)[0]
        dist, idx = NearestNeighbors(n_neighbors=1).fit(data[pts_b]).kneighbors(data[pts_a])
        best = np.argmin(dist[:, 0])
        i, j, d = pts_a[best], pts_b[idx[best, 0]], dist[best, 0]
        graph[i, j] = d
        graph[j, i] = d
    return graph.tocsr()


def nearest_neighbors(data, k):
    """Exact k nearest neighbour indices per row (self excluded), ascending.

    Used where an approximate index (PyNNDescent) isn't good enough -- e.g.
    mst_k_minimum's degree-floor fill, which documents itself as exact.
    sklearn chunks its internal distance computation even on the 'brute'
    fallback, so this stays memory-safe without materializing an (n, n)
    matrix.
    """
    n = data.shape[0]
    k = min(k, n - 1)
    # .kneighbors() with NO query arg -- sklearn only excludes a point from
    # its own neighbour list in this implicit form. Passing `data` back in
    # explicitly returns self as neighbour 0 (distance 0), silently handing
    # back k-1 real neighbours instead of k.
    _, idx = NearestNeighbors(n_neighbors=k).fit(data).kneighbors()
    return idx


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
    # Vectorized over the nested-loop + int()-per-element original: ~10x
    # faster building the (i, j) pairs (benchmarked in
    # logs/subagent_graph-building.log), identical output.
    i_arr = np.repeat(np.arange(n), idx.shape[1])
    j_arr = idx.ravel()
    mask = j_arr != i_arr                   # drop self by index, not distance
    return list(zip(i_arr[mask].tolist(), j_arr[mask].tolist()))


def base_graph(n):
    """Empty simple graph with all n nodes (keeps isolated nodes explicit)."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    return G


def add_edges(G, edges, source):
    """Add edges with a `source` tag, never overwriting an existing tag.

    Benchmarked against filter-then-`G.add_edges_from(...)`: that variant
    was slower at every n tried (0.5-0.9x), not faster -- it does the same
    has_edge-guard pass PLUS a second pass inside add_edges_from, while
    this single-pass loop does the check and the add together. Kept the
    loop (see logs/subagent_graph-building.log for the numbers).
    """
    for i, j in edges:
        if not G.has_edge(i, j):
            G.add_edge(i, j, source=source)
