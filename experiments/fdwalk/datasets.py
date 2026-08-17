#!/bin/env python3
"""datasets.py -- the graphs of the fdwalk experiment.

The parsing follows `fodined/modular.py`, and it holds the two repairs that
that file needed for a large graph:

  * The subgraph comes from a mask and a renumber, in O(nnz). The short form
    `A[keep][:, keep]` overflows the int32 that scipy uses for the count of
    the result, and it stops with "negative dimensions are not allowed" on
    roadNet-CA.
  * A graph that is not a tree takes a BFS ball, and not a walk down an edge
    list. The walk gave 500,000 nodes with 504 edges between them on
    roadNet-CA, and every score was then 1.0000 on an empty problem.

Every function is pure and it prints nothing.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order

DATA = "/home/h/gnn/fd-graph-embedding/fdmap/data_cache"

SNAP = {"com_youtube": "com_youtube/com-youtube.ungraph.txt",
        "as_skitter": "as_skitter/as-skitter.txt",
        "roadnet_ca": "roadnet_ca/roadNet-CA.txt"}


def _edges_cora():
    return np.loadtxt(f"{DATA}/cora/cora.cites", dtype=np.int64)


def _edges_pubmed():
    """`<edge_id>\\tpaper:<src>\\t|\\tpaper:<dst>`, after two header lines."""
    src, dst = [], []
    with open(f"{DATA}/pubmed/Pubmed-Diabetes/data/"
              f"Pubmed-Diabetes.DIRECTED.cites.tab") as f:
        f.readline(), f.readline()
        for line in f:
            p = line.split("\t")
            src.append(int(p[1].split(":")[1]))
            dst.append(int(p[3].split(":")[1]))
    return np.column_stack([src, dst]).astype(np.int64)


def _edges_snap(name):
    """The C parser of pandas reads 11M lines in seconds."""
    import pandas as pd
    return pd.read_csv(f"{DATA}/{SNAP[name]}", sep="\t", comment="#",
                       header=None, names=["u", "v"],
                       dtype=np.int64).to_numpy()


def read_edges(name: str):
    if name == "cora":
        return _edges_cora()
    if name == "pubmed":
        return _edges_pubmed()
    if name in SNAP:
        return _edges_snap(name)
    raise ValueError(f"Unknown graph {name!r}")


def to_csr(raw):
    """An edge list into a symmetric CSR of 1.0, with ids 0..n-1."""
    _, flat = np.unique(raw.ravel(), return_inverse=True)
    e = flat.reshape(raw.shape)
    e = e[e[:, 0] != e[:, 1]]
    n = int(e.max()) + 1
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    A = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    A.sort_indices()
    return A, n


def induced(A, keep, n: int):
    """The subgraph of the nodes `keep`, renumbered to 0..len(keep)-1."""
    mask = np.zeros(n, dtype=bool)
    mask[keep] = True
    new_id = np.full(n, -1, dtype=np.int64)
    new_id[keep] = np.arange(keep.size)
    c = A.tocoo()
    sel = mask[c.row] & mask[c.col]
    B = sp.csr_matrix(
        (c.data[sel], (new_id[c.row[sel]], new_id[c.col[sel]])),
        shape=(keep.size, keep.size))
    B.eliminate_zeros()
    return B


def load(name: str, max_nodes: int = 0, seed: int = 42):
    """`(A, n)`. A BFS ball of `max_nodes` nodes if the graph is larger.

    A BFS ball stays connected, thus the hop distances of the evaluation
    keep their meaning.
    """
    A, n = to_csr(read_edges(name))
    if max_nodes and n > max_nodes:
        start = int(np.random.default_rng(seed).integers(0, n))
        order = breadth_first_order(A, start, directed=False,
                                    return_predecessors=False)
        keep = np.sort(order[:max_nodes])
        A = induced(A, keep, n)
        n = A.shape[0]
    return A, n
