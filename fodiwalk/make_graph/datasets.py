#!/bin/env python3
"""make_graph.datasets -- the graphs of the fodiwalk experiment.

The parsing follows `fodined/modular.py`, and it holds the two repairs that
that file needed for a large graph:

  * The subgraph comes from a mask and a renumber, in O(nnz). The short form
    `A[keep][:, keep]` overflows the int32 that scipy uses for the count of
    the result, and it stops with "negative dimensions are not allowed" on
    roadNet-CA.
  * A graph that is not a tree takes a BFS ball, and not a walk down an edge
    list. The walk gave 500,000 nodes with 504 edges between them on
    roadNet-CA, and every score was then 1.0000 on an empty problem.

A TREE takes the other path, and the reason is the mirror image. A BFS ball
from a node of a high degree returns a STAR of depth 1: NCBI has nodes with
tens of thousands of children, thus every pair without an edge is exactly 2
hops apart and every method reaches a perfect score. A subtree keeps the
DEPTH of the tree. `TREE` names the graphs that take that path.

Every function is pure and it prints nothing.
"""
# Provenance: moved from `experiments/fdwalk/datasets.py` on 2026-08-19.
# One change: `DATA` is derived from this file and overridable, thus an
# absolute path of one machine is not compiled into the package.
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order

# The only change of the move: the path is derived from this file and it
# is overridable, thus the package works from any working directory and an
# absolute path of one machine is not compiled into it.
DATA = os.environ.get(
    "FDMAP_DATA",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data_cache"))

SNAP = {"com_youtube": "com_youtube/com-youtube.ungraph.txt",
        "as_skitter": "as_skitter/as-skitter.txt",
        "roadnet_ca": "roadnet_ca/roadNet-CA.txt"}

# The graphs whose edge list reads "child -> parent". `load` truncates them
# with `subtree` and not with a BFS ball. See the module docstring.
TREE = ("wordnet", "ncbi_taxonomy")


def _edges_cora():
    return np.loadtxt(f"{DATA}/cora/cora.cites", dtype=np.int64)


def _edges_citeseer():
    """`<from> <to>`, paper ids. Some ids are words, not numbers.

    `cora.cites` holds only integer ids; `citeseer.cites` mixes numeric and
    alphanumeric ones (e.g. `bradshaw97introduction`), so this reads text
    and lets `to_csr`'s `np.unique` renumber it, same as `cora`.
    """
    return np.loadtxt(f"{DATA}/citeseer/citeseer.cites", dtype="<U32")


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


def _edges_wordnet():
    """The hypernym edges of the WordNet 3.0 nouns, as `(child, parent)`.

    Each line of `data.noun` is one synset, and the pointer `@` (or `@i`)
    names its hypernym. The byte offset of a synset is its id.
    """
    src, dst = [], []
    with open(f"{DATA}/wordnet/dict/data.noun", encoding="latin-1") as f:
        for line in f:
            if line.startswith("  "):                    # licence header
                continue
            parts = line.partition("|")[0].split()
            i = 3
            i += 1 + 2 * int(parts[i], 16)               # skip the words
            p_cnt = int(parts[i])
            i += 1
            for _ in range(p_cnt):
                if parts[i] in ("@", "@i"):
                    src.append(int(parts[0]))
                    dst.append(int(parts[i + 1]))
                i += 4                                   # sym off pos st
    return np.column_stack([src, dst]).astype(np.int64)


def _edges_ncbi():
    """`tax_id | parent_tax_id | ...`, as `(child, parent)`.

    Each node has exactly one parent, thus this file is a true tree. The
    root is its own parent, and that line is dropped.
    """
    src, dst = [], []
    with open(f"{DATA}/ncbi_taxonomy/nodes.dmp") as f:
        for line in f:
            p = line.split("\t|\t", 2)
            a, b = int(p[0]), int(p[1])
            if a != b:
                src.append(a)
                dst.append(b)
    return np.column_stack([src, dst]).astype(np.int64)


def read_edges(name: str):
    if name == "cora":
        return _edges_cora()
    if name == "citeseer":
        return _edges_citeseer()
    if name == "pubmed":
        return _edges_pubmed()
    if name == "wordnet":
        return _edges_wordnet()
    if name == "ncbi_taxonomy":
        return _edges_ncbi()
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


def remap(raw):
    """An edge list with arbitrary ids into DIRECTED pairs on `0..n-1`.

    `to_csr` symmetrizes and it loses the direction. `subtree` needs it:
    the edge of a tree reads "child -> parent".
    """
    ids, flat = np.unique(raw.ravel(), return_inverse=True)
    e = flat.reshape(raw.shape)
    return e[e[:, 0] != e[:, 1]], int(ids.size)


def subtree(edges, n: int, cap: int, seed: int = 42):
    """The node ids of ONE subtree of about `cap` nodes.

    `edges` reads `(child, parent)`. The walk starts at a random node and
    it climbs to the parent until the subtree is large enough, at most 200
    times.

    Use it on a TREE, and use a BFS ball on anything else. A subtree keeps
    the DEPTH of the tree, which a BFS ball from a hub does not: NCBI has
    nodes with tens of thousands of children, thus the ball is a star of
    depth 1, every pair without an edge is then exactly 2 hops apart, and
    every method reaches a perfect score on it.

    The mirror failure is real too. A road network is not a tree, and this
    walk read its edge list as "parent -> child" and returned 500,000 nodes
    with 504 edges between them.

    ONE DIVERGENCE FROM `fodined/modular.py`, and it repairs a defect.
    `modular.py` collects the descendants with no `seen` set. WordNet is a
    DAG and not a tree -- a synset may name two hypernyms -- thus a node
    below two kept parents is collected TWO times. `induced` then writes
    `new_id[keep] = arange(keep.size)`, the last write wins, and every
    other copy becomes a row with NO edge. Measured on WordNet at
    `cap = 20000`: 20,000 entries for 19,581 distinct nodes, thus 419
    isolated rows and 420 connected components in a "subtree".

    An isolated row is not harmless. It has no `h = 1` entry, and the
    evaluation reads its distance to every other node as unreachable.
    `seen` below drops the duplicate, thus the result is `cap` DISTINCT
    nodes and one component. NCBI is a true tree and it never met this
    defect, which is why it stayed hidden.
    """
    parent = np.full(n, -1, dtype=np.int64)
    parent[edges[:, 0]] = edges[:, 1]        # the first parent wins (DAG)
    o = np.argsort(edges[:, 1], kind="stable")
    child_of, p_sorted = edges[o, 0], edges[o, 1]
    lo = np.searchsorted(p_sorted, np.arange(n))
    hi = np.searchsorted(p_sorted, np.arange(n), side="right")

    def below(root, limit):
        out, frontier = [root], [root]
        seen = {root}                        # see the DIVERGENCE note
        while frontier and len(out) < limit:
            nxt = []
            for u in frontier:
                for k in child_of[lo[u]:hi[u]]:
                    if len(out) >= limit:
                        break
                    k = int(k)
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(k)
                    nxt.append(k)
            frontier = nxt
        return out

    node = int(np.random.default_rng(seed).integers(0, n))
    for _ in range(200):                     # guard against a long climb
        got = below(node, cap + 1)
        if len(got) >= cap // 2 or parent[node] < 0:
            break
        node = int(parent[node])
    return np.sort(np.asarray(got[:cap]))


def load(name: str, max_nodes: int = 0, seed: int = 42):
    """`(A, n)`, truncated to about `max_nodes` nodes if the graph is larger.

    A graph that is not a tree takes a BFS ball, which stays connected,
    thus the hop distances of the evaluation keep their meaning. A tree
    (`TREE`) takes one subtree, which keeps the DEPTH. See `subtree` for
    what each choice does to the other kind of graph.
    """
    raw = read_edges(name)
    A, n = to_csr(raw)
    if max_nodes and n > max_nodes:
        if name in TREE:
            edges, n_ids = remap(raw)
            keep = subtree(edges, max(n, n_ids), max_nodes, seed)
        else:
            start = int(np.random.default_rng(seed).integers(0, n))
            order = breadth_first_order(A, start, directed=False,
                                        return_predecessors=False)
            keep = np.sort(order[:max_nodes])
        A = induced(A, keep, n)
        n = A.shape[0]
    return A, n
