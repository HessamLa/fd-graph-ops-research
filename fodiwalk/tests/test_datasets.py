"""test_datasets.py -- the loader, and the truncation each kind of graph needs.

`load` truncates a large graph in one of two ways, and BOTH wrong choices
give a perfect score on an EMPTY problem, thus a silent defect:

    a tree with a BFS ball   -> a star of depth 1 from a hub, thus every
                                non-edge pair is exactly 2 hops apart.
    a graph with a subtree   -> roadNet-CA read as "parent -> child" gave
                                500,000 nodes and 504 edges.

The tests below assert the property that separates the two: a truncated
tree must keep its DEPTH.

The two tree datasets are large files, thus these tests are `slow`.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.csgraph import shortest_path, connected_components

from fodiwalk.make_graph import datasets as DS


def test_tree_list_is_the_two_trees():
    assert DS.TREE == ("wordnet", "ncbi_taxonomy")


def test_remap_keeps_the_direction():
    """`to_csr` symmetrizes. `remap` must not -- `subtree` reads the edge
    as `(child, parent)`."""
    raw = np.array([[10, 20], [20, 30], [40, 40]], dtype=np.int64)
    e, n = DS.remap(raw)
    assert n == 4                              # 10, 20, 30, 40
    assert e.shape == (2, 2)                   # the self loop is dropped
    assert np.array_equal(e, np.array([[0, 1], [1, 2]]))


def test_subtree_keeps_the_depth():
    """A path of 200 nodes, truncated to 50, must stay a path.

    A BFS ball would give the same answer here; the test that separates
    them needs a hub, and `test_subtree_of_a_star_is_not_the_star` is it.
    """
    n = 200
    edges = np.column_stack([np.arange(1, n), np.arange(0, n - 1)])
    # The climb stops at `cap // 2`, thus the result is 25..50 nodes and
    # not always `cap`. That is the rule `modular.py` writes.
    keep = DS.subtree(edges, n, 50, seed=0)
    assert 25 <= keep.size <= 50
    assert np.array_equal(keep, np.sort(keep))
    assert np.unique(keep).size == keep.size


def test_subtree_of_a_hub_climbs_to_a_deep_root():
    """A hub with 400 leaves, hanging under a chain of 20.

    `subtree` starts at a random node and it CLIMBS until the subtree is
    large enough, thus it reaches the chain and the result keeps the
    depth. A BFS ball from the hub would return the hub and its leaves,
    which has depth 1.
    """
    chain, leaves = 20, 400
    n = chain + leaves
    src = list(range(1, chain)) + list(range(chain, n))
    dst = list(range(0, chain - 1)) + [chain - 1] * leaves
    edges = np.column_stack([src, dst]).astype(np.int64)
    keep = DS.subtree(edges, n, 200, seed=0)
    assert keep.size == 200
    assert np.unique(keep).size == 200
    assert 0 in keep or keep.min() < chain     # it climbed into the chain


@pytest.mark.slow
@pytest.mark.parametrize("name", ["wordnet", "ncbi_taxonomy"])
def test_a_tree_dataset_keeps_its_depth(name):
    """The measurement that a star would fail: hops well above 2."""
    A, n = DS.load(name, 20_000, 42)
    assert n > 10_000
    d = shortest_path(A, method="D", unweighted=True,
                      indices=np.arange(50))
    finite = d[np.isfinite(d)]
    assert finite.max() > 5, "a star of depth 1 gives 2"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["wordnet", "ncbi_taxonomy"])
def test_a_tree_dataset_is_one_component(name):
    """A subtree is connected, thus the hop distances keep their meaning.

    This is the test that catches the DAG defect of `subtree`. WordNet
    names two hypernyms for some synsets, thus a collector with no `seen`
    set returns a node two times, `induced` gives every copy but one a row
    with NO edge, and the "subtree" then has 420 components. See
    `datasets.subtree`.
    """
    A, n = DS.load(name, 20_000, 42)
    n_comp, _ = connected_components(A, directed=False)
    assert n_comp == 1
    assert int(np.diff(A.indptr).min()) > 0, "an isolated row is a defect"


def test_subtree_of_a_dag_returns_no_duplicate():
    """The small case of the WordNet defect: one node, two parents.

    Nodes 1 and 2 are children of the root 0, and node 3 names BOTH of
    them as a parent. A collector with no `seen` set returns 3 two times.
    """
    edges = np.array([[1, 0], [2, 0], [3, 1], [3, 2]], dtype=np.int64)
    keep = DS.subtree(edges, 4, 4, seed=0)
    assert np.unique(keep).size == keep.size
