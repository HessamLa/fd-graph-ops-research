"""Smoke test for graph_augmenting.hopfill.augment_graph (T2.1).

Not the real validation suite (that's T4.x, ``fdge2/validation/``). This
checks the contract this task is responsible for:

* ``D[i, j]`` indexing works for both ``is_sparse=True`` and ``False``
* ``D[u, u] == 0``
* direct edges get hop distance 1
* a known multi-hop pair gets the right distance, cross-checked against
  ``networkx.shortest_path_length`` as an independent oracle (not just
  "trust our own BFS port")
* disconnected pairs get the ``unreachable`` sentinel (default: n)
* ``D`` is symmetric
* the whole thing plugs into ``core.ForceDirected.embed`` without a
  signature mismatch (end-to-end smoke, trivial force law)

Run:
    source /home/h/gnn/fd-graph-embedding/fdmap/.venv/bin/activate
    python -m fdge2.graph_augmenting.test_hopfill_smoke      # from repo root (fdmap/)
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import scipy.sparse as sp

from fdge2.core import ForceDirected
from fdge2.graph_augmenting import augment_graph, get_shell_counts


def _get(D, i, j):
    """Scalar D[i, j] regardless of dense ndarray or scipy.sparse container."""
    v = D[i, j]
    if sp.issparse(D):
        v = v.toarray().item() if hasattr(v, "toarray") else v
    return int(v)


# ---------------------------------------------------------------------------
# Contract tests, run against both is_sparse=True and False
# ---------------------------------------------------------------------------
def _check_karate(is_sparse: bool):
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    D = augment_graph(G, is_sparse=is_sparse)

    assert D.shape == (n, n)
    if is_sparse:
        assert sp.issparse(D)
    else:
        assert isinstance(D, np.ndarray)

    # D[u, u] == 0
    for u in (0, 5, 33):
        assert _get(D, u, u) == 0

    # direct edges -> hop distance 1
    nodes = list(G.nodes())
    for u, v in list(G.edges())[:5]:
        iu, iv = nodes.index(u), nodes.index(v)
        assert _get(D, iu, iv) == 1
        assert _get(D, iv, iu) == 1

    # multi-hop pair, cross-checked against networkx as an independent oracle
    u, v = nodes[0], nodes[33]
    expected = nx.shortest_path_length(G, u, v)
    iu, iv = nodes.index(u), nodes.index(v)
    assert _get(D, iu, iv) == expected, (_get(D, iu, iv), expected)

    # symmetry (sample of pairs, full n=34 check is cheap too)
    Dd = D.toarray() if sp.issparse(D) else D
    assert np.array_equal(Dd, Dd.T), "D is not symmetric"

    # karate club is connected -> nothing should carry the sentinel
    assert not (Dd == n).any()

    print(f"[ok] karate_club_graph, is_sparse={is_sparse}: shape, self=0, "
          f"direct edges=1, multi-hop matches networkx oracle, symmetric")


def _check_disconnected(is_sparse: bool):
    G = nx.disjoint_union(nx.path_graph(4), nx.path_graph(3))
    n = G.number_of_nodes()  # 7
    D = augment_graph(G, is_sparse=is_sparse)
    Dd = D.toarray() if sp.issparse(D) else D

    # within-component distances match networkx
    for u, v in [(0, 3), (1, 2), (4, 6)]:
        expected = nx.shortest_path_length(G, u, v)
        assert _get(D, u, v) == expected

    # cross-component pairs are unreachable -> sentinel == n (default convention)
    for u, v in [(0, 4), (2, 6), (3, 5)]:
        assert _get(D, u, v) == n, (u, v, _get(D, u, v), n)
        assert _get(D, v, u) == n

    assert np.array_equal(Dd, Dd.T), "D is not symmetric"
    print(f"[ok] disconnected graph, is_sparse={is_sparse}: within-component "
          f"matches networkx, cross-component == sentinel (n={n}), symmetric")


def test_karate_dense():
    _check_karate(is_sparse=False)


def test_karate_sparse():
    _check_karate(is_sparse=True)


def test_disconnected_dense():
    _check_disconnected(is_sparse=False)


def test_disconnected_sparse():
    _check_disconnected(is_sparse=True)


def test_custom_unreachable_sentinel():
    G = nx.disjoint_union(nx.path_graph(2), nx.path_graph(2))
    D = augment_graph(G, is_sparse=False, unreachable=-1)
    assert _get(D, 0, 2) == -1
    assert _get(D, 1, 3) == -1
    print("[ok] custom `unreachable=` sentinel honored")


def test_get_shell_counts_helper():
    # small ring: every node has exactly 1 neighbor at hop 1, etc.
    G = nx.cycle_graph(6)
    D = augment_graph(G, is_sparse=False)
    n_bins = int(D.max()) + 1
    counts = get_shell_counts(D, n_bins)
    assert counts.shape == (6, n_bins)
    # each node: 1 node at hop 0 (itself), 2 at hop 1, 2 at hop 2, 1 at hop 3
    assert counts[0].tolist() == [1, 2, 2, 1]
    print("[ok] get_shell_counts helper matches expected shell sizes on C6")


# ---------------------------------------------------------------------------
# End-to-end plug-in check against core.ForceDirected
# ---------------------------------------------------------------------------
class _HopfillSmokeModel(ForceDirected):
    """Throwaway subclass: real augment_graph, trivial deterministic forces.

    Only exists to catch a signature mismatch against the real
    ForceDirected base class before it becomes the embedding-stage
    implementer's problem (per the task instructions) -- the force law
    itself is someone else's job (T3.1).
    """

    def augment_graph(self, G, is_sparse: bool = True, **kwargs):
        return augment_graph(G, is_sparse=is_sparse, **kwargs)

    def forces(self, Z, D, row_start, row_end, **kwargs):
        # trivial, deterministic: contract toward the origin
        return -0.1 * Z[row_start:row_end]


def test_plugs_into_force_directed_dense():
    G = nx.karate_club_graph()
    m = _HopfillSmokeModel(n_dim=2, verbosity=0, seed=0)
    Z = m.embed(G, epochs=3, is_sparse=False)
    assert Z.shape == (G.number_of_nodes(), 2)
    assert np.isfinite(Z).all()
    print("[ok] end-to-end: ForceDirected.embed(G, epochs=3, is_sparse=False) runs")


def test_plugs_into_force_directed_sparse():
    G = nx.karate_club_graph()
    m = _HopfillSmokeModel(n_dim=2, verbosity=0, seed=0)
    Z = m.embed(G, epochs=3, is_sparse=True)
    assert Z.shape == (G.number_of_nodes(), 2)
    assert np.isfinite(Z).all()
    print("[ok] end-to-end: ForceDirected.embed(G, epochs=3, is_sparse=True) runs "
          "(D is a scipy.sparse.csr_matrix all the way through the base loop)")


if __name__ == "__main__":
    test_karate_dense()
    test_karate_sparse()
    test_disconnected_dense()
    test_disconnected_sparse()
    test_custom_unreachable_sentinel()
    test_get_shell_counts_helper()
    test_plugs_into_force_directed_dense()
    test_plugs_into_force_directed_sparse()
    print("\nAll graph_augmenting.hopfill smoke tests passed.")
