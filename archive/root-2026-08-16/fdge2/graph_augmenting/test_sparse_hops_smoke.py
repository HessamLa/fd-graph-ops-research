"""Smoke test for graph_augmenting.sparse_hops.augment_graph (T2.2).

Analogous to ``test_hopfill_smoke.py`` but for the *sparse* policy. It
checks the contract this task is responsible for:

* ``D[i, j]`` indexing works (dense and sparse containers)
* ``D[u, u] == 0``; direct edges -> hop 1
* within-radius multi-hop distances match ``networkx.shortest_path_length``
  (independent oracle, not "trust our own BFS")
* pairs BEYOND ``radius`` are legitimately ABSENT from the sparse pattern
  (structural zero, not a stored 0) -- i.e. genuinely sparse, not
  dense-with-extra-steps like hop-fill's ``is_sparse=True`` path
* nnz grows ~linearly, not quadratically, with n at fixed avg degree
* the Q4 collapse risk is real: an absent far pair -> zero repulsion today
* plugs into ``core.ForceDirected.embed`` with no signature mismatch

Run:
    source /home/h/gnn/fd-graph-embedding/fdmap/.venv/bin/activate
    python -m fdge2.graph_augmenting.test_sparse_hops_smoke   # from repo root (fdmap/)
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import scipy.sparse as sp

from fdge2.core import ForceDirected
from fdge2.graph_augmenting import sparse_hops


def _get(D, i, j):
    """Scalar D[i, j] regardless of dense ndarray or scipy.sparse container."""
    v = D[i, j]
    if sp.issparse(D):
        v = v.toarray().item() if hasattr(v, "toarray") else v
    return int(v)


def _is_stored(D, i, j):
    """True iff (i, j) is an EXPLICIT entry in the CSR pattern (not absent)."""
    assert sp.issparse(D)
    Dc = D.tocsr()
    cols = Dc.indices[Dc.indptr[i]:Dc.indptr[i + 1]]
    return int(j) in cols.tolist()


# ---------------------------------------------------------------------------
# Within-radius correctness, cross-checked against networkx
# ---------------------------------------------------------------------------
def _check_karate(is_sparse: bool, radius: int):
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    D = sparse_hops.augment_graph(G, is_sparse=is_sparse, radius=radius)

    assert D.shape == (n, n)
    if is_sparse:
        assert sp.issparse(D)
    else:
        assert isinstance(D, np.ndarray)

    nodes = list(G.nodes())
    # self -> 0
    for u in (0, 5, 33):
        assert _get(D, u, u) == 0

    # direct edges -> hop 1
    for u, v in list(G.edges())[:5]:
        iu, iv = nodes.index(u), nodes.index(v)
        assert _get(D, iu, iv) == 1
        assert _get(D, iv, iu) == 1

    # every stored within-radius pair matches the networkx oracle exactly
    sp_len = dict(nx.all_pairs_shortest_path_length(G))
    Dd = D.toarray() if sp.issparse(D) else D
    for u in range(n):
        for v in range(n):
            true_h = sp_len[nodes[u]].get(nodes[v], None)
            if is_sparse:
                stored = _is_stored(D, u, v)
                # stored  <=>  reachable within radius
                assert stored == (true_h is not None and true_h <= radius), (u, v)
                if stored:
                    assert _get(D, u, v) == true_h, (u, v, _get(D, u, v), true_h)
            else:
                if true_h is not None and true_h <= radius:
                    assert Dd[u, v] == true_h, (u, v)

    # symmetry of stored values
    if is_sparse:
        Dc = D.tocsr()
        assert (Dc != Dc.T).nnz == 0, "sparse D not symmetric"
    print(f"[ok] karate is_sparse={is_sparse} radius={radius}: within-radius "
          f"hops match networkx oracle, self=0, direct=1, symmetric")


def test_karate_sparse_r2():
    _check_karate(is_sparse=True, radius=2)


def test_karate_sparse_r3():
    _check_karate(is_sparse=True, radius=3)


def test_karate_dense_r2():
    _check_karate(is_sparse=False, radius=2)


# ---------------------------------------------------------------------------
# The point of the exercise: genuinely sparse, not dense-in-a-sparse-box
# ---------------------------------------------------------------------------
def test_far_pairs_are_absent():
    """Pairs beyond `radius` must be STRUCTURALLY absent, not stored zeros."""
    # path 0-1-2-3-4-5: nodes 0 and 5 are 5 hops apart
    G = nx.path_graph(6)
    D = sparse_hops.augment_graph(G, is_sparse=True, radius=2)

    # within radius: stored, correct value
    assert _is_stored(D, 0, 2) and _get(D, 0, 2) == 2
    # beyond radius: NOT stored (and D[i,j] reads back as 0 only because
    # scipy returns 0 for missing -- prove it's genuinely absent instead)
    assert not _is_stored(D, 0, 3)
    assert not _is_stored(D, 0, 5)
    assert _get(D, 0, 5) == 0            # would look like "distance 0" if trusted

    # count check: a length-6 path at radius 2 stores, per node, itself +
    # up to 2 left + up to 2 right => far fewer than n*n = 36 entries
    assert D.nnz < 6 * 6, (D.nnz,)
    print(f"[ok] far pairs structurally absent (path6 r2 nnz={D.nnz} << 36); "
          f"absence is real, not a stored zero")


def test_disconnected_components_absent_no_sentinel():
    """Cross-component pairs are absent -- no `unreachable` sentinel (by design)."""
    G = nx.disjoint_union(nx.path_graph(4), nx.path_graph(3))  # n = 7
    D = sparse_hops.augment_graph(G, is_sparse=True, radius=3)
    # within a component, within radius -> stored
    assert _is_stored(D, 0, 3) and _get(D, 0, 3) == 3
    # across components -> absent (NOT stored as n, unlike hopfill)
    assert not _is_stored(D, 0, 4)
    assert not _is_stored(D, 3, 6)
    assert _get(D, 0, 4) == 0
    print("[ok] cross-component pairs absent, no sentinel (sparse-policy design)")


# ---------------------------------------------------------------------------
# nnz vs n scaling: linear-ish, not quadratic (measured, not asserted vibes)
# ---------------------------------------------------------------------------
def test_nnz_scales_subquadratic():
    """At fixed avg degree + radius, nnz/n stays ~flat => nnz ~ O(n)."""
    avg_deg, radius = 8, 2
    ratios = []
    nnzs = []
    for n in (500, 2000, 8000):
        G = nx.gnm_random_graph(n, n * avg_deg // 2, seed=0)
        D = sparse_hops.augment_graph(G, is_sparse=True, radius=radius)
        ratios.append(D.nnz / n)
        nnzs.append(D.nnz)
    # If it were dense (O(n^2)), nnz/n would grow ~16x from n=500 to n=8000.
    # Bounded-radius keeps nnz/n roughly constant -> ratio spread stays small.
    spread = max(ratios) / min(ratios)
    print(f"[ok] nnz/n across n=[500,2000,8000] = "
          f"{[round(r, 1) for r in ratios]} (nnz={nnzs}); spread={spread:.2f}x")
    assert spread < 3.0, (ratios, "nnz/n grew too fast -> not sub-quadratic")
    # sanity: genuinely sparse vs dense n^2
    assert nnzs[-1] < 8000 * 8000 / 10


# ---------------------------------------------------------------------------
# Q4: the collapse risk is real under this policy
# ---------------------------------------------------------------------------
def test_collapse_risk_is_real():
    """Two graph-far nodes are absent from D -> zero repulsion under the
    current 'repulsion only on D's support' force law, even if they drift
    close in embedding space. This is exactly the Q4 risk; the mitigation
    (sampled negatives) is documented in sparse_hops.__doc__."""
    G = nx.path_graph(10)
    D = sparse_hops.augment_graph(G, is_sparse=True, radius=2)
    u, v = 0, 9                          # 9 hops apart, radius 2
    assert not _is_stored(D, u, v), "expected far pair absent"
    # A force law that only iterates D's stored entries for row u never
    # visits v -> contributes exactly zero force between u and v, in either
    # direction. Emulate that "support = D's row" view:
    Dc = D.tocsr()
    row_u = set(Dc.indices[Dc.indptr[u]:Dc.indptr[u + 1]].tolist())
    assert v not in row_u                # v invisible to u's force computation
    assert "Q4 mitigation proposal" in sparse_hops.__doc__
    print("[ok] Q4 risk reproduced: far pair (0,9) absent from D -> zero "
          "repulsion today; mitigation documented in module docstring")


# ---------------------------------------------------------------------------
# End-to-end plug-in check against core.ForceDirected
# ---------------------------------------------------------------------------
class _SparseSmokeModel(ForceDirected):
    """Real sparse augment_graph, trivial deterministic forces (T3.1 owns
    the real force law -- this only catches a signature mismatch)."""

    def augment_graph(self, G, is_sparse: bool = True, **kwargs):
        return sparse_hops.augment_graph(G, is_sparse=is_sparse, **kwargs)

    def forces(self, Z, D, row_start, row_end, **kwargs):
        return -0.1 * Z[row_start:row_end]


def test_plugs_into_force_directed():
    G = nx.karate_club_graph()
    m = _SparseSmokeModel(n_dim=2, verbosity=0, seed=0)
    Z = m.embed(G, epochs=3, is_sparse=True, radius=2)
    assert Z.shape == (G.number_of_nodes(), 2)
    assert np.isfinite(Z).all()
    print("[ok] end-to-end: ForceDirected.embed(G, radius=2) runs with sparse D")


if __name__ == "__main__":
    test_karate_sparse_r2()
    test_karate_sparse_r3()
    test_karate_dense_r2()
    test_far_pairs_are_absent()
    test_disconnected_components_absent_no_sentinel()
    test_nnz_scales_subquadratic()
    test_collapse_risk_is_real()
    test_plugs_into_force_directed()
    print("\nAll graph_augmenting.sparse_hops smoke tests passed.")
