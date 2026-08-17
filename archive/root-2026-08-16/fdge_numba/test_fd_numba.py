"""Smoke tests for the Numba rewrite.

1. Kernel correctness: compare _forces_204_hidx against a naive dense
   NumPy oracle that mirrors the legacy torch forward() math exactly
   (including h = n for disconnected pairs and degree division).
2. Batching invariance: batch_count in {1, 3, 7} must give identical dZ.
3. End-to-end: embed Zachary's karate club into 2D, check attraction
   dominates structure (mean intra-hop-1 distance < mean distant distance).
"""
# %%
import sys
import numpy as np
import networkx as nx

# %%
%load_ext autoreload
%autoreload 2
# sys.path.insert(0, "/home/claude")
from forcedirected_numba.ForceDirected import ForceDirected
from forcedirected_numba.model_204_shell import (
    FDModel, graph_to_csr, get_hops_csr, _forces_204_hidx)

# %%
def naive_oracle(Z, hops, degrees, k1, k2, k3, k4):
    """Dense, unvectorized reference of the legacy forward() (drop disabled)."""
    n, d = Z.shape
    out = np.zeros((n, d))
    for u in range(n):
        # shell sizes for node u: occurrences of each hop value in row u
        vals, counts = np.unique(hops[u], return_counts=True)
        size = dict(zip(vals.tolist(), counts.tolist()))
        acc = np.zeros(d)
        for v in range(n):
            h = hops[u, v]
            if h <= 0:
                continue
            diff = Z[v] - Z[u]
            x = np.linalg.norm(diff)
            if x == 0:
                continue
            Fa = k1 * (1.0 / size[h]) * x * np.exp(-k2 * (h - 1))
            Fr = -k3 * h * np.exp(-k4 * x)
            acc += (Fa + Fr) / x * diff
        out[u] = acc / degrees[u] if degrees[u] > 0 else 0.0
    return out


def test_kernel_correctness():
    rng = np.random.default_rng(0)
    # graph with two components (exercises the disconnected h=n path)
    G = nx.disjoint_union(nx.karate_club_graph(), nx.path_graph(6))
    n = G.number_of_nodes()
    model = FDModel(G, n_dim=3, random_drop_rate=0.0, verbosity=0, seed=1)
    Z = rng.standard_normal((n, 3))
    model.Z = np.ascontiguousarray(Z)

    got = model.forward(0, n)
    hops = model.hops.astype(np.int64)  # includes sentinel value n
    want = naive_oracle(Z, hops, model.degrees, model.k1, model.k2, model.k3, model.k4)
    err = np.abs(got - want).max()
    assert err < 1e-9, f"kernel mismatch, max abs err={err}"
    print(f"[ok] kernel matches naive oracle (max abs err {err:.2e}, "
          f"two components, sentinel h=n exercised)")


def test_batching_invariance():
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    results = []
    for bc in (1, 3, 7):
        m = FDModel(G, n_dim=2, random_drop_rate=0.0, verbosity=0, seed=42)
        m.embed(epochs=3, batch_count=bc)
        results.append(m.get_embeddings())
    assert np.allclose(results[0], results[1]) and np.allclose(results[0], results[2])
    print("[ok] batch_count in {1,3,7} gives identical embeddings")


def test_end_to_end():
    G = nx.karate_club_graph()
    m = FDModel(G, n_dim=2, random_drop_rate=0.5, verbosity=0, seed=7)
    m.embed(epochs=120, epsilon=1e-4)
    Z = m.get_embeddings()
    assert np.isfinite(Z).all()
    hops = m.hops
    d = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
    near = d[hops == 1].mean()
    far = d[hops >= 3].mean()
    assert near < far, f"structure not recovered: near={near:.3f} far={far:.3f}"
    print(f"[ok] karate club end-to-end: mean dist h=1: {near:.3f} < h>=3: {far:.3f} "
          f"(stopped at epoch {m.latest_epoch})")
    df = m.get_embeddings_df()
    assert df.shape == (34, 3)
    print("[ok] get_embeddings_df:", df.shape)


def test_convergence_and_momentum():
    G = nx.karate_club_graph()
    m = FDModel(G, n_dim=2, random_drop_rate=0.0, verbosity=0, seed=3, beta=0.7)
    m.embed(epochs=50, lr=0.2, epsilon=1e-5)
    assert np.isfinite(m.get_embeddings()).all()
    print(f"[ok] momentum run finite (beta=0.7, stopped at epoch {m.latest_epoch})")


def test_callbacks():
    from forcedirected_numba.ForceDirected import Callback_Base
    events = []

    class Rec(Callback_Base):
        def on_train_begin(self, model, **kw): events.append("tb")
        def on_epoch_end(self, model, **kw): events.append("ee")
        def on_train_end(self, model, **kw): events.append("te")

    G = nx.path_graph(10)
    m = FDModel(G, n_dim=2, random_drop_rate=0.0, verbosity=0, seed=0)
    m.attach_callback(Rec())
    m.embed(epochs=4)
    assert events == ["tb", "ee", "ee", "ee", "ee", "te"], events
    print("[ok] callbacks fire in order:", events)


if __name__ == "__main__":
    test_kernel_correctness()
    test_batching_invariance()
    test_end_to_end()
    test_convergence_and_momentum()
    test_callbacks()
    print("\nAll tests passed.")

# %%
