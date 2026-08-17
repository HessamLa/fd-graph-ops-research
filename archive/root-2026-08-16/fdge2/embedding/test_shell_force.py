"""Correctness tests for embedding.shell_force (T3.1 + T3.2).

Independent of the Numba kernel -- the port is checked against a naive,
unvectorized Python/NumPy oracle (IMPLEMENTATION.md methodology: don't
trust your own port; compare to a dead-simple reference). Adapts the
``naive_oracle`` from ``fdge_numba/test_fd_numba.py``.

Tests:
1. Kernel vs. naive oracle on a two-component graph (exercises the
   disconnected h = n sentinel path), for both dense and sparse ``D``.
2. The single-slot per-``D`` cache actually avoids recomputation across
   many ``forces()`` calls on the same ``D`` (and does recompute on a new
   ``D``).
3. degrees derived-from-``D`` matches degrees from ``core.csr.graph_to_csr``.
4. A momentum ``updateGradient`` override runs cleanly against the *real*
   ``forces()`` for a handful of epochs on a real small graph, via a
   ``ForceDirected`` subclass (not core's DummyModel).

NOTE on imports: this is a TEST file, not part of the ``embedding/``
package source, so it is allowed to import ``fdge2.graph_augmenting.hopfill`` to
exercise real integration (RPD.md Sec 5 import discipline governs the
package's own modules, not its test suite).

Run:
    source /home/h/gnn/fd-graph-embedding/fdmap/.venv/bin/activate
    python -m fdge2.embedding.test_shell_force      # from repo root (fdmap/)
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from fdge2.core import ForceDirected, graph_to_csr
from fdge2.graph_augmenting.hopfill import augment_graph
from fdge2.embedding.shell_force import ShellForce, forces as forces_fn


# ---------------------------------------------------------------------------
# Naive dense oracle (adapted from fdge_numba/test_fd_numba.py)
# ---------------------------------------------------------------------------
def naive_oracle(Z, hops, degrees, k1, k2, k3, k4):
    """Dense, unvectorized reference of the shell force law (drop disabled)."""
    n, d = Z.shape
    out = np.zeros((n, d))
    for u in range(n):
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


def _two_component_graph():
    # karate club + a disjoint path -> disconnected pairs -> h = n sentinel
    return nx.disjoint_union(nx.karate_club_graph(), nx.path_graph(6))


# ---------------------------------------------------------------------------
# 1. kernel correctness vs. naive oracle
# ---------------------------------------------------------------------------
def _check_matches_oracle(is_sparse):
    rng = np.random.default_rng(0)
    G = _two_component_graph()
    n = G.number_of_nodes()
    d = 3
    Z = np.ascontiguousarray(rng.standard_normal((n, d)))

    D = augment_graph(G, is_sparse=is_sparse)          # real D from the real stage
    k1, k2, k3, k4 = 0.999, 1.0, float(n), 0.01        # k3=None -> n internally

    sf = ShellForce(k1=k1, k2=k2, k3=None, k4=k4, random_drop_rate=0.0)
    got = sf(Z, D, 0, n, degrees=None, rng=rng)        # degrees derived from D

    Dd = D.toarray() if hasattr(D, "toarray") else np.asarray(D)
    hops = Dd.astype(np.int64)                          # includes sentinel n
    degrees = (Dd == 1).sum(axis=1)
    want = naive_oracle(Z, hops, degrees, k1, k2, k3, k4)

    err = np.abs(got - want).max()
    assert err < 1e-9, f"kernel mismatch (is_sparse={is_sparse}), max abs err={err}"
    return err


def test_kernel_correctness_dense():
    err = _check_matches_oracle(is_sparse=False)
    print(f"[ok] dense D: kernel matches naive oracle (max abs err {err:.2e}, "
          f"two components, sentinel h=n exercised)")


def test_kernel_correctness_sparse():
    err = _check_matches_oracle(is_sparse=True)
    print(f"[ok] sparse D: kernel matches naive oracle (max abs err {err:.2e}, "
          f"densify-once path)")


# ---------------------------------------------------------------------------
# 2. per-D cache avoids recomputation
# ---------------------------------------------------------------------------
def test_cache_avoids_recomputation():
    rng = np.random.default_rng(1)
    G = _two_component_graph()
    n = G.number_of_nodes()
    Z = np.ascontiguousarray(rng.standard_normal((n, 3)))
    D = augment_graph(G, is_sparse=True)

    sf = ShellForce(random_drop_rate=0.0)
    # many calls on the SAME D (several epochs x several batches)
    for _ in range(20):
        sf(Z, D, 0, n // 2, rng=rng)
        sf(Z, D, n // 2, n, rng=rng)
    assert sf.n_derivations == 1, (
        f"cache recomputed {sf.n_derivations}x on the same D (expected 1)")

    # a genuinely different D object must trigger exactly one more derivation
    D2 = augment_graph(G, is_sparse=True)
    sf(Z, D2, 0, n, rng=rng)
    assert sf.n_derivations == 2, (
        f"new D did not re-derive (n_derivations={sf.n_derivations})")
    print(f"[ok] single-slot id cache: {40} calls on one D -> 1 derivation; "
          f"new D -> +1")


# ---------------------------------------------------------------------------
# 3. degrees derived from D == degrees from graph_to_csr
# ---------------------------------------------------------------------------
def test_degrees_derivation():
    G = _two_component_graph()
    _, _, degrees_csr = graph_to_csr(G)
    D = augment_graph(G, is_sparse=False)
    degrees_from_D = (np.asarray(D) == 1).sum(axis=1)
    assert np.array_equal(degrees_csr, degrees_from_D), "degree derivation mismatch"
    print("[ok] degrees derived from D (hop-1 counts) == graph_to_csr degrees")


# ---------------------------------------------------------------------------
# 4. momentum updateGradient override against the REAL forces()
# ---------------------------------------------------------------------------
class _RealShellModel(ForceDirected):
    """Minimal real model wiring the real augment + real shell forces.

    Stands in for the future fdge2/models.py composition root -- exercised
    here so the momentum seam runs against the real kernel, not DummyModel.
    """

    def __init__(self, *a, random_drop_rate=0.0, **kw):
        super().__init__(*a, **kw)
        self._shell = ShellForce(random_drop_rate=random_drop_rate)
        self._degrees = None

    def augment_graph(self, G, is_sparse: bool = True, **kwargs):
        D = augment_graph(G, is_sparse=is_sparse)
        # degrees available to the model (as fdge2/models.py would have them)
        _, _, self._degrees = graph_to_csr(G)
        return D

    def forces(self, Z, D, row_start, row_end, **kwargs):
        return self._shell(Z, D, row_start, row_end,
                           degrees=self._degrees, rng=self.rng, **kwargs)


class _MomentumShellModel(_RealShellModel):
    """updateGradient override implementing velocity (per core's documented pattern)."""

    def updateGradient(self, Z, D, **kwargs):
        if self.V is None:
            self.V = np.zeros_like(Z)
        rs, re = kwargs["row_start"], kwargs["row_end"]
        self.V[rs:re] = self.beta * self.V[rs:re] + self.forces(Z, D, **kwargs)
        return self.V[rs:re]


def test_passthrough_forces_runs():
    G = nx.karate_club_graph()
    m = _RealShellModel(n_dim=2, verbosity=0, seed=7)
    Z = m.embed(G, epochs=8, lr=0.5)
    assert np.isfinite(Z).all()
    # cache held across all epochs/batches -> exactly one derivation
    assert m._shell.n_derivations == 1, m._shell.n_derivations
    print(f"[ok] real passthrough forces: 8 epochs finite, "
          f"1 derivation across the whole embed()")


def test_momentum_override_runs():
    G = nx.karate_club_graph()
    m = _MomentumShellModel(n_dim=2, beta=0.7, verbosity=0, seed=3)
    Z = m.embed(G, epochs=8, lr=0.3)          # batch_count=1 (whole-array velocity)
    assert np.isfinite(Z).all()
    print("[ok] momentum updateGradient override (beta=0.7) runs finite vs real forces()")


def test_module_level_forces_runs():
    # the composition-root one-liner form also works and caches
    rng = np.random.default_rng(2)
    G = _two_component_graph()
    n = G.number_of_nodes()
    Z = np.ascontiguousarray(rng.standard_normal((n, 2)))
    D = augment_graph(G, is_sparse=True)
    out = forces_fn(Z, D, 0, n, random_drop_rate=0.0, rng=rng)
    assert out.shape == (n, 2) and np.isfinite(out).all()
    print("[ok] module-level forces(Z, D, ...) free function runs")


if __name__ == "__main__":
    test_kernel_correctness_dense()
    test_kernel_correctness_sparse()
    test_cache_avoids_recomputation()
    test_degrees_derivation()
    test_passthrough_forces_runs()
    test_momentum_override_runs()
    test_module_level_forces_runs()
    print("\nAll shell_force tests passed.")
