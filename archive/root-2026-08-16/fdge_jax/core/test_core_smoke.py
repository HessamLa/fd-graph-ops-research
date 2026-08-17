"""Throwaway confidence check for core.ForceDirected + core.csr (JAX port).

Not the real validation suite (that's under validation/). Port of
``fdge2/core/test_core_smoke.py``: defines a trivial concrete subclass
inline -- no graph_building/ / graph_augmenting/ / embedding/ needed --
and confirms the loop machinery ported correctly, including the JAX-only
seams (immutable Z/dZ, key splitting):

* embed() augments-then-loops for N epochs
* batching invariance: batch_count in {1, 3} gives identical Z
* callbacks fire in the expected order
* epsilon-based convergence stops early
* a updateGradient momentum override (beta > 0) runs without error
* csr helpers (graph_to_csr, row_of, n_rows) behave

Run:
    source /home/h/gnn/fd-graph-embedding/fdmap/.venv/bin/activate
    python -m fdge_jax.core.test_core_smoke      # from repo root (fdmap/)
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from fdge_jax.core import ForceDirected, Callback_Base, graph_to_csr, row_of, n_rows


# ---------------------------------------------------------------------------
# Trivial concrete model
# ---------------------------------------------------------------------------
class DummyModel(ForceDirected):
    """Deterministic stand-in for the real three stages.

    augment_graph: returns the adjacency ``G`` unchanged as a dense jnp ``D``.
    forces:        a linear contractive step  dZ = -0.5 * Z[rows]  (depends
                   only on Z and the row range -> batch-local, and its norm
                   shrinks geometrically so epsilon convergence triggers).
                   Deterministic (no dropout), so it doesn't care that
                   ``key`` now flows through kwargs -- batching invariance
                   still holds.
    """

    def augment_graph(self, G, is_sparse: bool = True, **kwargs):
        return jnp.asarray(G, dtype=jnp.float64)  # dense D, indexable D[i, j]

    def forces(self, Z, D, row_start, row_end, **kwargs):
        return -0.5 * Z[row_start:row_end]


class MomentumModel(DummyModel):
    """Same forces, but a updateGradient override implementing velocity."""

    def updateGradient(self, Z, D, **kwargs):
        if self.V is None:
            self.V = jnp.zeros_like(Z)
        rs, re = kwargs["row_start"], kwargs["row_end"]
        self.V = self.V.at[rs:re].set(
            self.beta * self.V[rs:re] + self.forces(Z, D, **kwargs))
        return self.V[rs:re]


class RecordingCallback(Callback_Base):
    def __init__(self):
        self.events = []
    def on_train_begin(self, model, **kw): self.events.append("tb")
    def on_epoch_end(self, model, **kw): self.events.append("ee")
    def on_train_end(self, model, **kw): self.events.append("te")


def _ring_graph(n):
    """Dense (n, n) adjacency of a simple ring -- a stand-in G."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = 1.0
        A[(i + 1) % n, i] = 1.0
    return A


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_csr_helpers():
    # ring of 4: 0-1-2-3-0
    class G:
        @staticmethod
        def nodes(): return [0, 1, 2, 3]
        @staticmethod
        def degree(): return [(0, 2), (1, 2), (2, 2), (3, 2)]
        @staticmethod
        def edges(): return [(0, 1), (1, 2), (2, 3), (3, 0)]
    indptr, indices, degrees = graph_to_csr(G)
    assert indptr.tolist() == [0, 2, 4, 6, 8]
    assert degrees.tolist() == [2, 2, 2, 2]
    assert set(indices[indptr[0]:indptr[1]].tolist()) == {1, 3}
    ro = row_of(indptr)
    assert ro.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert n_rows(np.zeros((7, 7))) == 7
    print("[ok] csr helpers: graph_to_csr / row_of / n_rows")


def test_runs_n_epochs():
    G = _ring_graph(10)
    m = DummyModel(n_dim=3, verbosity=0, seed=0)
    Z = m.embed(G, epochs=5)
    assert Z.shape == (10, 3)
    assert np.isfinite(np.asarray(Z)).all()
    assert m.latest_epoch == 5
    print("[ok] embed runs 5 epochs, Z shape", Z.shape)


def test_batching_invariance():
    G = _ring_graph(11)  # 11 not divisible by 3 -> uneven batches
    results = []
    for bc in (1, 3):
        m = DummyModel(n_dim=4, verbosity=0, seed=42)
        m.embed(G, epochs=6, batch_count=bc)
        results.append(m.get_embeddings())
    assert np.array_equal(results[0], results[1]), "batch_count changed the result"
    print("[ok] batch_count in {1, 3} gives identical Z")


def test_callbacks_order():
    G = _ring_graph(8)
    m = DummyModel(n_dim=2, verbosity=0, seed=0)
    rec = RecordingCallback()
    m.attach_callback(rec)
    m.embed(G, epochs=4)
    assert rec.events == ["tb", "ee", "ee", "ee", "ee", "te"], rec.events
    print("[ok] callbacks fire in order:", rec.events)


def test_convergence_stops_early():
    G = _ring_graph(12)
    m = DummyModel(n_dim=3, verbosity=0, seed=1)
    # dZ = -0.5*Z, Z halves each epoch -> Th halves each epoch -> converges
    m.embed(G, epochs=1000, epsilon=1e-3)
    assert m.latest_epoch < 1000, "epsilon convergence never triggered"
    print(f"[ok] epsilon convergence stopped early at epoch {m.latest_epoch}")


def test_momentum_override_runs():
    G = _ring_graph(9)
    m = MomentumModel(n_dim=3, beta=0.7, verbosity=0, seed=2)
    Z = m.embed(G, epochs=10)
    assert np.isfinite(np.asarray(Z)).all()
    print("[ok] updateGradient momentum override (beta=0.7) runs finite")


if __name__ == "__main__":
    test_csr_helpers()
    test_runs_n_epochs()
    test_batching_invariance()
    test_callbacks_order()
    test_convergence_stops_early()
    test_momentum_override_runs()
    print("\nAll core smoke tests passed.")
