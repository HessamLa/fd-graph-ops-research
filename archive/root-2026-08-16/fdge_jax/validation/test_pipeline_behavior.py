"""Whole-pipeline behavioral harness for ``fdge_jax.models.ReferenceFDModel``.

Kernel-level correctness is already covered rigorously by
``embedding/test_shell_force.py`` (bit-for-bit vs. a naive oracle) and
``graph_augmenting/test_*_smoke.py`` (vs. independent scipy/networkx
oracles). What's missing -- and what this file checks -- is that the
**assembled** ``ReferenceFDModel`` (all three JAX/NumPy stages wired
together via ``fdge_jax/models.py``) behaves correctly end-to-end. Direct
adaptation of ``fdge2/validation/test_pipeline_behavior.py``, run against
the JAX pipeline instead.

Numeric parity against fdge2 itself is a separate concern -- see
``test_regression_parity.py``.

Run:
    source /home/h/gnn/fd-graph-embedding/fdmap/.venv/bin/activate
    python -m fdge_jax.validation.test_pipeline_behavior     # from repo root (fdmap/)
    # or: pytest fdge_jax/validation/test_pipeline_behavior.py
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from fdge_jax.core import Callback_Base
from fdge_jax.models import ReferenceFDModel, SparseFDModel, EuclideanDistanceModel


# ---------------------------------------------------------------------------
# 1. runs N epochs without error, finite output
# ---------------------------------------------------------------------------
def test_runs_finite():
    G = nx.karate_club_graph()
    m = ReferenceFDModel(n_dim=3, verbosity=0, seed=0)
    Z = np.asarray(m.embed(G, epochs=15))
    assert Z.shape == (G.number_of_nodes(), 3)
    assert np.isfinite(Z).all()
    print(f"[ok] {m.latest_epoch} epochs ran, Z finite, shape {Z.shape}")


# ---------------------------------------------------------------------------
# 2. batching invariance
# ---------------------------------------------------------------------------
def test_batching_invariance():
    """batch_count in {1, 3, 7} must give numerically equal Z.

    Only holds with the default passthrough updateGradient (true here) AND
    random_drop_rate=0.0 (dropout draws a fresh key per batch, so a
    different batch_count means a different grouping of draws -- a real,
    expected source of batch-count-dependent randomness, not a bug).
    ``ShellForce`` computes the full-graph kernel once per batch call and
    slices afterward (docs/DESIGN.md), so mathematically this should be
    exact -- but on GPU, ``jax.ops.segment_sum``'s scatter-add is NOT
    bit-deterministic across separate invocations of the same compiled
    function (confirmed empirically: ~1e-15 diffs calling one jitted step
    twice with identical input -- an XLA-on-GPU characteristic, not a bug
    here; see docs/DESIGN.md "GPU determinism caveat"). So this uses a
    tight tolerance, not ``np.array_equal`` -- the fdge2/Numba version
    could assert bit-exact equality because Numba's row-streamed kernel
    has none of GPU scatter's nondeterminism.
    """
    G = nx.karate_club_graph()
    results = []
    for bc in (1, 3, 7):
        m = ReferenceFDModel(n_dim=2, random_drop_rate=0.0, verbosity=0, seed=42)
        Z = np.asarray(m.embed(G, epochs=5, batch_count=bc))
        results.append(Z)
    d13 = np.abs(results[0] - results[1]).max()
    d17 = np.abs(results[0] - results[2]).max()
    assert d13 < 1e-8, f"batch_count 1 vs 3 mismatch: max abs diff {d13:.3e}"
    assert d17 < 1e-8, f"batch_count 1 vs 7 mismatch: max abs diff {d17:.3e}"
    print(f"[ok] batch_count in {{1,3,7}} numerically equal "
          f"(max abs diff {max(d13, d17):.2e}, random_drop_rate=0.0)")


# ---------------------------------------------------------------------------
# 3. callback ordering
# ---------------------------------------------------------------------------
def test_callback_ordering():
    events = []

    class Rec(Callback_Base):
        def on_train_begin(self, model, **kw): events.append("tb")
        def on_epoch_begin(self, model, **kw): events.append("eb")
        def on_batch_begin(self, model, **kw): events.append("bb")
        def on_batch_end(self, model, **kw): events.append("be")
        def on_epoch_end(self, model, **kw): events.append("ee")
        def on_train_end(self, model, **kw): events.append("te")

    G = nx.path_graph(10)
    m = ReferenceFDModel(n_dim=2, random_drop_rate=0.0, verbosity=0, seed=0)
    m.attach_callback(Rec())
    m.embed(G, epochs=4, batch_count=2)

    assert events[0] == "tb" and events[-1] == "te"
    per_epoch = events[1:-1]
    expected_per_epoch = ["eb", "bb", "be", "bb", "be", "ee"]  # batch_count=2
    assert per_epoch == expected_per_epoch * 4, per_epoch
    print("[ok] callbacks fire in order:", events)


# ---------------------------------------------------------------------------
# 4. epsilon-based early convergence
# ---------------------------------------------------------------------------
def test_epsilon_convergence():
    """Small graph + damped lr/k3 so the n-body system demonstrably settles
    (see fdge2's equivalent test for why: strong repulsion at lr=1 can
    reach a dynamic force-balance oscillation rather than a fixed point --
    a property of the physics, not a bug)."""
    G = nx.path_graph(5)
    m = ReferenceFDModel(n_dim=2, random_drop_rate=0.0, verbosity=0, seed=7, k3=1.0)
    Z = np.asarray(m.embed(G, epochs=1000, lr=0.1, epsilon=1e-3))
    assert np.isfinite(Z).all()
    assert m.latest_epoch < 1000, (
        f"expected early stop before 1000 epochs, ran {m.latest_epoch}")
    print(f"[ok] converged early at epoch {m.latest_epoch}/1000 (epsilon=1e-3)")


# ---------------------------------------------------------------------------
# 5. end-to-end structure recovery on karate club
# ---------------------------------------------------------------------------
def test_end_to_end_structure_recovery():
    G = nx.karate_club_graph()
    m = ReferenceFDModel(n_dim=2, random_drop_rate=0.5, verbosity=0, seed=7)
    Z = np.asarray(m.embed(G, epochs=150, epsilon=1e-4))
    assert np.isfinite(Z).all()

    D = m.augment_graph(G)
    Dd = D.toarray() if hasattr(D, "toarray") else np.asarray(D)

    dist = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
    near = dist[Dd == 1].mean()
    far = dist[Dd >= 3].mean()
    assert near < far, f"structure not recovered: near={near:.3f} far={far:.3f}"
    print(f"[ok] karate club end-to-end: mean dist h=1: {near:.3f} < h>=3: "
          f"{far:.3f} (stopped at epoch {m.latest_epoch})")


# ---------------------------------------------------------------------------
# 6. momentum via a updateGradient override
# ---------------------------------------------------------------------------
class _MomentumFDModel(ReferenceFDModel):
    """updateGradient override implementing velocity, per core's documented
    pattern (core/force_directed.py's updateGradient docstring)."""

    def updateGradient(self, Z, D, **kwargs):
        import jax.numpy as jnp
        if self.V is None:
            self.V = jnp.zeros_like(Z)
        rs, re = kwargs["row_start"], kwargs["row_end"]
        self.V = self.V.at[rs:re].set(
            self.beta * self.V[rs:re] + self.forces(Z, D, **kwargs))
        return self.V[rs:re]


def test_momentum_override_finite():
    G = nx.karate_club_graph()
    # batch_count=1 (default): the momentum override reads/writes the
    # whole V array, so it must not be split across sub-epoch batches.
    m = _MomentumFDModel(n_dim=2, beta=0.7, random_drop_rate=0.0,
                          verbosity=0, seed=3)
    Z = np.asarray(m.embed(G, epochs=50, lr=0.2, epsilon=1e-5))
    assert np.isfinite(Z).all()
    print(f"[ok] momentum override (beta=0.7) finite, "
          f"stopped at epoch {m.latest_epoch}")


# ---------------------------------------------------------------------------
# 7. SparseFDModel (bounded-radius augmentation) runs end-to-end too
# ---------------------------------------------------------------------------
def test_sparse_model_runs_finite():
    G = nx.karate_club_graph()
    m = SparseFDModel(n_dim=2, verbosity=0, seed=5)
    Z = np.asarray(m.embed(G, epochs=20, radius=2))
    assert Z.shape == (G.number_of_nodes(), 2)
    assert np.isfinite(Z).all()
    print(f"[ok] SparseFDModel (radius=2): {m.latest_epoch} epochs, Z finite")


# ---------------------------------------------------------------------------
# 8. EuclideanDistanceModel (weighted-distance force law) runs end-to-end
# ---------------------------------------------------------------------------
def test_euclidean_model_runs_finite():
    rng = np.random.default_rng(11)
    data = rng.standard_normal((20, 3))
    m = EuclideanDistanceModel(n_dim=2, verbosity=0, seed=6)
    Z = np.asarray(m.fit(data, epochs=15, strategy="mst"))
    assert Z.shape == (20, 2)
    assert np.isfinite(Z).all()
    assert m.n_derivations == 1, (
        f"expected the per-D derivation to run once across the whole "
        f"embed() call, ran {m.n_derivations}x")
    print(f"[ok] EuclideanDistanceModel: {m.latest_epoch} epochs, Z finite, "
          f"1 derivation across the whole embed()")


if __name__ == "__main__":
    test_runs_finite()
    test_batching_invariance()
    test_callback_ordering()
    test_epsilon_convergence()
    test_end_to_end_structure_recovery()
    test_momentum_override_finite()
    test_sparse_model_runs_finite()
    test_euclidean_model_runs_finite()
    print("\nAll pipeline behavioral tests passed.")
