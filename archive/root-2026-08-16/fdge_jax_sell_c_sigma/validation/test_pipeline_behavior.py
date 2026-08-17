"""Whole-pipeline behavioral harness for ``fdge_jax_sell_c_sigma.models``.

Kernel-level correctness is already covered rigorously by
``embedding/test_sell_c_sigma.py`` (vs. a naive oracle, AND vs. the
sibling ``fdge_jax`` flat-edge-list engine) and
``graph_augmenting/test_*_smoke.py`` (vs. independent scipy/networkx
oracles). What's missing -- and what this file checks -- is that the
**assembled** ``ReferenceFDModel`` / ``SparseFDModel`` / ``EuclideanDistanceModel``
(all three stages wired together via ``fdge_jax_sell_c_sigma/models.py``)
behave correctly end-to-end. Adapted from
``fdge_jax/validation/test_pipeline_behavior.py``.

Run:
    source /home/h/gnn/fd-graph-embedding/fdmap/.venv/bin/activate
    python -m fdge_jax_sell_c_sigma.validation.test_pipeline_behavior   # from repo root (fdmap/)
    # or: pytest fdge_jax_sell_c_sigma/validation/test_pipeline_behavior.py
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import jax

from fdge_jax_sell_c_sigma.core import Callback_Base
from fdge_jax_sell_c_sigma.models import (
    ReferenceFDModel, SparseFDModel, EuclideanDistanceModel)


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

    Only holds with the default passthrough ``updateGradient`` (true here)
    AND ``random_drop_rate=0.0`` (dropout draws a fresh key per batch, so a
    different batch_count means a different grouping of draws -- a real,
    expected source of batch-count-dependent randomness, not a bug).
    ``ShellForce`` computes the full-graph bucketed pass once per
    batch call and slices afterward (same reasoning as ``ShellForce``'s
    own batching note, ``docs/DESIGN.md``), so mathematically this should
    be exact -- but the bucketed kernel's own reduction
    (``dZ.at[rows].add(..., mode="drop")``) is a scatter-add too (per-ROW,
    not per-edge, but still a JAX scatter under the hood), so it inherits
    the same GPU-scatter non-determinism caveat ``fdge_jax``'s docs
    document for ``segment_sum``: a tight tolerance, not
    ``np.array_equal``.
    """
    G = nx.karate_club_graph()
    results = []
    for bc in (1, 3, 7):
        m = ReferenceFDModel(n_dim=2, random_drop_rate=0.0, verbosity=0, seed=42)
        Z = np.asarray(m.embed(G, epochs=5, batch_count=bc))
        results.append(Z)
    d13 = np.abs(results[0] - results[1]).max()
    d17 = np.abs(results[0] - results[2]).max()
    assert d13 < 1e-4, f"batch_count 1 vs 3 mismatch: max abs diff {d13:.3e}"
    assert d17 < 1e-4, f"batch_count 1 vs 7 mismatch: max abs diff {d17:.3e}"
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
    (see fdge2's/fdge_jax's equivalent tests for why: strong repulsion at
    lr=1 can reach a dynamic force-balance oscillation rather than a fixed
    point -- a property of the physics, not a bug)."""
    G = nx.path_graph(5)
    m = ReferenceFDModel(n_dim=2, random_drop_rate=0.0, verbosity=0, seed=7, k3=1.0)
    Z = np.asarray(m.embed(G, epochs=1000, lr=0.1, epsilon=1e-3))
    assert np.isfinite(Z).all()
    assert m.latest_epoch < 1000, (
        f"expected early stop before 1000 epochs, ran {m.latest_epoch}")
    print(f"[ok] converged early at epoch {m.latest_epoch}/1000 (epsilon=1e-3)")


# ---------------------------------------------------------------------------
# 5. momentum via a updateGradient override
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
# 6. SparseFDModel (bounded-radius augmentation) runs end-to-end too
# ---------------------------------------------------------------------------
def test_sparse_model_runs_finite():
    G = nx.karate_club_graph()
    m = SparseFDModel(n_dim=2, verbosity=0, seed=5)
    Z = np.asarray(m.embed(G, epochs=20, radius=2))
    assert Z.shape == (G.number_of_nodes(), 2)
    assert np.isfinite(Z).all()
    print(f"[ok] SparseFDModel (radius=2): {m.latest_epoch} epochs, Z finite")


# ---------------------------------------------------------------------------
# 7. end-to-end structure recovery on karate club (SparseFDModel, since a
#    large-graph-oriented package's own "does it actually embed structure"
#    check should exercise the sparse policy it's meant to be used with)
# ---------------------------------------------------------------------------
def test_end_to_end_structure_recovery():
    G = nx.karate_club_graph()
    m = SparseFDModel(n_dim=2, random_drop_rate=0.5, verbosity=0, seed=7)
    Z = np.asarray(m.embed(G, epochs=150, epsilon=1e-4, radius=3))
    assert np.isfinite(Z).all()

    D = m.augment_graph(G, radius=3)
    Dd = D.toarray() if hasattr(D, "toarray") else np.asarray(D)

    dist = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
    near = dist[Dd == 1].mean()
    far = dist[Dd >= 3].mean()
    assert near < far, f"structure not recovered: near={near:.3f} far={far:.3f}"
    print(f"[ok] karate club end-to-end (SparseFDModel, radius=3): "
          f"mean dist h=1: {near:.3f} < h>=3: {far:.3f} "
          f"(stopped at epoch {m.latest_epoch})")


# ---------------------------------------------------------------------------
# 8. EuclideanDistanceModel: runs end-to-end on point-cloud data
# ---------------------------------------------------------------------------
def test_euclidean_distance_model_runs_finite():
    rng = np.random.default_rng(9)
    data = rng.standard_normal((40, 3))
    m = EuclideanDistanceModel(n_dim=2, verbosity=0, seed=2)
    Z = np.asarray(m.fit(data, epochs=10, strategy="mst"))
    assert Z.shape == (40, 2)
    assert np.isfinite(Z).all()
    print(f"[ok] EuclideanDistanceModel: {m.latest_epoch} epochs ran, "
          f"Z finite, shape {Z.shape}")


# ---------------------------------------------------------------------------
# 9. EuclideanDistanceModel: bucketed engine vs. fdge_jax's flat engine,
#    same graph (make_graph/augment_graph are UNCHANGED ports, so both
#    models see the identical D/self._hops/h_min -- only forces() differs)
# ---------------------------------------------------------------------------
def test_euclidean_distance_model_parity_vs_fdge_jax():
    # Deliberate cross-package import -- legitimate only in a parity test
    # (see models.py's / embedding/shell_force.py's module docstrings).
    from fdge_jax.models import EuclideanDistanceModel as FlatEuclideanModel

    rng = np.random.default_rng(11)
    data = rng.standard_normal((60, 4))

    bucketed = EuclideanDistanceModel(n_dim=3, random_drop_rate=0.0, verbosity=0, seed=1)
    flat = FlatEuclideanModel(n_dim=3, random_drop_rate=0.0, verbosity=0, seed=1)

    G = bucketed.make_graph(data, strategy="mst")
    D_bucketed = bucketed.augment_graph(G)
    D_flat = flat.augment_graph(G)          # same G, same augmentation code

    n = G.number_of_nodes()
    Z = np.ascontiguousarray(rng.standard_normal((n, 3))).astype(np.float32)
    key = jax.random.PRNGKey(5)

    got_bucketed = np.asarray(bucketed.forces(Z, D_bucketed, 0, n, key=key))
    got_flat = np.asarray(flat.forces(Z, D_flat, 0, n, key=key))

    err = float(np.abs(got_bucketed - got_flat).max())
    assert err < 1e-3, f"EuclideanDistanceModel bucketed vs. flat mismatch: {err:.3e}"
    print(f"[ok] EuclideanDistanceModel: bucketed vs. fdge_jax's flat engine, "
          f"max abs diff {err:.2e} (n={n})")


# ---------------------------------------------------------------------------
# 10. both models share ONE cache implementation -- and it still caches
# ---------------------------------------------------------------------------
def test_shared_plan_cache_derives_once_per_D():
    """One derivation per distinct ``D``, for BOTH model families.

    ``ReferenceFDModel`` and ``EuclideanDistanceModel`` used to own two
    hand-copied identity caches; they now share ``PlanCache`` via
    ``ShellForce`` / ``WeightedShellForce``. This pins the guarantee for
    both at the model level -- a whole ``embed()`` (many epochs x many
    batches) must cost exactly one plan build + one XLA compile, and a
    genuinely new ``D`` must cost exactly one more.

    For the Euclidean model this is also the check that ``bind_topology``
    keeps ``PlanCache``'s invariant: a fresh ``augment_graph`` produces a
    fresh ``self.D`` (new key -> re-derive) with matching ``_hops``, so the
    cache can never serve a plan built from the wrong topology.
    """
    G = nx.karate_club_graph()
    m = ReferenceFDModel(n_dim=2, random_drop_rate=0.0, verbosity=0, seed=0)
    m.embed(G, epochs=6, batch_count=3)
    assert m._sell.n_derivations == 1, m._sell.n_derivations
    m.embed(G, epochs=2)          # augment_graph runs again -> a NEW D object
    assert m._sell.n_derivations == 2, m._sell.n_derivations

    rng = np.random.default_rng(4)
    data = rng.standard_normal((40, 3))
    e = EuclideanDistanceModel(n_dim=2, random_drop_rate=0.0, verbosity=0, seed=2)
    # (no batch_count here: fit() forwards **kwargs to make_graph too)
    e.fit(data, epochs=6, strategy="mst")
    assert e._sell.n_derivations == 1, e._sell.n_derivations

    n = e.D.shape[0]
    Z = np.ascontiguousarray(rng.standard_normal((n, 2))).astype(np.float32)
    for _ in range(5):            # extra direct calls on the SAME D: still 1
        e.forces(Z, e.D, 0, n, key=jax.random.PRNGKey(0))
    assert e._sell.n_derivations == 1, e._sell.n_derivations

    e.augment_graph(e.G)          # fresh D (+ fresh _hops/_h_min) -> exactly +1
    e.forces(Z, e.D, 0, n, key=jax.random.PRNGKey(0))
    assert e._sell.n_derivations == 2, e._sell.n_derivations
    print("[ok] shared PlanCache: ReferenceFDModel and EuclideanDistanceModel "
          "both derive once per distinct D (+1 for a new D)")


if __name__ == "__main__":
    test_runs_finite()
    test_batching_invariance()
    test_callback_ordering()
    test_epsilon_convergence()
    test_momentum_override_finite()
    test_sparse_model_runs_finite()
    test_end_to_end_structure_recovery()
    test_euclidean_distance_model_runs_finite()
    test_euclidean_distance_model_parity_vs_fdge_jax()
    test_shared_plan_cache_derives_once_per_D()
    print("\nAll pipeline behavioral tests passed.")
