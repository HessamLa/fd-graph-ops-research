"""T4.1 -- whole-pipeline behavioral harness for ``fdge2.models.ReferenceFDModel``.

The embedding-stage agent already did rigorous *kernel-level* correctness
testing (``fdge2/embedding/test_shell_force.py``: bit-for-bit vs. a naive
oracle, on a synthetic ``Z``). What's missing -- and what this file checks
-- is that the **assembled** ``ReferenceFDModel`` (all three stages wired
together via ``fdge2/models.py``) behaves correctly end-to-end. This is a
direct adaptation of ``fdge_numba/test_fd_numba.py``'s behavioral suite,
run against the new pipeline instead of the legacy ``FDModel``.

Numeric parity against the legacy model itself is a separate concern --
see ``test_regression_parity.py`` (T4.2).

Run:
    source /home/h/gnn/fd-graph-embedding/fdmap/.venv/bin/activate
    python -m fdge2.validation.test_pipeline_behavior     # from repo root (fdmap/)
    # or: pytest fdge2/validation/test_pipeline_behavior.py
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from fdge2.core import Callback_Base
from fdge2.models import ReferenceFDModel


# ---------------------------------------------------------------------------
# 1. runs N epochs without error, finite output
# ---------------------------------------------------------------------------
def test_runs_finite():
    G = nx.karate_club_graph()
    m = ReferenceFDModel(n_dim=3, verbosity=0, seed=0)
    Z = m.embed(G, epochs=15)
    assert Z.shape == (G.number_of_nodes(), 3)
    assert np.isfinite(Z).all()
    print(f"[ok] {m.latest_epoch} epochs ran, Z finite, shape {Z.shape}")


# ---------------------------------------------------------------------------
# 2. batching invariance
# ---------------------------------------------------------------------------
def test_batching_invariance():
    """batch_count in {1, 3, 7} must give identical Z.

    Only holds with the default passthrough updateGradient (true here --
    ReferenceFDModel does not override it) AND with random_drop_rate=0.0.
    Dropout is intentionally excluded: DropSteadyRate draws
    ``rng.random(batch_size)`` once per batch, so a different batch_count
    means a different grouping/order of draws from the *same* rng stream
    -- a real, expected source of batch-count-dependent randomness, not a
    bug in batching invariance itself (mirrors the legacy test's use of
    random_drop_rate=0.0 for exactly this reason).
    """
    G = nx.karate_club_graph()
    results = []
    for bc in (1, 3, 7):
        m = ReferenceFDModel(n_dim=2, random_drop_rate=0.0, verbosity=0, seed=42)
        Z = m.embed(G, epochs=5, batch_count=bc)
        results.append(Z.copy())
    assert np.array_equal(results[0], results[1]), "batch_count 1 vs 3 mismatch"
    assert np.array_equal(results[0], results[2]), "batch_count 1 vs 7 mismatch"
    print("[ok] batch_count in {1,3,7} gives bit-identical embeddings "
          "(random_drop_rate=0.0)")


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
    """Small graph + small lr/k3 so the n-body system actually settles.

    Note: on karate_club_graph at lr=1.0 with a large repulsion coefficient
    (this was originally observed with k3 defaulting -- via a now-fixed bug
    -- to n_nodes; ReferenceFDModel's actual default is k3=10.0, matching
    model_204_shell.FDModel), the system can reach a dynamic force-balance
    *oscillation* rather than a fixed point -- Th(dZ) plateaus and never
    drops near any reasonable epsilon. That's a property of this physical
    system (attraction/repulsion equilibrium is not exactly static at
    lr=1.0 with strong-enough repulsion), not a bug -- so this test uses a
    small graph + damped lr + small k3 where the trajectory demonstrably
    damps toward zero, to isolate "does epsilon-based early stopping work"
    from "does this particular system converge to a fixed point."
    """
    G = nx.path_graph(5)
    m = ReferenceFDModel(n_dim=2, random_drop_rate=0.0, verbosity=0, seed=7, k3=1.0)
    Z = m.embed(G, epochs=1000, lr=0.1, epsilon=1e-3)
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
    Z = m.embed(G, epochs=150, epsilon=1e-4)
    assert np.isfinite(Z).all()

    # D is a constant input to the embedding stage (RPD.md Sec 3) -- we
    # recompute it here only to *measure* hop distances for the check, the
    # same way the legacy test read m.hops off the fitted model.
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
        if self.V is None:
            self.V = np.zeros_like(Z)
        rs, re = kwargs["row_start"], kwargs["row_end"]
        self.V[rs:re] = self.beta * self.V[rs:re] + self.forces(Z, D, **kwargs)
        return self.V[rs:re]


def test_momentum_override_finite():
    G = nx.karate_club_graph()
    # batch_count=1 (default): the momentum override reads/writes the
    # whole V array, so it must not be split across sub-epoch batches
    # (core.force_directed's updateGradient docstring flags this).
    m = _MomentumFDModel(n_dim=2, beta=0.7, random_drop_rate=0.0,
                          verbosity=0, seed=3)
    Z = m.embed(G, epochs=50, lr=0.2, epsilon=1e-5)
    assert np.isfinite(Z).all()
    print(f"[ok] momentum override (beta=0.7) finite, "
          f"stopped at epoch {m.latest_epoch}")


if __name__ == "__main__":
    test_runs_finite()
    test_batching_invariance()
    test_callback_ordering()
    test_epsilon_convergence()
    test_end_to_end_structure_recovery()
    test_momentum_override_finite()
    print("\nAll pipeline behavioral tests passed.")
