"""T4.2 -- regression parity vs. the legacy ``model_204_shell.FDModel``.

RPD.md success metric #2: given the same graph/seed/params, the new
``fdge2.models.ReferenceFDModel`` pipeline must reproduce the legacy
``fdge_numba.forcedirected_numba.model_204_shell.FDModel``'s numeric
output within float64 tolerance, epoch by epoch (IMPLEMENTATION.md
methodology: compare per-epoch, not just the final result, so a real bug
is caught at the point it first appears rather than averaged away).

Known gotcha (see task brief / RPD.md): the legacy ``FDModel.__init__``
initializes ``Z`` via ``generate_random_points`` (uniform points in a
d-ball: random unit directions + ``rng.random(n)**(1/d)`` radii) -- a
different sequence of RNG draws than ``fdge2.core.ForceDirected``'s lazy
default init (``rng.standard_normal((n, n_dim))``). Same seed does NOT
produce the same initial ``Z`` between the two models. This is sidestepped
by generating one ``Z0`` explicitly (via the legacy ``generate_random_points``
itself) and passing it via ``Z=Z0`` to *both* models' first ``embed()``
call -- isolating the comparison to "does the physics + loop match," which
is the actual thing this test checks.

Two passes:

1. ``random_drop_rate=0.0`` on both models -- removes dropout's RNG-draw
   ordering as a variable entirely, so any mismatch can't hide behind
   "maybe the RNG streams just diverged." This is the strict, must-pass
   comparison.
2. ``random_drop_rate>0`` with the two models' ``self.rng`` streams
   explicitly re-seeded to the same seed right after construction (see
   ``_maybe_sync_rng`` below -- necessary because the legacy model's
   ``__init__`` burns RNG draws on the throwaway initial ``Z`` that gets
   immediately overwritten by ``Z=Z0``, which would otherwise desync the
   two models' RNG streams before the first dropout draw). A stronger
   check since dropout is part of the real reference behavior, but not
   load-bearing for "is the port correct" the way pass 1 is.

Run:
    source /home/h/gnn/fd-graph-embedding/fdmap/.venv/bin/activate
    python -m fdge2.validation.test_regression_parity   # from repo root (fdmap/)
    # or: pytest fdge2/validation/test_regression_parity.py
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from fdge2.models import ReferenceFDModel
from fdge_numba.forcedirected_numba.model_204_shell import (
    FDModel as LegacyFDModel, generate_random_points,
)


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
def _run_epoch_by_epoch(G, n_dim, epochs, k1, k2, k3, k4, lr,
                         random_drop_rate, seed_z, sync_rng_seed=None):
    """Step both models one epoch at a time on a shared explicit Z0.

    Returns a list of (epoch, legacy_Z, new_Z) copies, one entry per epoch,
    so the caller can compare epoch-by-epoch and stop at first divergence
    (IMPLEMENTATION.md methodology) rather than only checking the end.
    """
    n = G.number_of_nodes()
    z_rng = np.random.default_rng(seed_z)
    Z0 = np.ascontiguousarray(generate_random_points(n, n_dim, z_rng),
                               dtype=np.float64)

    # Seeds are deliberately different here (999 / 111): the legacy
    # constructor's throwaway Z-init draws from its rng are discarded
    # below (Z0 overrides them via the first embed() call), and any
    # dropout-stream syncing is done explicitly via sync_rng_seed, not by
    # matching constructor seeds -- see module docstring.
    legacy = LegacyFDModel(G, n_dim=n_dim, k1=k1, k2=k2, k3=k3, k4=k4, lr=lr,
                            random_drop_rate=random_drop_rate,
                            verbosity=0, seed=999)
    new = ReferenceFDModel(n_dim=n_dim, k1=k1, k2=k2, k3=k3, k4=k4, lr=lr,
                            random_drop_rate=random_drop_rate,
                            verbosity=0, seed=111)

    if sync_rng_seed is not None:
        # Reset both streams fresh from the SAME seed right before the
        # first forces()/dropout call, so dropout draws line up call for
        # call. Must happen after construction (discards the legacy
        # ctor's throwaway Z-init draws) and before the first embed().
        legacy.rng = np.random.default_rng(sync_rng_seed)
        new.rng = np.random.default_rng(sync_rng_seed)

    results = []
    for e in range(1, epochs + 1):
        if e == 1:
            legacy.embed(epochs=1, lr=lr, Z=Z0)
            new.embed(G, epochs=1, lr=lr, Z=Z0)
        else:
            legacy.embed(epochs=1, lr=lr)
            new.embed(G, epochs=1, lr=lr)
        results.append((e, legacy.get_embeddings().copy(), new.get_embeddings().copy()))
    return results


def _assert_parity(results, tol, label):
    """Compare epoch-by-epoch, stopping at (and reporting) first divergence.

    Per IMPLEMENTATION.md: don't just check the final epoch -- a real bug
    should be caught at the point it first appears, not averaged away over
    many epochs of otherwise-matching physics.
    """
    max_seen = 0.0
    for epoch, z_legacy, z_new in results:
        assert np.isfinite(z_legacy).all() and np.isfinite(z_new).all(), (
            f"{label}: non-finite Z at epoch {epoch}")
        diff = np.abs(z_legacy - z_new).max()
        max_seen = max(max_seen, diff)
        assert diff < tol, (
            f"{label}: diverged at epoch {epoch}: max abs diff {diff:.3e} "
            f">= tol {tol:.1e} (max abs diff over prior epochs was within "
            f"tolerance, so this is the first epoch where legacy and new "
            f"disagree by more than float64 noise)")
    n_epochs = len(results)
    print(f"[ok] {label}: {n_epochs} epochs, max abs diff over the run "
          f"{max_seen:.3e} (tol {tol:.1e})")
    return max_seen


# ---------------------------------------------------------------------------
# Pass 1 (strict): random_drop_rate=0.0, no RNG-ordering variable at all
# ---------------------------------------------------------------------------
def test_parity_no_dropout_karate():
    G = nx.karate_club_graph()
    results = _run_epoch_by_epoch(
        G, n_dim=3, epochs=40, k1=0.999, k2=1.0, k3=None, k4=0.01, lr=1.0,
        random_drop_rate=0.0, seed_z=0)
    _assert_parity(results, tol=1e-9, label="karate_club_graph, no dropout")


def test_parity_no_dropout_disconnected():
    # disjoint_union exercises the h = n sentinel path (disconnected pairs)
    G = nx.disjoint_union(nx.karate_club_graph(), nx.path_graph(6))
    results = _run_epoch_by_epoch(
        G, n_dim=3, epochs=40, k1=0.999, k2=1.0, k3=None, k4=0.01, lr=1.0,
        random_drop_rate=0.0, seed_z=1)
    _assert_parity(results, tol=1e-9,
                   label="disjoint_union(karate, path6), no dropout")


# ---------------------------------------------------------------------------
# Pass 2 (stronger, optional per task brief): matched dropout RNG streams
# ---------------------------------------------------------------------------
def test_parity_with_dropout_matched_rng():
    G = nx.karate_club_graph()
    results = _run_epoch_by_epoch(
        G, n_dim=3, epochs=30, k1=0.999, k2=1.0, k3=None, k4=0.01, lr=1.0,
        random_drop_rate=0.4, seed_z=2, sync_rng_seed=4242)
    _assert_parity(results, tol=1e-9,
                   label="karate_club_graph, random_drop_rate=0.4, synced rng")


if __name__ == "__main__":
    test_parity_no_dropout_karate()
    test_parity_no_dropout_disconnected()
    test_parity_with_dropout_matched_rng()
    print("\nAll regression parity tests passed.")
