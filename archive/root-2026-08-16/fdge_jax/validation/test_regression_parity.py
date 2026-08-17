"""Regression parity vs. ``fdge2.models.ReferenceFDModel`` (the Numba version).

fdge_jax should reproduce the same *physics*, not the same *bits*.
Concretely, two structural differences rule out bit-exact comparison
(docs/DESIGN.md "Numeric parity expectation"):

* **Summation order.** fdge2's Numba kernel accumulates each row's force
  sum sequentially inside a ``prange``-parallel loop; fdge_jax's kernel
  scatter-adds via ``jax.ops.segment_sum`` on GPU, which uses a different
  (and, per ``test_pipeline_behavior.py``'s finding, not even run-to-run
  deterministic) reduction order. Floating point addition isn't
  associative, so this alone produces ~1e-13-1e-15-scale per-epoch
  differences even if every other part of the computation were identical.
* **RNG stream shape.** ``np.random.Generator`` (fdge2) and
  ``jax.random.PRNGKey`` (fdge_jax) are different algorithms entirely --
  there is no seed pairing that makes their draws line up. So this test
  keeps ``random_drop_rate=0.0`` on both sides throughout, removing
  dropout's RNG-draw ordering as a variable (same strategy fdge2's own
  parity test uses for its *strict* pass).

What this test actually checks: given the **same graph**, the **same
explicit initial Z0** (generated once, passed to both models' first
``embed()`` call, sidestepping both models' own lazy-init RNG paths -- see
fdge2's parity test for why this isolation matters), and the **same
hyperparameters**, do the two implementations' trajectories agree to
within float64-noise-scale tolerance for long enough to demonstrate "same
formula, same loop," before per-epoch summation-order noise compounds
(this is a real n-body-like dynamical system -- tiny per-epoch
differences can amplify over many epochs, same caveat
``sparse_elementwise_test.py``/IMPLEMENTATION.md document for a
structurally similar recurrence). A small graph + damped ``lr`` keeps the
dynamics well-behaved long enough to get a meaningful number of agreeing
epochs.

Run:
    source /home/h/gnn/fd-graph-embedding/fdmap/.venv/bin/activate
    python -m fdge_jax.validation.test_regression_parity   # from repo root (fdmap/)
    # or: pytest fdge_jax/validation/test_regression_parity.py
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from fdge_jax.models import ReferenceFDModel as JaxFDModel
from fdge_jax.models import EuclideanDistanceModel as JaxEucModel
from fdge_jax import graph_building as jax_graph_building
from fdge2.models import ReferenceFDModel as NumbaFDModel
from fdge2.models import EuclideanDistanceModel as NumbaEucModel


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
def _run_epoch_by_epoch(G, n_dim, epochs, k1, k2, k3, k4, lr, seed_z):
    """Step both models one epoch at a time from a shared explicit Z0.

    Returns a list of (epoch, numba_Z, jax_Z) copies (both as numpy
    arrays), one entry per epoch, so the caller can compare epoch-by-epoch
    and see exactly where (if anywhere) they part ways, rather than only
    checking the final result.
    """
    n = G.number_of_nodes()
    z_rng = np.random.default_rng(seed_z)
    Z0 = np.ascontiguousarray(z_rng.standard_normal((n, n_dim)), dtype=np.float64)

    numba = NumbaFDModel(k1=k1, k2=k2, k3=k3, k4=k4, lr=lr,
                          random_drop_rate=0.0, verbosity=0, seed=999)
    jaxm = JaxFDModel(k1=k1, k2=k2, k3=k3, k4=k4, lr=lr,
                       random_drop_rate=0.0, verbosity=0, seed=111)

    results = []
    for e in range(1, epochs + 1):
        if e == 1:
            # fdge2's `np.ascontiguousarray(Z, dtype=np.float64)` returns
            # the SAME array object when Z0 is already contiguous
            # float64 -- self.Z then aliases Z0 and updateZ's `+=`
            # mutates it in place. Pass each model an independent copy so
            # the numba model's in-place step can't corrupt Z0 before the
            # jax model reads it (jnp.asarray never aliases numpy memory,
            # so only the numba side actually needs this, but copy both
            # for symmetry/safety).
            numba.embed(G, epochs=1, lr=lr, Z=Z0.copy())
            jaxm.embed(G, epochs=1, lr=lr, Z=Z0.copy())
        else:
            numba.embed(G, epochs=1, lr=lr)
            jaxm.embed(G, epochs=1, lr=lr)
        results.append((e, numba.get_embeddings().copy(),
                         np.asarray(jaxm.get_embeddings()).copy()))
    return results


def _assert_parity(results, tol, label, min_epochs):
    """Compare epoch-by-epoch, reporting the first divergence if any.

    Requires at least ``min_epochs`` of agreement within ``tol`` -- fewer
    would mean the port disagrees almost immediately, a real bug; more is
    a bonus (some hyperparameter/graph combos happen to stay in lockstep
    far longer than the minimum, which is fine, not required).
    """
    max_seen = 0.0
    agreeing_epochs = 0
    for epoch, z_numba, z_jax in results:
        assert np.isfinite(z_numba).all() and np.isfinite(z_jax).all(), (
            f"{label}: non-finite Z at epoch {epoch}")
        diff = np.abs(z_numba - z_jax).max()
        max_seen = max(max_seen, diff)
        if diff < tol:
            agreeing_epochs = epoch
        else:
            break
    n_epochs = len(results)
    assert agreeing_epochs >= min_epochs, (
        f"{label}: only {agreeing_epochs}/{n_epochs} epochs agreed within "
        f"tol {tol:.1e} (needed >= {min_epochs}); max abs diff over the "
        f"agreeing prefix was {max_seen:.3e}")
    print(f"[ok] {label}: {agreeing_epochs}/{n_epochs} epochs agreed within "
          f"tol {tol:.1e} (max abs diff over that prefix {max_seen:.3e})")
    return agreeing_epochs, max_seen


# ---------------------------------------------------------------------------
# Pass 1: karate club, no dropout, damped lr (well-behaved dynamics)
# ---------------------------------------------------------------------------
def test_parity_karate_damped():
    G = nx.karate_club_graph()
    results = _run_epoch_by_epoch(
        G, n_dim=3, epochs=30, k1=0.999, k2=1.0, k3=10.0, k4=0.01, lr=0.1,
        seed_z=0)
    _assert_parity(results, tol=1e-6, label="karate_club_graph, lr=0.1",
                    min_epochs=10)


# ---------------------------------------------------------------------------
# Pass 2: disconnected graph, exercises the h = n sentinel path
# ---------------------------------------------------------------------------
def test_parity_disconnected_damped():
    G = nx.disjoint_union(nx.karate_club_graph(), nx.path_graph(6))
    results = _run_epoch_by_epoch(
        G, n_dim=3, epochs=30, k1=0.999, k2=1.0, k3=10.0, k4=0.01, lr=0.1,
        seed_z=1)
    _assert_parity(results, tol=1e-6,
                    label="disjoint_union(karate, path6), lr=0.1",
                    min_epochs=10)


# ---------------------------------------------------------------------------
# Pass 3: full lr=1.0 (fdge2's own default) on a small, simple graph
# ---------------------------------------------------------------------------
def test_parity_path_graph_full_lr():
    G = nx.path_graph(8)
    results = _run_epoch_by_epoch(
        G, n_dim=2, epochs=20, k1=0.999, k2=1.0, k3=10.0, k4=0.01, lr=1.0,
        seed_z=2)
    _assert_parity(results, tol=1e-4, label="path_graph(8), lr=1.0",
                    min_epochs=5)


# ---------------------------------------------------------------------------
# Pass 4: EuclideanDistanceModel (weighted-distance force law, JAX-kernel
# port of fdge2's row-streamed-NumPy variant -- see models.py's
# EuclideanDistanceModel docstring for what changed and why)
# ---------------------------------------------------------------------------
def test_parity_euclidean_distance_model():
    """Same epoch-by-epoch methodology, but for EuclideanDistanceModel.

    Builds one graph (MST topology + Euclidean edge weights) shared by
    both models -- can't reuse ``_run_epoch_by_epoch`` as-is since that
    helper takes a ready-made topology-only ``G``, while this model's
    weights depend on ``data`` and its own ``make_graph``. Both sides use
    fdge_jax's ``graph_building.make_graph`` (topology only depends on MST
    geometry, not on which package's ``make_graph`` builds it -- they're
    identical algorithms, just copied into two packages) so there's no
    strategy-RNG divergence to worry about, only the two models' own
    physics/loop.
    """
    n, d_data = 25, 3
    data_rng = np.random.default_rng(3)
    data = data_rng.standard_normal((n, d_data))

    G = jax_graph_building.make_graph(data, strategy="mst")
    for u, v in G.edges():
        G.edges[u, v]["weight"] = float(np.linalg.norm(data[u] - data[v]))

    n_dim = 2
    z_rng = np.random.default_rng(9)
    Z0 = np.ascontiguousarray(z_rng.standard_normal((n, n_dim)), dtype=np.float64)

    numba = NumbaEucModel(k1=0.999, k2=1.0, k3=10.0, k4=0.01, lr=0.1,
                           random_drop_rate=0.0, verbosity=0, seed=999)
    jaxm = JaxEucModel(k1=0.999, k2=1.0, k3=10.0, k4=0.01, lr=0.1,
                        random_drop_rate=0.0, verbosity=0, seed=111)

    results = []
    for e in range(1, 21):
        if e == 1:
            numba.embed(G, epochs=1, lr=0.1, Z=Z0.copy())
            jaxm.embed(G, epochs=1, lr=0.1, Z=Z0.copy())
        else:
            numba.embed(G, epochs=1, lr=0.1)
            jaxm.embed(G, epochs=1, lr=0.1)
        results.append((e, numba.get_embeddings().copy(),
                         np.asarray(jaxm.get_embeddings()).copy()))

    _assert_parity(results, tol=1e-6, label="EuclideanDistanceModel, mst, lr=0.1",
                    min_epochs=15)


if __name__ == "__main__":
    test_parity_karate_damped()
    test_parity_disconnected_damped()
    test_parity_path_graph_full_lr()
    test_parity_euclidean_distance_model()
    print("\nAll regression parity tests passed.")
