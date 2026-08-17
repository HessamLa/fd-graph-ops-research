"""Correctness tests for ``model_fdmap_auxdim.model`` (EuclideanAuxDimModel).

These are *correctness* tests -- "the wrapper does what it says", not "the
idea works". Whether annealed auxiliary dimensions actually improve layout
quality is an open empirical question and needs the comparison described in
``idea.md`` section 2 (quality metric, with vs. without aux, across seeds);
nothing here answers it, and a passing run here is not evidence for the
hypothesis.

The load-bearing test is ``test_aux0_parity_with_base_model``: at ``aux=0``
the model must be *bit-identical* to a plain ``EuclideanDistanceModel``,
which is what proves the new machinery is genuinely inert when switched off
rather than perturbing the physics in some small way.

Dual-mode, same convention as ``fdge_jax_sell_c_sigma``'s own test files:

    /home/h/gnn/fd-graph-embedding/fdmap/.venv/bin/python -m model_fdmap_auxdim.test_model
    # or: .venv/bin/pytest model_fdmap_auxdim/test_model.py
"""
from __future__ import annotations

import math

import numpy as np

from fdge_jax_sell_c_sigma.models import EuclideanDistanceModel
from model_fdmap_auxdim.model import (
    EuclideanAuxDimModel,
    DECAY_SCHEDULES,
    sigmoid_decay,
    exponential_decay,
)


# Small point cloud + MST topology: the cheapest fixture that exercises the
# real EuclideanDistanceModel path (Euclidean edge weights -> Dijkstra ->
# bucketed kernel) rather than a hand-made graph the model never sees in use.
N_POINTS, DATA_DIM = 40, 3


def _data(seed: int = 9) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((N_POINTS, DATA_DIM))


# ---------------------------------------------------------------------------
# 1. aux=0 is an exact no-op wrapper (THE test)
# ---------------------------------------------------------------------------
def test_aux0_parity_with_base_model():
    """aux=0 must reproduce EuclideanDistanceModel(n_dim=d) exactly.

    Same seed, same k1..k4, ``random_drop_rate=0.0`` so the only randomness
    left is the shared lazy-Z init (identical PRNGKey, identical width).
    Any divergence means the wrapper changed the physics.
    """
    data = _data()
    kw = dict(verbosity=0, seed=7, random_drop_rate=0.0,
              k1=0.999, k2=1.0, k3=10.0, k4=0.01)

    aux_model = EuclideanAuxDimModel(d=2, aux=0, **kw)
    base_model = EuclideanDistanceModel(n_dim=2, **kw)

    aux_model.fit(data, epochs=8, strategy="mst")
    base_model.fit(data, epochs=8, strategy="mst")

    Za, Zb = aux_model.get_embeddings(), base_model.get_embeddings()
    assert Za.shape == Zb.shape == (N_POINTS, 2)
    max_abs = float(np.abs(Za - Zb).max())
    assert np.allclose(Za, Zb, rtol=0, atol=1e-6), (
        f"aux=0 diverged from the base model (max |dZ| = {max_abs:.3e})")
    print(f"[ok] aux=0 parity vs EuclideanDistanceModel: max abs diff "
          f"{max_abs:.3e} over {N_POINTS} nodes x 2 dims")


# ---------------------------------------------------------------------------
# 2. aux>0 runs, stays finite, and returns d columns (not d+aux)
# ---------------------------------------------------------------------------
def test_aux_positive_runs_finite_and_slices():
    data = _data()
    m = EuclideanAuxDimModel(d=2, aux=2, verbosity=0, seed=3)
    m.fit(data, epochs=10, strategy="mst")

    Z = m.get_embeddings()
    assert Z.shape == (N_POINTS, 2), f"expected (n, d)=(40, 2), got {Z.shape}"
    assert np.isfinite(Z).all()
    # the underlying state really is d+aux wide -- i.e. the slice above is
    # doing work, the model is not secretly running in 2 dims.
    assert m.Z.shape == (N_POINTS, 4)
    assert m.n_dim == 4
    print(f"[ok] aux=2: internal Z {tuple(m.Z.shape)} -> get_embeddings() "
          f"{Z.shape}, finite, {m.latest_epoch} epochs")


# ---------------------------------------------------------------------------
# 3. the aux columns actually collapse
# ---------------------------------------------------------------------------
def test_aux_columns_shrink():
    """||Z[:, d:]|| at the end must be far below a fresh random init's scale.

    Reference scale: the lazy init is a standard normal, so a fresh
    ``(n, aux)`` block has expected Frobenius norm ~sqrt(n*aux). Running the
    full budget means the last epoch's linear coefficient is
    ``1 - E/E == 0``, so the expected end state is a hard zero -- but the
    threshold is written as "<1% of init scale" so the test still means
    something under a schedule that only *approaches* zero.
    """
    data = _data()
    d, aux, epochs = 2, 3, 30
    init_scale = math.sqrt(N_POINTS * aux)         # E||standard normal block||

    # "sigmoid" is checked alongside the default because linear's final
    # coefficient is *exactly* 0, which would satisfy any threshold for free;
    # sigmoid only ever approaches zero, so it is the case that actually
    # exercises the shrink accumulating epoch over epoch.
    for schedule in ("linear", "sigmoid"):
        m = EuclideanAuxDimModel(d=d, aux=aux, decay_schedule=schedule,
                                 verbosity=0, seed=5)
        m.fit(data, epochs=epochs, strategy="mst")

        Z_full = np.asarray(m.Z)                   # unsliced (n, d+aux) state
        aux_norm = float(np.linalg.norm(Z_full[:, d:]))
        main_norm = float(np.linalg.norm(Z_full[:, :d]))

        assert m._epoch_counter == epochs, (
            f"expected one shrink per epoch ({epochs}), got {m._epoch_counter}")
        assert aux_norm < 0.01 * init_scale, (
            f"{schedule}: aux block did not collapse: ||Z[:, {d}:]|| = "
            f"{aux_norm:.4g} vs init scale ~{init_scale:.4g}")
        assert main_norm > 0.0 and np.isfinite(Z_full).all()
        print(f"[ok] aux collapse ({schedule}) after {epochs} epochs: "
              f"||Z[:, {d}:]|| = {aux_norm:.4g} (init scale ~{init_scale:.3g}, "
              f"{aux_norm / init_scale:.2e}x); ||Z[:, :{d}]|| = {main_norm:.4g} intact")


# ---------------------------------------------------------------------------
# 4. every named schedule is a [1 -> 0] ramp, and survives a tiny budget
# ---------------------------------------------------------------------------
def test_all_schedules_ramp_from_one_to_zero():
    """coeff(0, E) ~= 1 and coeff(E, E) ~= 0 for all five, loosely.

    Loose on purpose: sigmoid ends at ~0.0067 and starts at ~0.9933,
    exponential ends at ~4.5e-5. These are heuristics, not identities.
    """
    E = 100
    assert set(DECAY_SCHEDULES) == {
        "linear", "cosine", "linear_cosine", "sigmoid", "exponential"}
    for name, fn in sorted(DECAY_SCHEDULES.items()):
        start, end = fn(0, E), fn(E, E)
        assert abs(start - 1.0) < 0.02, f"{name}: coeff(0, {E}) = {start}"
        assert abs(end) < 0.02, f"{name}: coeff({E}, {E}) = {end}"
        mid = fn(E // 2, E)
        assert 0.0 <= mid <= 1.0, f"{name}: coeff({E // 2}, {E}) = {mid}"
        print(f"[ok] schedule {name:<14} coeff(0)={start:.6f} "
              f"coeff(E/2)={mid:.6f} coeff(E)={end:.6f}")


def test_schedules_survive_small_max_epochs():
    """max_epochs/10 <= 1 (a smoke-test-sized run) must not blow up.

    ``sigmoid``/``exponential`` divide by ``max_epochs/10``; that is floored
    at 1e-9 in model.py, and the logistic is evaluated in the overflow-free
    direction, so even ``max_epochs=0`` stays finite instead of raising
    ZeroDivisionError / OverflowError.
    """
    for E in (0, 1, 5, 10):
        for epoch in range(0, E + 1):
            for name, fn in DECAY_SCHEDULES.items():
                c = fn(epoch, E)
                assert math.isfinite(c), f"{name}({epoch}, {E}) = {c}"
        # the two with the max_epochs/10 denominator, checked in range too
        s = [sigmoid_decay(e, E) for e in range(0, E + 1)]
        x = [exponential_decay(e, E) for e in range(0, E + 1)]
        assert all(0.0 <= v <= 1.0 for v in s + x), (E, s, x)
        print(f"[ok] max_epochs={E:<2} finite for all five; "
              f"sigmoid {s[0]:.4f}->{s[-1]:.4f}, exp {x[0]:.4f}->{x[-1]:.4f}")

    # far past the budget (a resumed embed()) must not raise either
    assert math.isfinite(sigmoid_decay(10_000, 5))
    assert math.isfinite(exponential_decay(10_000, 5))
    print("[ok] epoch >> max_epochs (resumed run) stays finite: "
          f"sigmoid={sigmoid_decay(10_000, 5):.3e}, "
          f"exponential={exponential_decay(10_000, 5):.3e}")


# ---------------------------------------------------------------------------
# 5. decay_fn= overrides decay_schedule=, and is really the one used
# ---------------------------------------------------------------------------
def test_custom_decay_fn_takes_precedence():
    def half(epoch, max_epochs):
        return 0.5

    m = EuclideanAuxDimModel(d=2, aux=1, decay_schedule="exponential",
                             decay_fn=half, verbosity=0, seed=1)
    assert m.decay_fn is half, "decay_fn= did not win over decay_schedule="

    # ...and observably so: a custom "never shrink" schedule must leave a far
    # bigger aux block than the default linear ramp on an identical run.
    data = _data()
    common = dict(d=2, aux=3, verbosity=0, seed=4)
    kept = EuclideanAuxDimModel(decay_fn=lambda e, E: 1.0, **common)
    shrunk = EuclideanAuxDimModel(decay_schedule="linear", **common)
    kept.fit(data, epochs=20, strategy="mst")
    shrunk.fit(data, epochs=20, strategy="mst")

    kept_norm = float(np.linalg.norm(np.asarray(kept.Z)[:, 2:]))
    shrunk_norm = float(np.linalg.norm(np.asarray(shrunk.Z)[:, 2:]))
    assert kept_norm > 100 * max(shrunk_norm, 1e-12), (kept_norm, shrunk_norm)
    print(f"[ok] custom decay_fn used: coeff=1 keeps ||aux|| = {kept_norm:.4g}, "
          f"default linear leaves {shrunk_norm:.4g}")


# ---------------------------------------------------------------------------
# 6. a bad schedule name fails loudly, naming the valid options
# ---------------------------------------------------------------------------
def test_bad_schedule_name_raises_informative_error():
    try:
        EuclideanAuxDimModel(d=2, aux=1, decay_schedule="nonsense")
    except ValueError as exc:
        msg = str(exc)
        assert "nonsense" in msg
        for name in DECAY_SCHEDULES:
            assert name in msg, f"error message omits valid name {name!r}: {msg}"
        print(f"[ok] bad decay_schedule raises ValueError: {msg}")
    else:
        raise AssertionError("expected ValueError for decay_schedule='nonsense'")


if __name__ == "__main__":
    test_aux0_parity_with_base_model()
    test_aux_positive_runs_finite_and_slices()
    test_aux_columns_shrink()
    test_all_schedules_ramp_from_one_to_zero()
    test_schedules_survive_small_max_epochs()
    test_custom_decay_fn_takes_precedence()
    test_bad_schedule_name_raises_informative_error()
    print("\nAll model_fdmap_auxdim tests passed.")
