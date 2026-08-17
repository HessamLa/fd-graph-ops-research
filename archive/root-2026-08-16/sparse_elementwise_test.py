"""
sparse_elementwise_test.py

Sample/test script for the "elementwise combine, then row-reduce" pattern
discussed for fdge2.

Zdiff, X, Y share one sparse topology (same shape, same nonzero positions),
represented as CSR (indptr, indices) with one (nnz, d) data array per
quantity. Zdiff is never materialized as its own array -- it's recomputed
per edge from the current Z inside the kernel, matching the "don't build
the full (n,n,d) pairwise tensor" guardrail from model_204_shell.py.

  F[i,j]   = clip(Zdiff[i,j] * X[i,j] * exp(-Y[i,j]) - exp(-Zdiff[i,j]), -100, 100)   (elementwise, per dim)
  Z[i]     = mean_j F[i,j]     (mean over node i's edges)
  LOOP 100x: recompute F from the current Z, then update Z

Rows with no edges (degree 0) keep Z[i] unchanged rather than dividing by
zero -- node 0 is deliberately left isolated below to exercise that path.

Includes:
  1. A Numba (parallel, njit) implementation over the CSR topology.
  2. A pure-Python/NumPy brute-force reference restricted to the same
     sparsity pattern, used only for correctness-checking on a small n.
  3. A timing smoke test on a larger n.
"""
from __future__ import annotations

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from numba import njit, prange

jax.config.update("jax_enable_x64", True)  # match float64 used by numba/numpy paths


# ---------------------------------------------------------------------------
# Shared sparse topology (CSR): Zdiff, X, Y all live on this same pattern
# ---------------------------------------------------------------------------
def build_random_topology(n: int, avg_degree: int, rng: np.random.Generator):
    """Random directed sparse (n, n) topology as CSR indptr/indices.

    No self-loops, no duplicate columns per row. Node 0 is left with no
    outgoing edges on purpose, to exercise the zero-degree row case.
    """
    indptr = np.zeros(n + 1, dtype=np.int64)
    row_indices = []
    for i in range(n):
        if i == 0:
            indptr[i + 1] = indptr[i]
            continue
        deg = min(int(rng.integers(1, avg_degree * 2)), n - 1)
        choices = rng.integers(0, n - 1, size=deg)
        choices[choices >= i] += 1  # shift to skip self-loop at i
        choices = np.unique(choices)
        row_indices.append(choices)
        indptr[i + 1] = indptr[i] + choices.size
    indices = (np.concatenate(row_indices) if row_indices
               else np.empty(0, dtype=np.int64)).astype(np.int64)
    return indptr, indices


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def compute_F(indptr, indices, Z, X_data, Y_data, F_data):
    """F_data[p, :] = clip(Zdiff*X*exp(-Y) - exp(-Zdiff), -100, 100) for edge
    p = (i, indices[p]).

    Zdiff is computed on the fly from Z, never stored as a separate
    (nnz, d) array -- one less array to allocate/touch per epoch. The clip
    is two-sided: exp(-Zdiff) blows up (and drags F to -inf) whenever Zdiff
    is very negative, not just very positive, so bounding only the top
    (min(100, f)) still let the recurrence diverge to -inf.
    """
    n = indptr.shape[0] - 1
    d = Z.shape[1]
    for i in prange(n):
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            for t in range(d):
                zdiff = Z[j, t] - Z[i, t]
                f = zdiff * X_data[p, t] * np.exp(-Y_data[p, t]) - np.exp(-zdiff)
                F_data[p, t] = min(100.0, max(-100.0, f))


@njit(parallel=True, fastmath=True, cache=True)
def row_mean_update(indptr, F_data, Z):
    """Z[i, :] = mean over row i's edges of F_data. Empty rows are left
    unchanged (degree-0 nodes have no F contributions to average)."""
    n = indptr.shape[0] - 1
    d = Z.shape[1]
    for i in prange(n):
        deg = indptr[i + 1] - indptr[i]
        if deg == 0:
            continue
        for t in range(d):
            acc = 0.0
            for p in range(indptr[i], indptr[i + 1]):
                acc += F_data[p, t]
            Z[i, t] = acc / deg


def run_numba(indptr, indices, Z0, X_data, Y_data, epochs):
    Z = Z0.copy()
    nnz, d = X_data.shape
    F_data = np.empty((nnz, d), dtype=np.float64)
    for _ in range(epochs):
        compute_F(indptr, indices, Z, X_data, Y_data, F_data)
        row_mean_update(indptr, F_data, Z)
    return Z


# ---------------------------------------------------------------------------
# Pure-Python/NumPy reference (small n only -- for correctness checking)
# ---------------------------------------------------------------------------
def reference_step(indptr, indices, Z, X_data, Y_data):
    """One epoch of the same update, plain Python/NumPy, sequential."""
    n = indptr.shape[0] - 1
    d = Z.shape[1]
    Znew = Z.copy()
    for i in range(n):
        deg = indptr[i + 1] - indptr[i]
        if deg == 0:
            continue
        acc = np.zeros(d)
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            zdiff = Z[j] - Z[i]
            f = zdiff * X_data[p] * np.exp(-Y_data[p]) - np.exp(-zdiff)
            acc += np.clip(f, -100.0, 100.0)
        Znew[i] = acc / deg
    return Znew


def run_reference(indptr, indices, Z0, X_data, Y_data, epochs):
    Z = Z0.copy()
    for _ in range(epochs):
        Z = reference_step(indptr, indices, Z, X_data, Y_data)
    return Z


# ---------------------------------------------------------------------------
# Vectorized NumPy implementation (no numba) -- same CSR topology
# ---------------------------------------------------------------------------
def numpy_step(row_of, indices, deg, deg_safe, Z, X_data, Y_data):
    """One epoch, fully vectorized NumPy (no Python-level loop over nodes
    or edges). Same CSR topology as the numba kernels.

    row_of[p]  : source node i for edge p (precomputed once, np.repeat over indptr)
    deg        : (n,) edge count per row, deg_safe = np.where(deg == 0, 1, deg)

    Row-sum uses np.bincount per dimension rather than np.add.reduceat:
    reduceat silently repeats the previous segment's sum on an empty
    segment instead of giving 0, which would corrupt deg-0 rows (bincount
    handles missing indices correctly by construction).
    """
    n, d = Z.shape
    Zdiff = Z[indices] - Z[row_of]                       # (nnz, d)
    F_data = np.clip(Zdiff * X_data * np.exp(-Y_data) - np.exp(-Zdiff), -100.0, 100.0)  # (nnz, d)

    sums = np.empty((n, d), dtype=np.float64)
    for t in range(d):
        sums[:, t] = np.bincount(row_of, weights=F_data[:, t], minlength=n)

    means = sums / deg_safe[:, None]
    return np.where(deg[:, None] == 0, Z, means)


def run_numpy(indptr, indices, Z0, X_data, Y_data, epochs):
    n = indptr.shape[0] - 1
    row_of = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    deg = np.diff(indptr)
    deg_safe = np.where(deg == 0, 1, deg)

    Z = Z0.copy()
    for _ in range(epochs):
        Z = numpy_step(row_of, indices, deg, deg_safe, Z, X_data, Y_data)
    return Z


# ---------------------------------------------------------------------------
# JAX implementation (same CSR topology) -- runs on CPU or GPU
# ---------------------------------------------------------------------------
def _jax_step(row_of, indices, deg, deg_safe, Z, X_data, Y_data, n):
    """One epoch, JAX version of numpy_step. Row-sum uses
    jax.ops.segment_sum, the JAX analog of np.bincount, which likewise
    gives 0 for segments with no entries (deg-0 rows)."""
    Zdiff = Z[indices] - Z[row_of]
    F_data = jnp.clip(Zdiff * X_data * jnp.exp(-Y_data) - jnp.exp(-Zdiff), -100.0, 100.0)
    sums = jax.ops.segment_sum(F_data, row_of, num_segments=n)
    means = sums / deg_safe[:, None]
    return jnp.where(deg[:, None] == 0, Z, means)


def run_jax(indptr, indices, Z0, X_data, Y_data, epochs, device: str = "cpu"):
    """JAX implementation, same CSR topology as run_numpy.

    device: "cpu" or "gpu" -- selects which JAX backend the computation is
    placed and run on.
    """
    device = device.lower()
    if device not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")

    try:
        target = jax.devices(device)[0]
    except RuntimeError as e:
        raise RuntimeError(f"no JAX '{device}' device available: {e}") from e

    n = indptr.shape[0] - 1
    row_of = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    deg = np.diff(indptr)
    deg_safe = np.where(deg == 0, 1, deg)

    with jax.default_device(target):
        row_of_d = jax.device_put(row_of, target)
        indices_d = jax.device_put(indices, target)
        deg_d = jax.device_put(deg, target)
        deg_safe_d = jax.device_put(deg_safe, target)
        X_data_d = jax.device_put(X_data, target)
        Y_data_d = jax.device_put(Y_data, target)
        Z = jax.device_put(Z0, target)

        step = jax.jit(functools.partial(_jax_step, n=n))
        for _ in range(epochs):
            Z = step(row_of_d, indices_d, deg_d, deg_safe_d, Z, X_data_d, Y_data_d)

    return np.asarray(Z)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_correctness(seed=0):
    """Compare the numba kernels against the sequential reference epoch by
    epoch, rather than only after all 100 iterations.

    `F` includes an unbounded -exp(Zdiff) feedback term: as Z updates,
    Zdiff grows, exp(Zdiff) grows faster, Z moves further -- this is a
    genuinely divergent recurrence, not an implementation bug. With
    unscaled inputs it overflows float64 well before epoch 100, and once
    values are inf/nan, matching a parallel (numba, fastmath) run against
    a sequential one bit-for-bit is meaningless. So: scale inputs down to
    stay in a well-behaved range longer, and stop the comparison (instead
    of failing) the moment either side leaves the finite range.
    """
    rng = np.random.default_rng(seed)
    n, d, avg_degree, epochs = 30, 4, 5, 100

    indptr, indices = build_random_topology(n, avg_degree, rng)
    nnz = indices.shape[0]
    X_data = rng.standard_normal((nnz, d)) * 0.1
    Y_data = rng.standard_normal((nnz, d)) * 0.05
    Z0 = rng.standard_normal((n, d)) * 0.1

    Z_numba = Z0.copy()
    Z_numpy = Z0.copy()
    Z_jax = Z0.copy()
    Z_ref = Z0.copy()
    F_data = np.empty((nnz, d), dtype=np.float64)

    row_of = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    deg = np.diff(indptr)
    deg_safe = np.where(deg == 0, 1, deg)

    jax_step = jax.jit(functools.partial(_jax_step, n=n))

    last_epoch = 0
    for epoch in range(1, epochs + 1):
        compute_F(indptr, indices, Z_numba, X_data, Y_data, F_data)
        row_mean_update(indptr, F_data, Z_numba)
        Z_numpy = numpy_step(row_of, indices, deg, deg_safe, Z_numpy, X_data, Y_data)
        Z_jax = jax_step(row_of, indices, deg, deg_safe, Z_jax, X_data, Y_data)
        Z_ref = reference_step(indptr, indices, Z_ref, X_data, Y_data)
        last_epoch = epoch

        all_finite = (np.isfinite(Z_numba).all() and np.isfinite(Z_numpy).all()
                      and np.isfinite(np.asarray(Z_jax)).all() and np.isfinite(Z_ref).all())
        if not all_finite:
            print(f"  epoch {epoch}: left the finite range (expected for this "
                  f"formula's -exp(Zdiff) feedback) -- stopping comparison here")
            break
        assert np.allclose(Z_numba, Z_ref, rtol=1e-8, atol=1e-10), \
            f"numba result diverged from reference at epoch {epoch}"
        assert np.allclose(Z_numpy, Z_ref, rtol=1e-8, atol=1e-10), \
            f"numpy result diverged from reference at epoch {epoch}"
        assert np.allclose(Z_jax, Z_ref, rtol=1e-6, atol=1e-8), \
            f"jax result diverged from reference at epoch {epoch}"

    # node 0 has degree 0 -> must stay exactly at its initial value throughout
    assert np.array_equal(Z_numba[0], Z0[0]), "isolated node (numba) was updated but shouldn't be"
    assert np.array_equal(Z_numpy[0], Z0[0]), "isolated node (numpy) was updated but shouldn't be"
    assert np.array_equal(np.asarray(Z_jax[0]), Z0[0]), "isolated node (jax) was updated but shouldn't be"
    print(f"correctness OK through epoch {last_epoch}  (n={n}, d={d}, nnz={nnz})")


def bench(seed=1, n=10_000, d=16, avg_degree=10, epochs=1000, input_data=None):
    
    if(input_data is not None):
        X_data, Y_data, Z0 = input_data
    else:
        rng = np.random.default_rng(seed)
        indptr, indices = build_random_topology(n, avg_degree, rng)
        nnz = indices.shape[0]
        X_data = rng.standard_normal((nnz, d)) * 0.1
        Y_data = rng.standard_normal((nnz, d)) * 0.05
        Z0 = rng.standard_normal((n, d)) * 0.1

    # warm up JIT compilation before timing
    run_numba(indptr, indices, Z0, X_data, Y_data, epochs=1)

    print("\n>>>>>> Running benchmark (numba) <<<<<")
    t0 = time.perf_counter()
    Z = run_numba(indptr, indices, Z0, X_data, Y_data, epochs)
    t1 = time.perf_counter()

    mean_numba_duration = (t1-t0) / epochs
    print(f"benchmark: n={n}, d={d}, nnz={nnz}, epochs={epochs} "
          f"-> {t1 - t0:.4f}s ({(t1 - t0) / epochs * 1e3:.3f} ms/epoch)")
    print(f"Z stats after run: mean={Z.mean():.6f}, std={Z.std():.6f}, "
          f"finite={np.isfinite(Z).all()}")

    print(f"numba: {t1 - t0:.4f}s ({(t1 - t0) / epochs * 1e3:.3f} ms/epoch)")

    # print(">>>>>> Running benchmark (numpy) <<<<<")
    # t0 = time.perf_counter()
    # Z = run_numpy(indptr, indices, Z0, X_data, Y_data, epochs)
    # t1 = time.perf_counter()
    # print(f"benchmark: n={n}, d={d}, nnz={nnz}, epochs={epochs} "
    #       f"-> {t1 - t0:.4f}s ({(t1 - t0) / epochs * 1e3:.3f} ms/epoch)")
    # print(f"Z stats after run: mean={Z.mean():.6f}, std={Z.std():.6f}, "
    #       f"finite={np.isfinite(Z).all()}")

    mean_jax_cpu_duration = None
    mean_jax_gpu_duration = None
    for device in ("cpu", "gpu"):
        try:
            jax.devices(device)
        except RuntimeError:
            print(f"jax[{device}]: no device available, skipping")
            continue
        run_jax(indptr, indices, Z0, X_data, Y_data, epochs=1, device=device)  # warm up
        print(f"\n>>>>>> Running benchmark (jax) on {device} <<<<<")
        t0 = time.perf_counter()
        run_jax(indptr, indices, Z0, X_data, Y_data, epochs, device=device)
        t1 = time.perf_counter()
        print(f"jax[{device}]: {t1 - t0:.4f}s ({(t1 - t0) / epochs * 1e3:.3f} ms/epoch)")

        if(device == "cpu"):
            mean_jax_cpu_duration = (t1 - t0) / epochs
        else:
            mean_jax_gpu_duration = (t1 - t0) / epochs

    return (mean_numba_duration, mean_jax_cpu_duration, mean_jax_gpu_duration)

if __name__ == "__main__":
    test_correctness()
    # bench()

    results = []
    for n in (1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10_000, 20_000, 30_000, 40_000, 50_000):
        r =  bench(n=n, d=16, avg_degree=10, epochs=1000)
        results.append((n, *r))

    print("\n\nSummary of mean epoch durations (seconds):")
    print(f"{'n':>8} {'numba':>12} {'jax[CPU]':>12} {'jax[GPU]':>12}")
    for n, numba_dur, jax_cpu_dur, jax_gpu_dur in results:
        print(f"{n:>8} {numba_dur:>12.6f} {jax_cpu_dur:>12.6f} {jax_gpu_dur:>12.6f}")
