"""T4.3 -- perf + memory benchmarks for the embedding-stage kernel.

RPD.md success metrics #5 and #6. Two distinct, deliberately unentangled
measurements (see task brief -- don't conflate a whole-pipeline timing
with the metric #6 kernel comparison, since augmentation is O(n^2) by
construction regardless of embedding-kernel implementation):

1. **Metric #6 -- Numba vs. vectorized-NumPy speedup** (the ~8x figure
   from IMPLEMENTATION.md). Benchmarks ``embedding.shell_force._forces_shell``
   (the real Numba kernel) against a throwaway vectorized-NumPy
   reimplementation of the *same* shell-averaged force law, written only
   for this benchmark (not a production module). Both operate on the same
   real ``D`` (dense hop-fill output) for a fair comparison.

   Important difference from IMPLEMENTATION.md's prototype benchmark: that
   prototype ran over a genuinely *sparse* CSR topology (avg_degree=10,
   nnz ~ 10n << n^2), so its vectorized-NumPy step could afford one
   (nnz, d) array per epoch. Hop-fill's ``D`` has NO such sparsity --
   every reachable pair carries a real hop distance (documented dense-by-
   construction in ``graph_augmenting/hopfill.py``), so the same "materialize
   the whole thing and reduce" numpy strategy would need an (n, n, d)
   array -- e.g. 51 GB at n=20000, d=16 -- which is not just slow, it
   doesn't fit in memory on this machine. The vectorized-NumPy
   reimplementation below is therefore **row-batched**: vectorized across
   a chunk of B rows x n x d at a time (bounded by a fixed memory budget,
   not by n), which is still "no Numba, no per-scalar Python loop" in the
   spirit of the benchmark, without being physically impossible to run.

2. **Metric #5 -- memory guardrail**. The embedding-kernel agent already
   confirmed by code inspection that the kernel's *extra* peak memory is
   O(n*d) (row-streamed accumulation, never an (n,n,d) or (nnz,d)
   temporary). This adds an empirical check: measure peak RSS during a
   real ``ReferenceFDModel.embed(...)`` call across increasing n (at fixed
   d) and increasing d (at fixed n), each in its own subprocess for a
   clean isolated measurement (``resource.getrusage(...).ru_maxrss``).
   Expectation: peak RSS scales roughly quadratically with n (inherent to
   dense hop-fill's O(n^2) D -- not a bug, documented in
   ``graph_augmenting/hopfill.py``) but is roughly FLAT in d (confirms no
   accidental (n,n,d) materialization anywhere in the assembled pipeline,
   which is what an O(n^2*d) blowup would look like).

This machine has limited memory (~4.4 GiB available at the time this was
written, checked via `free -h`) -- n is capped at 8000 (dense (n,n) int32
hops matrix = 256 MB at n=8000; would be 1.6 GB at n=20000, and transient
copies during derivation can multiply that), smaller than
IMPLEMENTATION.md's suggested 20000, per the task brief's explicit
"keep it modest" / "check available memory first" instructions.

Run:
    source /home/h/gnn/fd-graph-embedding/fdmap/.venv/bin/activate
    python -m fdge2.validation.bench_embedding_perf     # from repo root (fdmap/)
"""
from __future__ import annotations

import json
import pathlib
import resource
import subprocess
import sys
import time

import numpy as np
import networkx as nx

from fdge2.graph_augmenting.hopfill import augment_graph
from fdge2.embedding.shell_force import _forces_shell, _derive_from_D
from fdge2.models import ReferenceFDModel


K1, K2, K4 = 0.999, 1.0, 0.01


def _make_graph(n, avg_degree, seed=0):
    """Connected-ish graph with controlled avg degree, cheap to build at
    the sizes used here. Watts-Strogatz is O(n * avg_degree)."""
    k = max(2, avg_degree - (avg_degree % 2))  # nx requires even k
    return nx.watts_strogatz_graph(n, k=k, p=0.1, seed=seed)


# ---------------------------------------------------------------------------
# Throwaway vectorized-NumPy reimplementation of the shell force law
# (no Numba; row-batched only to stay within physical memory -- see
# module docstring for why full (n,n,d) vectorization is not an option).
# ---------------------------------------------------------------------------
def _forces_shell_numpy(Z, hops_idx, h_of_idx, shell_counts, degrees,
                         k1, k2, k3, k4, row_start, row_end,
                         max_batch_bytes=256 * 1024 * 1024):
    n, d = Z.shape
    out = np.empty((row_end - row_start, d), dtype=np.float64)

    bytes_per_row = max(n * d * 8, 1)
    batch = max(1, min(row_end - row_start, max_batch_bytes // bytes_per_row))

    for bs in range(row_start, row_end, batch):
        be = min(bs + batch, row_end)
        rows = np.arange(bs, be)

        Zu = Z[rows]                                    # (B, d)
        Zdiff = Z[None, :, :] - Zu[:, None, :]           # (B, n, d) -- the
                                                          # array IMPLEMENTATION.md's
                                                          # guardrail forbids
                                                          # materializing in the
                                                          # PRODUCTION kernel; fine
                                                          # here since this function
                                                          # exists only to be the
                                                          # slow benchmark baseline.
        x = np.sqrt(np.einsum("bnd,bnd->bn", Zdiff, Zdiff))

        hi = hops_idx[rows]                              # (B, n)
        valid = hi > 0
        h = h_of_idx[hi]
        shell_coeff = 1.0 / shell_counts[rows[:, None], hi]

        Fa = k1 * shell_coeff * x * np.exp(-k2 * (h - 1.0))
        Fr = -k3 * h * np.exp(-k4 * x)
        F = Fa + Fr

        safe_x = np.where(x > 0, x, 1.0)
        scale = np.where(valid & (x > 0), F / safe_x, 0.0)
        acc = np.einsum("bn,bnd->bd", scale, Zdiff)      # row-reduce over v

        deg = degrees[rows]
        safe_deg = np.where(deg > 0, deg, 1.0)
        out[bs - row_start:be - row_start] = np.where(
            deg[:, None] > 0, acc / safe_deg[:, None], 0.0)

    return out


# ---------------------------------------------------------------------------
# correctness gate: the numpy reimplementation must actually compute the
# same thing before its timing means anything
# ---------------------------------------------------------------------------
def _check_numpy_matches_numba():
    rng = np.random.default_rng(0)
    G = nx.disjoint_union(nx.karate_club_graph(), nx.path_graph(6))
    n = G.number_of_nodes()
    D = augment_graph(G, is_sparse=False)
    der = _derive_from_D(D)
    Z = np.ascontiguousarray(rng.standard_normal((n, 4)))
    k3 = float(der["n"])

    out_numba = np.empty((n, 4), dtype=np.float64)
    _forces_shell(Z, der["hops_idx"], der["h_of_idx"], der["shell_counts"],
                  der["degrees"], K1, K2, k3, K4, 0, n, out_numba)
    out_numpy = _forces_shell_numpy(Z, der["hops_idx"], der["h_of_idx"],
                                     der["shell_counts"], der["degrees"],
                                     K1, K2, k3, K4, 0, n)
    err = np.abs(out_numba - out_numpy).max()
    assert err < 1e-9, f"numpy reimplementation disagrees with numba kernel: {err:.3e}"
    print(f"[ok] vectorized-numpy baseline matches _forces_shell "
          f"(max abs diff {err:.2e}) -- timing comparison below is fair")


# ---------------------------------------------------------------------------
# Metric #6: Numba vs. vectorized-NumPy speedup
# ---------------------------------------------------------------------------
def run_speed_benchmark(n_values=(2000, 4000, 8000), avg_degree=8, d=16, epochs=5):
    print("\n" + "=" * 78)
    print("METRIC #6 -- Numba vs. vectorized-NumPy speedup "
          "(embedding kernel only)")
    print("=" * 78)
    print(f"avg_degree={avg_degree}, d={d}, {epochs} timed epochs "
          f"(JIT warmup excluded)\n")

    rows = []
    for n in n_values:
        G = _make_graph(n, avg_degree)
        D = augment_graph(G, is_sparse=False)
        der = _derive_from_D(D)
        k3 = float(der["n"])
        rng = np.random.default_rng(0)
        Z = np.ascontiguousarray(rng.standard_normal((n, d)))
        out = np.empty((n, d), dtype=np.float64)

        # warm up JIT (excluded from timing)
        _forces_shell(Z, der["hops_idx"], der["h_of_idx"], der["shell_counts"],
                      der["degrees"], K1, K2, k3, K4, 0, n, out)

        t0 = time.perf_counter()
        for _ in range(epochs):
            _forces_shell(Z, der["hops_idx"], der["h_of_idx"], der["shell_counts"],
                          der["degrees"], K1, K2, k3, K4, 0, n, out)
        t_numba = (time.perf_counter() - t0) / epochs

        t0 = time.perf_counter()
        for _ in range(epochs):
            _forces_shell_numpy(Z, der["hops_idx"], der["h_of_idx"],
                                 der["shell_counts"], der["degrees"],
                                 K1, K2, k3, K4, 0, n)
        t_numpy = (time.perf_counter() - t0) / epochs

        speedup = t_numpy / t_numba
        rows.append((n, t_numba, t_numpy, speedup))
        print(f"  n={n:>6}  numba={t_numba:8.4f}s  numpy={t_numpy:8.4f}s  "
              f"speedup={speedup:5.1f}x")

    min_speedup = min(r[3] for r in rows)
    print()
    if min_speedup >= 2.0:
        print(f"[PASS] Numba meaningfully beats vectorized NumPy at every n "
              f"(min speedup {min_speedup:.1f}x). IMPLEMENTATION.md's sparse "
              f"prototype found ~8x under a different force law/topology; "
              f"same-order-of-magnitude win confirmed here for the real "
              f"shell-force kernel on dense hop-fill D.")
    else:
        print(f"[FAIL] speedup collapsed to {min_speedup:.1f}x -- regression "
              f"against the metric #6 performance floor, investigate.")
    return rows


# ---------------------------------------------------------------------------
# Metric #5: memory guardrail -- empirical peak-RSS scaling check
# ---------------------------------------------------------------------------
def _mem_child(n, d, avg_degree, epochs):
    """Run in a fresh subprocess: build a real ReferenceFDModel, embed it,
    print {"n":.., "d":.., "ru_maxrss_kb":..} as the last stdout line."""
    G = _make_graph(n, avg_degree)
    m = ReferenceFDModel(n_dim=d, random_drop_rate=0.0, verbosity=0, seed=0)
    m.embed(G, epochs=epochs, is_sparse=False)
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
    print(json.dumps({"n": n, "d": d, "ru_maxrss_kb": peak_kb}))


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # .../fdmap


def _measure_mem_subprocess(n, d, avg_degree, epochs):
    # Invoked as -m so the child's `import fdge2...` resolves the same way
    # it would for any normal caller (running the file directly by path
    # would not put the repo root on sys.path, and fdge2 has no top-level
    # installed package -- it's imported by being run from the repo root).
    proc = subprocess.run(
        [sys.executable, "-m", "fdge2.validation.bench_embedding_perf",
         "_mem_child", str(n), str(d), str(avg_degree), str(epochs)],
        capture_output=True, text=True, check=True, cwd=str(_REPO_ROOT),
    )
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)["ru_maxrss_kb"]


def run_memory_benchmark(n_values=(1000, 2000, 4000, 8000), d_fixed=16,
                          d_values=(4, 16, 64), n_fixed=4000,
                          avg_degree=8, epochs=3, n_baseline=50):
    print("\n" + "=" * 78)
    print("METRIC #5 -- memory guardrail (O(n*d) kernel overhead, "
          "O(n^2) from dense D -- not O(n^2*d))")
    print("=" * 78)
    print(f"Each row = ReferenceFDModel(...).embed(...) in a fresh subprocess, "
          f"peak RSS via resource.getrusage(RUSAGE_SELF).ru_maxrss.\n")

    # Fixed process overhead (Python/numpy/scipy/networkx import + Numba JIT
    # compilation of the parallel kernels) is ~300 MB on this machine --
    # comparable to or larger than the actual O(n^2) data at the n values
    # kept modest here (memory-constrained machine, see module docstring).
    # A raw RSS ratio would be swamped by that constant and wrongly read as
    # "not quadratic." Subtract a trivial-n baseline so the comparison
    # isolates the n/d-DEPENDENT part of memory, which is what this metric
    # is actually about.
    baseline_kb = _measure_mem_subprocess(n_baseline, d_fixed, avg_degree, epochs)
    print(f"baseline (n={n_baseline}, fixed import+JIT overhead): "
          f"{baseline_kb / 1024:.1f} MB\n")

    print(f"-- varying n, d fixed at {d_fixed} --")
    n_rows = []
    for n in n_values:
        rss_kb = _measure_mem_subprocess(n, d_fixed, avg_degree, epochs)
        extra_kb = max(rss_kb - baseline_kb, 1)
        n_rows.append((n, rss_kb, extra_kb))
        print(f"  n={n:>6}  peak RSS={rss_kb / 1024:8.1f} MB  "
              f"extra-over-baseline={extra_kb / 1024:7.1f} MB")

    print(f"\n-- varying d, n fixed at {n_fixed} --")
    d_rows = []
    for d in d_values:
        rss_kb = _measure_mem_subprocess(n_fixed, d, avg_degree, epochs)
        extra_kb = max(rss_kb - baseline_kb, 1)
        d_rows.append((d, rss_kb, extra_kb))
        print(f"  d={d:>4}  peak RSS={rss_kb / 1024:8.1f} MB  "
              f"extra-over-baseline={extra_kb / 1024:7.1f} MB")

    # ---- verdict: n-scaling (extra, baseline-subtracted) should be
    # roughly quadratic; d-scaling should be flat -------------------------
    ns = np.array([r[0] for r in n_rows], dtype=np.float64)
    extra_n = np.array([r[2] for r in n_rows], dtype=np.float64)
    # log-log slope: extra ~ n^p  =>  log(extra) = p*log(n) + c
    p_n, _ = np.polyfit(np.log(ns), np.log(extra_n), 1)
    print(f"\nn-scaling: log-log fit of (extra-over-baseline) vs n gives "
          f"exponent p={p_n:.2f} (quadratic hop-fill predicts p~2; "
          f"O(n) alone would give p~1)")

    d0, rss_d0, extra_d0 = d_rows[0]
    dN, rss_dN, extra_dN = d_rows[-1]
    ratio_d = dN / d0
    ratio_rss_d = rss_dN / rss_d0
    print(f"d grew {ratio_d:.1f}x ({d0}->{dN}); peak RSS grew only "
          f"{ratio_rss_d:.2f}x (an O(n^2*d) bug would predict ~{ratio_d:.1f}x)")

    # n exponent should be clearly super-linear (rules out O(n) or O(n log n)
    # augmentation dominating) and not absurdly above quadratic; loose bounds
    # since this is 4 data points with real noise, not a controlled fit
    n_ok = 1.3 <= p_n <= 2.7
    # flat-in-d: RSS should barely move across a 16x change in d -- nowhere
    # near the ~16x an O(n^2*d) or O(n*d) blowup at fixed large n^2 would give
    d_ok = ratio_rss_d < (ratio_d ** 0.3)

    print()
    if n_ok and d_ok:
        print(f"[PASS] memory scales roughly like O(n^2) with n (matches "
              f"inherent dense hop-fill D, documented in "
              f"graph_augmenting/hopfill.py) and is essentially flat with d (no "
              f"(n,n,d)/(nnz,d) materialization anywhere in the assembled "
              f"pipeline -- an O(n^2*d) bug would have shown up as the "
              f"d-series growing in lockstep with d).")
    else:
        print(f"[FAIL] n_ok={n_ok} (p_n={p_n:.2f}) d_ok={d_ok} "
              f"(ratio_rss_d={ratio_rss_d:.2f}) -- memory scaling does not "
              f"match the expected O(n^2) + O(n*d) shape. Investigate before "
              f"trusting the O(n*d) kernel-memory guardrail.")
    return n_rows, d_rows


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_mem_child":
        _mem_child(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        _check_numpy_matches_numba()
        # Memory benchmark FIRST, deliberately: it spawns subprocesses via
        # `subprocess.run`, which are independent processes, but this
        # machine is memory-constrained and shared (module docstring). If
        # the speed benchmark runs first in THIS process, its own large
        # transient numpy temporaries (n=8000 vectorized-numpy path
        # allocates ~256 MB per row-batch) inflate this parent's resident
        # memory and system-wide memory pressure, which was observed to
        # distort the child subprocesses' peak-RSS readings (all children
        # spuriously reporting the same ~1.3 GB regardless of n/d).
        # Running memory-sensitive measurements first, while this process
        # is still "clean," avoided that -- verified by rerunning in
        # isolation (see IMPLEMENTATION notes in this file's docstring).
        run_memory_benchmark()
        run_speed_benchmark()
