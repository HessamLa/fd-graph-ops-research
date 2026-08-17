#!/usr/bin/env python3
"""
fdmap_bucketed_bench_jax.py

Force-directed embedding update, bucketed/padded (SELL-C-sigma style) JAX
implementation. Numerics, batch plan, and printout format are identical to
the original fdmap_bucketed_bench.py, so results remain comparable.

Changes vs. the original:
  * float64 validation oracle is CHUNKED over edges (bounded host RAM) and
    capped at VALIDATE_MAX_N -- fixes the host-OOM that killed the Colab
    D=128 GPU run at n=700k
  * env overrides: FDMAP_SWEEP="1000,5000,..."  FDMAP_D=128
  * per-size OOM is caught and skipped; the sweep continues
  * buffers and jit caches are released between sizes (helps small-VRAM GPUs;
    on CUDA, TF_GPU_ALLOCATOR=cuda_malloc_async may further reduce
    fragmentation)

Runs unchanged on CPU / GPU / TPU.
"""

import os
import time
from functools import partial

import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

# ----------------------------- config ---------------------------------------
M           = 3            # BA attachment parameter
SEED        = 42
D           = 128            # embedding dimension (override: FDMAP_D)
N_STEPS     = 10           # iterations of the update loop
B_CELLS     = 16_384       # cell budget per batch  (peak transient ~ B_CELLS*D)
K_MAX       = 256          # max row width; wider rows are split into virtual rows
LADDER_BASE = 1.5          # width-quantization ladder base; finer -> less padding,
                           # more compiled shapes
C1_SCALE    = 0.05         # C1 ~ U(0, C1_SCALE)
C2_SCALE    = 0.50         # C2 ~ U(0, C2_SCALE)
SWEEP_N     = (
                # 1_000, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000,
                # 40_000, 50_000, 75_000, 
                100_000, 200_000, 400_000,
                700_000, 1_000_000)
REPS        = 5            # timed repetitions of the full N_STEPS run
VALIDATE_MAX_N = 200_000   # float64 oracle cap (host-RAM guard)
ORACLE_BLOCK   = 1 << 17   # edges per oracle chunk (~bounded RAM)

if os.environ.get("FDMAP_SWEEP"):
    SWEEP_N = tuple(int(x) for x in os.environ["FDMAP_SWEEP"].split(","))
if os.environ.get("FDMAP_D"):
    D = int(os.environ["FDMAP_D"])


# ------------- stand-ins for your existing graph script ----------------------
# Same signatures as your snippet -- delete these two and import your own.
def generate_ba_graph(n, m=3, seed=42, verbose=False):
    """Barabasi-Albert preferential attachment; returns directed edge arrays."""
    rng = np.random.default_rng(seed)
    row, col = [], []
    repeated = list(range(m))
    targets = list(range(m))
    for src in range(m, n):
        for t in targets:
            row.append(src)
            col.append(t)
        repeated.extend(targets)
        repeated.extend([src] * m)
        tset = set()
        while len(tset) < m:
            tset.add(repeated[rng.integers(len(repeated))])
        targets = list(tset)
    if verbose:
        print(f"  BA edges: {len(row)}")
    return np.asarray(row, np.int64), np.asarray(col, np.int64)


def build_sparse_adjacency(row, col, n):
    """Symmetric binary CSR adjacency from a directed edge list."""
    A = sp.coo_matrix((np.ones(len(row), np.float32), (row, col)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32).tocsr()
    A.sort_indices()
    return A
# -----------------------------------------------------------------------------


def random_csr_like(adj, rng, scale):
    """Random positive values on exactly adj's CSR sparsity structure."""
    A = adj.tocsr()
    data = rng.uniform(0.0, scale, A.nnz).astype(np.float32)
    return sp.csr_matrix((data, A.indices.copy(), A.indptr.copy()), shape=A.shape)


# ----------------------------- preprocessing ---------------------------------
def build_ladder(k_max, base):
    ks = [1]
    while ks[-1] < k_max:
        ks.append(int(np.ceil(ks[-1] * base)))
    # adjust last rung to exactly k_max (if not already)
    if  ks[-1] > k_max: 
        ks[-1] = k_max
    return np.asarray(ks, dtype=np.int64)



def make_plan(adj, C1, C2, b_cells=B_CELLS, k_max=K_MAX, ladder_base=LADDER_BASE):
    """
    One-time host-side batch plan.

    Returns (plan, inv_deg_ext, stats):
      plan: tuple of rungs; each rung is a 4-tuple of stacked batch arrays
            rows (nb, R) int32       owner node id per row slot (== n for pad rows)
            nbrs (nb, R, k) int32    neighbor ids (row's own id in pad cells)
            c1, c2 (nb, R, k) f32    kernel coefficients (0 in pad cells)
      inv_deg_ext: (n+1,) f32, 1/deg with a trailing 0 slot for pad rows
    """
    n = adj.shape[0]
    A = adj.tocsr()
    A.sort_indices()
    indptr, indices = A.indptr, A.indices
    deg = np.diff(indptr).astype(np.int64)

    assert C1.nnz == A.nnz and np.array_equal(C1.indptr, indptr)
    assert C2.nnz == A.nnz and np.array_equal(C2.indptr, indptr)
    c1d = C1.data.astype(np.float32)
    c2d = C2.data.astype(np.float32)

    # 1) virtual rows: split any row wider than k_max (hub handling)
    owners, starts, lens = [], [], []
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        for cs in range(s, e, k_max):
            owners.append(i)
            starts.append(cs)
            lens.append(min(k_max, e - cs))
    owners = np.asarray(owners, np.int64)
    starts = np.asarray(starts, np.int64)
    lens = np.asarray(lens, np.int64)

    # 2) sort virtual rows by width; quantize widths on the ladder
    order = np.argsort(lens, kind="stable")
    owners, starts, lens = owners[order], starts[order], lens[order]
    ladder = build_ladder(k_max, ladder_base)
    kq = ladder[np.searchsorted(ladder, lens)]

    # 3) pack per rung: nb balanced batches of R rows each (R*k ~ b_cells)
    rungs, padded = [], 0
    for k in np.unique(kq):
        sel = np.where(kq == k)[0]
        k = int(k)
        cap = max(1, b_cells // k)              # row cap from the cell budget
        nb = -(-len(sel) // cap)                # batches needed
        R = -(-len(sel) // nb)                  # balanced rows/batch (<= cap)
        rows = np.full(nb * R, n, dtype=np.int32)          # pad rows -> dropped
        nbrs = np.zeros((nb * R, k), dtype=np.int32)
        c1 = np.zeros((nb * R, k), dtype=np.float32)
        c2 = np.zeros((nb * R, k), dtype=np.float32)
        for slot, v in enumerate(sel):                     # vectorizable if ever hot
            o, s0, L = int(owners[v]), int(starts[v]), int(lens[v])
            rows[slot] = o
            nbrs[slot, :L] = indices[s0:s0 + L]
            nbrs[slot, L:] = o                             # pad cells -> self
            c1[slot, :L] = c1d[s0:s0 + L]
            c2[slot, :L] = c2d[s0:s0 + L]
        padded += nb * R * k - int(lens[sel].sum())
        rungs.append((rows.reshape(nb, R),
                      nbrs.reshape(nb, R, k),
                      c1.reshape(nb, R, k),
                      c2.reshape(nb, R, k)))

    inv_deg_ext = np.zeros(n + 1, np.float32)
    inv_deg_ext[:n] = 1.0 / np.maximum(deg, 1)

    cells = int(lens.sum())
    stats = dict(cells=cells,
                 n_virtual=len(owners),
                 n_split=int((deg > k_max).sum()),
                 rungs=len(rungs),
                 pad_frac=padded / max(1, cells + padded))
    return tuple(rungs), inv_deg_ext, stats


# ----------------------------- jitted update loop ----------------------------
@partial(jax.jit, static_argnums=(3,))
def run(Z, plan, inv_deg_ext, n_steps):
    n = Z.shape[0]

    def one_step(_, Z):
        dZ = jnp.zeros_like(Z)
        for rows_b, nbrs_b, c1_b, c2_b in plan:            # unrolled over rungs
            def body(dZ, batch):
                rows, nbrs, c1, c2 = batch
                # --- gather ---
                Zc = Z[jnp.minimum(rows, n - 1)]           # (R, d) centers, per row
                Zj = Z[nbrs]                               # (R, k, d) neighbors
                zd = Zj - Zc[:, None, :]                   # element-wise
                # --- element-wise functions ---
                r = jnp.sqrt(jnp.sum(zd * zd, axis=-1))    # (R, k)
                s = c1 * r - c2 * jnp.exp(-r)              # kernel scalar per cell
                # --- row-wise summation ---
                F = jnp.sum(zd * s[..., None], axis=1)     # row-wise sum -> (R, d)
                F = F * inv_deg_ext[rows][:, None]         # /deg (0 for pad rows)
                dZ = dZ.at[rows].add(F, mode="drop")       # per-ROW write; id n dropped
                return dZ, None
            dZ, _ = jax.lax.scan(body, dZ, (rows_b, nbrs_b, c1_b, c2_b))
        return Z + dZ

    return jax.lax.fori_loop(0, n_steps, one_step, Z)


# ----------------------------- chunked float64 oracle ------------------------
def reference_run(Z0, adj, C1, C2, n_steps, block=ORACLE_BLOCK):
    """Direct float64 edge-list evaluation, streamed in fixed-size edge blocks."""
    A = adj.tocsr()
    A.sort_indices()
    n = A.shape[0]
    deg = np.maximum(np.diff(A.indptr), 1).astype(np.float64)
    src = np.repeat(np.arange(n), np.diff(A.indptr))
    dst = A.indices
    c1e = C1.data.astype(np.float64)
    c2e = C2.data.astype(np.float64)
    Z = Z0.astype(np.float64).copy()
    m = len(dst)
    for _ in range(n_steps):
        F = np.zeros_like(Z)
        for s0 in range(0, m, block):
            sl = slice(s0, min(s0 + block, m))
            zd = Z[dst[sl]] - Z[src[sl]]
            r = np.linalg.norm(zd, axis=1)
            sca = c1e[sl] * r - c2e[sl] * np.exp(-r)
            np.add.at(F, src[sl], zd * sca[:, None])
        Z += F / deg[:, None]
    return Z


def _is_oom(e):
    s = str(e)
    return "RESOURCE_EXHAUSTED" in s or "out of memory" in s.lower()


# ----------------------------- benchmark sweep -------------------------------
def main():
    print(f"jax {jax.__version__} | backend: {jax.default_backend()} "
          f"| devices: {jax.devices()}")
    print("config:")
    for name in ("M", "SEED", "D", "N_STEPS", "B_CELLS", "K_MAX", "LADDER_BASE",
                 "C1_SCALE", "C2_SCALE", "REPS", "VALIDATE_MAX_N", "ORACLE_BLOCK"):
        print(f"  {name:<14} = {globals()[name]}")
    print(f"  SWEEP_N        = {SWEEP_N}\n")

    rng = np.random.default_rng(SEED)
    results = []

    for n in SWEEP_N:
        try:
            row, col = generate_ba_graph(n, m=M, seed=SEED)
            adj = build_sparse_adjacency(row, col, n)
            C1 = random_csr_like(adj, rng, C1_SCALE)
            C2 = random_csr_like(adj, rng, C2_SCALE)

            t0 = time.perf_counter()
            plan_np, inv_np, st = make_plan(adj, C1, C2)
            prep_s = time.perf_counter() - t0

            plan = tree_map(jnp.asarray, plan_np)
            inv = jnp.asarray(inv_np)
            Z0 = jnp.asarray(rng.standard_normal((n, D)).astype(np.float32))

            # compile + first execution (excluded from steady-state timing)
            t0 = time.perf_counter()
            Zf = run(Z0, plan, inv, N_STEPS).block_until_ready()
            compile_s = time.perf_counter() - t0

            if n <= VALIDATE_MAX_N:
                Z1 = run(Z0, plan, inv, 1).block_until_ready()
                Z0np = np.asarray(Z0)
                err1 = float(np.max(np.abs(
                    np.asarray(Z1) - reference_run(Z0np, adj, C1, C2, 1))))
                errN = float(np.max(np.abs(
                    np.asarray(Zf) - reference_run(Z0np, adj, C1, C2, N_STEPS))))
                assert err1 < 1e-3, f"validation failed at n={n}: {err1:.2e}"
                errs = f"err(1)={err1:.1e}  err({N_STEPS})={errN:.1e}"
            else:
                errs = "err=skipped (n > VALIDATE_MAX_N)"

            # steady-state timing
            t0 = time.perf_counter()
            for _ in range(REPS):
                run(Z0, plan, inv, N_STEPS).block_until_ready()
            per_iter_ms = (time.perf_counter() - t0) / REPS / N_STEPS * 1e3
            mcells = st["cells"] / (per_iter_ms * 1e-3) / 1e6

            print(f"n={n:>7}  cells={st['cells']:>8}  rungs={st['rungs']:>2}  "
                  f"split_hubs={st['n_split']}  pad={st['pad_frac']*100:4.1f}%  "
                  f"prep={prep_s*1e3:6.1f}ms  compile+1st={compile_s:5.2f}s  "
                  f"iter={per_iter_ms:8.3f}ms  {mcells:7.1f} Mcells/s  {errs}")
            results.append((n, st["cells"], per_iter_ms, mcells))

            del plan, inv, Z0, Zf
            if hasattr(jax, "clear_caches"):
                jax.clear_caches()

        except MemoryError:
            print(f"n={n:>7}: host out of memory")
            break
        except Exception as e:
            if _is_oom(e):
                print(f"n={n:>7}  SKIPPED: out of memory")
                if hasattr(jax, "clear_caches"):
                    jax.clear_caches()
                break
            else:
                raise

    print("\nsummary (steady-state, excludes compile):")
    print(f"{'n':>8} {'cells':>9} {'ms/iter':>10} {'Mcells/s':>9}")
    for n, c, ms, mc in results:
        print(f"{n:>8} {c:>9} {ms:>10.3f} {mc:>9.1f}")


if __name__ == "__main__":
    main()
