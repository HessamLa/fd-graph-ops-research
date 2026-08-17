#!/usr/bin/env python3
"""
fdmap_tiled_bench_jax.py

Uniform-tile force-update benchmark (JAX). Counterpart of
fdmap_bucketed_bench_jax.py (SELL-C-sigma ladder): identical graph
generation, seeds, kernel expression, float64 oracle, and printout
conventions, so the two benchmarks compare apples-to-apples.

Layout: every virtual row in the plan has the SAME width TILE_H (the tile
height h of the covering-problem framing). The whole matrix therefore
compiles to ONE batch shape (nb, R, h) processed by a single lax.scan --
no ladder, no K_MAX, no per-rung shapes. h replaces K_MAX as the hub
splitter: a degree-K row simply becomes ceil(K/h) chunks.

Plan modes (FDMAP_TILE_MODE):
  row     (default) each row i is split into ceil(deg_i/h) chunks of h
          slots, padded independently. Total padding is the pure width-
          quantization term  sum_i (h*ceil(deg_i/h) - deg_i)  plus a tiny
          balanced batch-fill term. No staircase term exists in this mode.
  column  literal geometric covering of the sorted degree profile: rows
          sorted by degree (desc), grouped into columns of TILE_W rows,
          every member row padded to ceil(col_max/h)*h before chunking.
          Strictly more padding (staircase + quantization); included to
          price the staircase and to check the asymptotic prediction
          rho ~ (w*dF + h*n)/2 against a real power-law profile. The
          output line adds the measured stair/quant split and `pred`.

Scatter conflicts: without K_MAX, a degree-K hub owns ceil(K/h) chunks.
FDMAP_INTERLEAVE=1 (default) reorders chunks with stride nb so an owner's
consecutive chunks land in different batches; batches are sequential scan
steps, so cross-batch duplicates cost nothing.

Env overrides:
  FDMAP_SWEEP="1000,5000"     FDMAP_D=128          FDMAP_BCELLS=65536
  FDMAP_TILE_H=4              FDMAP_H_SWEEP="2,3,4,6,8,12,16"
  FDMAP_TILE_MODE=row|column  FDMAP_TILE_W=64      FDMAP_INTERLEAVE=0

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
TILE_H      = 4            # slots per chunk = uniform row width (the h knob)
TILE_W      = 64           # rows per tile -- column mode only
TILE_MODE   = "row"        # "row" | "column"
INTERLEAVE  = True         # spread same-owner chunks across batches
C1_SCALE    = 0.05         # C1 ~ U(0, C1_SCALE)
C2_SCALE    = 0.50         # C2 ~ U(0, C2_SCALE)
SWEEP_N     = (1_000, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000,
               40_000, 50_000, 75_000, 100_000, 200_000, 400_000,
               700_000, 1_000_000)
REPS        = 5            # timed repetitions of the full N_STEPS run
VALIDATE_MAX_N = 200_000   # float64 oracle cap (host-RAM guard)
ORACLE_BLOCK   = 1 << 17   # edges per oracle chunk (~bounded RAM)

if os.environ.get("FDMAP_SWEEP"):
    SWEEP_N = tuple(int(x) for x in os.environ["FDMAP_SWEEP"].split(","))
if os.environ.get("FDMAP_D"):
    D = int(os.environ["FDMAP_D"])
if os.environ.get("FDMAP_BCELLS"):
    B_CELLS = int(os.environ["FDMAP_BCELLS"])
if os.environ.get("FDMAP_TILE_H"):
    TILE_H = int(os.environ["FDMAP_TILE_H"])
if os.environ.get("FDMAP_TILE_W"):
    TILE_W = int(os.environ["FDMAP_TILE_W"])
if os.environ.get("FDMAP_TILE_MODE"):
    TILE_MODE = os.environ["FDMAP_TILE_MODE"]
    assert TILE_MODE in ("row", "column"), f"bad FDMAP_TILE_MODE={TILE_MODE!r}"
if os.environ.get("FDMAP_INTERLEAVE"):
    INTERLEAVE = os.environ["FDMAP_INTERLEAVE"] != "0"
H_LIST = (tuple(int(x) for x in os.environ["FDMAP_H_SWEEP"].split(","))
          if os.environ.get("FDMAP_H_SWEEP") else (TILE_H,))


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
def make_tile_plan(adj, C1, C2, h, b_cells=B_CELLS, mode=TILE_MODE,
                   w=TILE_W, interleave=INTERLEAVE):
    """
    One-time host-side plan: uniform-width chunks (single compiled shape).

    Returns (plan, inv_deg_ext, stats):
      plan = (rows, nbrs, c1, c2):
        rows (nb, R)    int32  owner node id per chunk (== n for fill chunks)
        nbrs (nb, R, h) int32  neighbor ids (owner id in pad cells, 0 in fill)
        c1, c2 (nb, R, h) f32  kernel coefficients (0 in pad/fill cells)
      inv_deg_ext: (n+1,) f32, 1/deg with a trailing 0 slot
    Padding safety is the same double guarantee as the ladder plan: pad
    cells point at the row's own node (zd = 0) AND carry c1 = c2 = 0.
    """
    assert h >= 1
    n = adj.shape[0]
    A = adj.tocsr()
    A.sort_indices()
    indptr = A.indptr.astype(np.int64)
    indices = A.indices
    deg = np.diff(indptr)
    m = int(indptr[-1])

    assert C1.nnz == A.nnz and np.array_equal(C1.indptr, A.indptr)
    assert C2.nnz == A.nnz and np.array_equal(C2.indptr, A.indptr)
    c1d = C1.data.astype(np.float32)
    c2d = C2.data.astype(np.float32)

    # 1) chunks per row: independent (row mode) or column-max coupled (column)
    if mode == "row":
        order = np.arange(n, dtype=np.int64)       # rank -> node id
        dord = deg
        nchunk = -(-dord // h)                     # ceil(deg/h) per row
        stair = 0
    elif mode == "column":
        order = np.argsort(-deg, kind="stable")    # rank -> node id, deg desc
        dord = deg[order]
        col_of = np.arange(n, dtype=np.int64) // w
        col_max = dord[::w]                        # first of block = block max
        nchunk = (-(-col_max // h))[col_of]        # ceil(col_max/h) tiles/row
        stair = int((col_max[col_of] - dord).sum())
    else:
        raise ValueError(f"unknown tile mode {mode!r}")

    n_chunks = int(nchunk.sum())
    rowpad = n_chunks * h - m                      # stair (col mode) + quant
    base = np.zeros(n + 1, np.int64)
    np.cumsum(nchunk, out=base[1:])
    rank_of = np.empty(n, np.int64)
    rank_of[order] = np.arange(n)

    # 2) vectorized cell -> (chunk, slot) scatter
    rows_per_cell = np.repeat(np.arange(n, dtype=np.int64), deg)
    pos = np.arange(m, dtype=np.int64) - np.repeat(indptr[:-1], deg)
    cid = base[rank_of[rows_per_cell]] + pos // h
    slot = pos % h
    owner = np.repeat(order, nchunk)

    # 3) balanced packing (same load-bearing rule as the ladder plan):
    #    nb from the cell-budget cap, then R = ceil(chunks/nb) so the last
    #    batch is never near-empty full-width fill work
    cap = max(1, b_cells // h)
    nb = max(1, -(-n_chunks // cap))
    R = -(-n_chunks // nb)
    total = nb * R

    rows_flat = np.full(total, n, np.int32)        # fill chunks -> dropped
    rows_flat[:n_chunks] = owner
    nbrs_flat = np.zeros((total, h), np.int32)
    nbrs_flat[:n_chunks] = owner.astype(np.int32)[:, None]   # pad cells -> self
    c1_flat = np.zeros((total, h), np.float32)
    c2_flat = np.zeros((total, h), np.float32)
    nbrs_flat[cid, slot] = indices                 # real cells overwrite
    c1_flat[cid, slot] = c1d
    c2_flat[cid, slot] = c2d

    # 4) stride-nb interleave: old chunk c -> batch (c mod nb), so an
    #    owner's consecutive chunks land in different (sequential) batches
    if interleave and nb > 1:
        perm = np.arange(total).reshape(R, nb).T.reshape(-1)
        rows_flat = rows_flat[perm]
        nbrs_flat = nbrs_flat[perm]
        c1_flat = c1_flat[perm]
        c2_flat = c2_flat[perm]

    plan = (rows_flat.reshape(nb, R),
            nbrs_flat.reshape(nb, R, h),
            c1_flat.reshape(nb, R, h),
            c2_flat.reshape(nb, R, h))

    inv_deg_ext = np.zeros(n + 1, np.float32)
    inv_deg_ext[:n] = 1.0 / np.maximum(deg, 1)

    fill = (total - n_chunks) * h
    padded = rowpad + fill
    tot = max(1, m + padded)
    stats = dict(cells=m, chunks=n_chunks, nb=nb, R=R,
                 pad_frac=padded / tot,
                 stair_frac=stair / tot,
                 quant_frac=(rowpad - stair) / tot,
                 fill_frac=fill / tot)
    if mode == "column":
        dF = int(dord[0] - dord[-1]) if n > 1 else 0
        pred = 0.5 * (w * dF + h * n)              # asymptotic rho, sec. 2/3
        stats["pred_frac"] = pred / max(1.0, m + pred)
    return plan, inv_deg_ext, stats


# ----------------------------- jitted update loop ----------------------------
@partial(jax.jit, static_argnums=(3,))
def run(Z, plan, inv_deg_ext, n_steps):
    n = Z.shape[0]
    rows_b, nbrs_b, c1_b, c2_b = plan              # ONE shape -> one scan

    def one_step(_, Z):
        def body(dZ, batch):
            rows, nbrs, c1, c2 = batch
            Zc = Z[jnp.minimum(rows, n - 1)]       # (R, d) centers, per chunk
            Zj = Z[nbrs]                           # (R, h, d) neighbors
            zd = Zj - Zc[:, None, :]               # element-wise
            r = jnp.sqrt(jnp.sum(zd * zd, axis=-1))          # (R, h)
            s = c1 * r - c2 * jnp.exp(-r)          # kernel scalar per cell
            F = jnp.sum(zd * s[..., None], axis=1)           # -> (R, d)
            F = F * inv_deg_ext[rows][:, None]     # /deg (0 for fill chunks)
            dZ = dZ.at[rows].add(F, mode="drop")   # per-chunk write; id n drops
            return dZ, None
        dZ, _ = jax.lax.scan(body, jnp.zeros_like(Z),
                             (rows_b, nbrs_b, c1_b, c2_b))
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
    for name in ("M", "SEED", "D", "N_STEPS", "B_CELLS", "TILE_H", "TILE_W",
                 "TILE_MODE", "INTERLEAVE", "C1_SCALE", "C2_SCALE", "REPS",
                 "VALIDATE_MAX_N", "ORACLE_BLOCK"):
        print(f"  {name:<14} = {globals()[name]}")
    print(f"  H_LIST         = {H_LIST}")
    print(f"  SWEEP_N        = {SWEEP_N}\n")

    rng = np.random.default_rng(SEED)
    results = []

    for n in SWEEP_N:
        try:
            row, col = generate_ba_graph(n, m=M, seed=SEED)
            adj = build_sparse_adjacency(row, col, n)
            C1 = random_csr_like(adj, rng, C1_SCALE)
            C2 = random_csr_like(adj, rng, C2_SCALE)
            Z0np = rng.standard_normal((n, D)).astype(np.float32)
        except MemoryError:
            print(f"n={n:>7}  SKIPPED: host out of memory (graph build)")
            continue
        ref1 = refN = None                         # oracle cache: h-invariant

        for h in H_LIST:
            try:
                t0 = time.perf_counter()
                plan_np, inv_np, st = make_tile_plan(
                    adj, C1, C2, h, B_CELLS, TILE_MODE, TILE_W, INTERLEAVE)
                prep_s = time.perf_counter() - t0

                plan = tree_map(jnp.asarray, plan_np)
                inv = jnp.asarray(inv_np)
                Z0 = jnp.asarray(Z0np)

                # compile + first execution (excluded from steady-state timing)
                t0 = time.perf_counter()
                Zf = run(Z0, plan, inv, N_STEPS).block_until_ready()
                compile_s = time.perf_counter() - t0

                if n <= VALIDATE_MAX_N:
                    if ref1 is None:
                        ref1 = reference_run(Z0np, adj, C1, C2, 1)
                        refN = reference_run(Z0np, adj, C1, C2, N_STEPS)
                    Z1 = run(Z0, plan, inv, 1).block_until_ready()
                    err1 = float(np.max(np.abs(np.asarray(Z1) - ref1)))
                    errN = float(np.max(np.abs(np.asarray(Zf) - refN)))
                    assert err1 < 1e-3, \
                        f"validation failed at n={n} h={h}: {err1:.2e}"
                    errs = f"err(1)={err1:.1e}  err({N_STEPS})={errN:.1e}"
                else:
                    errs = "err=skipped (n > VALIDATE_MAX_N)"

                # steady-state timing
                t0 = time.perf_counter()
                for _ in range(REPS):
                    run(Z0, plan, inv, N_STEPS).block_until_ready()
                per_iter_ms = (time.perf_counter() - t0) / REPS / N_STEPS * 1e3
                mcells = st["cells"] / (per_iter_ms * 1e-3) / 1e6

                extra = ""
                if TILE_MODE == "column":
                    extra = (f" [stair {st['stair_frac']*100:4.1f} "
                             f"quant {st['quant_frac']*100:4.1f} "
                             f"pred {st['pred_frac']*100:4.1f}]")
                print(f"n={n:>7} h={h:>3}  cells={st['cells']:>8}  "
                      f"chunks={st['chunks']:>8}  nb={st['nb']:>4}  "
                      f"pad={st['pad_frac']*100:4.1f}%{extra}  "
                      f"prep={prep_s*1e3:6.1f}ms  compile+1st={compile_s:5.2f}s  "
                      f"iter={per_iter_ms:8.3f}ms  {mcells:7.1f} Mcells/s  "
                      f"{errs}")
                results.append((n, h, st["cells"], per_iter_ms, mcells))

                del plan, inv, Z0, Zf
                if hasattr(jax, "clear_caches"):
                    jax.clear_caches()

            except MemoryError:
                print(f"n={n:>7} h={h:>3}  SKIPPED: host out of memory")
            except Exception as e:
                if _is_oom(e):
                    print(f"n={n:>7} h={h:>3}  SKIPPED: out of memory")
                    if hasattr(jax, "clear_caches"):
                        jax.clear_caches()
                else:
                    raise
        del ref1, refN

    print("\nsummary (steady-state, excludes compile):")
    print(f"{'n':>8} {'h':>4} {'cells':>9} {'ms/iter':>10} {'Mcells/s':>9}")
    for n, h, c, ms, mc in results:
        print(f"{n:>8} {h:>4} {c:>9} {ms:>10.3f} {mc:>9.1f}")


if __name__ == "__main__":
    main()
