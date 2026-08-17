#!/usr/bin/env python3
"""
fdmap_tiled_bench_torch.py

PyTorch port of the uniform-tile force-update benchmark. Preprocessing,
graph generation, seeds, plan, and printout format are IDENTICAL to
fdmap_tiled_bench_jax.py, so results compare apples-to-apples (and to the
ladder pair fdmap_bucketed_bench_{jax,torch}.py).

Layout: every virtual row has the SAME width TILE_H -- one batch shape
(nb, R, h) for the whole matrix, h replaces K_MAX as the hub splitter.
Modes: row (default, independent per-row chunking, quantization-only
padding) and column (literal covering of the sorted degree profile,
staircase + quantization; prints the split and the rho ~ (w*dF + h*n)/2
prediction). FDMAP_INTERLEAVE=1 (default) spreads a hub's chunks across
batches to break same-address index_add_ atomics within a batch.

Execution paths (reported per-row as [mode]):
  eager          plain eager kernels (CPU, or graphs disabled/failed)
  cuda-graph     eager kernels, whole iteration captured as one CUDA Graph
  compile        FDMAP_COMPILE=1: per-batch math fused by torch.compile
  compile+graph  compiled kernels additionally captured as one CUDA Graph

torch.compile requires Triton on GPU (compute capability >= 7.0; newest
Triton needs >= 8.0) -- on unsupported devices it falls back automatically
and says so. On the GTX 950 use the default eager+graph path. The uniform
shape means torch.compile builds ONE specialization instead of one per
ladder rung.

Local note: torch 2.0.1+cu117 wheels are built against the NumPy 1.x C-API;
run with numpy<2 (or in the dedicated torch venv).

Env overrides:
  FDMAP_SWEEP="1000,5000"     FDMAP_D=128          FDMAP_BCELLS=65536
  FDMAP_TILE_H=4              FDMAP_H_SWEEP="2,3,4,6,8,12,16"
  FDMAP_TILE_MODE=row|column  FDMAP_TILE_W=64      FDMAP_INTERLEAVE=0
  FDMAP_DEVICE=cpu|cuda       FDMAP_COMPILE=1
"""

import os
import time

import numpy as np
import scipy.sparse as sp
import torch

# ----------------------------- config (mirrors JAX script) -------------------
M            = 3
SEED         = 42
D            = 2
N_STEPS      = 10
B_CELLS      = 16_384
TILE_H       = 4
TILE_W       = 64
TILE_MODE    = "row"
INTERLEAVE   = True
C1_SCALE     = 0.05
C2_SCALE     = 0.50
SWEEP_N      = (1_000, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000,
                40_000, 50_000, 75_000, 100_000, 200_000, 400_000,
                700_000, 1_000_000)
REPS         = 5
USE_CUDA_GRAPHS = True
VALIDATE_MAX_N  = 200_000       # float64 oracle cap (host-RAM guard)
ORACLE_BLOCK    = 1 << 17       # edges per oracle chunk (~bounded RAM)

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
USE_COMPILE = os.environ.get("FDMAP_COMPILE", "") == "1"
DEVICE = os.environ.get("FDMAP_DEVICE",
                        "cuda" if torch.cuda.is_available() else "cpu")


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
    A = adj.tocsr()
    data = rng.uniform(0.0, scale, A.nnz).astype(np.float32)
    return sp.csr_matrix((data, A.indices.copy(), A.indptr.copy()), shape=A.shape)


# ----------------------------- preprocessing (identical to JAX) --------------
def make_tile_plan(adj, C1, C2, h, b_cells=B_CELLS, mode=TILE_MODE,
                   w=TILE_W, interleave=INTERLEAVE):
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

    if mode == "row":
        order = np.arange(n, dtype=np.int64)
        dord = deg
        nchunk = -(-dord // h)
        stair = 0
    elif mode == "column":
        order = np.argsort(-deg, kind="stable")
        dord = deg[order]
        col_of = np.arange(n, dtype=np.int64) // w
        col_max = dord[::w]
        nchunk = (-(-col_max // h))[col_of]
        stair = int((col_max[col_of] - dord).sum())
    else:
        raise ValueError(f"unknown tile mode {mode!r}")

    n_chunks = int(nchunk.sum())
    rowpad = n_chunks * h - m
    base = np.zeros(n + 1, np.int64)
    np.cumsum(nchunk, out=base[1:])
    rank_of = np.empty(n, np.int64)
    rank_of[order] = np.arange(n)

    rows_per_cell = np.repeat(np.arange(n, dtype=np.int64), deg)
    pos = np.arange(m, dtype=np.int64) - np.repeat(indptr[:-1], deg)
    cid = base[rank_of[rows_per_cell]] + pos // h
    slot = pos % h
    owner = np.repeat(order, nchunk)

    cap = max(1, b_cells // h)
    nb = max(1, -(-n_chunks // cap))
    R = -(-n_chunks // nb)
    total = nb * R

    rows_flat = np.full(total, n, np.int32)
    rows_flat[:n_chunks] = owner
    nbrs_flat = np.zeros((total, h), np.int32)
    nbrs_flat[:n_chunks] = owner.astype(np.int32)[:, None]
    c1_flat = np.zeros((total, h), np.float32)
    c2_flat = np.zeros((total, h), np.float32)
    nbrs_flat[cid, slot] = indices
    c1_flat[cid, slot] = c1d
    c2_flat[cid, slot] = c2d

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
        pred = 0.5 * (w * dF + h * n)
        stats["pred_frac"] = pred / max(1.0, m + pred)
    return plan, inv_deg_ext, stats


# ----------------------------- torch runtime ---------------------------------
class TorchPlan:
    """Device-resident plan: one uniform (nb, R, h) shape."""

    def __init__(self, plan_np, inv_deg_ext, n, device):
        rows, nbrs, c1, c2 = plan_np
        nb, R, h = nbrs.shape
        self.n, self.nb, self.R, self.h = n, nb, R, h
        inv = torch.from_numpy(inv_deg_ext).to(device)
        rows_t = torch.from_numpy(rows.astype(np.int64)).to(device)
        self.rows_w = rows_t                                # (nb,R) incl. id n
        self.rows_r = rows_t.clamp(max=n - 1)               # for the Z gather
        self.nbrs = (torch.from_numpy(nbrs.astype(np.int64))
                     .reshape(nb, R * h).to(device))
        self.c1 = torch.from_numpy(c1).to(device)
        self.c2 = torch.from_numpy(c2).to(device)
        self.invw = inv[rows_t].unsqueeze(-1)               # (nb,R,1); 0 on fill


def batch_force_eager(Z, rows_r, nbrs, c1, c2, invw, R, k):
    """In-place-optimized eager path: (R,d) chunk forces for one batch."""
    Zc = Z.index_select(0, rows_r)                          # (R, d)
    Zj = Z.index_select(0, nbrs).view(R, k, -1)             # (R, k, d)
    zd = Zj - Zc.unsqueeze(1)
    r = (zd * zd).sum(-1).sqrt_()                           # (R, k)
    s = c1 * r
    r = r.neg_().exp_()
    s = s.addcmul_(c2, r, value=-1.0)                       # c1*r - c2*e^-r
    F = zd.mul_(s.unsqueeze(-1)).sum(1)                     # (R, d)
    return F.mul_(invw)


def batch_force_pure(Z, rows_r, nbrs, c1, c2, invw, R, k):
    """Functional variant (no in-place ops) for torch.compile."""
    Zc = Z.index_select(0, rows_r)
    Zj = Z.index_select(0, nbrs).view(R, k, -1)
    zd = Zj - Zc.unsqueeze(1)
    r = (zd * zd).sum(-1).sqrt()
    s = c1 * r - c2 * torch.exp(-r)
    F = (zd * s.unsqueeze(-1)).sum(1)
    return F * invw


def one_step(Z, dZ, tp, force_fn):
    """One force-update iteration, in place: Z += dZ; all shapes static."""
    dZ.zero_()
    for b in range(tp.nb):
        F = force_fn(Z, tp.rows_r[b], tp.nbrs[b],
                     tp.c1[b], tp.c2[b], tp.invw[b], tp.R, tp.h)
        dZ.index_add_(0, tp.rows_w[b], F)                   # fill -> row n
    Z.add_(dZ[: tp.n])


def build_stepper(Z, dZ, tp, device, use_graphs, use_compile):
    """Returns (step_fn, mode_str). step_fn() advances Z by one iteration."""
    force_fn, mode = batch_force_eager, "eager"

    if use_compile:
        try:
            compiled = torch.compile(batch_force_pure, dynamic=False)
            one_step(Z, dZ, tp, compiled)                   # triggers compilation
            force_fn, mode = compiled, "compile"
        except Exception as e:
            print(f"  (torch.compile unavailable -> eager fallback: "
                  f"{type(e).__name__}: {e})")

    if device.type != "cuda" or not use_graphs:
        return (lambda: one_step(Z, dZ, tp, force_fn)), mode
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):                              # warmup (mutates Z)
                one_step(Z, dZ, tp, force_fn)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            one_step(Z, dZ, tp, force_fn)
        return g.replay, (mode + "+graph" if mode == "compile" else "cuda-graph")
    except Exception as e:
        print(f"  (CUDA Graph capture failed -> {mode} fallback: "
              f"{type(e).__name__}: {e})")
        return (lambda: one_step(Z, dZ, tp, force_fn)), mode


def run_steps(step_fn, Z, Z0, n_steps, device):
    Z.copy_(Z0)
    for _ in range(n_steps):
        step_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()


# ----------------------------- chunked float64 oracle ------------------------
def reference_run(Z0, adj, C1, C2, n_steps, block=ORACLE_BLOCK):
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


# ----------------------------- benchmark sweep -------------------------------
def main():
    device = torch.device(DEVICE)
    print(f"torch {torch.__version__} | device: {device}", end="")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        print(f" | {p.name} cc {p.major}.{p.minor} | "
              f"VRAM free {free/2**20:.0f}/{total/2**20:.0f} MiB", end="")
        if free < 1_200 * 2**20:
            print("\n  WARNING: little free VRAM -- another process may be "
                  "holding the GPU (check nvidia-smi).", end="")
    else:
        print(f" | threads: {torch.get_num_threads()}", end="")
    print("\nconfig:")
    for name in ("M", "SEED", "D", "N_STEPS", "B_CELLS", "TILE_H", "TILE_W",
                 "TILE_MODE", "INTERLEAVE", "C1_SCALE", "C2_SCALE", "REPS",
                 "USE_CUDA_GRAPHS", "USE_COMPILE", "VALIDATE_MAX_N",
                 "ORACLE_BLOCK"):
        print(f"  {name:<15} = {globals()[name]}")
    print(f"  H_LIST          = {H_LIST}")
    print(f"  SWEEP_N         = {SWEEP_N}\n")

    rng = np.random.default_rng(SEED)
    results = []
    grad_ctx = torch.no_grad if USE_COMPILE else torch.inference_mode

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
        ref1 = refN = None                          # oracle cache: h-invariant

        for h in H_LIST:
            try:
                t0 = time.perf_counter()
                plan_np, inv_np, st = make_tile_plan(
                    adj, C1, C2, h, B_CELLS, TILE_MODE, TILE_W, INTERLEAVE)
                prep_s = time.perf_counter() - t0

                tp = TorchPlan(plan_np, inv_np, n, device)
                Z0 = torch.from_numpy(Z0np).to(device)
                Z = Z0.clone()
                dZ = torch.zeros((n + 1, D), dtype=torch.float32, device=device)

                with grad_ctx():
                    t0 = time.perf_counter()
                    step_fn, mode = build_stepper(Z, dZ, tp, device,
                                                  USE_CUDA_GRAPHS, USE_COMPILE)
                    run_steps(step_fn, Z, Z0, N_STEPS, device)  # first full run
                    warm_s = time.perf_counter() - t0
                    Zf = Z.detach().cpu().numpy().copy()

                    if n <= VALIDATE_MAX_N:
                        if ref1 is None:
                            ref1 = reference_run(Z0np, adj, C1, C2, 1)
                            refN = reference_run(Z0np, adj, C1, C2, N_STEPS)
                        run_steps(step_fn, Z, Z0, 1, device)
                        Z1 = Z.detach().cpu().numpy().copy()
                        err1 = float(np.max(np.abs(Z1 - ref1)))
                        errN = float(np.max(np.abs(Zf - refN)))
                        assert err1 < 1e-3, \
                            f"validation failed at n={n} h={h}: {err1:.2e}"
                        errs = f"err(1)={err1:.1e}  err({N_STEPS})={errN:.1e}"
                    else:
                        errs = "err=skipped (n > VALIDATE_MAX_N)"

                    t0 = time.perf_counter()
                    for _ in range(REPS):
                        run_steps(step_fn, Z, Z0, N_STEPS, device)
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
                      f"prep={prep_s*1e3:6.1f}ms  warmup+capture={warm_s:5.2f}s  "
                      f"[{mode}]  iter={per_iter_ms:8.3f}ms  "
                      f"{mcells:7.1f} Mcells/s  {errs}")
                results.append((n, h, st["cells"], per_iter_ms, mcells))

                del tp, Z, dZ, Z0
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            except (torch.cuda.OutOfMemoryError, MemoryError) as e:
                print(f"n={n:>7} h={h:>3}  SKIPPED: out of memory ({e})")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"n={n:>7} h={h:>3}  SKIPPED: out of memory")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise
        del ref1, refN

    print("\nsummary (steady-state, excludes warmup/capture):")
    print(f"{'n':>8} {'h':>4} {'cells':>9} {'ms/iter':>10} {'Mcells/s':>9}")
    for n, h, c, ms, mc in results:
        print(f"{n:>8} {h:>4} {c:>9} {ms:>10.3f} {mc:>9.1f}")


if __name__ == "__main__":
    main()
