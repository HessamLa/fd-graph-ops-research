#!/usr/bin/env python3
"""
fdmap_bucketed_bench_torch.py

PyTorch port of the bucketed/padded (SELL-C-sigma style) force-update
benchmark. Preprocessing, graph generation, seeds, batch plan, and printout
format are IDENTICAL to fdmap_bucketed_bench_jax.py, so results compare
apples-to-apples.

Execution paths (reported per-row as [mode]):
  eager          plain eager kernels (CPU, or graphs disabled/failed)
  cuda-graph     eager kernels, whole iteration captured as one CUDA Graph
  compile        FDMAP_COMPILE=1: per-batch math fused by torch.compile
  compile+graph  compiled kernels additionally captured as one CUDA Graph

torch.compile requires Triton on GPU (compute capability >= 7.0; newest
Triton needs >= 8.0) -- on unsupported devices it falls back automatically
and says so. On the GTX 950 use the default eager+graph path.

Local note: torch 2.0.1+cu117 wheels are built against the NumPy 1.x C-API;
run with numpy<2 (or in the dedicated torch venv).

Env overrides: FDMAP_SWEEP="1000,5000"  FDMAP_D=128  FDMAP_DEVICE=cpu|cuda
               FDMAP_COMPILE=1
"""

import os
import time

import numpy as np
import scipy.sparse as sp
import torch

# ----------------------------- config (mirrors JAX script) -------------------
M            = 3
SEED         = 42
D            = 128
N_STEPS      = 10
B_CELLS      = 16_384
K_MAX        = 256
LADDER_BASE  = 1.5
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
def build_ladder(k_max, base):
    ks = [1]
    while ks[-1] < k_max:
        ks.append(min(k_max, max(ks[-1] + 1, int(np.ceil(ks[-1] * base)))))
    return np.asarray(ks, dtype=np.int64)


def make_plan(adj, C1, C2, b_cells=B_CELLS, k_max=K_MAX, ladder_base=LADDER_BASE):
    n = adj.shape[0]
    A = adj.tocsr()
    A.sort_indices()
    indptr, indices = A.indptr, A.indices
    deg = np.diff(indptr).astype(np.int64)

    assert C1.nnz == A.nnz and np.array_equal(C1.indptr, indptr)
    assert C2.nnz == A.nnz and np.array_equal(C2.indptr, indptr)
    c1d = C1.data.astype(np.float32)
    c2d = C2.data.astype(np.float32)

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

    order = np.argsort(lens, kind="stable")
    owners, starts, lens = owners[order], starts[order], lens[order]
    ladder = build_ladder(k_max, ladder_base)
    kq = ladder[np.searchsorted(ladder, lens)]

    rungs, padded = [], 0
    for k in np.unique(kq):
        sel = np.where(kq == k)[0]
        k = int(k)
        cap = max(1, b_cells // k)
        nb = -(-len(sel) // cap)
        R = -(-len(sel) // nb)
        rows = np.full(nb * R, n, dtype=np.int32)
        nbrs = np.zeros((nb * R, k), dtype=np.int32)
        c1 = np.zeros((nb * R, k), dtype=np.float32)
        c2 = np.zeros((nb * R, k), dtype=np.float32)
        for slot, v in enumerate(sel):
            o, s0, L = int(owners[v]), int(starts[v]), int(lens[v])
            rows[slot] = o
            nbrs[slot, :L] = indices[s0:s0 + L]
            nbrs[slot, L:] = o
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


# ----------------------------- torch runtime ---------------------------------
class TorchPlan:
    """Device-resident batch plan with precomputed index/scale tensors."""

    def __init__(self, plan_np, inv_deg_ext, n, device):
        self.n = n
        inv = torch.from_numpy(inv_deg_ext).to(device)
        self.rungs = []
        for rows, nbrs, c1, c2 in plan_np:
            nb, R, k = nbrs.shape
            rows_t = torch.from_numpy(rows.astype(np.int64)).to(device)
            self.rungs.append(dict(
                R=R, k=k, nb=nb,
                rows_w=rows_t,                                  # (nb,R) incl. id n
                rows_r=rows_t.clamp(max=n - 1),                 # for the Z gather
                nbrs=torch.from_numpy(nbrs.astype(np.int64))
                     .reshape(nb, R * k).to(device),
                c1=torch.from_numpy(c1).to(device),
                c2=torch.from_numpy(c2).to(device),
                invw=inv[rows_t].unsqueeze(-1),                 # (nb,R,1); 0 on pads
            ))


def batch_force_eager(Z, rows_r, nbrs, c1, c2, invw, R, k):
    """In-place-optimized eager path: (R,d) row forces for one batch."""
    Zc = Z.index_select(0, rows_r)                              # (R, d)
    Zj = Z.index_select(0, nbrs).view(R, k, -1)                 # (R, k, d)
    zd = Zj - Zc.unsqueeze(1)
    r = (zd * zd).sum(-1).sqrt_()                               # (R, k)
    s = c1 * r
    r = r.neg_().exp_()
    s = s.addcmul_(c2, r, value=-1.0)                           # c1*r - c2*e^-r
    F = zd.mul_(s.unsqueeze(-1)).sum(1)                         # (R, d)
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
    for rg in tp.rungs:
        R, k = rg["R"], rg["k"]
        for b in range(rg["nb"]):
            F = force_fn(Z, rg["rows_r"][b], rg["nbrs"][b],
                         rg["c1"][b], rg["c2"][b], rg["invw"][b], R, k)
            dZ.index_add_(0, rg["rows_w"][b], F)                # pads -> row n
    Z.add_(dZ[: tp.n])


def build_stepper(Z, dZ, tp, device, use_graphs, use_compile):
    """Returns (step_fn, mode_str). step_fn() advances Z by one iteration."""
    force_fn, mode = batch_force_eager, "eager"

    if use_compile:
        try:
            compiled = torch.compile(batch_force_pure, dynamic=False)
            one_step(Z, dZ, tp, compiled)                       # triggers compilation
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
            for _ in range(3):                                  # warmup (mutates Z)
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
    for name in ("M", "SEED", "D", "N_STEPS", "B_CELLS", "K_MAX", "LADDER_BASE",
                 "C1_SCALE", "C2_SCALE", "REPS", "USE_CUDA_GRAPHS", "USE_COMPILE",
                 "VALIDATE_MAX_N", "ORACLE_BLOCK"):
        print(f"  {name:<15} = {globals()[name]}")
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

            t0 = time.perf_counter()
            plan_np, inv_np, st = make_plan(adj, C1, C2)
            prep_s = time.perf_counter() - t0

            tp = TorchPlan(plan_np, inv_np, n, device)
            Z0 = torch.from_numpy(
                rng.standard_normal((n, D)).astype(np.float32)).to(device)
            Z = Z0.clone()
            dZ = torch.zeros((n + 1, D), dtype=torch.float32, device=device)

            with grad_ctx():
                t0 = time.perf_counter()
                step_fn, mode = build_stepper(Z, dZ, tp, device,
                                              USE_CUDA_GRAPHS, USE_COMPILE)
                run_steps(step_fn, Z, Z0, N_STEPS, device)      # first full run
                warm_s = time.perf_counter() - t0
                Zf = Z.detach().cpu().numpy().copy()

                if n <= VALIDATE_MAX_N:
                    run_steps(step_fn, Z, Z0, 1, device)
                    Z1 = Z.detach().cpu().numpy().copy()
                    Z0np = Z0.cpu().numpy()
                    err1 = float(np.max(np.abs(
                        Z1 - reference_run(Z0np, adj, C1, C2, 1))))
                    errN = float(np.max(np.abs(
                        Zf - reference_run(Z0np, adj, C1, C2, N_STEPS))))
                    assert err1 < 1e-3, f"validation failed at n={n}: {err1:.2e}"
                    errs = f"err(1)={err1:.1e}  err({N_STEPS})={errN:.1e}"
                else:
                    errs = "err=skipped (n > VALIDATE_MAX_N)"

                t0 = time.perf_counter()
                for _ in range(REPS):
                    run_steps(step_fn, Z, Z0, N_STEPS, device)
                per_iter_ms = (time.perf_counter() - t0) / REPS / N_STEPS * 1e3

            mcells = st["cells"] / (per_iter_ms * 1e-3) / 1e6
            print(f"n={n:>7}  cells={st['cells']:>8}  rungs={st['rungs']:>2}  "
                  f"split_hubs={st['n_split']}  pad={st['pad_frac']*100:4.1f}%  "
                  f"prep={prep_s*1e3:6.1f}ms  warmup+capture={warm_s:5.2f}s  "
                  f"[{mode}]  iter={per_iter_ms:8.3f}ms  {mcells:7.1f} Mcells/s  "
                  f"{errs}")
            results.append((n, st["cells"], per_iter_ms, mcells))

            del tp, Z, dZ, Z0
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except (torch.cuda.OutOfMemoryError, MemoryError) as e:
            print(f"n={n:>7}  SKIPPED: out of memory ({e})")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"n={n:>7}  SKIPPED: out of memory")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise

    print("\nsummary (steady-state, excludes warmup/capture):")
    print(f"{'n':>8} {'cells':>9} {'ms/iter':>10} {'Mcells/s':>9}")
    for n, c, ms, mc in results:
        print(f"{n:>8} {c:>9} {ms:>10.3f} {mc:>9.1f}")


if __name__ == "__main__":
    main()
