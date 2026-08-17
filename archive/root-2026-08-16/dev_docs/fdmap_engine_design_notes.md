# FDMap force-update engine: design notes and reference

Engineering reference for the bucketed force-directed update kernel developed
and benchmarked across JAX and PyTorch on CPU / GPU (Maxwell, Turing) / TPU.
Intended for future development of the kernel function, the SELL-C-sigma
cell-budget layout, and memory management. All figures in this document are
measured, not estimated, unless marked as a model.

Companion files: `fdmap_bucketed_bench_jax.py`, `fdmap_bucketed_bench_torch.py`,
`probe_env.sh`, `probe_frameworks.py`.

---

## 1. Problem definition

Given a connected graph with `n` nodes, power-law degrees, adjacency `mask`
(sparse, binary, symmetric, static across iterations), sparse float matrices
`C1`, `C2` sharing `mask`'s exact sparsity pattern, and a dense embedding
`Z (n, d)` in float32, each iteration computes for every node `i`:

```
F_i  = sum over j in N(i) of  (Z_j - Z_i) * ( C1_ij * r_ij  -  C2_ij * exp(-r_ij) )
       where r_ij = || Z_j - Z_i ||
dZ_i = F_i / deg_i
Z   += dZ
```

repeated `N_STEPS` (10) times. With `m = nnz(mask)` directed cells, cost is
exactly `O(m * d)` per iteration: repulsion is masked to edges, so no
n-body approximation (Barnes-Hut etc.) is needed or used. Structurally this
is one round of message passing: per-cell gather, per-cell element-wise
scalar kernel, per-row reduction, dense update.

Two facts drive the whole design:

- The sparsity pattern is static, so arbitrary preprocessing (sorting,
  splitting, packing, relabeling) is paid once and amortized.
- The computation is memory-bound at every measured scale (arithmetic
  intensity ~3d+10 flops per ~20d bytes of traffic); every design decision
  is a memory-traffic decision.

## 2. The kernel function

### 2.1 Contract

The kernel is the per-cell scalar `s(r; c1, c2) = c1*r - c2*exp(-r)`, applied
element-wise to `(rows, k)` tiles of distances, then multiplied into `zd` and
reduced over `k`. Any replacement kernel must satisfy:

1. **Element-wise in `r` and the per-cell coefficients.** The layout supports
   any `s(r, c1, c2, ...)`; additional per-cell coefficient planes can be added
   as extra `(nb, R, k)` arrays in the plan with zero structural change.
2. **Exact zero on padding.** Pad cells carry `c1 = c2 = 0` and neighbor index
   = the row's own node, so `zd = 0` AND `s = 0` (double safety). A new kernel
   must return exactly 0 when all coefficients are 0, and must be finite at
   `r = 0`. If a future kernel divides by `r` (e.g. unit-direction forms),
   the self-index padding gives `r = 0` and will produce NaN — switch pad
   handling to an explicit mask or add an epsilon in that case.
3. **No autodiff requirement.** Forces are computed directly, never via
   `grad`, so non-differentiable points (e.g. `sqrt` at 0) are value-safe.

### 2.2 Sign / overflow note

The original prose spec wrote `exp(+||x||)`; the reference loop used
`exp(-znorm)`. All implementations follow the loop (`exp(-r)`), which is also
the numerically safe choice: `exp(-r) ∈ (0, 1]` for `r ≥ 0`, whereas
`exp(+r)` overflows float32 near `r ≈ 88`. If a positive-exponent kernel is
ever required, clamp or rescale first.

### 2.3 Do not split the kernel

`F = F_attract - F_repel` was considered and rejected: `C1` and `C2` share
one sparsity pattern, so splitting doubles the gather traffic of `Z` for zero
benefit. Splitting only pays if the two terms ever get *different* patterns
(e.g. all-pairs repulsion), which would change the algorithm class entirely.

### 2.4 Numerical behavior near equilibrium

As layouts converge, `F_i → 0` while individual messages stay O(1): the
per-row sum has inherent, benign float32 cancellation. This is unavoidable
and acceptable; what it *rules out* is any reduction scheme whose error
scales with more than the row's own terms (see cumsum-diff in §4).

### 2.5 Where the kernel lives in code

One expression per implementation; keep all four in sync when changing it:
JAX `run()` body, torch `batch_force_eager()`, torch `batch_force_pure()`
(the `torch.compile` variant, no in-place ops), and the numpy float64 oracle
`reference_run()`.

## 3. The layout: SELL-C-sigma with cell-budget slicing

### 3.1 Lineage

The scheme is a SELL-C-sigma variant (sort rows by length, slice, pad each
slice to its local max) with slice boundaries set by a **cell budget** rather
than a row count — independently re-derived in this project from a
batch-utilization argument, and matching CSR-Stream / CSR-Adaptive
(Greathouse & Daga) and merge-based equal-nnz partitioning (Merrill &
Garland) from the SpMV literature. The defining trade: a small bounded
padding cost buys a **dense, regular `sum(axis=1)` reduction** — fused,
deterministic, portable — instead of a segmented/scatter reduction.

### 3.2 Preprocessing (host, numpy, one-time)

```
1. hub split      rows with deg > K_MAX -> ceil(deg/K_MAX) virtual rows,
                  each a contiguous slice of the CSR row; owner recorded
2. width sort     stable argsort of virtual-row widths
3. ladder quant   k(row) = smallest ladder value >= width, ladder =
                  {1, ceil(1*r), ceil(...*r), ..., K_MAX}, base r = LADDER_BASE
4. pack per rung  cap = B_CELLS // k ; nb = ceil(rows/cap) ; R = ceil(rows/nb)
                  (BALANCED: R rows per batch, never a near-empty last batch)
5. pad            pad CELLS: neighbor = own node id, c1 = c2 = 0  -> exact 0
                  pad ROWS:  owner id = n                        -> dropped
```

Outputs per rung: `rows (nb, R) int32` (owner ids, `n` on pad rows),
`nbrs (nb, R, k) int32`, `c1, c2 (nb, R, k) f32`; plus `inv_deg_ext (n+1) f32`
with `inv_deg_ext[n] = 0` (pad rows self-neutralize even before the drop),
where `deg` is always the ORIGINAL degree — hub partial sums are each scaled
by `1/deg` and then summed, which is correct by linearity.

The step-4 balancing rule is load-bearing. The original greedy packing
(`R = B//k`, fill, pad the last batch) left near-empty final batches whose
pad rows do full-width work: measured padding jumped from ~25% to ~40% at
n ≥ 10k. Balancing restored padding to a flat, n-independent 12.6–13.7%
(base 1.5) across n = 1k..1M.

### 3.3 Per-iteration compute (per batch, all shapes static)

```
Zc = Z[rows]                    # (R, d)   centers: ONE read per ROW (broadcast)
Zj = Z[nbrs]                    # (R, k, d) neighbors: one read per CELL
zd = Zj - Zc[:, None, :]
r  = sqrt(sum(zd*zd, -1))       # (R, k)
s  = c1*r - c2*exp(-r)          # the kernel
F  = sum(zd * s[..., None], 1)  # (R, d)   dense axis-k reduction
F *= inv_deg_ext[rows][:, None]
dZ[rows] += F                   # per-ROW write; owner id n dropped/garbage-row
Z += dZ[:n]                     # after all batches
```

Properties worth preserving in any reimplementation:

- **Broadcast centers.** `Z` read traffic is `m * (1 + 1/k_mean)` row-reads,
  not `2m` — the rectangular layout is what deletes per-cell center gathers.
- **No scatter in the hot path.** The only indexed write is per-ROW (n writes
  per iteration, not m); rows are unique within a batch except for split-hub
  chunks that land in the same batch, so results are deterministic in
  practice and bit-identical across backends (verified: identical error
  columns across two GPUs, two torch versions, two jax versions).
- **Pad-drop mechanics.** JAX: `dZ.at[rows].add(F, mode="drop")`, center
  gather clamped. Torch: `dZ` allocated `(n+1, d)`; pad rows write into the
  garbage row `n`; `rows_r = rows.clamp(max=n-1)` precomputed for the gather.
- **Fusion is the entire performance story** (§9). XLA fuses the whole chain
  above into ~2 memory passes per cell; eager execution materializes every
  intermediate and pays 3.5–4x for it.

### 3.4 What the transformations correspond to

Row-reordering = the width sort (labels themselves are NOT permuted — batch
processing order is decoupled from node-label order, which leaves label order
free for a future locality relabeling, §11). Reshaping = hub splitting.
Row-wise batching = the cell budget `B_CELLS`.

## 4. Alternatives considered and rejected (decision record)

| Alternative | Why rejected |
|---|---|
| Sorted COO + `segment_sum` (= flat cell-chunks with per-cell scatter) | JAX's `segment_sum` lowers to `scatter_add`; XLA GPU scatter is atomics-based and non-deterministic, the deterministic flag is catastrophically slow, sorted ids measured *slower* than shuffled in fp32, and scatter is the weak path on TPU. Kept conceptually as the GPU baseline; the padded layout matched or beat it everywhere it mattered. |
| Flat chunk + cumsum-then-diff segmented reduce | Deterministic and scatter-free, but adds a full extra memory pass and its error scales with the whole chunk's prefix magnitude — worst exactly near equilibrium where row sums are small (§2.4). |
| Algebraic factorization `F_i = Σ s_ij Z_j − (Σ s_ij) Z_i` (SpMM + row-scale) | Valid identity, only `(m,)` intermediates; keep in the toolbox. But computing `r` matrix-style needs SDDMM: JAX's `bcoo_dot_general_sampled` default path materializes the FULL dense product (disqualifying at n x n), and the Gram identity `r² = |zi|²+|zj|²−2⟨zi,zj⟩` cancels catastrophically for close neighbors — precisely the pairs attraction creates. PyTorch's `torch.sparse.sampled_addmm` (cuSPARSE SDDMM) is real if ever needed. |
| True fused CSR-Stream custom kernel | Requires Pallas (GPU path = Triton, needs cc >= 7.0/8.0; unavailable on the sm_52 dev card) or hand CUDA. Remains the correct future step for eager-torch parity (§11). |
| `jax.experimental.sparse` as the vehicle | Kernel is nonlinear in gathered features, not expressible as one SpMM; module is experimental with API subject to change. |
| Kernel split `F_a − F_r` | §2.3. |

## 5. Tunables and tuning rules

| Knob | Default | Rule |
|---|---|---|
| `B_CELLS` | 16384 | **Inverts with d.** Small d (launch/granularity-bound): raise it (candidate 131072 at d=2 for eager torch; also worth sweeping 64–256k for JAX GPU small-n). Large d (bandwidth-bound): lower it so per-batch transients fit L2 — rule of thumb `B_CELLS * d * 4 * ~3 <= L2` → ~512–1024 on a 1 MB-L2 card, ~2–4k on 4 MB, at d=128. Untested; highest-value sweep outstanding. |
| `LADDER_BASE` | 1.5 | Finer ladder = less padding, more compiled shapes. Measured: 1.5 → ~13% pad, 2.0 → ~25%, 1.3 → ~7.4%. 1.5 is right at BOTH extremes of d (small d: index/coeff stream dominates per cell; large d: bandwidth-bound so pad% is a direct throughput tax). The early heuristic "2.0 for d >= 32" was measured wrong and retracted. Coarser (2.0) only wins in dispatch-bound regimes (measured slightly faster on CPU and plausibly small-n GPU). |
| `K_MAX` | 256 | Hub-split threshold = top rung width. BA graphs trip it from n≈10k (max deg ~ 3·sqrt(n)); split count grew 0→196 over n=1k→1M with zero pad regression. |
| `N_STEPS`, `REPS` | 10, 5 | Loop count inside one jit / one graph replay set; timing repetitions. |
| `VALIDATE_MAX_N` | 200000 | Oracle cap (host RAM guard). |
| `ORACLE_BLOCK` | 131072 | Edges per oracle chunk; peak extra host RAM ≈ `block * d * 8 * ~3` (~400 MB at d=128). |

Env overrides (both scripts): `FDMAP_SWEEP="1000,5000,..."`, `FDMAP_D=128`;
torch additionally `FDMAP_DEVICE=cpu|cuda`, `FDMAP_COMPILE=1`.

## 6. Memory management

### 6.1 Device working-set model

- Embedding state: `Z = 4nd` bytes; steady state holds ~4–5 Z-sized buffers
  (input, loop carry, `dZ`, sum temporary; +1 without input donation).
- Plan storage: ~12 B per padded cell in JAX (int32 idx + 2 f32 coeffs);
  torch doubles the index component (indexing requires int64) plus an
  `invw` plane.
- Transient per batch: `B_CELLS * d * 4` per materialized intermediate.
  Under XLA fusion these mostly never reach HBM; under eager torch every
  one of them does.
- CUDA Graphs **pin** their memory pool for the life of the graph — that is
  the price of the launch-overhead win. Measured consequence: torch OOMed at
  n=1M / d=128 on the 2 GB card (490 MiB request, 1.14 GiB reserved) where
  JAX had survived the same size in an earlier run.

### 6.2 Measured ceilings and the fragmentation lesson (2 GB Maxwell, d=128)

The first JAX sweep died at n=200k with a Z-sized allocation failing inside
a fragmented BFC pool (`**_..xxx` map) even though the true working set
(~0.5 GiB) fit. Fix that recovered n=200k AND n=400k: delete all per-size
arrays and call `jax.clear_caches()` between sizes (drops stale executables
and their constants). Remaining ceiling is real: n=700k needs a 343 MiB
buffer against Z=350 MiB state and fails honestly; fp32 ceiling for 2 GB at
d=128 sits between 400k and 700k. Levers beyond that, unimplemented:
bf16 storage of Z with fp32 compute (halves state), input donation, and
`TF_GPU_ALLOCATOR=cuda_malloc_async` (XLA's own fragmentation suggestion).
Both benches now catch per-size OOM, skip, clean up, and continue.

### 6.3 Host RAM: the oracle post-mortem

The original float64 oracle materialized `(m, d)` float64 edge arrays; at
n=700k, d=128 that is ~4.3 GB for `zd` alone and ~13 GB peak — which
OOM-killed the Colab GPU VM (misdiagnosed at first as a time limit; the TPU
VM survived only because it has more host RAM). Permanent fix in both
scripts: stream the oracle in `ORACLE_BLOCK`-edge chunks (bit-identical
result to ~fp64 eps; verified against original error values) and cap
validation at `VALIDATE_MAX_N`. Preprocessing itself is host-linear
(~3.5–5 s at n=1M; the python fill loop is vectorizable if it ever matters).

### 6.4 Operational checklist for small-VRAM benchmarking

Kill GPU-squatting processes first (a leftover session holding 1537/2048 MiB
produced instant OOMs); check `nvidia-smi`. The torch bench warns at startup
if free VRAM < 1.2 GiB.

## 7. Implementations

### 7.1 JAX (`fdmap_bucketed_bench_jax.py`)

Entire `N_STEPS` loop inside one `jax.jit` (`lax.fori_loop`); per rung a
`lax.scan` over stacked batches; rung loop unrolled (~10–14 fixed shapes,
compile ~1–3 s, size-independent). Core ops only (gather / element-wise /
axis reductions / per-row indexed add with `mode="drop"`) → runs unchanged
on CPU, GPU, TPU. This is the reference implementation and the performance
ceiling among tested runtimes.

### 7.2 PyTorch (`fdmap_bucketed_bench_torch.py`)

Same preprocessing verbatim. Four execution paths, reported per row:

- `eager` — in-place-optimized kernels (~13 launches/batch), `inference_mode`.
- `cuda-graph` — the whole iteration (all rungs/batches) captured once via
  the side-stream warmup recipe, replayed per step. Kills CPU dispatch; does
  NOT reduce kernel count or memory traffic; pins memory (§6.1).
- `compile` (`FDMAP_COMPILE=1`) — per-batch math through
  `torch.compile(dynamic=False)` on the pure (non-in-place) kernel variant;
  `index_add_` stays outside the compiled region. Recompiles per rung shape
  and per sweep size (R varies with n): warmup cost is real wall-clock,
  excluded from steady-state.
- `compile+graph` — compiled kernels additionally graph-captured (all shapes
  are compiled during warmup, so capture is safe). Closest analogue to XLA.

Fallback chain is automatic and printed. Triton floor: cc >= 7.0
historically, >= 8.0 in the newest releases — sm_52 is always out;
whether a T4 (cc 7.5) compiles depends on the installed torch/Triton
vintage. Compatibility constraints in §10.

## 8. Validation methodology

Chunked float64 numpy oracle evaluating the problem statement directly from
CSR; checks `max|Δ|` after 1 step (asserted < 1e-3; measured ~1.5e-7 at d=2,
~5e-7 at d=128 ≈ fp32 resolution of the sums) and after `N_STEPS`
(reported; ~3–8e-7). Multi-step drift reflects summation-order divergence
amplified through iterations, not error in either side. Cross-backend
observations worth relying on: the two GPU runs (GTX 950 / torch 2.0.1 vs
T4 / torch 2.11; likewise JAX 0.6.2 vs 0.7.2) produce IDENTICAL error
columns at every n — the no-scatter tree-reduction design is bit-stable
across hardware and framework versions. TPU differs slightly
(transcendental implementations), stable at ~1–2.5e-6 over 10 steps.

## 9. Measured performance and the model that explains it

All figures: BA graph m=3, cells = 6n−18, float32, steady-state Mcells/s
(real cells per second per iteration, padding excluded from the numerator).

### 9.1 d = 2

| backend | peak (at n) | at n=1M | shape of curve |
|---|---|---|---|
| i7-6700K, JAX CPU | 164 @20k | 68.5 | knee when Z=8n B leaves L2; declines through L3/DRAM |
| GTX 950, JAX | 520 @50k | 365 | knee at n≈100–125k ⇔ Z ≈ 1 MB = Maxwell L2, exactly |
| GTX 950, torch cuda-graph | 173 @100k | 166 | flat tail: already HBM-resident at all n, no cache tier to fall off |
| Tesla T4, JAX | 716 @700k | 677 | still rising; 4 MB L2 + latency hiding defer the knee |
| Colab TPU, JAX | 160 @10k | 57 (floor) | gather-throughput-bound; 8 B rows use ~2/128 lanes per tile |

Model: `Z = 8n` bytes vs cache size predicts every knee. Small-n is
dispatch-bound (~150 launches/iter under jit; bare metal beats virtualized
Colab below n≈10k). Torch's 3x gap to JAX on identical silicon at d=2 is
kernel-granularity: ~5400 launches/iter at n=1M → ~6.7 µs/kernel, the floor
for tiny kernels on 6 SMs — CUDA graphs removed the CPU side of dispatch but
not the GPU-side fixed cost per kernel.

### 9.2 d = 128

| backend | plateau | ceiling / notes |
|---|---|---|
| GTX 950, JAX | 39–41.6 | OOM past 400k (§6.2). ≈ 100 GB/s effective ≈ card's 105 GB/s with ~2 passes/cell |
| GTX 950, torch graph | 9.4–9.8 | OOM at 1M (graphs pin memory). ~4.6 KB/cell ≈ 9 passes |
| Tesla T4, JAX | 87–105 (104.8 @1M) | full sweep clean with chunked oracle |
| Tesla T4, torch graph | 22–31.2 | flat from 200k |
| Colab TPU, JAX | 127 @10k → 35.7 @1M | knee ≈ n=40k (Z ≈ 20 MB crossing an on-chip tier; unprofiled) |

The load-bearing ratios: JAX/torch fusion tax = 4.1x (950) and 3.4x (T4) —
**the eager materialization tax at d=128 is ~3.5–4x, independent of GPU
generation**. Torch-to-torch across cards = 3.2 ≈ bandwidth ratio 2.9
(300/105 GB/s): both eager runs sit on their memory bus. JAX/JAX across
cards = 2.6, same story with fewer passes. FLOPs are irrelevant throughout
(~40 GFLOP/s used of 8 TFLOP/s available on T4).

TPU across d: Mcells/s fell 57→36, but effective gather bandwidth rose
0.46 → 18.3 GB/s (40x) because a 512 B row finally fills one 128-lane×4 B
tile — confirming the lane-utilization hypothesis — yet 18 GB/s is still
~2% of v5e-class HBM: the gather machinery, not the memory system, remains
the TPU limiter. TPU is worthwhile only with d >= 64 AND index locality
(§11); at d=2 it loses to the desktop CPU at n=1M.

### 9.3 Headline absolutes

n=1M, d=2: 36 ms/iter on the GTX 950 (torch) / 16.4 ms (T4, JAX) — a
1000-iteration layout of a million-node graph ≈ 9 s of T4 compute at d=2,
~57 s at d=128.

## 10. Environment and compatibility matrix

| Item | Fact |
|---|---|
| Dev box | Ubuntu 22.04, i7-6700K (4C/8T, 1 MB L2 / 8 MB L3), 16 GB RAM, GTX 950 2 GB (sm_52, 6 SMs, ~105 GB/s, 1 MB L2), driver 535.309.01 / CUDA 12.2, `/usr/local/cuda-11.7` toolkit on disk |
| JAX env (FDMap venv) | jax 0.6.2 + jax-cuda12-plugin, cu12 pip wheels (cudnn 9.24) |
| Torch on the dev box | torch 2.0.1+cu117 (system site-packages). Runs on sm_52 because its `sm_50` cubins are binary-compatible: NVIDIA guarantees a cubin runs on any GPU with the same major and >= minor compute capability. **Requires numpy < 2** — the 2.0.1 wheel is built against the NumPy 1.x C-API and crashes under numpy 2.x ("Numpy is not available"); pin `numpy<2` or use a dedicated torch venv. |
| Do NOT upgrade torch casually | torch 2.8 dropped sm_50–sm_60 from cu12.8/12.9 builds (binary size); PyPI defaults for recent versions are those builds; CUDA 13 removed Maxwell/Pascal/Volta from the toolkit entirely. Narrow escape hatch: 2.8+cu126 kept sm_50; it buys nothing for this workload. |
| torch.compile | Needs Triton: cc >= 7.0 historically, >= 8.0 in newest Triton. sm_52 never; T4 (7.5) vintage-dependent — the script prints which path actually ran. |
| Colab (as measured) | GPU runtime: T4 16 GB cc 7.5, jax 0.7.2, torch 2.11.0+cu128 (Triton 3.6). TPU runtime: single-core v5e-class, jax 0.7.2. Host RAM: GPU VM ~12.7 GB (killed the unchunked oracle), TPU VM larger. |
| torch on TPU | Deliberately out of scope: torch_xla is a genuine port (lazy tensors, step markers, no CUDA graphs) and would mostly re-measure XLA anyway. TPU column is JAX-only. |
| Determinism flags | XLA GPU scatter is non-deterministic; `--xla_gpu_deterministic_ops` is orders-of-magnitude slow — the design avoids needing either. |

## 11. Roadmap (ordered by expected value)

1. **Compiled-torch on a Triton-capable GPU** (`FDMAP_COMPILE=1`, L4/A100 if
   the T4 refuses). The direct test of "the gap is fusion, not framework";
   prediction on record: `compile+graph` lands near the JAX curve. Pending.
2. **`B_CELLS` sweeps in both directions** (§5): up at d=2 for eager torch
   (predicted plateau 170 → 300+ on the 950), down at d=128 for everything
   (L2-resident transients). One-constant experiments with falsifiable
   predictions.
3. **Locality relabeling** (reverse Cuthill-McKee,
   `scipy.sparse.csgraph.reverse_cuthill_mckee`): relabel node ids so
   neighbor indices cluster; storage order is already decoupled from the
   degree-sorted processing order, so this is preprocessing-only. Attacks
   the gather-locality tails on CPU/GTX 950 and is the main hope for TPU.
4. **Hand-fused CUDA kernel** via `torch.utils.cpp_extension` against the
   on-disk CUDA 11.7 (still targets sm_52): gives eager torch the ~2-pass
   traffic profile and reduces the comparison to pure framework overhead.
5. **bf16 storage of Z, fp32 compute**: doubles the small-VRAM size ceiling
   (Maxwell has no fast fp16 compute — storage-only cast).
6. Smaller items: vectorize the preprocessing fill loop (~10x, matters at
   n >= 1M); input donation in JAX; `TF_GPU_ALLOC=cuda_malloc_async` /
   `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb` experiments; optional
   `lax.scan` variant emitting per-step Z snapshots for epoch-slider
   visualization; Pallas-TPU CSR-Stream if TPU ever becomes a target.

## 12. Running the benchmarks

```
# JAX (CPU/GPU/TPU, auto)
python fdmap_bucketed_bench_jax.py
FDMAP_D=128 FDMAP_SWEEP="1000,100000,1000000" python fdmap_bucketed_bench_jax.py

# torch
python fdmap_bucketed_bench_torch.py                 # eager / cuda-graph
FDMAP_COMPILE=1 python fdmap_bucketed_bench_torch.py # compile / compile+graph
FDMAP_DEVICE=cpu python fdmap_bucketed_bench_torch.py
```

Both scripts share seeds, graph generation (drop-in `generate_ba_graph` /
`build_sparse_adjacency` stand-ins with the project's signatures), plan
construction, validation, and line format; per-size OOM skips and continues.
`prep` recurs per process; `compile` / `warmup+capture` columns are excluded
from steady-state and are per-size (torch compile path recompiles per n).

## 13. Sources for the compatibility and platform claims

- NVIDIA CUDA architecture compatibility guides (cubin same-major /
  >=-minor rule): docs.nvidia.com/cuda/ada-compatibility-guide
- PyTorch 2.8 release notes and issue #157517 (Maxwell/Pascal removal from
  cu12.8/12.9 builds; cu12.6 arch lists)
- dev-discuss.pytorch.org: "CUDA toolkit version and architecture support
  update" (2.8 policy change)
- PyTorch forums: "torch.compile, Triton cuda capability" (Triton >= 7.0);
  newer Triton documentation states >= 8.0
- jax.ops.segment_sum documentation (implementation via scatter-add;
  `num_segments` static under jit; `bucket_size` stability option)
- openxla determinism documentation and jax#17844
  (`--xla_gpu_deterministic_ops` slowdown); jax#26227 (sorted-vs-shuffled
  scatter fp32 measurement)
- jax.experimental.sparse source: `bcoo_dot_general_sampled` dense-product
  default path; module marked experimental
- NumPy 2.0 troubleshooting guide (NumPy 1.x-compiled extensions under 2.x)
- Greathouse & Daga, CSR-Adaptive (SC'14); Merrill & Garland, merge-based
  SpMV (SC'16); SELL-C-sigma: Kreutzer et al.
