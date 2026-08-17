# TODOS — `fdge_jax_sell_c_sigma`

Open work items, roughly in priority order. Items already done in the
2026-08-08 refactor pass are tracked in `logs/master.log`, not here.

For *why* a decision was made — including the alternatives that were tried
and rejected, and the things that turned out wrong — see
[`journal/`](journal/README.md). Several items below have a matching
journal entry that carries reasoning this file deliberately omits.

---

## 1. The complexity problem (highest priority)

### 1.1 The two force-calculation modes

Force-directed embedding here can compute forces over either:

1. **Every node pair.** Distance `d(u_i, u_j)` is defined for all `(i, j)`,
   which is a complete graph on `n` nodes — a dense `n x n` matrix. No
   pairing policy needed; every pair contributes a force term.
2. **A chosen subset of node pairs.** Augmentation adds only a fraction of
   the possible edges and weights each with `d(u_i, u_j)`. Which pairs get
   an edge is a *policy* decision, made in the augmentation stage.

**Measured result so far: mode 2 matches mode 1's embedding quality at a
small fraction of the memory.** That is the whole reason this package is
built on a sparse `D`. Both modes must stay available — mode 1 is much
simpler (no policy to design) and is the correctness reference.

Every memory-related decision below must be evaluated against *both* modes.

### 1.2 Naming these two modes

`D` is already overloaded with `is_sparse`, which refers to the *storage
container*, not to which pairs exist. Do not reuse "sparse"/"dense" for
this axis — that collision is exactly what makes the current code
confusing.

Recommended: a single `coupling` parameter on `augment_graph`.

| Value | Meaning |
|---|---|
| `coupling="complete"` | every pair is coupled by a force (mode 1) |
| `coupling="selective"` | only policy-chosen pairs are coupled (mode 2) |

Rationale: "coupling" is the physics term for which pairs exert force on
each other, it is orthogonal to storage, and it reads correctly next to
`is_sparse`. Second choice, if a more literal name is preferred:
`pair_policy="complete" | "selected"`.

There is established precedent for the distinction itself (not for these
names): the graph-drawing literature calls mode 1 the **full stress
model** and mode 2 the **sparse stress model** — Ortmann, Klimenta &
Brandes, *A Sparse Stress Model*, JGAA 21(5):791–821, 2017
([arXiv:1608.08909](https://arxiv.org/abs/1608.08909)). Worth reading
before designing new policies; their term-aggregation approach to picking
representatives is directly relevant. Their naming is only rejected here
because "sparse" is already taken in this codebase.

### 1.3 OOM at 100k nodes — must fix

The current implementation raises OOM at 100k nodes. This is fully
explained, not mysterious.

**Measured on this machine: 15 GB total RAM, ~7 GB free — a dense
`(n, n)` float64 matrix therefore tops out at roughly n = 30k.** The 100k
OOM is not a leak or a subtle bug; it is `graph_building` asking for 80 GB
on a 15 GB box. Any fix has to remove the dense allocation, not shrink it.

Known `O(n^2)` sites, in the order they will bite:

- ~~`graph_building/building_blocks.mst_edges` — `squareform(pdist(data))`
  materializes a dense `(n, n)` float64 distance matrix (80 GB at n=100k,
  2 TB at n=500k).~~ **FIXED 2026-08-08.** Now an MST over an exact
  `sklearn` kNN graph, plus explicit component bridging. Measured:
  n=100,000 in 27 s at **59 MB peak** (was 80 GB); n=30k at 17.9 MB, a
  401x reduction. Verified connected and **exactly equal in total weight**
  to the dense MST on clustered fixtures.
  Non-obvious finding worth keeping: raising `k` does NOT fix kNN-graph
  disconnection on well-separated clusters — no cross-cluster point is ever
  a k-nearest neighbour at any `k`. Connectivity comes from a centroid-MST
  bridging pass with one exact 1-NN query per bridge, not from a bigger `k`.
- `graph_augmenting/hopfill.augment_graph` — `O(n^2)` by construction; it
  stores every reachable pair plus an explicit sentinel for every
  disconnected pair. This IS mode 1, so it is expected to be quadratic —
  but it should fail with a clear, early, actionable error at large `n`,
  not an opaque OOM deep in a BFS.
- `embedding` coefficient derivation — the shell-count table was `(n,
  max_hop+1)`, which on a *disconnected* graph became `(n, n+1)` because
  hopfill's unreachable sentinel is `n` itself. Fixed in the 2026-08-08
  pass by compacting hop values; re-check if the sentinel convention
  changes.

Action items:
- [ ] Decide and document a hard `n` ceiling for `coupling="complete"`,
      and raise a clear error above it instead of OOM-ing. **This is now the
      binding constraint**: with `graph_building` fixed, dense `hopfill` is
      the remaining `O(n^2)` site on the default path.
- [x] Replace the dense-`pdist` MST with a scalable one. *(done 2026-08-08)*
- [ ] Add a memory-ceiling regression test at ~100k nodes to CI so this
      cannot silently regress again.
- [ ] Audit peak *device* memory separately from host memory (see §2). Note
      the dev GPU here is a 2 GB GTX 950 shared with a Jupyter kernel, so
      device-memory headroom is far tighter than host headroom and CUDA
      init intermittently fails; `JAX_PLATFORMS=cpu` is the workaround.

### 1.4 New `coupling="selective"` policies to explore

The existing policy is bounded-hop-radius BFS (`graph_augmenting/
sparse_hops.py` — read its docstring for why bounded-radius beats kNN for
a shell-averaged force law: it keeps every shell it keeps *complete*, so
`|S_h(u)|` stays exact).

`ForceDirected.augment_graph` (`core/force_directed.py`) is the intended
extension point — a researcher subclasses and overrides just that method.

Planned:
- [ ] **k-hop balls** — connect all pairs at most `k` hops apart. (This is
      what `sparse_hops` already does; formalize it under the new
      `coupling` naming and make `k` the primary knob.)
- [ ] **k-hop balls + high-degree hubs** — add edges between high-degree
      nodes on top of the ball, to restore the long-range coupling that a
      pure radius cut removes.
- [ ] **k-hop balls + inter-community edges** — detect communities, then
      add a sample of edges between them. Restores global structure at
      near-linear cost. Needs a community detection dependency (see §3).
- [ ] **Sampled negatives** for repulsion — the documented Q4 risk: under a
      selective coupling, two graph-far nodes have no stored pair, so they
      exert zero repulsion even if they collide in embedding space. This is
      an `embedding/`-stage mitigation, not an augmentation one.

---

## 2. JAX profiling, behind a flag

Add opt-in profiling to the embedding stage. Off by default, zero cost when
off; enabled by an argument (e.g. `embed(..., profile="/tmp/fdge-trace")`
or a `profile: bool | str` constructor arg).

Metrics wanted: device memory, host/device I/O, and CPU/GPU/TPU utilization.

Verified available in the installed JAX (0.6.2, CUDA device present):

| API | Present | Use |
|---|---|---|
| `jax.profiler.start_trace(dir, profiler_options=...)` / `stop_trace()` | yes | explicit on/off — matches a flag cleanly |
| `jax.profiler.trace(dir)` context manager | yes | wrap the epoch loop |
| `jax.profiler.ProfileOptions()` | yes | `host_tracer_level` 0–3, `device_tracer_level` 0–1, `python_tracer_level` 0–1 |
| `jax.profiler.StepTraceAnnotation` | yes | label each epoch as a step — gives per-epoch breakdown |
| `jax.profiler.annotate_function` | yes | label `make_plan` / `_step` / drop separately |
| `jax.profiler.save_device_memory_profile("mem.prof")` | yes | device memory, viewed with `pprof --web` |
| `jax.live_arrays()` | yes | cheap always-on leak check, no pprof needed |

Notes that will matter when implementing:
- Call `.block_until_ready()` before stopping a trace or saving a memory
  profile — JAX dispatch is async, so an unblocked trace measures dispatch,
  not execution.
- `jit`-compiled functions are opaque to the memory profiler: everything
  inside `_step` is attributed to `_step` as a whole. To attribute within
  it, use `annotate_function` / `TraceAnnotation`, not the memory profile.
- `pprof --web --diff_base a.prof b.prof` diffs two snapshots — the right
  tool for "which epoch leaked".
- Viewing: `tensorboard --logdir=<dir>` or `xprof --port 8791 <dir>`;
  `jax.profiler.trace(..., create_perfetto_link=True)` for Perfetto.
- The benchmark (`validation/bench_embedding_perf.py`) already isolates
  epoch-1 compile time from steady state — profiling must preserve that
  split, or the trace will be dominated by XLA compilation.

Action items:
- [ ] `profile` flag threaded through `ForceDirected.embed`, off by default.
- [ ] `StepTraceAnnotation` per epoch; `annotate_function` on plan build,
      kernel, and drop.
- [ ] Optional `save_device_memory_profile` at train_end.
- [ ] Report peak device memory in the benchmark output, alongside
      Mcells/s — this is the number §1.3 is actually about.

---

## 3. NetworKit — EVALUATED 2026-08-08. Verdict: do not adopt.

Installed `networkit==11.2.1`, benchmarked against both real jobs on 8
threads, results verified identical to the current code in every case.
**Conclusion: no. Not close.** Details below so nobody re-opens this
without new information.

### Job B — bounded radius (`sparse_hops`), the path actually used at scale

| n | r | current | NetworKit (SPSP, parallel C++) | result |
|---|---|---|---|---|
| 2,000 | 2 | 0.02 s | 0.37 s | **18x slower** |
| 5,000 | 2 | 0.06 s | 2.20 s | **37x slower** |
| 10,000 | 2 | 0.11 s | 9.41 s | **85x slower** |

Tried twice, giving NetworKit its best shot: first one `BFS` per source
(30–100x slower, and the Python loop kills parallelism), then `SPSP`, which
runs every source inside C++ with OpenMP. Still 85x slower.

The reason is structural, not an implementation detail: **bounded radius 2
needs exactly two sparse matrix products.** It never computes a distance it
will not keep. NetworKit computes *every* pairwise distance and then
discards everything past the radius. No amount of parallelism recovers a
constant factor that large against an algorithm doing asymptotically less
work. `SPSP` also materializes a dense `(n, n)` result — 400 MB at n=10k,
80 GB at n=100k — which reintroduces exactly the wall §1.3 is about.

### Job A — all-pairs (`hopfill`), i.e. `coupling="complete"`

| n | scipy | NetworKit APSP (+conversion) | result |
|---|---|---|---|
| 1,000 | 0.22 s | 0.14 s | 1.6x faster |
| 2,000 | 0.91 s | 0.46 s | 2.0x faster |
| 4,000 | 3.85 s | 1.79 s | 2.2x faster |
| 8,000 | 18.12 s | 7.92 s | **2.3x faster** |

Here NetworKit genuinely wins, and conversion is negligible (0.04 s at
n=8k, building the graph from CSR arrays rather than via `nxadapter`).
Distances agree exactly.

But it does not matter, for two independent reasons:

1. **It does not move the memory wall.** Peak RSS was 3.9 GB at n=8,000 —
   both implementations are `O(n^2)`. NetworKit makes mode 1 *faster*, not
   *feasible*. The thing stopping `coupling="complete"` at scale is memory,
   and 2.3x on time changes nothing about that.
2. **Amdahl's law.** Augmentation runs ONCE; the embedding loop runs every
   epoch. Measured at n=20,000, radius 2, d=64:

   ```
   augmentation (one time) : 0.43 s
   embedding, steady state : 292.5 ms/epoch
      100 epochs -> augment is  1.4% of runtime
     1000 epochs -> augment is  0.1% of runtime
     2000 epochs -> augment is  0.1% of runtime
   ```

   A 2.3x speedup on 0.1% of the runtime saves **0.08%** of a 1000-epoch
   run — 0.24 seconds out of 293. That is not worth a compiled dependency.

### What would change this verdict

Not speed. Only a NetworKit feature the current stack lacks entirely:

- `networkit.community` (label propagation, modularity) for the
  **inter-community augmentation policy** in §1.4 — there is no equivalent
  in scipy, and writing one is real work.
- `networkit.centrality` for hub selection in the hub policy.
- `PrunedLandmarkLabeling` IF a future policy needs many repeated
  point-to-point distance queries (the current ones do not).

So: revisit **only** when implementing the community-based policy, and even
then adopt it for the community detection alone, not to replace any
distance computation. Do not adopt it as a general speed play.

---

## 4. Structural / code-health items

- [x] Move `_levelwise_reach` into `sparse_hops.py` — after hopfill moved to
      `scipy.sparse.csgraph.shortest_path`, it has exactly one caller, so a
      shared module is no longer justified. *(done 2026-08-08)*
- [ ] `graph_building/` cannot reach the scale `embedding/` targets (§1.3).
      It is the weakest stage in the pipeline and needs its own pass.
- [ ] `validation/bench_embedding_perf.py` monkeypatches
      `model.augment_graph` to defeat the identity-keyed plan cache. That
      works, but it means there is no supported way to run one epoch
      without re-augmenting. Consider exposing a lower-level per-step call.
- [ ] `ForceDirected.end_epoch` is incremented *after* the epsilon break, so
      a converged run under-counts by one and a resumed run restarts one
      epoch early. Decide the intended resume semantics and fix.
- [ ] `self.epochs` / `self.epoch` are assigned in `embed()` but never
      declared in `__init__`, unlike `latest_epoch` / `end_epoch`.

---

## 5. Kept deliberately (do not "fix")

- `build_ladder` — one-time host cost, and `np.geomspace` does **not**
  reproduce it (the `max(ks[-1]+1, ...)` guard changes the sequence).
  Verified. Leave it alone.
- `shell_counts` (was `_shell_counts_from_hops`) — keep. Moved next to the
  force law in `embedding/shell_force.py` and its bin allocation compacted
  to the distinct hop values present, but the function itself stays.
- `row_of` — `D.tocoo().row` is the library equivalent but pays a full
  CSR→COO conversion for something derivable from `indptr` alone.
- The balanced cell-budget packing rule (`R = ceil(rows/nb)`, not
  `b_cells // k`) — documented measured regression from ~13% to ~40%
  padding if "simplified". See `make_plan`'s docstring.
