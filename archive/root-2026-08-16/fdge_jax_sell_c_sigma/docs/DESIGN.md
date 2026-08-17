# fdge_jax_sell_c_sigma Design Contract

`fdge_jax_sell_c_sigma` is a sibling of `fdge_jax` (itself a JAX port of
`fdge2`, which uses Numba). Same three-stage architecture, same physics
(`make_graph -> augment_graph -> embed`, shell-averaged force law) -- see
`fdge_jax/docs/DESIGN.md` for the shared contract this doc doesn't repeat.
The ONE thing this package changes is `embedding/`'s execution engine:
instead of a flat edge-list + `jax.ops.segment_sum` kernel, it uses a
bucketed/padded "SELL-C-sigma" layout (host-side hub-split + width-sort +
ladder-quantize + cell-budget batching, dense `sum(axis=1)` reduction
instead of a scatter) for higher throughput on large, power-law-degree
graphs. See `dev_docs/fdmap_engine_design_notes.md` (repo root) and
`archive/fdmap_bucketed_bench_jax.py` for the full algorithmic rationale
and reference implementation this engine ports from -- this doc points to
those rather than re-deriving them.

This doc states *what the contract is*. For *how each rule was arrived at* --
the alternatives measured and rejected, and the earlier claims that turned
out to be wrong -- see `../journal/` (start at its `README.md`). Open work
items are in `../TODOS.md`.

Goals, in order (this is a benchmarking/engineering exercise, not research
code and not production code -- but it must run as fast as possible while
staying easy to read and modify):
1. **Fast to run at scale.** The whole reason this package exists: the
   epoch loop (`embedding/`'s force law) must be `jax.jit`-compiled AND
   shaped so XLA can fuse it into a small, fixed number of memory passes
   over the real (non-padded) work -- the property a flat scatter-based
   kernel doesn't reliably get on power-law-degree graphs (a few hub nodes
   dominate scatter write-contention).
2. **Fast to modify.** One small file per idea, long docstrings explaining
   *why*, not just *what* -- same style discipline as `fdge_jax`.
3. **Minimal.** Prefer deleting a knob over generalizing it if it isn't
   needed by any real caller.

## Directory layout (mirrors fdge_jax)

```
fdge_jax_sell_c_sigma/
  __init__.py          # does NOT force jax_enable_x64 -- float32 default (see below)
  core/
    csr.py              # row_of, n_rows  (numpy only)
    force_directed.py   # ForceDirected, Callback_Base  (numpy + jax + core.csr; self.Z/self.dZ float32 by default)
  graph_building/        # unchanged from fdge_jax (no numba there to begin with)
  graph_augmenting/
    hopfill.py           # dense hop-fill -> scipy.sparse.csr_matrix (via nx.to_scipy_sparse_array)
    sparse_hops.py        # bounded-radius hop-fill -> scipy.sparse.csr_matrix (via nx.to_scipy_sparse_array)
  embedding/
    shell_force.py         # THE FORCE LAW + the quantities defining it + the binding classes
    sell_c_sigma.py        # the SELL-C-sigma layout machinery (THIS package's one real change)
    drop.py                # steady-rate random drop + the key=None fallback key stream
  models.py               # composition root (only file allowed to import all 3 stages)
  validation/              # integration tests + the large-graph throughput benchmark
```

`embedding/` is split three ways along "what would a researcher change?".
The force law is the research knob, so it lives alone in `shell_force.py`
(~250 lines, no layout code in it); the batching/padding/JIT machinery is
infrastructure nobody should need to open, so it lives in `sell_c_sigma.py`
and takes the force law as a `force_fn` argument. The seam is exactly one
function signature:

```python
force_fn(x, planes, params) -> (R, k) force magnitude
```

`x` is the `(R, k)` embedding-distance tile; `planes` is the tuple of
`(R, k)` coefficient tiles the plan was built with (`shell_coeff`, `h` for
the shipped law, but `make_plan` takes any number and never looks inside
one); `params` is a plain dict of **traced** scalars. Everything around the
call -- the `/x` projection, the `x == 0` guard, the degree division, the
pad-row drop -- stays in `_step`, because it protects the padding contract
rather than expressing physics. Swapping the force law therefore touches
zero lines of `sell_c_sigma.py`; `embedding/test_sell_c_sigma.py`'s
`test_force_law_is_swappable` runs an alternative law through the
unmodified engine to keep that true.

`models.py` wires three concrete models, matching `fdge_jax/models.py`'s own
set: `ReferenceFDModel` (dense hop-fill + bucketed force law), `SparseFDModel`
(same, bounded-radius augmentation -- use this one at scale), and
`EuclideanDistanceModel` (weighted shortest-path distance instead of hop
count). All three go through the SAME plan builder, the same compiled
kernel and the same cache: the Euclidean variant is
`embedding.shell_force.WeightedShellForce`, a two-hook subclass of
`ShellForce` that plans from the unweighted hop matrix while feeding the
`h` plane from the weighted one and setting `h_shift = h_min`. `models.py`
contains no `jax.jit`, no plan cache and no engine code of its own -- it is
wiring only, as its docstring claims.

Import discipline (one deliberate deviation from `fdge_jax`'s own table --
`graph_augmenting/` now also imports `networkx`, see below):
- `core/csr.py`: **numpy only**.
- `core/force_directed.py`: numpy + jax + `core.csr`. Never a stage package.
- `graph_building/`: numpy / networkx / scipy / pynndescent. Never `core`
  or the other two stage packages.
- `graph_augmenting/`: numpy / scipy.sparse / networkx / `fdge_jax_sell_c_sigma.core`
  only. NetworkX is already a hard requirement of this package (via
  `graph_building/`), so building the initial adjacency directly with
  `nx.to_scipy_sparse_array` -- rather than a hand-rolled duck-typed CSR
  builder -- costs nothing new and drops bespoke code that duplicated a
  library one-liner.
- `embedding/`: numpy / scipy.sparse / jax / `fdge_jax_sell_c_sigma.core`
  only. Within the package, `shell_force.py` may import `sell_c_sigma.py`
  and `drop.py`, never the reverse -- the layout engine must stay ignorant
  of the physics for the `force_fn` seam to mean anything.
- `models.py`: the only file allowed to import `core` + all three stages.
- **New rule specific to this package**: nothing under
  `fdge_jax_sell_c_sigma/` may import the sibling `fdge_jax` package at
  runtime -- this package is fully self-contained, not a thin wrapper
  around `fdge_jax`. The only sanctioned exception is test/benchmark code
  (`embedding/test_sell_c_sigma.py`'s cross-engine parity check,
  `validation/bench_embedding_perf.py`'s correctness spot-check), which
  deliberately imports `fdge_jax.embedding.shell_force.ShellForce` to prove
  the two engines agree. A handful of small helpers (`D` -> `csr_matrix`
  coercion, shell-count derivation, `drop_steady_rate`) are therefore
  duplicated across that package boundary rather than shared -- same "own
  copy across an import boundary" pattern `graph_augmenting/` already uses.
  Note this package has its own `embedding/shell_force.py`; when both are
  in scope, say which one you mean.

## float32 by default (the one other change from fdge_jax, besides the engine)

`fdge_jax/__init__.py` calls `jax.config.update("jax_enable_x64", True)` at
import time so its physics matches `fdge2`'s float64 Numba path bit-for-bit
in spirit. This package's `__init__.py` deliberately does **not** do that:
this package's whole point is throughput, and the bucketed kernel was
designed and measured in float32 (`dev_docs/fdmap_engine_design_notes.md`
Sec 2, 9 -- the archive bench's own headline Mcells/s figures are all
float32). `core.force_directed.ForceDirected` constructs `self.Z` /
`self.dZ` as float32 by default, and `embedding.shell_force.ShellForce`
casts its `Z` argument to float32 on every call regardless of what dtype
the caller handed it. A researcher who genuinely needs float64 precision
can still get it -- call `jax.config.update("jax_enable_x64", True)` before
importing anything and pass float64 arrays in explicitly via `Z=` -- but
nothing here does that automatically, unlike `fdge_jax`.

Consequence for correctness testing: this package's own oracle tests use a
*relative*, not absolute, error tolerance (`embedding/test_sell_c_sigma.py`)
-- an absolute tolerance tuned for `fdge_jax`'s float64 kernel (`~1e-9`)
is meaningless for a float32 kernel producing O(10⁴)-magnitude values on
an artificial worst-case fixture; the right bar is "matches to float32 ULP
scale," which is a relative statement.

## `embedding/` -- the force law and the hot-loop kernel

Full contract, the plan builder's exact algorithm, and the per-cell kernel
code are documented in `sell_c_sigma.py`'s own module docstring (physics in
`shell_force.py`'s) (and, one
level further back, `dev_docs/fdmap_engine_design_notes.md` Sec 2-3 and
`archive/fdmap_bucketed_bench_jax.py`) -- this section summarizes the
shape of the contract, not the algorithm itself, and exists to answer "how
does this fit into the three-stage pipeline," not "how does SELL-C-sigma
work."

**Same physics as `fdge_jax.embedding.shell_force`, different execution
shape.** Per stored pair `(u, v)` with hop distance `h = D[u,v] >= 1`:
`Fa = k1 * (1/|S_h(u)|) * x * exp(-k2*(h-1))`, `Fr = -k3*h*exp(-k4*x)`,
summed per row and divided by degree (hop-1 count), then a per-node random
steady-rate drop. The physics is never redesigned here -- only *how* the
per-row sum is computed changes.

**Host-side plan builder (`make_plan`), run once per distinct `D`:**
1. **Hub split** -- rows wider than `k_max` become multiple virtual rows,
   each a contiguous CSR slice, tagged with the real owner node id.
2. **Width sort** -- stable sort of virtual rows by width (storage order,
   i.e. node ids, is untouched -- only processing order changes).
3. **Ladder quantize** -- each width rounds up to the nearest value on a
   geometric ladder (`build_ladder`, base `LADDER_BASE`), so runs of
   similarly-sized rows share one padded shape.
4. **Balanced cell-budget packing** -- rows of a given quantized width `k`
   are packed into batches of `R = ceil(rows_in_rung / nb)` rows each. This
   balancing is load-bearing, not a style choice: naive greedy packing
   (`R = B_CELLS // k`, pad only the last, possibly near-empty batch) is a
   *measured regression* from ~13% to ~40% padding at n >= 10k
   (`dev_docs/fdmap_engine_design_notes.md` Sec 3.2) -- do not "simplify"
   this away.
5. **Pad** -- pad CELLS get neighbor = the row's own node id, and EVERY
   per-cell coefficient plane zeroed, so for the shipped law `Fa` and `Fr`
   vanish independently ("exact zero on padding" -- inherited from the
   archive kernel's contract). Because the neighbour is the row's own node
   id, `x` is also exactly 0 there, and `_step`'s `x == 0` guard zeroes the
   contribution a second time -- deliberate double safety, and the reason a
   *new* force law (even one with a constant term) is safe on padding
   without knowing any of this. Pad ROWS get owner id `n` (out of `[0, n)`, dropped by
   `.at[rows].add(..., mode="drop")`), and `inv_deg_ext[n] = 0` as a second,
   independent neutralization.

**Per-epoch kernel (`_step`), jitted once per distinct `D`'s plan shapes:**
a Python `for rung in plan:` loop (unrolled at trace time -- `plan` is a
fixed-length tuple of fixed-shape arrays) around one `jax.lax.scan` per
rung over that rung's stacked `(nb, R, k)` batches: gather row centers
(once per ROW, broadcast) and neighbors (once per CELL), call `force_fn`
element-wise on the `(R, k)` distance tile, reduce with a plain
`sum(axis=1)`, write with `.at[rows].add(F, mode="drop")`. `n` and
`force_fn` are baked in via `functools.partial` (static); `params` is a
traced dict, so a hyperparameter sweep against one `D` recompiles nothing
-- `test_params_are_traced_not_static` pins `_cache_size() == 1` across
five distinct `k1` values.

**Single-pass-per-call contract (why there is no outer epoch loop here):**
unlike the archive bench (which jits its ENTIRE multi-step loop via an
outer `lax.fori_loop`, since it owns its own timing harness), this engine
must plug into the EXISTING `core.ForceDirected.embed()` Python loop, which
already owns epochs, batching, callbacks, and epsilon-convergence -- none
of that is reimplemented or wrapped in a bigger jit here.
`ShellForce.__call__` computes one full-graph force-law pass; `embed`
calls it once per batch, every epoch, exactly like `fdge_jax`'s own
`ShellForce.__call__`.

**Caching:** one implementation, `embedding.sell_c_sigma.PlanCache` --
single-slot and identity-keyed (`key is cached_key`), the same pattern
`fdge_jax`'s `ShellForce` uses. The plan, the device-resident batch
arrays and the compiled step function are derived once per distinct `D`
and held until a genuinely new `D` object shows up. `PlanCache` is
force-law agnostic (it is constructed with a `force_fn`), so both bindings
and both model families share it; there is exactly one `jax.jit` call site
in the package. See its docstring for the "why identity, not a dict"
argument and -- importantly -- for **the invariant it cannot check**: the
key object must be freshly constructed whenever any plan input changes,
including the coefficient planes. `EuclideanDistanceModel.augment_graph`
upholds this by assigning `self.D` and calling `bind_topology(self._hops,
self._h_min)` in the same breath.

**Batching note:** same accepted tradeoff as `ShellForce` -- the kernel
always computes the full-graph pass, then slices `[row_start:row_end]`
afterward, because the plan's batches are grouped by width across the
WHOLE graph and have no clean correspondence to a row range. `batch_count
> 1` therefore redoes the whole-graph computation on every batch call;
not something to "fix" here.

## Which augmentation policy to use at scale

**Use `graph_augmenting.sparse_hops` (bounded radius), not dense
`hopfill`, for anything beyond a small test/parity graph.** This is a
correctness-of-judgment note, not a code contract, but it matters enough
to repeat here: dense hop-fill connects *every* reachable pair and fills an
`unreachable = n` sentinel for every disconnected pair, which is `O(n^2)`
stored entries by construction -- regardless of how fast the embedding
engine consuming `D` is. At 100k-500k nodes this is not just slow, it's
generally infeasible to even materialize (`D` alone would need on the
order of `n^2` int32 entries). `sparse_hops.augment_graph(G, radius=...)`
instead stores only within-radius pairs -- `O(n * avg_degree^radius)`,
linear in `n` at fixed radius and degree -- which is what makes the
100k-500k node benchmark in `validation/bench_embedding_perf.py` possible
at all. `models.SparseFDModel` wires this in; `models.ReferenceFDModel`
(dense hop-fill) exists for small-graph numeric-parity testing against
`fdge_jax`, not for real runs at this package's target scale.

(A related, sharper reason dense hop-fill doesn't scale even before
considering `D` itself: `hopfill`'s own shell-count derivation,
`get_shell_counts`, densifies `D` and builds an `(n, n_bins)` array where
`n_bins` includes the `unreachable = n` sentinel as a bin -- so `n_bins`
scales with `n` itself for a disconnected or near-diameter-n graph, making
that helper's memory footprint `O(n^2)` on TOP of `D`'s own `O(n^2)`.
`sparse_hops`-derived `D`'s hop values are bounded by `radius` (e.g. 3 for
radius=2), so the equivalent shell-count table stays `O(n * radius)` --
tiny, independent of `n`'s scale.)

`embedding/` no longer has this failure mode at all.
`shell_force.shell_counts` used to build the same `(n, D.data.max() + 1)`
table and inherited the same `8 n^2` blowup on any disconnected `D`
(32 MB measured at n=2000, ~20 GB extrapolated at n=50k -- invisible on
connected graphs, which is why no test caught it). It now compacts to the
**distinct hop values actually stored**, `(n, #distinct_hops)`, returning
`(hop_values, counts)`; on the two-component fixture above that is `(n, 2)`
instead of `(n, n+1)`, a 1000x reduction at n=2000 and growing linearly in
`n`. `test_shell_counts_not_quadratic_on_disconnected_graph` guards it.

## Numeric parity expectation

Not bit-identical to `fdge_jax`'s `ShellForce` (different summation order
-- padded dense `sum(axis=1)` per batch vs. a flat `segment_sum` scatter --
and different floating point precision, float32 here vs. `fdge_jax`'s
float64-by-default). The correctness bar, same spirit as `fdge_jax`'s own
"Numeric parity expectation" section:
- Matches a naive oracle to float32 ULP scale (`~1e-4` to `~1e-6`
  *relative* error, dropout disabled) -- see
  `embedding/test_sell_c_sigma.py`.
- Matches `fdge_jax.embedding.shell_force.ShellForce`'s output on the same
  `D`, same `Z`, same `k1..k4`, dropout disabled, to a tight tolerance
  (`atol=1e-4`) -- the test that actually proves this engine reproduces
  the existing engine's physics, just with a different (faster, at scale)
  execution shape. Also `embedding/test_sell_c_sigma.py`.
