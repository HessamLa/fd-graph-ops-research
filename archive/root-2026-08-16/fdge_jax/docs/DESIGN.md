# fdge_jax Design Contract

`fdge_jax` is a JAX port of `fdge2` (which uses Numba). Same three-stage
architecture (see `fdge2/docs/ARCHITECTURE.md` / `API_DESIGN.md` — read
those for the *why*; this doc is the concrete contract for the port).

Goals, in order (this is research code, not a library):
1. **Fast to modify** — one small file per idea (new force law, new
   augmentation policy). No defensive coding, no speculative abstractions.
2. **Fast to run** — the epoch loop (`embedding/`'s force law) is the hot
   path and must be `jax.jit`-compiled. Everything else (graph building,
   one-time graph augmentation) runs once per `embed()` call and does not
   need to be jitted — plain NumPy/SciPy is fine and often simpler there.
3. **Minimal.** Prefer deleting a knob over generalizing it if it isn't
   needed by any real caller.

## Directory layout (mirrors fdge2)

```
fdge_jax/
  __init__.py          # jax.config.update("jax_enable_x64", True) lives here
  core/
    csr.py              # graph_to_csr, row_of, n_rows, HopMatrix  (numpy only)
    force_directed.py   # ForceDirected, Callback_Base  (numpy + jax + core.csr)
  graph_building/        # unchanged from fdge2 (no numba there to begin with)
  graph_augmenting/
    hopfill.py           # dense hop-fill -> HopMatrix
    sparse_hops.py        # bounded-radius hop-fill -> HopMatrix
  embedding/
    shell_force.py        # the JAX hot-loop kernel
  models.py               # composition root (only file allowed to import all 3 stages)
  validation/              # integration / regression-parity tests
```

Import discipline (unchanged from fdge2 RPD.md Sec 5):
- `core/csr.py`: **numpy only**.
- `core/force_directed.py`: numpy + jax + `core.csr`. Never a stage package.
- `graph_building/`: numpy / networkx / scipy / pynndescent. Never `core`
  or the other two stage packages.
- `graph_augmenting/`: numpy / scipy.sparse / `fdge_jax.core` only.
- `embedding/`: numpy / jax / `fdge_jax.core` only.
- `models.py`: the only file allowed to import `core` + all three stages.

## Why JAX only shows up in `embedding/`

`augment_graph` runs **once** per `embed()` call; `forces` runs **every
epoch**. JAX's win (fused, compiled, GPU-resident kernels) matters for the
thing that runs thousands of times, not the thing that runs once. Graph
BFS (hop-fill, bounded-radius hop-fill) also has irregular, data-dependent
control flow that doesn't map cleanly onto `jax.jit`'s static-shape
tracing. So: **`graph_augmenting/` stays plain NumPy/SciPy** (replacing
Numba's role with SciPy's own compiled sparse-matrix primitives — no
speed lost, since it's not the hot path), and **`embedding/` is the one
place that must be `jax.jit`-compiled**, matching the pattern already
validated in `sparse_elementwise_test.py` (CSR topology + elementwise
combine + `segment_sum` row-reduce, `jax.jit`'d once, called every epoch).

## `HopMatrix` — the one shared `D` representation

Replaces fdge2's "dense ndarray or `scipy.sparse.csr_matrix`" duck-typed
`D`. Both augmentation policies (dense hop-fill, bounded-radius hop-fill)
now return the **same** small CSR-triple class, defined in `core/csr.py`:

```python
class HopMatrix:
    """(n, n) hop-distance matrix stored as a CSR triple.

    indptr  : int64 (n+1,)  row pointer
    indices : int32 (nnz,)  neighbor node id per stored entry
    data    : int32 (nnz,)  hop distance for that entry (>= 0)
    n       : node count

    Row u's stored pairs: indices[indptr[u]:indptr[u+1]] with hop
    distances data[indptr[u]:indptr[u+1]] at the same positions. A pair
    absent from row u means "no stored distance" (self, or beyond the
    augmentation policy's reach) -- NOT the same as distance 0.

    Satisfies the indexability contract (D[i, j] works, dense-array
    semantics: 0 for an absent pair) for debugging/tests via
    `__getitem__`/`toarray()`, while giving `embedding/` direct CSR
    access with no densify-then-reparse step.
    """
```

- `hopfill.augment_graph` stores **every** reachable pair (`is_sparse`
  still selects the return container: `HopMatrix` if True, `.toarray()`
  dense ndarray if False), including the `unreachable = n` sentinel for
  disconnected pairs, matching fdge2 exactly.
- `sparse_hops.augment_graph` stores only pairs within `radius` hops —
  genuinely sparse, no sentinel (matches fdge2 exactly).
- Self is never stored in either (`h = 0` isn't a force-law contributor,
  same as fdge2's `hi <= 0: continue` skip).

## RNG: `jax.random.PRNGKey`, not `np.random.Generator`

JAX has no hidden global RNG state — keys are explicit and split, not
mutated. `core.ForceDirected` keeps **both**:
- `self.rng` — a `np.random.Generator` (seeded the same way as fdge2),
  used only for non-hot-path randomness (e.g. `graph_building` strategies
  that want a numpy seed, e.g. `union_mst_nndescent`'s `random_state`).
- `self.key` — a `jax.random.PRNGKey(seed or 0)`, split once per
  epoch/batch and threaded into `updateGradient` / `forces` via a `key=`
  kwarg (replacing fdge2's `rng=` kwarg). The per-row random drop
  (`DropSteadyRate`) becomes a pure function of `(dZ, key, drop_rate)`.

## `core.ForceDirected` — what changes from fdge2, concretely

Same public API and loop *shape* (batching, callbacks, epsilon
convergence — port these nearly verbatim, they're plumbing, not physics).
What must change because `jnp.ndarray` is immutable:

- `self.Z`, `self.dZ` are `jnp.ndarray` (float64 — `fdge_jax/__init__.py`
  sets `jax.config.update("jax_enable_x64", True)` at import time, so the
  whole package defaults to float64 for numerical parity with fdge2;
  leave a one-line comment noting float32 is faster if a researcher wants
  to trade precision for speed).
- Lazy Z init: `self.key, sub = jax.random.split(self.key)` then
  `self.Z = jax.random.normal(sub, (n, self.n_dim))` instead of
  `rng.standard_normal`.
- Per-batch write into `dZ`: `self.dZ = self.dZ.at[row_start:row_end].set(rows)`
  instead of `self.dZ[row_start:row_end] = rows`.
- `updateZ`: `self.Z = self.Z + lr * self.dZ` (no `+=` in place — same
  effect, just not a mutation).
- `Th(dZ)`: `float(jnp.linalg.norm(dZ, axis=-1).mean())` — the `float()`
  is a deliberate host sync, same cost/role as fdge2's numpy version
  (needed every epoch for the print/epsilon check regardless of backend).
- The loop should split `self.key` once per batch and pass
  `key=batch_key` through `**kwargs` into `updateGradient`/`forces`,
  mirroring how fdge2 threads `rng=self.rng`.
- `get_embeddings()` still returns a **numpy** array
  (`np.asarray(self.Z)`) — external consumers (pandas, plotting) shouldn't
  need to know the internal backend.

`forces`/`augment_graph`/`make_graph` stay abstract hooks raising
`NotImplementedError` in `core`, exactly as in fdge2 — `core` never
imports a stage package.

## `embedding/shell_force.py` — the hot-loop kernel

Same physics as fdge2 (`_scalar_force`: `Fa = k1 * shell_coeff * x *
exp(-k2*(h-1))`, `Fr = -k3*h*exp(-k4*x)`, `f = Fa+Fr`), same per-`D`
derived-quantity caching pattern (single-slot cache keyed on `D is
cached_D` — identity, not `id()`), but:

- The derivation converts `D` (a `HopMatrix`, or dense ndarray) into a
  **flat edge list** (excluding self): `u_of`, `v_of` = `D.indices`,
  `h` = `D.data`, all as device `jnp.ndarray`s (`jax.device_put` once,
  cached). `shell_counts[u, h]` (needed for `shell_coeff = 1/|S_h(u)|`)
  is still cheapest to compute once in NumPy (`np.bincount` per row, same
  as fdge2 — this is a cacheable, not-hot-path derivation) and then
  gathered into a **per-edge** `shell_coeff` array
  (`shell_coeff[p] = 1/shell_counts[u_of[p], h[p]]`) — precomputing this
  per-edge means the jitted per-epoch step never needs `shell_counts`
  itself, just a flat `(nnz,)` array, which is the natural JAX shape.
- Degrees: same derivation as fdge2 (count of hop-1 entries per row, or
  passed in), cached as a device array.
- The actual per-epoch force computation is a `jax.jit`'d pure function,
  same shape as `_jax_step` in `sparse_elementwise_test.py`:

  ```python
  def _step(Z, u_of, v_of, h, shell_coeff, degrees_safe, degrees_zero,
            k1, k2, k3, k4, n):
      diff = Z[v_of] - Z[u_of]                       # (nnz, d)
      x = jnp.linalg.norm(diff, axis=-1)              # (nnz,)
      x_safe = jnp.where(x == 0, 1.0, x)
      Fa = k1 * shell_coeff * x * jnp.exp(-k2 * (h - 1.0))
      Fr = -k3 * h * jnp.exp(-k4 * x)
      scale = jnp.where(x == 0, 0.0, (Fa + Fr) / x_safe)
      contrib = scale[:, None] * diff                  # (nnz, d)
      sums = jax.ops.segment_sum(contrib, u_of, num_segments=n)
      return jnp.where(degrees_zero[:, None], 0.0, sums / degrees_safe[:, None])
  ```

  `k1..k4` are traced (not static) arguments so sweeping hyperparameters
  never forces a recompile — only a genuinely new `D` (new shape) does.
  `jax.jit(_step, static_argnames=("n",))` (or close over `n`) once per
  `D`, cached alongside the derived arrays (extend the existing
  `_cache`/`_cache_D` single-slot pattern with the compiled step fn).
- `row_start`/`row_end` batching: compute the **full** `(n, d)` step
  output, then slice `[row_start:row_end]` in plain Python/JAX indexing
  *after* the jitted call. Do not try to slice the edge arrays by row
  range before the kernel — CSR row ranges have variable length (bad for
  static shapes) and JAX's whole point is one fused pass over everything
  anyway. Note in a comment that `batch_count > 1` therefore recomputes
  the full graph's forces on every batch call (wasted work for that rare
  path) — acceptable; don't build machinery to avoid it.
- `DropSteadyRate` becomes a pure function taking `key`:
  ```python
  def drop_steady_rate(dZ, key, drop_rate):
      if drop_rate <= 0.0:
          return dZ
      keep = jax.random.uniform(key, (dZ.shape[0],)) >= drop_rate
      return dZ * keep[:, None]
  ```
  (the `if` runs at Python/trace level against a static float, not a
  traced value — fine, mirrors how `k3=None` resolves before tracing).
- Keep the `ShellForce` class (holds the cache) and the module-level
  `forces` free-function convenience wrapper, same shape as fdge2.
  `ShellForce.__call__` signature: `(Z, D, row_start, row_end, key=None,
  degrees=None, k1=None, k2=None, k3=None, k4=None,
  random_drop_rate=None, **kwargs)` — `key=` replaces fdge2's `rng=`.

## `graph_augmenting/` — plain NumPy/SciPy, no Numba, no JAX

Both policies build `D` via **sparse boolean matrix powers** instead of
per-source BFS (no Python loop over nodes at all — a handful of
`scipy.sparse` matmuls total, independent of `n`):

```
A = symmetric adjacency, scipy.sparse.csr_matrix, dtype=bool/int8, no self loops
visited = identity (dist 0, self)
frontier = A                      # candidates newly reachable at hop 1
level = 1
loop while frontier has any nonzero entries (and, for sparse_hops, level <= radius):
    new = frontier with already-visited positions removed         # boolean AND-NOT
    record (row, col) -> level for every nonzero in `new`
    visited = visited | new
    frontier = new @ A            # expand ONLY from this level's new nodes
    level += 1
```

This computes all-pairs (or all-pairs-within-radius) hop distances for
**every source simultaneously** — `radius` (or the graph diameter, for
hopfill) sparse matmuls total, not `n` BFS launches. `hopfill.py` runs it
to convergence (empty frontier) and fills the `unreachable = n` sentinel
for any pair never visited; `sparse_hops.py` runs it for exactly `radius`
levels and stores nothing beyond that (no sentinel — matches fdge2).
Factor the shared level-synchronous loop into one small internal helper
(e.g. `graph_augmenting/_levelwise_reach.py`) that both modules call with
a different stopping condition, so the two policies don't duplicate the
matrix-power logic. Assemble the recorded `(row, col, level)` triples
into a `core.csr.HopMatrix` (sort by row for a valid CSR `indptr`, e.g.
via `scipy.sparse.coo_matrix((data, (row, col))).tocsr()` and lift its
`.indptr/.indices/.data` into a `HopMatrix`).

`get_shell_counts(D, n_bins)` (fdge2's optional helper) ports directly —
works on `HopMatrix` via `.toarray()` or directly on its CSR triple with
`np.bincount` per row.

## `graph_building/` — unchanged

`fdge2/graph_building/` already uses only numpy/networkx/scipy/pynndescent
— no Numba there to begin with. Port as a straight copy with import paths
updated (`fdge2.` -> `fdge_jax.`); no algorithmic changes.

## `models.py` — composition root

Same shape as `fdge2/models.py`'s `ReferenceFDModel`: wires
`graph_building.make_graph`, `graph_augmenting.hopfill.augment_graph`,
and an `embedding.shell_force.ShellForce` instance together. `forces`
passes `key=self.key`-derived per-call keys (via `**kwargs` from `embed`'s
loop) instead of `rng=self.rng`.

## GPU determinism caveat (found during validation)

`jax.ops.segment_sum` (and other scatter/reduce ops) on GPU use a
non-deterministic scatter-add: the **same** jitted call with the **same**
input, run twice in the same process, can differ at the ~1e-15 level
(confirmed empirically -- this is an XLA-on-GPU characteristic, not a bug
in this port). Consequence: fdge2's bit-exact "`batch_count` doesn't
change the result" guarantee degrades to *numerically equal to float64
noise* (not bit-identical) on GPU, because `ShellForce` calls the same
compiled step once per batch and those separate invocations aren't
guaranteed to scatter-add in the same order. Tests that check batching
invariance should use `np.allclose(..., atol=1e-9)`, not `np.array_equal`.

If a researcher needs true bit-exact reproducibility (e.g. debugging),
set `XLA_FLAGS=--xla_gpu_deterministic_ops=true` in the environment
**before** the process starts (it's read at XLA backend init, too late to
set from inside a script after `jax` has already been imported elsewhere)
-- confirmed this reduces the repeated-call diff to exactly `0.0`. Not
set by default here since it costs some throughput and most research use
doesn't need bit-exact repeats.

## Numeric parity expectation

Not bit-identical to fdge2/Numba (different summation order — `fastmath`
Numba vs. XLA `segment_sum` — and JAX's RNG stream is structurally
different from `np.random.Generator`, so dropout draws can't line up
either). The correctness bar is: matches a naive oracle to float64
precision (`~1e-9`, dropout disabled) — same methodology fdge2's own
`test_shell_force.py` uses — and tracks fdge2 epoch-by-epoch within a
loose tolerance (`~1e-6` relative) on a small graph with dropout disabled,
long enough to show the trajectories are the same physics, not
identical-bits physics.
