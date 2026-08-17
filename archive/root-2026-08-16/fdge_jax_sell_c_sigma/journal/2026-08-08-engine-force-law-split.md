# 2026-08-08 — Separating the force law from the layout engine

The structural half of the refactor. Execution trace is in
`logs/master.log`; this is the design record.

## The problem, in the user's words

> To change the force law, a researcher must edit a function that also
> contains the padding contract, `lax.scan` plumbing, and `mode="drop"`
> sentinel logic. That is the opposite of the intended research seam.

Exactly backwards: the force law is the research knob and should be the
easiest thing in the package to change; the layout machinery is
infrastructure nobody should need to open.

## The seam

One function signature:

```python
force_fn(x, planes, params) -> (R, k) force magnitude
```

- `x` — the `(R, k)` embedding-distance tile.
- `planes` — the tuple of `(R, k)` coefficient tiles the plan was built
  with. `shell_coeff` and `h` for the shipped law, but `make_plan` now
  takes **any number** and never looks inside one.
- `params` — a plain dict of **traced** scalars.

Everything around the call — the `/x` projection, the `x == 0` guard, the
degree division, the pad-row drop — stays in `_step`, because it protects
the padding contract rather than expressing physics.

## The JAX constraint that made it work

`force_fn` must be **static** (it is a Python callable, not traceable
data). `params` must be **traced** (so a hyperparameter sweep does not
recompile). Getting this backwards would silently recompile per `k1` value.

Verified before specifying it, rather than assuming:

```
g = jax.jit(functools.partial(f, n=3))
g(x, 1.0, 2.0)   -> _cache_size() == 1
g(x, 5.0, 9.0)   -> _cache_size() == 1     # different scalars, no recompile
g(bigger_x, ...) -> _cache_size() == 2     # different shape, recompile
```

and that a plain callable passed via `functools.partial` works as a closure
constant. Final form:
`jax.jit(functools.partial(_step, n=n, force_fn=force_fn))`.

## Result

```
embedding/
  shell_force.py   360  THE PHYSICS — the file you edit
  sell_c_sigma.py  543  layout only, force-law agnostic (was 763)
  drop.py           57  regularizer + FallbackKeys
models.py          236  wiring only (was 303)
```

`sell_c_sigma.py`'s docstring now opens with: *"If you came here to change
the physics, you are in the wrong file."*

**Total line count went up** (1095 → 1246). The split costs a module
header, the `PlanCache` boundary, and two hook methods. What shrank is the
thing that matters: the file you must read to change physics, 763 → 360,
none of it layout.

## `SellCSigmaForce` dissolved

Its two halves separated:

- cache / compile / dispatch → `PlanCache` in `sell_c_sigma.py`,
  force-law agnostic, constructed with a `force_fn`.
- `k1..k4` + `h_shift` → `ShellForce` in `shell_force.py`, thin.

`EuclideanDistanceModel`'s duplicate engine is gone. It is now
`WeightedShellForce(ShellForce)`, overriding exactly two hooks:

- `_plan_inputs` — plan from the unweighted hop matrix (topology and shell
  grouping), but feed the `h` plane from the weighted Dijkstra matrix.
- `_h_shift` — return `h_min` instead of `1.0`.

Everything else — `__call__`, k-resolution, drop, fallback keys, the cache,
the compiled kernel — is inherited, one copy.

## The cache-correctness trap

`PlanCache` keys on object identity, which **cannot see that the
coefficient planes changed**. Same `D`, different `h_data` → stale plan.

Decision: **document the invariant, do not enforce it.** Hashing the
`(nnz,)` plane arrays on every call would cost more than the plan it
guards. The invariant is written into `PlanCache`'s docstring:

> the cache key object MUST be a fresh object whenever ANY plan input
> changes — topology, planes, or degrees.

It holds naturally for both callers. `WeightedShellForce` additionally
raises loudly if `bind_topology` was never called, and
`EuclideanDistanceModel.augment_graph` assigns `self.D` and calls
`bind_topology` on adjacent lines so they cannot drift.

## Verification

Existing tests mostly assert shapes and finiteness — a subtly wrong force
law passes all of them. So **golden arrays were captured before touching
anything**: real `Z` from `ReferenceFDModel`, `SparseFDModel`,
`EuclideanDistanceModel`, and a `batch_count=3` invariance case, with
`random_drop_rate=0.0` for determinism.

Checked the harness had teeth first: embeddings have magnitude ~120, so
`atol=1e-5` is a ~3e-7 *relative* tolerance — tight enough to catch any
real physics change, loose enough to absorb float32 reassociation from
reordered ops.

Result, after graph_building + graph_augmenting + the whole embedding
rewrite + models rewiring:

```
reference/sparse/euclidean/reference_b3: exact=True maxdiff=0.000e+00
GOLDEN MATCH
```

**Bit-exact**, not merely within tolerance. That was designed in:
`force_fn` returns `Fa + Fr` and `_step` keeps `/x_safe`, so the operation
sequence is unchanged.

Also verified independently:

- **No recompile.** `_cache_size()` stays 1 across four distinct
  `(k1, k2)` pairs; `n_derivations` stays 1.
- **`O(n^2)` fix.** Table 83x / 285x / 857x smaller at n=500/2000/6000. The
  ratio *growing* with n is the quadratic-vs-linear signature; a constant
  factor would not do that.
- **The seam itself.** Wrote a new force law in a throwaway script, ran it
  through the stock engine via `PlanCache`, hashed `sell_c_sigma.py` before
  and after — **byte-identical**. New physics, zero lines changed in the
  engine.
- `pad_frac = 0.1275` in the benchmark — the balanced cell-budget packing
  rule survived. The documented regression if "simplified" to greedy
  packing is ~40%.

## An improvement the agent made on the spec

I asked for `_cache_size()` to stay flat. It also printed `dZ[0,0]`
alongside, noting that a flat cache count is **also** consistent with the
parameter being silently ignored — the output moving while the compile
count does not is the actual proof. Sharper than the test I specified.

## Design decisions worth revisiting

1. **Two binding classes** (`ShellForce`, `WeightedShellForce`) rather than
   one with mode flags. The alternative was three kwargs that must be set
   together.
2. **`make_plan` no longer defaults `degrees`** — now required, because
   "what counts as a degree" is a force-law question and that file is
   force-law agnostic.
3. `shell_counts` returns `(hop_values, counts)`; `shell_coeff_data`
   recovers the column with `searchsorted` rather than carrying an inverse
   index around.
