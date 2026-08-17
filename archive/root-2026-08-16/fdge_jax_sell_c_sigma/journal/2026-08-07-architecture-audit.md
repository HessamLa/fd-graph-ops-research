# 2026-08-07 — Full architecture audit

## The brief

> Go over other structures and methods in `fdge_jax_sell_c_sigma/` and find
> the ones that could be replaced by well-known libraries. Only generate a
> report. [...] Some methods are placed in wrong files. [...] Find the code
> blocks and methods that are core, and those that are not.

Read all 3,719 lines. Every claim below was checked by running code, not by
reading.

## Library-replaceable code

**Verified equivalent (ran both, compared output):**

- `_levelwise_reach` + `hopfill`'s BFS →
  `scipy.sparse.csgraph.shortest_path(A, method='D', unweighted=True)`.
  Exact match including the disconnected case.
- `sparse_hops`' bounded BFS →
  `scipy.sparse.csgraph.dijkstra(A, unweighted=True, limit=radius)`.
  Exact match. **Rejected anyway**: `limit` only stops expansion early, the
  return is still dense `(n,n)` — ~1 TB at 500k nodes. Rejected on memory,
  not correctness. Recorded in the module docstring so nobody "simplifies"
  it back.

**Replaceable but NOT a win (measured, contradicting my own expectation):**

- `_shell_counts_from_hops`' per-row Python loop → a `coo_matrix`
  histogram. Same output, but **1.6x faster at n=5k and 0.8x — slower — at
  n=50k**. Worth doing for readability only. Recording this because the
  "remove the Python loop" instinct was simply wrong here.

**Not replaceable (checked before assuming):**

- `build_ladder` — `np.geomspace` produces a *different* sequence
  (`[1,2,3,4,6,9,…]` vs `[1,2,3,5,8,12,…]`); the `max(ks[-1]+1, ...)` guard
  is load-bearing. Keep.
- `row_of` — see the [previous thread](2026-08-06-graph-to-csr-removal.md).

**Scale mismatch:**

- `mst_edges` used `squareform(pdist(data))` — dense `O(n^2)`. **80 GB at
  n=100k, 2 TB at 500k.** `graph_building/` could not reach the scale
  `embedding/` targets.

## Misplaced code

The user named two examples; both held, and the second was worse than
stated.

**Coefficient derivation living in the engine file.**
`_shell_counts_from_hops`, `_shell_coeff_data`, `_degrees_from_D` encode
*physics* decisions — "shells are grouped by hop count", "degree means
hop-1 count". They sat next to `make_plan` in the layout file.

**Engine core tangled with the force law.** `make_plan` and `_step` are
pure SELL-C-sigma layout machinery, except the force law was hardcoded
inside `_step`'s `lax.scan` body behind a comment reading "edit these two
lines to try a new force law". The coupling ran deeper than those lines:

- `_step`'s signature hardcoded exactly `k1..k4` — a law needing `k5`, or
  only `k1`, could not fit.
- `make_plan` built exactly **two** coefficient planes (`shell_coeff`,
  `h`). A three-term law required editing the plan builder too.

**`SellCSigmaForce` should not exist as-is.** It was not a force law. It
was a cache + dispatcher (single-slot identity cache, `_fallback_key`,
`device_put`, `jax.jit`) with the physics knobs bolted on.

**The composition root implementing pipeline logic.** `models.py`'s own
docstring says "nothing here implements pipeline logic itself." It was
false: `EuclideanDistanceModel` carried a hand-copied second engine — its
own cache slots, its own near-verbatim `_fallback_key`, its own
`jax.jit(functools.partial(_step, n=n))`.

## Two real bugs found

**`get_embeddings_df()` was broken unconditionally.** It read `self.Gx`;
nothing in the package ever assigns `Gx`. Confirmed:
`AttributeError: 'ReferenceFDModel' object has no attribute 'Gx'`. It had
no test, which is why it went unnoticed.

Deeper than a typo: `self.G` is only set by `make_graph`, but `embed(G)` is
a valid entry point that never calls it — so renaming alone would still
have failed for every test and demo script.

**An `O(n^2)` memory blowup.** `_shell_counts_from_hops` allocated
`(n, D.data.max() + 1)`. On a **disconnected** graph `hopfill` stamps an
`unreachable = n` sentinel, so `max()` *is* `n` and the table becomes
`(n, n+1)` int64 = `8n^2` bytes. Measured n=2000 → 32 MB (exactly
`n^2 * 8`); extrapolates to ~20 GB at n=50k. Connected graphs bound `max()`
by the diameter and never notice — which is why no test caught it.

## Smaller inconsistencies

- Two near-identically-named functions disagreeing: `hopfill.get_shell_counts`
  vs `embedding._shell_counts_from_hops` return `[1,2,2,1]` vs `[0,2,2,1]`
  on `cycle_graph(6)` row 0 (self/absent handling). Neither wrong per its
  own contract; the naming is a trap.
- Dead code: `DropSteadyRate` (used by nothing, its own docstring admits
  it), `get_shell_counts` (tests only), the module-level `forces()` +
  `_default_sell` process-wide mutable singleton (one test).
- `make_graph` (then in `registry.py`, since renamed to `make_graph.py`)
  documents `node_labels`, stores `"target"`, consumers read `"label"`.
  Three names for one thing.
- `graph_building/__init__.py` cites `../docs/ARCHITECTURE.md` and
  `RPD.md Sec 5` — neither exists.
- `end_epoch += 1` sits *after* the epsilon break, so a converged run
  under-counts by one and a resumed run restarts an epoch early.

## Core vs not-core

**Core** (infrastructure; a researcher should never edit): the `embed` loop,
callbacks, `row_of`/`n_rows`, `make_plan`'s hub-split/sort/quantize/pack,
`_step`'s scan/gather/pad/scatter skeleton, `build_ladder`, the per-`D`
cache and jit compile.

**Not core** (research knobs that were *hard* to edit): the `Fa`/`Fr`
formula, `k1..k4`, `h_shift`, the coefficient derivations, and
`drop_steady_rate` (a regularizer, neither layout nor force law).

The whole finding of the audit is that these two lists were interleaved in
one 763-line file.

## Recommended order

1. Fix `self.Gx`. Real bug, ~one line.
2. Compact the hop-value bins. Real `O(n^2)`, small fix.
3. Extract the force law from `_step`; move the coefficient derivations
   beside it.
4. Dissolve `SellCSigmaForce`; delete the duplicate engine.
5. `hopfill` → scipy; move `_levelwise_reach` into `sparse_hops`.
6. Delete dead code.
7. Fix stale docstrings and the label naming.

3 and 4 touch the same seam and must be done together. 1, 2 and 5 are
mechanical and independently verifiable.
