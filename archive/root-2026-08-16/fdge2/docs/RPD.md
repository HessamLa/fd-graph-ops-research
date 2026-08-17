# fdge2.0 — Requirements & Planning Document (RPD)

Status: **Stage 0-4 implemented and verified** (T0.1-T4.3 in §8 below all
complete; full `fdge2/` test suite — 38 tests across `core/`,
`graph_building/`, `graph_augmenting/`, `embedding/`, `validation/` — passes).
Based on `README.md`, `ARCHITECTURE.md`, `API_DESIGN.md`,
`IMPLEMENTATION.md`. Companion document: `ORCHESTRATION.md` (agent roles
and task assignment used to build this).

**Verified against the success metrics in §7**: numeric parity with the
legacy `model_204_shell.FDModel` is exact (bit-identical, max abs diff
`0.0`, not just within tolerance); the `embedding/shell_force` Numba
kernel is 33-35x faster than a vectorized-NumPy equivalent (vs. the
~8x floor from IMPLEMENTATION.md's synthetic prototype); peak memory
scales as clean `O(n²)` (matching dense hop-fill's inherent cost) with
negligible dependence on embedding dimension `d`, ruling out an
accidental `(n,n,d)` materialization. See `fdge2/validation/` for the
harnesses and `ARCHITECTURE.md`'s "Open questions — status" section for
the Q1-Q5 resolutions.

## 1. Purpose

Rebuild the force-directed embedding pipeline (currently
`fdge_numba/forcedirected_numba/model_204_shell.py`, all-pairs BFS,
`O(n^2)` regardless of source sparsity) as `fdge2`: three independently
swappable stages — **make graph**, **augment graph**, **embed** — so a
researcher can change one stage without reading or risking the other two,
and so genuinely sparse augmentation policies become possible later.

This is research code. Priorities, in order (per `ARCHITECTURE.md`):
fast to modify > fast to run > not defensive. No input validation or
back-compat scaffolding beyond what's already decided in the docs.

## 2. Scope & Non-Goals

**In scope**: the three-stage pipeline, the `ForceDirected`/`FDModel`
class skeleton, one reference implementation per stage (dense hop-fill
augmentation, shell-averaging force law — i.e. numeric parity with today's
`model_204_shell.py`), and the correctness/performance harnesses needed to
trust that parity.

**Non-goals for this pass**: production-grade validation/error handling;
a general sparse augmentation policy (bounded-hop-radius, kNN) — designed
and flagged, not necessarily fully implemented; solving the repulsion/
sparse-collapse open question beyond documenting it.

## 3. Design Constraints (carried over, not renegotiated)

- Every stage's output (`G` or `D`) must be indexable as `G[i,j]`/`D[i,j]`,
  dense or sparse (API_DESIGN.md, "indexability contract").
- The embedding stage's hot loop must be row-streamed (accumulate per row,
  peak extra memory `O(n*d)`); never materialize `(n,n,d)` or a full
  `(nnz,d)` set of temporaries in the Numba path (ARCHITECTURE.md,
  IMPLEMENTATION.md memory guardrail).
- `D` is a **constant input** to the embedding stage — nothing in
  `forces`/`updateGradient`/`embed` may know how `D` was produced.
- Quantities derived from `D` inside the embedding stage (weights,
  per-edge distances, coefficients) share `D`'s exact CSR pattern; no
  general sparse-sparse arithmetic is needed anywhere in this design.
- `embed(G, ...)` takes the un-augmented graph and calls
  `self.augment_graph(G, ...)` itself — callers never call `augment_graph`
  directly as part of the normal `fit`/`embed` flow.

## 4. Module Breakdown

Target layout under `fdge2/`:

```
fdge2/
  core/            # base classes + shared indexable/CSR contract, nothing else
  graph_building/  # make_graph — thin, reuses existing graphmaking/
  graph_augmenting/      # augment_graph implementations
  embedding/       # forces / updateGradient / the epoch loop
  models.py        # composition root — wires the three stages onto ForceDirected (see 4.6)
  validation/      # correctness + perf/memory harnesses (cross-cutting, not a pipeline stage)
  docs/            # this + ARCHITECTURE/API_DESIGN/IMPLEMENTATION (existing)
```

### 4.6 `models.py` — composition root
Not one of the three pipeline stages — the glue that wires them
together. A single runnable model needs `make_graph`, `augment_graph`,
and `forces` all implemented at once, but the three stage packages must
never import each other (§5), so composition happens one level up, in
this file. `ReferenceFDModel` is the reference pipeline (dense hop-fill +
shell-averaged force law), reproducing `model_204_shell.FDModel`'s
behavior through the new seams — see `API_DESIGN.md`'s "Composition
root" section for the pattern.

### 4.1 `core/`
Owns: `ForceDirected` base class (`embed`, `updateGradient`, `fit`, loop
machinery — batching, callbacks, convergence, carried over from
`fdge_numba/forcedirected_numba/ForceDirected.py`) and the shared
CSR/indexable utilities (`graph_to_csr`-equivalent, row/indptr helpers)
that both `graph_augmenting/` and `embedding/` need. This is the one module
allowed to know about all three stages' *contracts* (not their
internals) — it's what wires `make_graph -> augment_graph -> embed`
together in `fit`.

### 4.2 `graph_building/`
Owns: `make_graph(data, strategy, ...)`. Not a rewrite — this should
**wrap or re-export** the existing `graphmaking/registry.py` dispatcher
(MST, kNN strategies already implemented there) rather than
reimplementing graph construction. New strategies register there, not
here.

### 4.3 `graph_augmenting/`
Owns: `augment_graph(G, ...) -> D`. Reference implementation: port the
dense hop-fill (`get_hops_csr` + shell counts from
`model_204_shell.py`) behind the `augment_graph` contract. Sparse
augmentation policies (bounded-hop-radius, kNN-based) are a second,
explicitly-future deliverable — see §9.

### 4.4 `embedding/`
Owns: `forces(Z, D, ...)` (the scalar/vector force law — the one
function meant to be edited per experiment, per the `_scalar_force`
convention) and `updateGradient(Z, D, ...)` (the momentum/gradient-shaping
seam). Reference implementation: port `_scalar_force` /
`_forces_204_hidx` to operate on the new `D` contract instead of the
dense `(n,n)` hops matrix directly.

### 4.5 `validation/`
Owns: the correctness methodology from `IMPLEMENTATION.md` (Numba vs.
sequential-reference vs. vectorized-NumPy, epoch-by-epoch comparison,
stop-at-divergence) generalized into a reusable harness, plus a
regression test that the new `FDModel` reproduces
`fdge_numba/forcedirected_numba/model_204_shell.FDModel`'s numeric
output given the same graph/seed/params. Also owns perf/memory
benchmarking (the subprocess + `RLIMIT_AS` technique for safe max-`n`
search).

## 5. Import Discipline

Rule: **imports flow forward through the pipeline, never sideways or
backward.**

- `graph_building/` imports: `numpy`, `networkx`, and the existing
  `graphmaking` package. **Not** `graph_augmenting/` or `embedding/`.
- `graph_augmenting/` imports: `numpy`, `scipy.sparse`, `numba`, and `core/`
  (for CSR helpers / base contracts). **Not** `embedding/`. May accept a
  `graph_building/`-produced `G` as input but must not import
  `graph_building/` internals — the contract is "any indexable weighted
  graph," not a specific builder's output type.
- `embedding/` imports: `numpy`, `numba`, and `core/` only. **Not**
  `graph_building/` or `graph_augmenting/` — `D` arrives as a plain argument;
  this module must not know or care how `D` was produced.
- `core/` imports: `numpy` only. It defines the contracts the other three
  depend on and must **not** import any of them back (no circular
  dependency).
- `validation/` is the only module allowed to import across all of the
  above — that's its job (comparing/benchmarking across stages).

A PR that adds an import violating this (e.g. `embedding/` importing
`graph_augmenting.hopfill` to special-case dense `D`) is a design smell, not a
convenience — it means the indexable-`D` contract isn't actually solid.

## 6. Open Design Questions to Resolve

These come from `ARCHITECTURE.md` and `API_DESIGN.md`'s "Open
items"/"Open questions" sections and must be settled during Stage 0
(core scaffolding), not deferred silently into implementation:

| # | Question | Where flagged |
|---|---|---|
| Q1 | Does `embed`'s simplified loop still get batching (`batch_count`), callbacks (`on_epoch_begin/end`, etc.), and convergence (`epsilon`/`Th`) — reintroduced explicitly now that `embed` no longer delegates to `super().embed(...)`? | API_DESIGN.md |
| Q2 | Exact signature/defaults for `make_graph`'s `is_sparse` flag and how `augment_graph` is expected to honor it | API_DESIGN.md |
| Q3 | Does `forces(Z, D, ...)` batch internally (row ranges) or operate on the whole graph each epoch? | API_DESIGN.md |
| Q4 | Is repulsion restricted to `N(u)` (edges present in `D`)? Real risk of embedding collapse once augmentation becomes sparse, unless compensated (sampled negatives, wider repulsion neighborhood) | ARCHITECTURE.md |
| Q5 | Which concrete sparse augmentation policy/policies to implement beyond dense hop-fill | ARCHITECTURE.md |

Q1–Q3 block Stage 0/core work and must be resolved first. Q4–Q5 block
only the *future* sparse-augmentation deliverable, not the reference
(dense) implementation.

## 7. Success Metrics

Concrete, checkable — not vibes:

1. **Contract compliance**: for both a dense and a sparse reference
   `augment_graph` output, `D[i,j]` indexing works and returns the
   expected value for at least one connected and one disconnected pair.
2. **Numeric parity**: given the same graph, seed, and force-law
   parameters, `fdge2`'s reference pipeline (dense hop-fill + shell
   force law) reproduces `fdge_numba/forcedirected_numba/model_204_shell`
   output within float64 tolerance over the first N epochs (N chosen
   large enough to catch drift, small enough to predate any chaotic
   divergence in the physical system itself).
3. **Modularity, demonstrated not asserted**: swapping the augmentation
   policy (e.g. dense hop-fill → a stub sparse policy) requires touching
   only `graph_augmenting/`, with zero changes to `embedding/` or
   `graph_building/`. Same test for swapping the force law (only
   `embedding/` changes) and swapping the graph-building strategy (only
   `graph_building/` changes, already true today via the registry).
4. **No import-boundary violations**: `core/`, `graph_building/`,
   `graph_augmenting/`, `embedding/` respect §5 — checkable with a simple
   static import-graph check (e.g. `grep`/`ast` scan in CI or by a
   reviewing agent).
5. **Memory guardrail holds**: the embedding stage's Numba kernel's peak
   extra memory stays `O(n*d)` — no `(n,n,d)` or full `(nnz,d)`
   temporary array, verified by the same subprocess+`RLIMIT_AS` /
   peak-RSS technique used in `IMPLEMENTATION.md`.
6. **Performance floor**: row-streamed Numba kernel stays within the same
   order of magnitude advantage over vectorized NumPy observed in
   `IMPLEMENTATION.md` (~8x at `avg_degree=10, d=16`) — regression, not
   just presence, i.e. a benchmark that fails loudly if the gap collapses.
7. **Open questions resolved**: Q1–Q3 (§6) have a written decision in
   `API_DESIGN.md`/`ARCHITECTURE.md` before Stage 0 is considered done;
   Q4–Q5 have a written decision or an explicit "deferred, tracked as
   future work" note before Stage 3 (sparse augmentation) is considered
   done.

## 8. Task Breakdown

| ID | Module | Task | Depends on | Complexity | Status |
|---|---|---|---|---|---|
| T0.1 | core | `ForceDirected`/`FDModel` skeleton (`embed`, `updateGradient`, `fit`); resolve Q1–Q3 | — | High | Done — `fdge2/core/force_directed.py` |
| T0.2 | core | Shared CSR/indexable utilities (`graph_to_csr`-equivalent, row helpers) | — | Medium | Done — `fdge2/core/csr.py` |
| T1.1 | graph_building | `make_graph` wrapper over existing `graphmaking/registry.py` | T0.1 | Low | Done — `fdge2/graph_building/__init__.py` |
| T2.1 | graph_augmenting | Port dense hop-fill `augment_graph` (from `get_hops_csr` + shell counts) | T0.1, T0.2 | Medium | Done — `fdge2/graph_augmenting/hopfill.py` |
| T2.2 | graph_augmenting | Design doc + prototype for one sparse augmentation policy (bounded-hop or kNN); addresses Q4/Q5 | T2.1 | High | Done — `fdge2/graph_augmenting/sparse_hops.py` (bounded-hop-radius; Q4 mitigation proposed, not yet built into `embedding/`) |
| T3.1 | embedding | Port `forces`/`_scalar_force` (row-streamed Numba) onto the new `D` contract | T0.1, T0.2, T2.1 | High | Done — `fdge2/embedding/shell_force.py`, bit-identical to legacy kernel |
| T3.2 | embedding | `updateGradient` momentum seam | T3.1 | Medium | Done — passthrough default lives in `core/`, momentum override verified in `embedding/test_shell_force.py` |
| T4.1 | validation | Correctness harness (Numba vs. sequential vs. vectorized NumPy, per `IMPLEMENTATION.md` methodology) | T3.1 | Medium | Done — `fdge2/validation/test_pipeline_behavior.py` |
| T4.2 | validation | Regression parity test vs. `model_204_shell.py` (success metric #2) | T3.1, T2.1 | Medium | Done — `fdge2/validation/test_regression_parity.py`, bit-identical |
| T4.3 | validation | Perf + memory benchmarks (success metrics #5, #6) | T3.1 | Medium | Done — `fdge2/validation/bench_embedding_perf.py`, 33-35x speedup, O(n²) memory |
| T5.1 | docs | Fold resolved open questions back into `ARCHITECTURE.md`/`API_DESIGN.md` | T0.1 | Low | Done |

## 9. Out of Scope / Future Work

- General sparse augmentation beyond one prototype policy (T2.2)
  — genuinely `O(n log n)`/`O(n)` augmentation is future work per
  `ARCHITECTURE.md`.
- Resolving Q4 (repulsion neighborhood under sparse `D`) with a real
  mitigation (sampled negatives, etc.) — documented as an open risk,
  not solved here.
- Any GPU/distributed backend — not mentioned in any source doc, not
  assumed.
