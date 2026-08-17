# fdge2 Architecture

## Motivation

Mathematically this is still a Force-Directed graph embedding, as in
`forcedirected_numba`. The difference is structural: `forcedirected_numba`'s
current model (`model_204_shell.py`) computes an all-pairs hop-distance
matrix (via BFS) and evaluates the force law over every reachable pair,
which is `O(n^2)` regardless of how sparse the original graph is. fdge2
splits graph construction apart from force computation so that sparsity
(when the augmentation strategy provides it) can actually be exploited,
and so new graph-building or force-law ideas can be tried without touching
the other parts.

This is a **research codebase**, not a production one. The design
priorities, in order, are:

1. **Fast to modify** — swapping in a new graph-building algorithm, a new
   augmentation policy, or a new force law should mean editing one small,
   well-isolated piece, not tracing through the whole pipeline.
2. **Fast to run** — large-graph experiments are the point, so the hot
   path (force computation, run every epoch) still has to be efficient.
3. **Not defensive** — no need for the input validation, error handling,
   or backwards-compatibility scaffolding a shipped library would need.

## Three stages

The embedding pipeline is broken into three compartments:

1. **Making graph** — `make_graph(data, ...)`: builds a graph `G` from
   raw data (e.g. MST, kNN). Left to the user/researcher to implement per
   experiment; not prescribed by this architecture.
2. **Augmenting graph** — `augment_graph(G, ...)`: takes a graph `G`
   (weighted or unweighted) and produces a weighted matrix `D` — adding or
   modifying edges. This is where today's `get_hops_csr` logic
   (`forcedirected_numba/model_204_shell.py`) lives conceptually: it
   connects every reachable pair `(u, v)` with an edge weighted by hop
   distance `d(u, v)`. `d_uv = 1` for original edges (hop distance 1),
   `d_uw` = hop distance for the rest.
3. **Embedding graph** — the force functions and the epoch loop. `D`
   (the augmented weighted matrix) is a **constant input** to this stage;
   nothing here knows or cares how `D` was produced. Today's
   `_forces_204_hidx` / `_scalar_force` play this role.

```
Embedding:            G (weighted/unweighted) -> [Augmenting Graph] -> D -> [Embed] -> Z
Fitting (or Mapping):  data -> [Making Graph] -> G -> [Augmenting Graph] -> D -> [Embed] -> Z
```

Reference mapping from the current codebase to these three stages:

| Stage | Function |
|---|---|
| Making graph | `make_graph` in `fdge_numba/graphmaking/registry.py` (directory exists, not yet implemented — left to the researcher) |
| Augmenting graph | `get_hops_csr` in `fdge_numba/forcedirected_numba/model_204_shell.py` |
| Embedding graph | `_forces_204_hidx`, `_scalar_force` in the same file |

Other functions in the existing code prepare intermediate variables and
coefficient matrices (e.g. shell counts) and don't concretely belong to
one stage — see "Derived quantities" below.

## Derived quantities live between stages, not inside a stage's output

Example: `S_h(u)`, the set of nodes in the shell (hop-distance group) of
`u`, is not part of what `augment_graph` returns. It's derivable from `G`
(or from `D`), and whichever stage needs it computes it from the matrix it
already has. This keeps `augment_graph`'s contract simple ("returns a
weighted graph") instead of growing a second, force-law-specific return
value.

## Representation: dense vs. sparse

Density/sparsity of `D` is an implementation choice, not an architectural
one. Both representations need to satisfy the same outward contract (see
API_DESIGN.md): `G[i,j]` / `D[i,j]` must be indexable.

> **Resolved (Q2, Stage 0 / T0.1):** the `is_sparse` knob lives on
> `augment_graph(self, G, is_sparse: bool = True, ...)`, **not** on
> `make_graph` — `make_graph` just returns a graph, and whether the
> resulting `D` is a dense ndarray or a `scipy.sparse.csr_matrix` is the
> augmentation implementation's decision. `core/` documents the
> indexability expectation but does not enforce the representation. See
> API_DESIGN.md "Open items" for the full resolution.

**Known limitation, acknowledged and left as future work**: the current
`get_hops_csr`-based augmentation fills in a hop distance for *every*
reachable pair, which makes `D` dense in practice (`O(n^2)`) even when the
original graph is sparse, and even if `D` is stored in a sparse container.
Genuinely sparse augmentation (bounded hop radius, kNN-style, etc.) can
bring this down to `O(n log n)` or `O(n)`, but that's a property of the
*augmentation policy*, not something the embedding stage or the storage
format can fix on their own.

## Shared sparsity pattern across matrices

Within the embedding stage, several quantities derived from `D` (weights,
per-edge distances computed from the current `Z`, per-edge coefficients)
end up sharing `D`'s exact nonzero pattern — same node pairs, same order.
That guarantee is what makes efficient sparse computation practical here:
elementwise combination of two same-pattern sparse quantities reduces to a
plain, aligned 1-D array operation on their `.data` arrays, with no
pattern-merging step. General sparse-sparse arithmetic (two matrices with
*different* patterns) is more expensive and isn't needed anywhere in this
design as currently scoped.

## Two execution strategies for the embedding stage

Two ways to implement "elementwise combine, then row-reduce" for the force
computation, both mathematically equivalent:

- **Row-streamed** (what `_forces_204_hidx` does today, and what fdge2's
  Numba kernels use): loop over rows/edges, compute each edge's
  contribution as a scalar/vector, accumulate directly into the per-row
  output. Peak extra memory is `O(n*d)` — no large intermediate array is
  ever materialized. This is the pattern to prefer for the actual force
  kernels.
- **Vectorized** (plain NumPy, no Numba): gather the needed values via
  fancy indexing, combine with normal NumPy operators across the whole
  `(nnz, d)` array at once, then reduce per row (e.g. via `np.bincount`
  per dimension). Simpler to read and debug, but materializes several full
  `(nnz, d)` temporary arrays per epoch. See IMPLEMENTATION.md for the
  measured cost of that difference (~8x slower than the streamed Numba
  version across a range of graph sizes, mostly allocation/bandwidth
  overhead, not algorithmic).

Row-streamed execution is the target for the real, performance-sensitive
kernels. Vectorized NumPy is useful as a second, independent
implementation for correctness-checking (see IMPLEMENTATION.md) and for
quick prototyping of new force laws before porting them to a `njit`
kernel.

## Open questions — status

- **Q4 — repulsion restricted to `N(u)`? RISK CONFIRMED, MITIGATION
  DESIGNED, IMPLEMENTATION DEFERRED (T2.2).** Yes, today's force law
  (`embedding/shell_force.py`) only sees pairs present in `D`'s support,
  so once augmentation goes genuinely sparse (see Q5), pairs absent from
  `D` get zero force in either direction. `fdge2/graph_augmenting/sparse_hops.py`
  demonstrates the collapse risk concretely (`test_collapse_risk_is_real`)
  and proposes a **sampled-negatives mitigation**: per row `u`, draw `m`
  random node ids and add a `1/m`-scaled repulsion term reusing the
  existing `_scalar_force` repulsive branch. It plugs into
  `forces(self, Z, D, row_start, row_end, **kwargs)` **without changing
  the signature** — `neg_samples=m` would arrive via the `**kwargs`
  `embed` → `updateGradient` → `forces` already threads, `n` comes from
  `D.shape[0]`, randomness from the model's existing `self.rng`. Building
  this into `embedding/` is explicit future work, not yet implemented —
  see `graph_augmenting/sparse_hops.py`'s docstring for the full proposal.
- **Q5 — concrete sparse augmentation policy. RESOLVED (T2.2).**
  Implemented as bounded-hop-radius BFS
  (`graph_augmenting/sparse_hops.py::augment_graph(G, radius=2, ...)`): fills
  `D[u,v]` only for pairs within `radius` hops, leaving farther pairs
  **structurally absent** (not stored zeros). Chosen over a kNN-based
  policy because the shell-averaging force law divides by *complete*
  shell counts `|S_h(u)|` — bounded-radius keeps every retained shell
  exact, where a naive kNN cutoff could truncate mid-shell and corrupt
  the denominator. One deliberate semantic change from dense hop-fill:
  no `unreachable` sentinel — "beyond radius" and "different component"
  are both simply absent. Measured empirically sub-quadratic: `nnz/n`
  stays flat (68 → 73, a 1.07x spread) across a 16x increase in `n`
  (500 → 8000, avg_degree=8, radius=2) — O(n), not O(n²).
