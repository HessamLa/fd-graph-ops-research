# 2026-08-06 — `graph_to_csr` / `row_of` / `n_rows`: keep or drop?

Follow-on to the [CSR container decision](2026-08-06-csr-container-decision.md).
With `HopMatrix` gone, the question moved to the three remaining helpers.

## The question

> Why do we need `graph_to_csr` / `row_of` / `n_rows` while we could use
> `scipy.sparse`?

## First answer, and where it was wrong

I initially defended all three:

- `graph_to_csr` — "scipy has no graph→CSR constructor; the closest is
  `networkx.to_scipy_sparse_array`, which hard-requires a `networkx.Graph`
  rather than any duck-typed graph."
- `row_of` — cheap `np.repeat` trick; scipy's equivalent (`D.tocoo().row`)
  pays a format conversion.
- `n_rows` — "literally just `D.shape[0]`; the weakest of the three."

The user pushed on the first point:

> How about `NetworkX.to_scipy_sparse_array(.)`? [...] NetworkX is one of
> the requirements of this module.

That is decisive and my objection was weak. The "duck-typed graph" contract
was self-imposed and unused — every caller passes a real `networkx.Graph`.
Defending a hand-rolled builder to preserve a generality nobody uses is
exactly the "speculative abstraction" the project's own design doc rejects.

## Two real gotchas found while switching

Not objections, but things that would have silently corrupted results:

1. **`weight="weight"` is the default.** `nx.to_scipy_sparse_array` bakes
   edge weights into the matrix values. `EuclideanDistanceModel` builds a
   *weighted* graph, so without `weight=None` the "unweighted hop topology"
   feeding shell-grouping would have silently become weighted. Both
   augmentation policies now pass `weight=None` explicitly.
2. **Node ordering.** Must pass `nodelist=list(G.nodes())` to get the
   deterministic `0..n-1` row order the rest of the package assumes.
   `graph_to_csr` guaranteed this by construction; scipy does not unless
   told.

## The `indptr` detour

> What is `indptr` anyway?

`indptr` is CSR's row-boundary index: an `(n+1,)` array where row `i`'s
entries are `indices[indptr[i]:indptr[i+1]]`. `indptr[0] = 0`,
`indptr[n] = nnz`, and `np.diff(indptr)` gives each row's entry count
(here, its degree). It is what makes CSR compact — instead of storing a row
index per entry the way COO does, it stores only the *boundaries*, since
entries are already grouped by row.

Then:

> scipy csr already has the indptr equivalent. Does it not? Write a very
> short sample script and test it.

Confirmed empirically on a triangle graph: `A.indptr = [0 2 4 6]`,
`A.indices = [1 2 0 2 0 1]`, `A.data = [1 1 1 1 1 1]`. `graph_to_csr` was
building by hand exactly what the library already returns.

## Why `row_of` is orthogonal (asked, and worth recording)

> Why do we need "per-edge row lookup"? Why is it "orthogonal"?

`row_of` does not *build* a CSR matrix — it *consumes* one. CSR stores
"which row owns entry `k`" implicitly, via `indptr` boundaries. Vectorized
per-edge math (`Z[indices] - Z[row_of(indptr)]`, every edge's difference
vector in one shot) needs that row id materialized as a flat array aligned
with `indices`.

It is orthogonal because it operates downstream, on `D.indptr`, and does
not care what produced the matrix. Changing the construction method does
not remove the need for it. The scipy-native equivalent, `D.tocoo().row`,
is a full format conversion for something derivable from `indptr` alone.

## Outcome

- `graph_to_csr` **deleted**. Both policies now call
  `nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None, format="csr")`.
- `build_adjacency` (in `_levelwise_reach.py`) deleted — its only job was
  wrapping `graph_to_csr`'s output.
- `row_of` **kept**, docstring extended with the `tocoo().row` comparison.
- `n_rows` **kept**, though it remains a one-line wrapper over `.shape[0]`;
  its only value is not caring whether `D` is dense or sparse.
- `graph_augmenting/` gained `networkx` as a declared dependency. Noted in
  `docs/DESIGN.md` as a deliberate deviation from `fdge_jax`'s import table.

## Test fallout

`core/test_core_smoke.py::test_csr_helpers` had built a fake duck-typed
graph object to exercise `graph_to_csr`. With the duck-typed contract gone,
that fixture tested nothing real — replaced with an actual
`nx.cycle_graph(4)`.
