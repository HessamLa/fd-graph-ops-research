# journal/

Design discussions, in the order they happened. One file per thread.

These are **discussion records**, not changelogs. They keep the reasoning,
the rejected alternatives, the things that turned out wrong, and the open
questions. `logs/master.log` is the execution trace of the 2026-08-08
refactor; `TODOS.md` is the forward-looking work list; these are the *why*.

Filename convention: `YYYY-MM-DD-slug.md`, dated by when the discussion
happened.

| File | Thread | Outcome |
|---|---|---|
| [2026-08-06-csr-container-decision.md](2026-08-06-csr-container-decision.md) | Why a bespoke `HopMatrix` instead of `scipy.sparse`? | Class deleted. Two of its three stated justifications did not hold. |
| [2026-08-06-graph-to-csr-removal.md](2026-08-06-graph-to-csr-removal.md) | Do we need `graph_to_csr` / `row_of` / `n_rows`? | `graph_to_csr` deleted for `nx.to_scipy_sparse_array`. Other two kept. |
| [2026-08-07-architecture-audit.md](2026-08-07-architecture-audit.md) | Full audit: library-replaceable code, misplaced methods, core vs not-core | Found the `self.Gx` crash, an `O(n^2)` memory bug, and the engine/force-law tangle. |
| [2026-08-08-engine-force-law-split.md](2026-08-08-engine-force-law-split.md) | Separating the force law from the SELL-C-sigma layout machinery | Three-file split. `force_fn` seam. `SellCSigmaForce` dissolved. |
| [2026-08-08-complexity-modes-and-oom.md](2026-08-08-complexity-modes-and-oom.md) | The two force-calculation modes; naming; the 100k OOM | Recommended `coupling="complete"/"selective"`. OOM diagnosed and fixed. |
| [2026-08-08-networkit-evaluation.md](2026-08-08-networkit-evaluation.md) | Would NetworKit help? | Measured. No. Revisit only for `networkit.community`. |
| [2026-08-08-jax-profiling-research.md](2026-08-08-jax-profiling-research.md) | What profiling APIs exist and which metrics we want | API surface verified against the installed JAX. Not yet implemented. |
| [2026-08-06-adjacent-auxdim-thread.md](2026-08-06-adjacent-auxdim-thread.md) | Aux-dimension annealing (`model_fdmap_auxdim/`) | Adjacent package. Contains one **unresolved** design decision. |

## Recurring lessons

Three mistakes recur across these threads. They are recorded in each file,
but collected here because the pattern matters more than the instances.

1. **Documentation-level research presented as an answer.** The NetworKit
   section was written from a docs page and a plausible mapping onto our
   needs, with the real decision deferred. The user pushed back, correctly.
   Installing and benchmarking reversed the conclusion for the path that
   actually matters. Ask "which of these numbers did I actually measure?"
2. **Justifications that sound right and are not.** `core/csr.py`'s
   docstring gave three reasons for `HopMatrix`; two were false (scipy also
   exposes raw CSR arrays; the "numpy only" rule was already broken
   elsewhere in the package). Stated rationale is not evidence.
3. **Overclaiming citations.** An earlier design doc asserted literature
   precedent without a reference. Searching found the real precedent was
   thinner and different from what was implied. Verify before citing.
