# fdge2 Implementation Notes

This document tracks concrete implementation findings and experiment
results from prototyping the "elementwise combine, then row-reduce"
pattern that the embedding stage's kernels are built on
(see ARCHITECTURE.md for how this fits into the three-stage design).

Prototype script: `sparse_elementwise_test.py` (repo root).

## Sparse representation

Graph-derived quantities that share a sparsity pattern (weights,
per-edge distances recomputed from the current `Z`, per-edge
coefficients — see ARCHITECTURE.md's "shared sparsity pattern" note) are
represented as one CSR topology (`indptr`, `indices`) plus one `(nnz, d)`
`data` array per quantity, matching the pattern `graph_to_csr` already
uses in `forcedirected_numba/model_204_shell.py`:

```
row i's neighbors:  indices[indptr[i] : indptr[i+1]]
row i's data:        data[indptr[i] : indptr[i+1]]   # same positions, paired
```

No separate lookup step is needed to go from a neighbor `v` to its weight
— `v` and its weight are already at the same position `p` in `indices`
and `data`. A `row_of` array (`np.repeat(np.arange(n), np.diff(indptr))`,
giving the source node for every edge position) makes vectorized NumPy
gathers straightforward: `Z[indices] - Z[row_of]` computes every edge's
`Zdiff` in one call.

## Correctness pitfalls found

- **`np.add.reduceat` breaks on empty segments.** The natural way to
  row-sum a `(nnz, d)` array grouped by `indptr` is
  `np.add.reduceat(data, indptr[:-1])`, but a zero-length segment (a
  degree-0 node) doesn't produce `0` — it silently repeats the *previous*
  segment's sum. Since fdge2 graphs can have isolated nodes (and the test
  topology deliberately includes one to exercise this), row-sums were
  implemented instead with `np.bincount(row_of, weights=data[:, t],
  minlength=n)` per dimension `t`, which handles missing indices
  correctly by construction. (`scipy.sparse.csr_matrix.sum(axis=1)` also
  handles this correctly out of the box, if a scipy container is used
  instead of raw CSR arrays.)
- **`scipy.sparse` operator overloading footgun.** `*` between two
  `csr_matrix` objects is matrix multiplication, not elementwise —
  `.multiply()` is needed for the Hadamard product. This didn't end up
  mattering for the row-streamed Numba kernels (everything is scalar
  arithmetic inside the loop, no matrix-level operators at all), but
  would matter for any future vectorized-with-scipy implementation.
- **`fastmath=True` + parallel reduction order is not bit-reproducible**
  against a sequential reference, and this gets worse, not better, as a
  recurrence's magnitude grows — see below.

## Memory guardrail: don't materialize `(n, n, d)`

The dense kernel in `model_204_shell.py` deliberately never builds a full
pairwise-difference tensor; it accumulates per row, peak extra memory
`O(n*d)`. This carries over as a guardrail for fdge2's row-streamed
kernels: "elementwise then row-reduce" is a mathematical description, not
an instruction to materialize the full array before reducing. The
row-streamed Numba kernels satisfy this by construction (one row's worth
of contributions computed and immediately reduced). The vectorized NumPy
implementation below does **not** satisfy it — see the benchmark.

## Prototype: `sparse_elementwise_test.py`

Validates the pattern with a synthetic (non-physical) test formula, over
a random sparse CSR topology with one deliberately isolated node:

```
F[i,j] = Zdiff[i,j] * X[i,j] * exp(Y[i,j]) - exp(Zdiff[i,j])   # elementwise, per dim
Z[i]   = mean_j F[i,j]                                          # mean over node i's edges
LOOP 100x: recompute F from the current Z, then update Z
```

Three implementations, all operating on the same CSR topology:

1. **`compute_F` / `row_mean_update`** — parallel `njit` kernels
   (`prange` over rows), row-streamed, no intermediate arrays beyond a
   reused `F_data` buffer. This is the pattern the real force kernels
   should follow.
2. **`reference_step`** — plain sequential Python/NumPy, one node/edge at
   a time. Slow, but simple enough to trust as a correctness baseline.
3. **`numpy_step`** — vectorized NumPy (fancy-indexing gather +
   elementwise ops across the whole `(nnz, d)` array + `bincount`
   row-sum). No Numba.

**Correctness methodology**: the formula's `-exp(Zdiff)` term is an
unconditional, unopposed positive feedback — `Zdiff` grows each epoch,
`exp(Zdiff)` grows faster, pushing `Z` further, so the recurrence
genuinely diverges past `float64` range within a handful of epochs
regardless of input scale. That's a property of this synthetic test
formula, not a bug. Two consequences for how correctness was checked:
comparing all three implementations **epoch by epoch** (not just after
100 iterations) so a real bug would be caught at the point it first
appears, and **stopping the comparison** the moment any implementation
leaves the finite range, since matching `fastmath=True` parallel output
against a sequential reference bit-for-bit is meaningless once values are
`inf`/`nan` (summation order differs between the two, and differences that
are negligible in a well-behaved range get amplified once the recurrence
is already diverging). With inputs scaled down (`* 0.1` / `* 0.05`), all
three implementations agree through epoch 6 on a 30-node test graph, at
which point the formula overflows for all three simultaneously — expected.
The isolated node is verified to stay exactly at its initial value in
every implementation, throughout.

## Experiment: Numba vs. vectorized NumPy runtime

Fixed `avg_degree=10`, `d=16`, 5 epochs (timing only — the formula's
divergence within a handful of epochs makes longer runs meaningless for
this synthetic test; per-epoch throughput is what's being measured).
JIT warm-up excluded from the Numba timing.

| n | nnz | numba (s) | numpy (s) | numpy / numba | peak RSS |
|---|---|---|---|---|---|
| 25,000 | 249,727 | 0.144 | 1.090 | 7.5x | 336 MB |
| 50,000 | 499,474 | 0.490 | 3.860 | 7.9x | 512 MB |
| 100,000 | 997,709 | 0.913 | 7.395 | 8.1x | 867 MB |
| 200,000 | 1,996,329 | 1.671 | 13.701 | 8.2x | 1.58 GB |
| 400,000 | 3,996,997 | 3.269 | 26.152 | 8.0x | 2.90 GB |

Both scale linearly in `n` (fixed average degree), and the gap holds
steady at **~8x** across the whole range. The likely cause: the
vectorized NumPy step allocates several full `(nnz, d)` temporaries per
epoch (two gathers, `Zdiff`, `exp(Y)`, `exp(Zdiff)`, `F_data`), each a
full memory-bandwidth pass, versus the Numba kernel's single reused
`F_data` buffer and row-streamed accumulation — this matches the
`(n, n, d)`-materialization guardrail above, just at `(nnz, d)` scale
instead of `(n, n, d)` since the topology is sparse.

**Takeaway for the real embedding-stage kernels**: row-streamed `njit`
is the right default for anything performance-sensitive. Vectorized
NumPy is still useful for quickly prototyping a new force law (faster to
write and debug) before porting it into a `njit` kernel — the ~8x gap is
the cost of that convenience, not a reason to avoid it during early
experimentation.

## Experiment: finding a safe max `n` without risking a real OOM

Needed to answer "how large a graph before this runs out of memory,"
but the machine used only had ~5.9 GiB *available* (not total) and is
shared with other processes — actually exhausting system memory risks
the kernel OOM-killer taking down an unrelated process, not just this
one. Instead: each trial `n` ran in its own subprocess with
`resource.setrlimit(resource.RLIMIT_AS, (cap, cap))` set to a fixed 4 GiB
budget (comfortably under the 5.9 GiB available), so an oversized trial
fails with a clean `MemoryError` rather than a real system-level OOM.
Search was doubling to bracket a failure, then bisection to refine.

Result: under a 4 GiB cap, `avg_degree=10`, `d=16` → **max n ≈ 530,000**
(peak measured at 3.52 GiB). This is specific to that budget and those
parameters — memory scales roughly linearly in `n * avg_degree * d`
(a handful of `(nnz, d)` float64 arrays), so the max-`n` figure moves
proportionally if any of those change.
