# 2026-08-06 — The `D` container: `HopMatrix` vs `scipy.sparse.csr_matrix`

## The question

> Why are we creating a csr class of our own and not using a famous
> library? What are we gaining by this?

`core/csr.py` defined `HopMatrix`, a hand-written CSR triple with
`__getitem__`, `toarray()`, `shape`, `nnz`, and `__slots__`.

## What the code claimed

Its module docstring gave three justifications. Checked each:

| Claim | Verdict |
|---|---|
| "gives `embedding/` direct CSR access with no densify-then-reparse step" | **False as a distinction.** `scipy.sparse.csr_matrix` exposes `.indptr` / `.indices` / `.data` as plain numpy arrays too. This is equally true of scipy, so it distinguishes nothing. |
| "Import discipline: this module imports **numpy only**" | **False as a constraint.** Self-imposed in this file's own docstring, and already broken package-wide: `graph_augmenting`, `graph_building`, and `EuclideanDistanceModel` all import `scipy.sparse` directly. scipy was already an unconditional dependency. |
| "one consistent representation, replacing fdge2's duck-typed dense-or-sparse `D`" | **True**, but scipy is also one consistent representation. |

Two of the three did not survive contact.

## What was genuinely gained

Not nothing, to be fair:

- **Enforced dtypes.** `HopMatrix.__init__` forced `int64` indptr /
  `int32` indices+data on every instance. `csr_matrix` enforces nothing.
- **A closed API surface.** `__slots__` plus six methods, versus scipy's
  entire sparse-matrix API (arithmetic operators, format conversions,
  sparse linalg) — none of which this package uses, all of which a reader
  now has to know does not apply.
- **The domain contract on the type**: "0 is never a stored value; absent
  means not-yet-reached, not hop-distance-zero."

## The user's reframing, which settled it

> HopMatrix is a utility class which is supposed to be used to find the
> graph geodesic distance of two nodes [...] In a general case
> `D[i,j] ~ d(u_i, u_j)` [...] With unweighted graphs, `d(.)` is the
> shortest graph path length. With weighted graphs, `d(.)` may indicate the
> least costly path, or hop distance, or any other distance function that
> we choose.

This is the important point and it is not really about containers. `D` is
a **distance matrix with a pluggable distance function**. Its identity is
the semantics of `d(.)`, not the storage type. A bespoke class named after
*hops* actively misleads, because hop distance is only one choice of
`d(.)`. `EuclideanDistanceModel` was already using a different one.

Decision: use `scipy.sparse.csr_matrix`. Delete `HopMatrix`.

## What changed

- `HopMatrix` deleted from `core/csr.py`; removed from `core/__init__.py`.
- `hopfill` / `sparse_hops` already built a `csr_matrix` internally
  (`coo.tocsr()`) before wrapping it — they now just return it.
- `embedding`: `_to_hopmatrix` → `_to_csr` (dispatch on `sp.issparse(D)`
  rather than `hasattr(D, "indptr")`); every `D.n` → `D.shape[0]`.
- `isinstance(D, HopMatrix)` checks → `sp.issparse(D)`.

## One thing that broke, and why it was left broken

`embedding/test_sell_c_sigma.py`'s cross-engine parity test feeds our `D`
into the **sibling** `fdge_jax` package, which still has its own
`HopMatrix` and calls `D.n`. Fixed on our side by passing `D.toarray()` to
the sibling engine — same values, a container both accept. `fdge_jax` is
out of scope and was deliberately not touched.

## Notes for later

- `nx.to_scipy_sparse_array` returns `scipy.sparse.csr_array`, not
  `csr_matrix`. Slightly different class, smaller API — notably **no
  `.getnnz()`**. Everything this package uses (`.indptr`, `.indices`,
  `.data`, `.shape`, `.nnz`, `[i,j]`, `.toarray()`, `sp.issparse`) works on
  both. Use `np.diff(A.indptr)` for per-row counts; it works on either.
- The enforced-dtype property was lost with the class. Nothing has depended
  on it so far, but it is now a convention rather than an invariant.
