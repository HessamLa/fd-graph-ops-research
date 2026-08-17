# 2026-08-08 — The two force-calculation modes, naming, and the 100k OOM

## The user's framing

Two modes of force-directed embedding:

1. **All node pairs.** `d(u_i, u_j)` defined for every `(i, j)` — a
   complete graph, a dense `n x n` matrix. No pairing policy needed.
2. **A chosen subset of pairs.** Augmentation adds only a fraction of the
   possible edges. Which pairs is a *policy* decision, made in stage 2.

And the load-bearing empirical claim:

> Between the two modes above, we have seen that the **Some-pairs** mode has
> equal performance as **All-pairs** mode, while being much more memory
> efficient.

That is why the package is built on a sparse `D`. Both modes must stay
available — mode 1 is simpler and is the correctness reference.

## Naming

The user asked for better names than "All-pairs"/"Some-pairs".

**Recommended: `coupling="complete"` vs `coupling="selective"`.**

Reasoning: `is_sparse` already exists and means *storage container*.
Reusing "sparse"/"dense" for *which pairs exist* is precisely the collision
that makes the current code confusing. "Coupling" is the physics term for
which pairs exert force on each other, and it is orthogonal to storage —
`coupling="selective", is_sparse=True` reads correctly. Second choice:
`pair_policy="complete" | "selected"`, more literal.

**Prior art exists for the distinction** (not for these names): the
graph-drawing literature calls mode 1 the **full stress model** and mode 2
the **sparse stress model** — Ortmann, Klimenta & Brandes, *A Sparse Stress
Model*, JGAA 21(5):791–821, 2017 ([arXiv:1608.08909](https://arxiv.org/abs/1608.08909)).
Their term-aggregation method for picking representatives is directly
relevant to designing new policies. Their naming is rejected here only
because "sparse" is taken.

I searched to confirm this citation rather than asserting it from memory —
an earlier overclaim in this project made that necessary.

## The 100k OOM, diagnosed

> The current implementation already crashes at 100K nodes with OOM.

Measured the machine: **15 GB total RAM, ~7 GB free → a dense `(n,n)`
float64 tops out near n = 30k.**

So `squareform(pdist(data))` in `graph_building` wanted **80 GB on a 15 GB
box**. Not a leak, not subtle — an impossibility. That changes what a fix
must do: *remove* the dense allocation, not shrink it.

Three `O(n^2)` sites, in the order they bite:

1. `mst_edges` — dense pdist. **Fixed** (below).
2. `hopfill` — `O(n^2)` by construction. This *is* mode 1, so quadratic is
   expected; it should fail early with a clear error rather than OOM.
3. The shell-count table — `(n, n+1)` on disconnected graphs. **Fixed**
   (see the [engine split](2026-08-08-engine-force-law-split.md)).

## The MST fix, and a non-obvious finding

Replaced with an MST over an exact `sklearn` kNN graph. Measured:

| n | peak now | dense equivalent |
|---|---|---|
| 2,000 | 1.3 MB | 0.03 GB (25x) |
| 10,000 | 6.0 MB | 0.80 GB (132x) |
| 30,000 | 17.9 MB | 7.20 GB (401x) |
| **100,000** | **59.3 MB, 27 s** | **80 GB** |

Quality is better than "acceptable": on clustered fixtures the kNN-graph
MST has **total weight exactly equal** to the true dense MST (ratio
1.000000), not merely close.

**The finding worth keeping: raising `k` does NOT fix kNN-graph
disconnection.** On well-separated clusters the graph stayed split into
exactly `n_clusters` components at every `k` tested up to 30, because no
cross-cluster point is ever anyone's k-nearest neighbour *at any `k`*. The
obvious mitigation is simply wrong.

The actual fix: a centroid-MST chooses which components to bridge, then one
exact 1-NN query per bridge finds the true nearest cross-component pair.
`O(n log n)`, never `O(n^2)`. Verified connected on 5/20/50-cluster
adversarial data.

A subtle bug surfaced during this: `.kneighbors(data)` with `data` passed
explicitly does **not** auto-exclude self in sklearn (unlike
`kneighbors_graph`), silently returning `k-1` real neighbours. Caught by
cross-checking edge counts against the old dense implementation — the
degree floor was still technically met, so a weaker check would have missed
it.

## What is now the binding constraint

With `graph_building` fixed, **dense `hopfill` is the remaining `O(n^2)`
site on the default path**. The open item is deciding a hard `n` ceiling
for `coupling="complete"` and raising a clear error above it.

## Policies still to explore (selective coupling)

Recorded in `TODOS.md` §1.4:

- **k-hop balls** — what `sparse_hops` already does; formalize under the
  new naming with `k` as the primary knob.
- **k-hop balls + high-degree hubs** — restores the long-range coupling a
  pure radius cut removes. Needs only a degree sort, so no new dependency.
  Cheapest to try first.
- **k-hop balls + inter-community edges** — needs community detection; the
  sole surviving reason to consider NetworKit (see
  [that thread](2026-08-08-networkit-evaluation.md)).
- **Sampled negatives** for repulsion — the documented Q4 risk. An
  `embedding/`-stage mitigation, not an augmentation one.

`ForceDirected.augment_graph` is the intended extension point: subclass and
override that one method.
