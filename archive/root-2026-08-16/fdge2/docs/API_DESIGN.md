# fdge2 API Design

## Goals

This is research code: the audience is someone trying a new augmentation
policy or force law next week, not an external consumer of a stable API.
The class design should make the common research loop —
"change one function, rerun" — as friction-free as possible, while
keeping the hot path (the embedding stage's inner loop) fast. Concretely:

- Each of the three stages (ARCHITECTURE.md) should map to **one small,
  overridable method**, so a researcher can override exactly one of them
  and inherit the rest unchanged.
- The base class (`ForceDirected`, from `forcedirected_numba`) already
  provides working, tested loop machinery — batching, callbacks, momentum,
  convergence checking. fdge2 models should **reuse it**, not reimplement
  it, so that machinery doesn't have to be re-validated every time.
- Follow the existing convention in `model_204_shell.py` of a short
  "researcher note" comment at the one function that's meant to be edited
  for a given kind of experiment (there: `_scalar_force`, "edit me to try
  new force laws").

## Class skeleton

```python
class ForceDirected:
    def updateGradient(self, Z, D, ...): 
        dZ = self.forces(Z, D, ...)
        # The researcher might want to over-ride this funcion, such as the following
        # self.v = alpha*self.v + beta*self.forces(self.Z, D, ...)
        # dZ = self.v
        return dZ

    def embed(self, G, epochs: int = 1000, dZ_threshold=None, ...):
        # This method is computationally heavy
        D = self.augment_graph(G, ...)
        LOOP:
          # Calculate gradient ()
          self.dZ = self.updateGradient(self.Z, self.D, ...)
          # Update embeddings
          self.Z += self.dZ

    def fit(self, data, epochs: int = 1000, ...):
        G = self.make_graph(data, ...)
        return self.embed(G, epochs=epochs, ...)

class FDModel(ForceDirected):

    def make_graph(self, data, ...):
        raise NotImplementedError  # left to the researcher / experiment

    def augment_graph(self, G, ...):
        raise NotImplementedError  # e.g. hop-distance fill, kNN augmentation, ...

    def forces(self, ...):
        raise NotImplementedError  # the force law, evaluated per edge

```

Usage:

```python
# Embedding an existing graph
model = FDModel(...)
Z = model.embed(G, ...)

# Fitting raw data
model = FDModel(...)
Z = model.fit(data, ...)
```

## How `embed` and `updateGradient` compose

This is the one point in the design that took a few iterations to settle,
so it's worth stating the resolved contract explicitly. The current
skeleton moves the loop and the momentum-style seam onto `ForceDirected`
itself, with `FDModel` overriding only the three per-experiment hooks:

- `embed(self, G, ...)` still takes the **un-augmented** graph `G`, not a
  pre-augmented `D`. Its first step is `D = self.augment_graph(G, ...)`.
  (An earlier draft had `embed` take `D` directly, with the caller
  responsible for calling `augment_graph` first — rejected in favor of
  the single-call ergonomics of `model.embed(G, ...)`.)
- `embed` now owns the epoch loop directly (`self.dZ = self.updateGradient(...)`,
  `self.Z += self.dZ`, repeated) rather than delegating to a separately
  imported base method. It's marked "computationally heavy" in the
  skeleton on purpose — this loop is the hot path, and it's the reason
  `forces` (and anything `updateGradient` does) needs to follow the
  row-streamed, no-large-intermediates guidance from ARCHITECTURE.md /
  IMPLEMENTATION.md.
- `updateGradient(self, Z, D, ...)` is the new seam between the raw force
  law and the update actually applied to `Z`. Its default is a pure
  passthrough (`return self.forces(Z, D, ...)`), which is equivalent to
  today's `model_204_shell.py` behavior — plain gradient descent, no
  momentum. Momentum/velocity (or gradient clipping, or any other
  transform between "raw force" and "step taken") is now expressed as an
  **override of `updateGradient`**, e.g.:

  ```python
  def updateGradient(self, Z, D, ...):
      self.v = alpha * self.v + beta * self.forces(Z, D, ...)
      return self.v
  ```

  rather than as a fixed `beta`/`V` parameter baked into the loop. This
  is more flexible than `forcedirected_numba.ForceDirected`'s current
  hard-coded momentum formula, at the cost of the researcher having to
  write that formula out when they want it.
- `forces` keeps the role `_forces_204_hidx` plays in `model_204_shell.py`
  today: the one function to edit for a new force law. It's now called
  explicitly as `self.forces(Z, D, ...)` from `updateGradient`, rather than
  being reached indirectly through a `forward(row_start, row_end, ...)`
  batch hook.
- `fit(data, ...)` is unchanged: a thin convenience, `make_graph` then
  `embed`.

**Open question this raises** (see "Open items" below): the previous
draft got batching (`batch_count`), callbacks (`on_epoch_begin/end`,
`on_batch_begin/end`), and convergence checking (`epsilon`/`Th`) for free
by delegating to `forcedirected_numba.ForceDirected.embed(...)`. The
current skeleton's `embed` shows a bare loop instead, so it's not yet
settled whether those are still assumed to happen underneath this
pseudocode (elided for brevity) or whether they need to be reintroduced
explicitly now that `embed` isn't calling `super().embed(...)` anymore.

## The indexability contract

Every graph-building output (`G` or `D`, dense or sparse) must support
`G[i, j]` / `D[i, j]` — this is a **Python-level, outward-facing**
contract (testing, debugging, small/dense graphs), not a requirement on
how the performance-critical kernels access the data:

- **Dense**: an `(n, n)` (or `(n, n, d)`) NumPy array satisfies this
  natively, and it's also what a `njit` kernel would index directly — no
  tension here.
- **Sparse**: `scipy.sparse.csr_matrix` (or similar) satisfies `[i, j]` at
  the Python level. But `njit` kernels never do random-access `[i, j]`
  lookups on sparse data — that would mean an `O(deg)` (or `O(log deg)`)
  search per query. Instead they **iterate** each row's actual entries
  (`indptr`/`indices`/`data`), which is both what "only compute forces
  between connected edges" (the whole point of going sparse) requires,
  and what's actually available for free while walking the row. `D` is
  passed explicitly through `updateGradient(Z, D, ...)` into `forces(Z, D,
  ...)` — whichever representation it is, `forces` is where that
  row-iteration happens. See IMPLEMENTATION.md for the CSR access
  pattern used in practice.

So: `[i, j]` indexing is a correctness/convenience guarantee on the
*output* of graph-building, decoupled from how the embedding stage's
kernels are actually written.

## Guidance for researchers

The three things this design expects to change often, and where:

- **New graph-building algorithm** → implement/override `make_graph`.
  Not prescribed by fdge2; whatever produces a NetworkX-style graph (or
  directly a `D`-compatible structure) is fine.
- **New augmentation policy** (e.g. bounded-hop-radius instead of full
  BFS fill, kNN-based augmentation) → override `augment_graph`. Its
  contract is fixed: takes `G`, returns an indexable weighted `D` (dense
  or sparse, implementation's choice).
- **New force law** → edit `forces` (and whatever scalar/vector force
  function it calls). Should not require touching `embed`,
  `updateGradient`, or graph-building at all — matching the existing
  `_scalar_force` convention of being the one function to edit for this
  kind of experiment.
- **New gradient-shaping behavior** (momentum, velocity, gradient
  clipping, anything that transforms a raw force into the step actually
  applied to `Z`) → override `updateGradient`, without touching `forces`
  or `embed`. The default `updateGradient` is a pure passthrough to
  `forces`, so this override point is opt-in.

## Composition root

A single runnable model needs all three stage hooks — `make_graph`,
`augment_graph`, `forces` — implemented at once. But by the import
discipline (RPD.md §5) the three stage packages (`graph_building/`,
`graph_augmenting/`, `embedding/`) must never import one another. The
resolution: **composition happens one level up**, in `fdge2/models.py`
(not part of `core/`), which is allowed to import `core/` and all three
stage packages and wire them onto a concrete subclass:

```python
from fdge2.core import ForceDirected
from fdge2 import graph_building, graph_augmenting, embedding

class ReferenceFDModel(ForceDirected):
    def make_graph(self, data, **kw):
        return graph_building.make_graph(data, **kw)
    def augment_graph(self, G, **kw):
        return graph_augmenting.hopfill.augment_graph(G, **kw)
    def forces(self, Z, D, row_start, row_end, **kw):
        return embedding.shell_force.forces(Z, D, row_start, row_end, **kw)
```

Nothing in `core/` assumes or hardcodes a specific augmentation or force
implementation — the abstract-hook design (`make_graph`/`augment_graph`/
`forces` raise `NotImplementedError` in `core.ForceDirected`) gives that
for free. `core/` knows the three stages' *contracts*, never their
internals, and never imports a stage package back (no circular
dependency).

## Open items

> **Resolved during Stage 0 (T0.1, core skeleton).** The three items
> below are kept for historical context; each now carries its resolution.
> See `core/force_directed.py` for the implementation.

- **Q2 — `make_graph`'s `is_sparse` flag. RESOLVED:** `make_graph` takes
  **no** `is_sparse` flag — it just returns a graph. Choosing dense
  ndarray vs. `scipy.sparse.csr_matrix` for `D` is a concrete
  `augment_graph` implementation's decision, so the flag lives on
  `augment_graph(self, G, is_sparse: bool = True, ...)`. `core/` only
  documents the expectation (the returned `D` must satisfy the
  indexability contract either way); it does not enforce the
  representation. *Rationale: keeps `make_graph`'s contract to "returns a
  graph" and puts the density knob where the density decision is actually
  made.*

- **Q3 — does `forces(Z, D, ...)` batch internally? RESOLVED: yes, it
  keeps row-range batching.** `forces(self, Z, D, row_start, row_end,
  ...)` computes and returns the dZ rows for `[row_start, row_end)`,
  mirroring today's `forward(row_start, row_end, ...)`. `embed`'s batch
  loop (Q1) calls it once per contiguous row batch. *Rationale: the batch
  loop needs a per-range entry point, and this preserves the reference's
  contiguous-row-batch semantics unchanged.*

- **Q1 — do batching / callbacks / convergence still apply under the
  simplified `embed`? RESOLVED: yes, reintroduced explicitly.** Now that
  `embed(G, ...)` owns the loop directly (no `super().embed(...)`
  delegation), `batch_count` (contiguous row ranges), the callback hooks
  (`on_train_begin/end`, `on_epoch_begin/end`, `on_batch_begin/end`), and
  `epsilon`/`Th` convergence are ported into `core.ForceDirected.embed`
  nearly verbatim from `forcedirected_numba.ForceDirected`, adapted to the
  augment-first signature. Momentum, however, is **not** reintroduced into
  the loop — it moved to the `updateGradient` seam (a subclass override),
  replacing the reference's hardcoded `beta`/`V` term in `updateZ`.
  *Rationale: `test_fd_numba.py` exercises batching invariance, callback
  ordering, and epsilon convergence; keeping them means the ported loop
  passes equivalent tests. See `core/test_core_smoke.py`.*

  Batching-invariance note: it holds for the **default passthrough**
  `updateGradient` (forces are batch-local; `Z` updates once per epoch). A
  momentum `updateGradient` override that maintains a whole-array velocity
  should run with `batch_count == 1`.
