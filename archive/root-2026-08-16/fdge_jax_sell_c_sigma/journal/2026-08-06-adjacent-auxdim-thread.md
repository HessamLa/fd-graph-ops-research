# 2026-08-06 — Adjacent: aux-dimension annealing (`model_fdmap_auxdim/`)

**Scope note.** This thread is about `model_fdmap_auxdim/`, a research
add-on that *imports* this package but is not part of it. Recorded here
because it happened in the same discussion and because it contains one
**unresolved design decision** that touches `core.ForceDirected`'s callback
contract. The full design doc is `model_fdmap_auxdim/idea.md`; the agent
work log is `model_fdmap_auxdim/agent_log.md`.

## The idea

Embed in `d + aux` dimensions and anneal the `aux` columns toward zero over
training. Extra scaffolding room to route around local minima early,
collapsed away by the end so the returned embedding is genuinely
`d`-dimensional. Shipped as `EuclideanAuxDimModel(EuclideanDistanceModel)`
with five decay schedules (`linear` default), plus demo scripts at 2+2 dims.

## Correction 1: an overclaimed citation

I wrote that this had "real precedent" in the MDS / force-directed
literature without citing anything. The user asked directly:

> Do you have any supporting evidence? In what other work or research paper
> is the idea of auxiliary dimensions used?

Searching found the precedent is **real but thinner** than I had implied.
The doc was corrected to separate genuine matches from
mechanistically-different work:

**Actual precedent**
- Buja et al. 2008, JCGS — "dimension crunching" in MDS / stress
  majorization. The closest true precedent.
- Kim & Han 2025, CAVW — "Dimension Expansion" for mass-spring systems.
  Closest domain match.
- Böhm, Berens & Kobak 2023, ICLR (t-SimCNE) — uses the literal term
  "dimensionality annealing", different mechanism.

**Related but NOT the same mechanism** (I had implied otherwise)
- Klock & Buhmann 2000 — temperature annealing, not dimensions.
- t-SNE early exaggeration — force-strength annealing, not dimensions.
- Harel & Koren 2004 — one-shot PCA projection after a high-D pivot
  embedding, not a gradual anneal.

Lesson, and it recurred later with NetworKit: state plainly when something
is an intuition rather than dressing it as established practice.

## Correction 2: losing sight of the actual question — UNRESOLVED

The user proposed a much simpler design than the subclass:

> For `EuclideanAuxDimModel`, was it possible to simply use an
> `on_epoch_end` callback function? We could setup `n_dim = n_dim + aux_dim`,
> and then, in a callback function, simply apply the diminishing coefficient
> on `dZ[:, aux_dim:]`.

I answered "yes for the shrink step" but then argued a subclass was still
needed for `get_embeddings()` slicing and `Th()` scoping. The user rejected
that, sharply and correctly:

> "Why do I even have to worry about this?" ... "Why should it even be an
> issue? It could easily be overwritten, as it already is! Also, we could
> always use `on_epoch_end` to calculate the desired dimensions, no need to
> even touch `Th()`. I think you are losing sight here!"

They were right on both counts. What I had framed as limitations were not:

- `get_embeddings()` — a one-line slice at the call site, or a
  monkey-patch. Not a reason for a subclass, and a concern under *either*
  design.
- `Th()` / `epsilon` — a non-issue twice over. The demo scripts do not use
  epsilon-based stopping, and `model.stop_training` (checked at the top of
  every loop iteration, with `model.dZ` available un-zeroed inside
  `on_epoch_end`) is the already-documented, callback-reachable mechanism
  for custom early stopping. No `Th()` override needed at all.

### What the loop contract actually guarantees

Confirmed by reading `core/force_directed.py`, and worth recording because
it is the basis for the callback-only design:

- `on_epoch_end` fires strictly **after** `updateZ`.
- At that point `kwargs` already carries both `epoch` and `epochs`,
  matching a self-tracked counter exactly — including resume behaviour
  across repeated `embed()` calls.
- Callbacks **can** read and mutate `model.Z` / `model.dZ`, and **can** set
  `model.stop_training`.
- Callbacks **cannot** intercept `get_embeddings()` (called externally,
  after training) or the internal `self.Th(self.dZ)` call site — but as
  above, neither limitation matters in practice.

### The open decision

The whole `EuclideanAuxDimModel` subclass reduces to a small
`AuxDimDecay(Callback_Base)` plus a slicing convention — fully generic
across *any* `ForceDirected` model, not just the Euclidean one.

**I proposed this and the user has not confirmed it. Nothing has been
changed.** `EuclideanAuxDimModel` still exists as a subclass. If this
thread is picked up again, that is the first thing to settle.

## Current state

- `EuclideanAuxDimModel` implemented, 7 tests. The load-bearing one is
  `test_aux0_parity_with_base_model`: at `aux=0` it must be **bit-identical**
  to a plain `EuclideanDistanceModel` (max abs diff `0.000e+00`), proving
  the machinery is genuinely inert when switched off.
- Two demo scripts, both verified end to end.
- **Known failing test, pre-existing and unrelated to this package's
  refactor**: `test_all_schedules_ramp_from_one_to_zero`. Its
  `__init__.py` exports `linear_sine_decay` while `__all__` lists
  `linear_cosine`. Confirmed to fail identically before any of the
  2026-08-08 work. Left alone as out of scope.

## Why this matters to `fdge_jax_sell_c_sigma`

The callback-vs-subclass question is really a question about **this**
package's extension contract. If a decaying-auxiliary-dimension model can
be written as a callback, then `Callback_Base` is a stronger extension
point than currently advertised, and the same argument likely applies to
other research variants. Worth settling before more `ForceDirected`
subclasses accumulate.
