"""fodiwalk.base -- `Fodiwalk_base`, the contract of the Fodiwalk family.

`core.ForceDirected` is a pure engine: give it `D`, it relaxes
`Z` against it. This class is the layer above -- the PIPELINE contract
every Fodiwalk-family model must satisfy to go from raw data to an
embedding: `make_graph` (stage 1), `graph_walk` and `augment_graph` (stage
2), `forces` (stage 3, already declared abstract on the engine -- not
duplicated here), and `embed` (the ORCHESTRATION: build `D`, then hand it
to the engine's own `embed`).

Every method here RAISES. A concrete subclass (`fodiwalk.Fodiwalk`, or a
future one) implements every one of them; this class exists so the
contract is stated in ONE place and not re-derived from what a leaf class
happens to define.

`fit()` does NOT exist on this class, or anywhere in the hierarchy
(REMOVED 2026-08-21): `make_graph` is a stub in every concrete model of
this project, and a `fit` that chained it into `embed` promised a stage
that was never real. `dev-docs/CATALOG.md`, the section on the class
split, records the reason.

Import discipline: `core` only. `core.force_directed` is a FORWARDER
since 2026-08-26; the engine itself is the root package `forcedirected`.
"""
from __future__ import annotations

from .core.force_directed import ForceDirected


class Fodiwalk_base(ForceDirected):
    """The abstract Fodiwalk pipeline. Every high-level stage hook, unimplemented."""

    def make_graph(self, data, **kwargs):
        """Stage 1: build a graph `G` from raw `data` (MST, kNN, an edge
        list on disk, ...). Returns any indexable weighted/unweighted
        graph; the density of the eventual `D` is `augment_graph`'s
        decision, not this one's."""
        raise NotImplementedError("make_graph(.) is not implemented")

    def graph_walk(self, A, n: int | None = None, rng=None, **kwargs):
        """The walk-family hook: the random walks over `A`, and the
        statistics of the pairs they give. Every policy of this project is
        walk-based; a future non-walk policy may leave this raising."""
        raise NotImplementedError("graph_walk(.) is not implemented")

    def augment_graph(self, G, **kwargs):
        """Stage 2: turn `G` into a weighted matrix `D` (add/reweight
        edges), and prepare the DATA the law needs -- the planes, the
        degrees, the force params (`dev-docs/fodiwalk-module.md`: stage 3
        does no data preparation, thus this stage does all of it). `D`
        must satisfy the indexability contract (`D[i, j]` works) -- a
        dense ndarray or a `scipy.sparse.csr_matrix`."""
        raise NotImplementedError("augment_graph(.) is not implemented")

    def embed(self, G, epochs: int = 1000, lr: float | None = None, Z=None,
              batch_count: int = 1, epsilon: float | None = None, **kwargs):
        """The orchestration: `augment_graph(G)` into `D`, ONE time, then
        the engine's `ForceDirected.embed(D, ...)` relaxes it.
        `G` is the UN-augmented graph -- everything `forces` needs is
        already in `D` by the time the engine sees it."""
        raise NotImplementedError("embed(.) is not implemented")
