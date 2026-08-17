"""fdge_jax_sell_c_sigma.models -- the composition root.

A single runnable model needs all three stage hooks (``make_graph``,
``augment_graph``, ``forces``) implemented at once, but the three stage
packages (``graph_building/``, ``graph_augmenting/``, ``embedding/``) must
never import one another (docs/DESIGN.md import discipline). This module
is the one place allowed to import ``core`` and all three stage packages,
and does so only to wire them onto a concrete ``ForceDirected`` subclass --
nothing here implements pipeline logic itself.

Three models here, matching ``fdge_jax/models.py``'s own set:

- ``ReferenceFDModel`` -- dense hop-fill augmentation + the SELL-C-sigma
  bucketed force law. Numerically comparable to ``fdge_jax``'s own
  ``ReferenceFDModel`` (same physics, same augmentation policy), just a
  different embedding engine.
- ``SparseFDModel`` -- swaps only ``augment_graph`` for the genuinely
  sparse bounded-radius policy (``graph_augmenting.sparse_hops``). This is
  the model to reach for on anything beyond a small test graph: dense
  hop-fill's ``D`` is ``O(n^2)`` in stored entries regardless of engine
  (see ``docs/DESIGN.md``'s note on why large-graph runs, including this
  package's own benchmark, use ``SparseFDModel``, not
  ``ReferenceFDModel``).
- ``EuclideanDistanceModel`` -- same shell-averaged force law, but ``D`` is
  a dense all-pairs WEIGHTED shortest-path distance instead of a hop count
  (see its own class docstring). Ported from ``fdge_jax``'s model of the
  same name; uses ``embedding.shell_force.WeightedShellForce``, i.e. the
  SAME kernel, cache and compile path as the other two models, rather than
  a second engine of its own.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.sparse.csgraph import dijkstra

from fdge_jax_sell_c_sigma.core import ForceDirected
from fdge_jax_sell_c_sigma import graph_building, graph_augmenting, embedding


class ReferenceFDModel(ForceDirected):
    """Dense hop-fill augmentation + SELL-C-sigma bucketed force law.

    The reference pipeline, wired from independently-built stage packages:

    - ``make_graph``    -> ``graph_building.make_graph``
    - ``augment_graph`` -> ``graph_augmenting.hopfill.augment_graph``
    - ``forces``        -> an ``embedding.shell_force.ShellForce``
      instance held on ``self._sell`` so its per-``D`` plan + compiled-
      kernel cache persists across a whole ``embed()`` call.

    Swap any one stage by subclassing and overriding just that method --
    the whole point of the three-stage split. ``SparseFDModel`` below is
    exactly that: only ``augment_graph`` changes.
    """

    def __init__(self, *args,
                 k1: float = 0.999, k2: float = 1.0, k3: float | None = 10.0,
                 k4: float = 0.01, random_drop_rate: float = 0.5,
                 b_cells: int = 16_384, k_max: int = 256, ladder_base: float = 1.5,
                 **kwargs) -> None:
        # k3=10.0 matches the model's actual out-of-the-box constructor
        # default -- see ShellForce's docstring for why this must not
        # silently become None (None resolves to n_nodes downstream, a
        # much larger repulsion coefficient on any graph of real size).
        super().__init__(*args, **kwargs)
        self._sell = embedding.ShellForce(
            k1=k1, k2=k2, k3=k3, k4=k4, random_drop_rate=random_drop_rate,
            b_cells=b_cells, k_max=k_max, ladder_base=ladder_base)

    def make_graph(self, data, **kwargs):
        self.G = graph_building.make_graph(data, **kwargs)
        return self.G

    def augment_graph(self, G, **kwargs):
        self.D = graph_augmenting.hopfill.augment_graph(G, **kwargs)
        return self.D

    def forces(self, Z, D, row_start, row_end, **kwargs):
        # degrees intentionally omitted: ShellForce derives + caches them
        # from D (count of hop-1 entries per row) when not supplied.
        return self._sell(Z, D, row_start, row_end, **kwargs)


class SparseFDModel(ReferenceFDModel):
    """Same as ``ReferenceFDModel``, but augmented with the genuinely
    sparse bounded-radius hop-fill (``graph_augmenting.sparse_hops``)
    instead of dense hop-fill. Only ``augment_graph`` changes -- the force
    law, the loop, and everything else is inherited unchanged, which is
    the whole point of keeping the stages independent.

    This is the model to use for anything beyond a small test/parity
    graph, and specifically the one this package's own large-graph
    benchmark (``validation/bench_embedding_perf.py``) uses: dense
    hop-fill's ``D`` is dense by construction (every reachable pair
    stored, including an ``unreachable = n`` sentinel for disconnected
    pairs) -- ``O(n^2)`` regardless of which embedding engine consumes it
    afterward, so it's unusable at the 100k-500k node scale this engine
    targets no matter how fast the bucketed kernel itself is.

    ``radius`` (default 2, forwarded to ``sparse_hops.augment_graph`` via
    ``embed(G, radius=...)`` / ``**kwargs``) trades off physics fidelity
    (bigger radius, more complete shells, closer to dense hop-fill)
    against the genuinely sub-quadratic memory/compute win. See
    ``graph_augmenting/sparse_hops.py``'s module docstring for the
    documented Q4 caveat (repulsion collapse risk for pairs the augmenter
    never stores) before relying on this for a real run.
    """

    def augment_graph(self, G, **kwargs):
        self.D = graph_augmenting.sparse_hops.augment_graph(G, **kwargs)
        return self.D


class EuclideanDistanceModel(ForceDirected):
    """Same as ``ReferenceFDModel`` except for the distance policy.

    Ported from ``fdge_jax.models.EuclideanDistanceModel`` -- read that
    class's docstring for the full "why" of the physics; this docstring
    covers only what's different about running it through the bucketed
    engine instead of the flat ``segment_sum`` one.

    - **Making graph** and **augmenting graph** are IDENTICAL to
      ``fdge_jax``'s version: ``graph_building.make_graph`` for topology,
      then every edge's ``weight`` set to the Euclidean distance between
      the two endpoints in the original (pre-embedding) data space (the
      output graph MUST be one connected component); ``self.D`` is the
      dense ``(n, n)`` all-pairs weighted shortest path (Dijkstra), and
      ``self._hops`` is the ordinary unweighted hop matrix (reusing
      ``graph_augmenting.hopfill.augment_graph`` on the same graph) so the
      shell-averaging (`|S_h(u)|`, grouped by *hop* count) still means
      something -- shell membership is a topology notion, weighted
      distance is not. One small improvement over ``fdge_jax``'s version:
      ``self._hops`` is fetched directly as a ``scipy.sparse.csr_matrix``
      (``is_sparse=True``) instead of densified then re-sparsified, since
      the bucketed plan builder wants CSR directly anyway.

    - **Force law**: same shell-averaged Fa/Fr formula as
      ``ReferenceFDModel``, with ``h`` replaced by the *weighted* distance
      ``D[u, v]`` and the attractive term's baseline shifted from "hop
      distance 1" to ``h_min`` (the closest weighted distance any two
      nodes actually achieve, computed once in ``augment_graph``).

    - **Execution engine**: ``embedding.shell_force.WeightedShellForce`` --
      the same force law, the same plan builder, the same compiled kernel
      and the same per-``D`` cache as the other two models. The only
      difference is which matrix feeds which part: the plan comes from
      ``self._hops`` (topology + shell-count grouping) while the ``h``
      coefficient plane comes from ``self.D`` (weighted distance) gathered
      at ``self._hops``'s own stored ``(u, v)`` positions. Both live inside
      ``WeightedShellForce``; this file just hands it the two matrices via
      ``bind_topology`` and lets it do the rest.

      Caveat worth knowing (unchanged from ``fdge_jax``): ``h``'s numeric
      scale is now a raw Euclidean distance rather than a small hop count,
      so the k1..k4 defaults inherited from ``ReferenceFDModel`` (tuned for
      the hop-count regime) may not behave sensibly on a differently-scaled
      dataset -- retuning per-dataset is left to the caller.
    """

    def __init__(self, *args,
                 k1: float = 0.999, k2: float = 1.0, k3: float | None = 10.0,
                 k4: float = 0.01, random_drop_rate: float = 0.5,
                 b_cells: int = 16_384, k_max: int = 256, ladder_base: float = 1.5,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sell = embedding.WeightedShellForce(
            k1=k1, k2=k2, k3=k3, k4=k4, random_drop_rate=random_drop_rate,
            b_cells=b_cells, k_max=k_max, ladder_base=ladder_base)
        self._hops = None          # unweighted csr_matrix, shell-grouping + topology
        self._h_min = None         # min(D) over connected, distinct pairs

    # ------------------------------------------------------------- stage 1
    def make_graph(self, data, **kwargs):
        """Topology from ``graph_building`` + Euclidean edge weights."""
        G = graph_building.make_graph(data, **kwargs)
        X = np.asarray(data)
        for u, v in G.edges():
            G.edges[u, v]["weight"] = float(np.linalg.norm(X[u] - X[v]))
        self.G = G
        return G

    # ------------------------------------------------------------- stage 2
    def augment_graph(self, G, is_sparse: bool = True, **kwargs):
        """D = all-pairs weighted shortest path (dense); self._hops = hop count.

        ``is_sparse`` is accepted for signature compatibility with the
        base class's convention but has no effect -- ``D`` is always a
        dense ndarray (Dijkstra distances, not a structural pattern; see
        class docstring).
        """
        # Shell grouping needs hop count, not weighted distance -- reuse the
        # existing BFS-based augmentation rather than re-deriving it. Fetched
        # as a csr_matrix directly (unlike fdge_jax's dense round-trip): the
        # bucketed plan builder consumes CSR, so there's nothing to gain from
        # densifying first.
        self._hops = graph_augmenting.hopfill.augment_graph(G, is_sparse=True)

        nodes = list(G.nodes())
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight="weight",
                                      format="csr")
        dist = dijkstra(A, directed=False)

        n = dist.shape[0]
        off_diag = ~np.eye(n, dtype=bool)
        finite = np.isfinite(dist) & off_diag
        # d_max/h_min are both properties of the *connected* pairs only --
        # computed before filling in the disconnected sentinel below (which
        # can't change either: it's set to d_max itself, so it can't lower
        # the min, and it's not larger than d_max by construction).
        d_max = float(dist[finite].max()) if finite.any() else 0.0
        self._h_min = float(dist[finite].min()) if finite.any() else 0.0

        # Disconnected pairs (shouldn't occur given the "one component"
        # precondition, but handled defensively): dijkstra gives inf. Use
        # the maximum weighted distance actually present in the graph --
        # same numeric scale as every other entry.
        dist[np.isinf(dist)] = d_max

        # float64 host-side (dijkstra's natural precision); the plan builder
        # downcasts to float32 itself when gathering the h-plane, same as
        # every other array in this float32-by-default package.
        self.D = dist.astype(np.float64)

        # Same breath as `self.D`, deliberately: the engine's plan cache is
        # keyed on self.D's object identity and cannot see that _hops or
        # h_min changed underneath it (embedding/sell_c_sigma.py's
        # PlanCache docstring states this invariant). Binding them here,
        # right where the fresh self.D is produced, is what keeps it true.
        self._sell.bind_topology(self._hops, self._h_min)
        return self.D

    # ------------------------------------------------------------- stage 3
    def forces(self, Z, D, row_start, row_end, **kwargs):
        return self._sell(Z, D, row_start, row_end, **kwargs)

