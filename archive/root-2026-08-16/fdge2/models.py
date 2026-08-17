"""fdge2.models -- the composition root.

Per API_DESIGN.md's "Composition root" section: a single runnable model
needs all three stage hooks (``make_graph``, ``augment_graph``, ``forces``)
implemented at once, but the three stage packages (``graph_building/``,
``graph_augmenting/``, ``embedding/``) must never import one another (RPD.md
Sec 5, import discipline). This module is the one place that is allowed
to import ``core`` and all three stage packages, and it does so only to
wire them onto a concrete ``ForceDirected`` subclass -- nothing here
implements pipeline logic itself.

``ReferenceFDModel`` reproduces the existing, proven
``fdge_numba/forcedirected_numba/model_204_shell.FDModel`` behavior
(dense hop-fill augmentation + shell-averaged force law) through the new
three-stage seams, for numeric-parity testing (RPD.md success metric #2)
and as the template new researchers copy to swap in a different stage.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from fdge2.core import ForceDirected
from fdge2 import graph_building, graph_augmenting, embedding


class ReferenceFDModel(ForceDirected):
    """Dense hop-fill augmentation + shell-averaged force law.

    The reference pipeline, wired from independently-built stage packages:

    - ``make_graph``    -> ``graph_building.make_graph``    (thin re-export
      of ``graphmaking/registry.py``)
    - ``augment_graph`` -> ``graph_augmenting.hopfill.augment_graph`` (T2.1)
    - ``forces``        -> an ``embedding.shell_force.ShellForce`` instance
      (T3.1/T3.2), held on ``self._shell`` so its per-``D`` derived-quantity
      cache persists across a whole ``embed()`` call.

    Swap any one stage by subclassing and overriding just that method --
    the whole point of the three-stage split (ARCHITECTURE.md).
    """

    def __init__(self, *args,
                 k1: float = 0.999, k2: float = 1.0, k3: float | None = 10.0,
                 k4: float = 0.01, random_drop_rate: float = 0.5,
                 **kwargs) -> None:
        # k3=10.0 matches model_204_shell.FDModel's actual constructor
        # default -- see ShellForce's docstring for why this must not
        # silently become None (None resolves to n_nodes downstream, a
        # ~180x larger repulsion coefficient on a 1797-node graph).
        super().__init__(*args, **kwargs)
        self._shell = embedding.shell_force.ShellForce(
            k1=k1, k2=k2, k3=k3, k4=k4, random_drop_rate=random_drop_rate)

    def make_graph(self, data, **kwargs):
        self.G = graph_building.make_graph(data, **kwargs)
        return self.G

    def augment_graph(self, G, **kwargs):
        self.D = graph_augmenting.hopfill.augment_graph(G, **kwargs)
        return self.D

    def forces(self, Z, D, row_start, row_end, **kwargs):
        # degrees intentionally omitted: ShellForce derives + caches them
        # from D (count of hop-1 entries per row) when not supplied.
        return self._shell(Z, D, row_start, row_end, rng=self.rng, **kwargs)

class EuclideanDistanceModel(ForceDirected):
    """Same as ``ReferenceFDModel`` except for the distance policy.

    - **Making graph**: build the graph topology the usual way, via
      ``graph_building.make_graph`` (MST/kNN/... -- caller picks the
      strategy through ``**kwargs``, same as ``ReferenceFDModel``), then
      set every edge's ``weight`` attribute to the Euclidean distance
      between the two endpoints in the *original* (raw, pre-embedding)
      data space: ``edge(u,v).weight = ||data[u] - data[v]||``. Only the
      edge weights are new here; the topology comes from whichever
      strategy is requested. The output graph MUST BE one component.

    - **Augmenting graph**: ``D[u,v]`` is the shortest **weighted** path
      distance between ``u`` and ``v`` (all-pairs Dijkstra over the
      weighted graph above) -- not a hop count, unlike
      ``ReferenceFDModel``'s dense hop-fill. For disconnected pairs (no
      path at all -- shouldn't happen given the "one component"
      precondition above, but handled defensively), ``D[u,v]`` is set to
      the **maximum** weighted distance actually present among connected
      pairs in the graph -- the "as far as it gets" value, on the same
      numeric scale as every other entry (unlike hopfill's node-count
      sentinel, which was a hop count on a hop-count scale).

      The shell-averaged force law still needs ``|S_h(u)|`` grouped by
      *hop* count -- shell membership is a graph-topology notion, not a
      distance notion (ARCHITECTURE.md, "Derived quantities live between
      stages"). So ``augment_graph`` also computes the ordinary
      unweighted hop matrix by calling
      ``graph_augmenting.hopfill.augment_graph`` on the same graph
      (reusing that BFS rather than re-deriving it) and stashes it on
      ``self._hops`` for ``forces`` to read. ``augment_graph``'s return
      value is still just ``D`` (the weighted matrix), matching the
      fixed one-value contract (API_DESIGN.md) -- ``self._hops`` (and the
      cached ``self._h_min``, see below) are auxiliary state, the same
      pattern ``ReferenceFDModel`` already uses for ``self.G``/``self.D``.

    - **Force functions**: the same shell-averaged Fa/Fr formula as
      ``ReferenceFDModel`` (k1..k4), with one adjustment for the new
      distance scale. ``h`` in the formula is now the *weighted* distance
      ``D[u,v]`` instead of the hop count, and the attractive term's
      ``exp(-k2 * (h - 1))`` -- where ``1`` was "the hop distance of a
      direct edge," the baseline where attraction is strongest -- becomes
      ``exp(-k2 * (h - h_min))``, where ``h_min = min(D)`` over connected,
      distinct pairs (computed once in ``augment_graph``, cached as
      ``self._h_min``): the closest actual distance any two nodes achieve
      plays the role "hop distance 1" played on the hop-count scale.
      ``shell_coeff = 1/|S_h(u)|`` is still grouped by hop count (from
      ``self._hops``). That decoupling -- shell index from one matrix,
      the force law's ``h`` argument from another -- is why this class
      can't reuse ``embedding.shell_force.ShellForce`` as a black box
      (it assumes one matrix drives both); it does reuse
      ``embedding.shell_force._derive_from_D`` for shell-counting and
      mirrors ``_scalar_force``'s exact formula in ``forces`` below.
      Row-streamed plain NumPy, not Numba -- this is a new/experimental
      variant, not the proven hot-path kernel, and ARCHITECTURE.md
      documents vectorized NumPy as the right tool for "quick prototyping
      of new force laws before porting to njit."

      Caveat worth knowing: ``h``'s numeric scale is now a raw Euclidean
      distance rather than a small hop count (1..maxhops), so the k1..k4
      defaults inherited from ``ReferenceFDModel`` (tuned for the
      hop-count regime) may not behave sensibly on a differently-scaled
      dataset -- retuning per-dataset is left to the caller, not done
      automatically here.
    """

    def __init__(self, *args,
                 k1: float = 0.999, k2: float = 1.0, k3: float | None = 10.0,
                 k4: float = 0.01, random_drop_rate: float = 0.5,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k1, self.k2, self.k3, self.k4 = k1, k2, k3, k4
        self.random_drop = embedding.shell_force.DropSteadyRate(
            drop_rate=random_drop_rate)

        self._hops = None          # unweighted hop matrix, shell-grouping only
        self._shell_cache_hops = None
        self._shell_cache = None   # {"hops_idx", "shell_counts", "degrees", ...}
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
        """D = all-pairs weighted shortest path; self._hops = hop count."""
        # Shell grouping needs hop count, not weighted distance -- reuse the
        # existing BFS-based augmentation rather than re-deriving it.
        self._hops = graph_augmenting.hopfill.augment_graph(G, is_sparse=False)

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
        # same numeric scale as every other entry, unlike hopfill's
        # node-count sentinel (which made sense for hop counts, not for
        # distances on a genuinely different scale).
        dist[np.isinf(dist)] = d_max

        D = dist.astype(np.float64)
        self.D = sp.csr_matrix(D) if is_sparse else D
        return self.D

    # ------------------------------------------------------------- stage 3
    def _shell_derived(self):
        """Shell-counting derived from self._hops, cached.

        Same identity-cache reasoning as embedding.shell_force.ShellForce:
        self._hops is constant for a whole embed() call, so this should
        compute once (it's O(n) to O(n^2)-ish work), not once per epoch.
        """
        if self._hops is None:
            raise RuntimeError(
                "augment_graph must run before forces() -- call embed(G, ...), "
                "don't call forces() directly.")
        if self._shell_cache is not None and self._shell_cache_hops is self._hops:
            return self._shell_cache
        der = embedding.shell_force._derive_from_D(self._hops)
        self._shell_cache = der
        self._shell_cache_hops = self._hops
        return der

    def forces(self, Z, D, row_start, row_end, **kwargs):
        """embedding.shell_force._scalar_force's Fa/Fr formula, adapted for
        a weighted ``h``: ``h`` = the weighted distance D[u,v] (not the hop
        count), the attractive term's baseline shifts from hop distance 1
        to ``self._h_min`` (the closest actual distance any two nodes
        achieve), and shell_coeff is still grouped by hop count
        (self._hops) -- see class docstring. Row-streamed (one row's
        neighbors materialized at a time, never (n,n,d)) but plain NumPy,
        not njit.
        """
        if self._h_min is None:
            raise RuntimeError(
                "augment_graph must run before forces() -- call embed(G, ...), "
                "don't call forces() directly.")
        der = self._shell_derived()
        hops_idx = der["hops_idx"]
        shell_counts = der["shell_counts"]
        degrees = der["degrees"]

        Dd = D.toarray() if hasattr(D, "toarray") else np.asarray(D)
        n, d = Z.shape
        _k3 = self.k3 if self.k3 is not None else float(n)

        out = np.zeros((row_end - row_start, d), dtype=np.float64)
        for r in range(row_end - row_start):
            u = row_start + r
            hi_row = hops_idx[u]
            mask = hi_row > 0                     # exclude self (hi == 0)
            if not mask.any():
                continue
            diff = Z[mask] - Z[u]                 # (k, d) -- this row only
            x = np.linalg.norm(diff, axis=1)
            nz = x > 0
            if not nz.any():
                continue
            diff, x = diff[nz], x[nz]
            h = Dd[u][mask][nz]                   # weighted distance, not hop count
            shell_coeff = 1.0 / shell_counts[u][hi_row[mask][nz]]

            Fa = self.k1 * shell_coeff * x * np.exp(-self.k2 * (h - self._h_min))
            Fr = -_k3 * h * np.exp(-self.k4 * x)
            f = Fa + Fr

            acc = ((f / x)[:, None] * diff).sum(axis=0)
            deg = degrees[u]
            out[r] = acc / deg if deg > 0 else 0.0

        return self.random_drop(out, self.rng)
