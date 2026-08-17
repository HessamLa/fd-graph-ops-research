"""fdge_jax.models -- the composition root.

A single runnable model needs all three stage hooks (``make_graph``,
``augment_graph``, ``forces``) implemented at once, but the three stage
packages (``graph_building/``, ``graph_augmenting/``, ``embedding/``) must
never import one another (docs/DESIGN.md import discipline). This module
is the one place allowed to import ``core`` and all three stage packages,
and does so only to wire them onto a concrete ``ForceDirected`` subclass
-- nothing here implements pipeline logic itself.

``ReferenceFDModel`` reproduces ``fdge2.models.ReferenceFDModel``'s
behavior (dense hop-fill augmentation + shell-averaged force law) through
the JAX stages, for numeric-parity testing (see validation/) and as the
template a researcher copies to swap in a different stage -- e.g.
``SparseFDModel`` below swaps only ``augment_graph`` for the genuinely
sparse bounded-radius policy, changing nothing else.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import dijkstra

from fdge_jax.core import ForceDirected
from fdge_jax import graph_building, graph_augmenting, embedding


class ReferenceFDModel(ForceDirected):
    """Dense hop-fill augmentation + shell-averaged force law (JAX kernel).

    The reference pipeline, wired from independently-built stage packages:

    - ``make_graph``    -> ``graph_building.make_graph``
    - ``augment_graph`` -> ``graph_augmenting.hopfill.augment_graph``
    - ``forces``        -> an ``embedding.shell_force.ShellForce`` instance
      held on ``self._shell`` so its per-``D`` derived-array + compiled-
      kernel cache persists across a whole ``embed()`` call.

    Swap any one stage by subclassing and overriding just that method --
    the whole point of the three-stage split.
    """

    def __init__(self, *args,
                 k1: float = 0.999, k2: float = 1.0, k3: float | None = 10.0,
                 k4: float = 0.01, random_drop_rate: float = 0.5,
                 **kwargs) -> None:
        # k3=10.0 matches model_204_shell.FDModel's actual constructor
        # default -- see ShellForce's docstring for why this must not
        # silently become None (None resolves to n_nodes downstream, a
        # much larger repulsion coefficient on any graph of real size).
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
        return self._shell(Z, D, row_start, row_end, **kwargs)


def _euclidean_step(Z, u_of, v_of, h_dist, shell_coeff, degrees_safe,
                     degrees_zero, k1, k2, k3, k4, h_min, n):
    """One fused pass over every (u, v) pair -> full (n, d) dZ, weighted-distance variant.

    Same flat-edge-list + segment_sum shape as ``embedding.shell_force``'s
    ``_step``, adapted for ``EuclideanDistanceModel``'s two-matrix physics:
    ``h_dist`` (the weighted shortest-path distance, gathered once from
    ``self.D`` during derivation -- it doesn't depend on ``Z``, so this
    gather is cacheable, not a per-epoch cost) plays the role fdge2's
    ``_scalar_force`` gives to hop distance, with the attractive term's
    baseline shifted from "hop distance 1" (nonsensical on this scale) to
    ``h_min`` (the closest weighted distance any two nodes actually
    achieve). ``shell_coeff`` is still grouped by *hop* count (from
    ``self._hops``, gathered by the shared ``embedding.shell_force``
    derivation) -- see ``EuclideanDistanceModel``'s class docstring for why
    this decoupling means it can't reuse ``ShellForce`` as a black box.

    Researcher note -- edit the two FORCE LAW lines below to try a new
    distance-based force law for this model; nothing else needs to change.

    ``indices_are_sorted=True`` on the ``segment_sum`` below: ``u_of``
    comes from ``embedding.shell_force._derive_from_D``, which builds it
    via ``core.csr.row_of(D.indptr)`` on a valid CSR ``indptr`` --
    non-decreasing by construction, so this hint is always safe here (see
    ``embedding/shell_force.py``'s ``_step`` docstring for the full
    explanation and the measured ~50x this avoids: without it, XLA's GPU
    lowering does a general atomic-scatter-add, and on GPUs lacking
    native fp64 atomics the massive per-row write contention -- every
    stored pair sharing a source row targets the same output address --
    forces a compare-and-swap retry loop that serializes almost
    completely).
    """
    diff = Z[v_of] - Z[u_of]                          # (nnz, d)
    x = jnp.linalg.norm(diff, axis=-1)                 # (nnz,)
    x_safe = jnp.where(x == 0, 1.0, x)

    # ---- FORCE LAW: edit these two lines to try a new force law --------
    Fa = k1 * shell_coeff * x * jnp.exp(-k2 * (h_dist - h_min))  # attractive
    Fr = -k3 * h_dist * jnp.exp(-k4 * x)                          # repulsive
    # ----------------------------------------------------------------------

    scale = jnp.where(x == 0, 0.0, (Fa + Fr) / x_safe)
    contrib = scale[:, None] * diff
    sums = jax.ops.segment_sum(contrib, u_of, num_segments=n,
                                indices_are_sorted=True)
    return jnp.where(degrees_zero[:, None], 0.0, sums / degrees_safe[:, None])


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
      pairs -- the "as far as it gets" value, on the same numeric scale as
      every other entry (unlike hopfill's node-count sentinel, which was a
      hop count on a hop-count scale). ``D`` is always a plain dense
      ``(n, n)`` float64 ndarray -- ``core.csr.HopMatrix`` doesn't fit here
      (its ``.data`` is int32 hop counts by design), and unlike hop-fill
      there's no sparse container that would carry real information either
      way (this matrix is dense by nature: dijkstra distances, not a
      structural pattern).

      The shell-averaged force law still needs ``|S_h(u)|`` grouped by
      *hop* count -- shell membership is a graph-topology notion, not a
      distance notion. So ``augment_graph`` also computes the ordinary
      unweighted hop matrix by calling
      ``graph_augmenting.hopfill.augment_graph`` on the same graph
      (reusing that BFS rather than re-deriving it) and stashes it on
      ``self._hops`` for ``forces`` to read. ``augment_graph``'s return
      value is still just ``D`` (the weighted matrix) -- ``self._hops``
      (and the cached ``self._h_min``, see below) are auxiliary state, the
      same pattern ``ReferenceFDModel`` uses for ``self.G``/``self.D``.

    - **Force law**: the same shell-averaged Fa/Fr formula as
      ``ReferenceFDModel`` (k1..k4), with one adjustment for the new
      distance scale: ``h`` in the formula is the *weighted* distance
      ``D[u,v]`` instead of the hop count, and the attractive term's
      ``exp(-k2 * (h - 1))`` becomes ``exp(-k2 * (h - h_min))`` where
      ``h_min = min(D)`` over connected, distinct pairs (computed once in
      ``augment_graph``, cached as ``self._h_min``) plays the role "hop
      distance 1" played on the hop-count scale. ``shell_coeff`` is still
      grouped by hop count (from ``self._hops``).

    - **Execution engine**: unlike fdge2's version of this class (plain
      row-streamed NumPy -- documented there as a quick-prototyping
      variant, not the proven hot kernel), this port runs a ``jax.jit``'d
      flat-edge-list + ``segment_sum`` kernel, the same shape as
      ``embedding.shell_force``'s ``_step``. This is possible without
      violating the "(n, n, d) never materialized" guardrail because
      hop-fill's ``D``/``self._hops`` is dense by construction anyway (see
      ``graph_augmenting/hopfill.py``'s documented known limitation) --
      the edge list here has ``nnz = n*(n-1)``, same scale ``ShellForce``
      already handles for the dense-hopfill case, so reusing that pattern
      is a straightforward win over a slow Python row loop, not a new
      scalability claim. ``h_dist`` (the weighted distance per edge) is
      gathered from ``self.D`` **once** per ``embed()`` call (it doesn't
      depend on ``Z``) and cached alongside the compiled kernel, mirroring
      ``ShellForce``'s per-``D`` identity cache -- see
      ``embedding.shell_force._derive_from_D``, reused directly here for
      the shell-grouping half of the derivation (same reuse fdge2's own
      version makes of this "private" helper -- ``models.py`` is allowed
      to reach into ``embedding/``'s internals, unlike the stage packages
      themselves).

      Caveat worth knowing: ``h``'s numeric scale is now a raw Euclidean
      distance rather than a small hop count (1..maxhops), so the k1..k4
      defaults inherited from ``ReferenceFDModel`` (tuned for the
      hop-count regime) may not behave sensibly on a differently-scaled
      dataset -- retuning per-dataset is left to the caller.
    """

    def __init__(self, *args,
                 k1: float = 0.999, k2: float = 1.0, k3: float | None = 10.0,
                 k4: float = 0.01, random_drop_rate: float = 0.5,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k1, self.k2, self.k3, self.k4 = k1, k2, k3, k4
        self.random_drop_rate = random_drop_rate

        self._hops = None          # unweighted hop matrix, shell-grouping only
        self._h_min = None         # min(D) over connected, distinct pairs

        # single-slot identity cache (same pattern as ShellForce -- see
        # embedding/shell_force.py's module docstring for "why identity").
        self._cache_D = None
        self._cache = None
        self.n_derivations = 0
        self._fallback_root_key = None

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
        base class's convention but has no effect -- see class docstring
        for why this matrix is always a dense ndarray.
        """
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
        # same numeric scale as every other entry.
        dist[np.isinf(dist)] = d_max

        self.D = dist.astype(np.float64)
        return self.D

    # ------------------------------------------------------------- stage 3
    def _derived(self):
        """Return cached per-``D`` device arrays + compiled kernel, computing once.

        Reuses ``embedding.shell_force._derive_from_D(self._hops)`` for the
        shell-grouping half (``u_of``, ``v_of``, ``shell_coeff``,
        ``degrees_*``) -- its own ``h``/``step`` fields are discarded (they'd
        be grouped by hop count, not the weighted distance this model's
        force law actually needs). ``h_dist`` is gathered from ``self.D``
        at the same edge positions, once, since ``self.D`` is constant for
        the whole ``embed()`` call.
        """
        if self._hops is None or self._h_min is None:
            raise RuntimeError(
                "augment_graph must run before forces() -- call embed(G, ...), "
                "don't call forces() directly.")
        if self._cache is not None and self._cache_D is self.D:
            return self._cache

        der = embedding.shell_force._derive_from_D(self._hops)
        u_of_np = np.asarray(der["u_of"])
        v_of_np = np.asarray(der["v_of"])
        h_dist = self.D[u_of_np, v_of_np].astype(np.float64)
        n = der["n"]

        self._cache = {
            "u_of": der["u_of"],
            "v_of": der["v_of"],
            "shell_coeff": der["shell_coeff"],
            "degrees_safe": der["degrees_safe"],
            "degrees_zero": der["degrees_zero"],
            "h_dist": jax.device_put(h_dist),
            "n": n,
            "step": jax.jit(functools.partial(_euclidean_step, n=n)),
        }
        self._cache_D = self.D
        self.n_derivations += 1
        return self._cache

    def _fallback_key(self):
        """Non-reproducible fallback key when the caller doesn't pass one
        (mirrors ``ShellForce``'s ``_fallback_key`` -- see its docstring)."""
        if self._fallback_root_key is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
            self._fallback_root_key = jax.random.PRNGKey(seed)
        self._fallback_root_key, sub = jax.random.split(self._fallback_root_key)
        return sub

    def forces(self, Z, D, row_start, row_end, key=None, **kwargs):
        der = self._derived()
        k3 = float(der["n"]) if self.k3 is None else self.k3

        Zj = jnp.asarray(Z)
        full = der["step"](Zj, der["u_of"], der["v_of"], der["h_dist"],
                            der["shell_coeff"], der["degrees_safe"],
                            der["degrees_zero"], self.k1, self.k2, k3,
                            self.k4, self._h_min)
        out = full[row_start:row_end]

        if key is None:
            key = self._fallback_key()
        return embedding.shell_force.drop_steady_rate(
            out, key, self.random_drop_rate)


class SparseFDModel(ReferenceFDModel):
    """Same as ``ReferenceFDModel``, but augmented with the genuinely
    sparse bounded-radius hop-fill (``graph_augmenting.sparse_hops``)
    instead of dense hop-fill. Only ``augment_graph`` changes -- the force
    law, the loop, and everything else is inherited unchanged, which is
    the whole point of keeping the stages independent.

    ``radius`` (default 2, forwarded to ``sparse_hops.augment_graph`` via
    ``embed(G, radius=...)`` / ``**kwargs``) trades off physics fidelity
    (bigger radius, more complete shells, closer to dense hop-fill) against
    the genuinely sub-quadratic memory/compute win. See
    ``graph_augmenting/sparse_hops.py``'s module docstring for the
    documented Q4 caveat (repulsion collapse risk for pairs the augmenter
    never stores) before relying on this for a real run.
    """

    def augment_graph(self, G, **kwargs):
        self.D = graph_augmenting.sparse_hops.augment_graph(G, **kwargs)
        return self.D
