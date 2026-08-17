"""embedding.shell_force -- THE FORCE LAW. This is the file you edit.

Everything that defines the physics lives here and nowhere else: the two
force terms, the shell counts they average over, the degree they divide
by, and the thin classes that bind them to a graph. The execution
machinery -- hub-splitting, width-sorting, padding, batching, the
``lax.scan``, the compiled-kernel cache -- is in ``sell_c_sigma.py`` and
you should not need to open it.

The law, per stored node pair ``(u, v)`` with hop distance ``h = D[u,v]``
and embedding distance ``x = ||Z[v] - Z[u]||``:

    Fa(u, v) =  k1 * (1 / |S_h(u)|) * x * exp(-k2 * (h - h_shift))   attractive
    Fr(u, v) = -k3 * h * exp(-k4 * x)                                repulsive

``|S_h(u)|`` is the size of ``u``'s hop-``h`` shell, so the attractive term
is *shell-averaged*: a node with 200 two-hop neighbours is not pulled 200
times harder than one with 2. The engine then projects ``Fa + Fr`` onto
the unit vector ``(Z[v] - Z[u]) / x``, sums over ``v``, divides by
``deg(u)``, and applies the steady-rate drop (``drop.py``).

``h_shift`` is the attractive term's baseline for ``h``. ``ShellForce``
(hop-count physics) passes ``1.0``: hop distance 1 is the closest any
stored pair can be. ``WeightedShellForce`` passes ``h_min``, the closest
weighted distance any two nodes actually achieve, because "distance 1" is
meaningless on an arbitrary Euclidean scale. It is a traced parameter, not
a compile-time constant, so switching it never triggers a recompile.

Disconnected pairs (only possible when ``D`` came from dense ``hopfill``,
which stores an ``unreachable = n`` sentinel) are just another (large)
``h``; nothing here special-cases them.

--------------------------------------------------------------------------
Swapping the force law
======================
Write a function with ``shell_force``'s signature and hand it to a
``PlanCache``; nothing in ``sell_c_sigma.py`` changes::

    def linear_attraction(x, planes, params):
        shell_coeff, h = planes
        return params["k1"] * shell_coeff * x

    cache = PlanCache(linear_attraction, b_cells=..., k_max=...)

``planes`` is the tuple of ``(R, k)`` coefficient tiles in the order the
plan was built with; ``params`` is a plain dict of traced scalars. If your
law needs a different set of per-pair coefficients, build different planes
-- ``make_plan`` takes any number of them and never looks inside.

--------------------------------------------------------------------------
Import discipline: this module imports numpy / scipy.sparse / jax and
``fdge_jax_sell_c_sigma.core`` only -- never the other stage packages,
never the sibling ``fdge_jax``. ``D`` arrives as a plain argument; this
module must not know or care how it was produced.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import jax.numpy as jnp

from fdge_jax_sell_c_sigma.core.csr import row_of
from fdge_jax_sell_c_sigma.embedding.drop import drop_steady_rate, FallbackKeys
from fdge_jax_sell_c_sigma.embedding.sell_c_sigma import PlanCache, _to_csr


# ---------------------------------------------------------------------------
# Quantities the force law is defined in terms of (host-side, one-time per D)
# ---------------------------------------------------------------------------
def shell_counts(D: sp.csr_matrix):
    """Per-node shell sizes, over only the hop values ``D`` actually stores.

    Returns ``(hop_values, counts)``:

        hop_values : ``(m,)`` the DISTINCT hop values present in ``D.data``,
                     sorted ascending.
        counts     : ``(n, m)`` int64, ``counts[u, j] = |S_hop_values[j](u)|``
                     -- how many stored neighbours node ``u`` has at that
                     hop distance.

    Why the values are compacted instead of indexed by hop directly (this
    is a fixed **O(n^2) memory bug**, not a micro-optimization): the
    obvious table is ``(n, D.data.max() + 1)``, indexed by the raw hop
    value. Dense ``hopfill`` stamps an ``unreachable = n`` sentinel for
    every disconnected pair, so on a disconnected graph ``max()`` IS ``n``
    and the table becomes ``(n, n + 1)`` int64 = ``8 n^2`` bytes -- 32 MB
    at n = 2000, and on the order of 20 GB at n = 50k, for a table whose
    real content is a handful of columns. Connected graphs bound ``max()``
    by the diameter and never notice, which is why no test caught it.
    Compacting to the distinct values present makes the table ``(n,
    #distinct_hops)``: two columns on the disconnected case above.

    Built with one flat ``bincount`` over ``u * m + column``, not
    ``np.add.at`` (an unbuffered ufunc call, far slower at nnz scale) and
    not a per-row Python loop (O(n) interpreter round-trips at n = 500k).
    Cheap, one-time-per-``D`` NumPy work -- never runs inside the jitted
    kernel or the plan builder's hot inner loops.
    """
    n = D.shape[0]
    hop_values = np.unique(D.data)
    m = hop_values.size
    if m == 0:
        return hop_values, np.zeros((n, 0), dtype=np.int64)
    col = np.searchsorted(hop_values, D.data)
    flat = np.bincount(row_of(D.indptr) * m + col, minlength=n * m)
    return hop_values, flat.reshape(n, m).astype(np.int64)


def shell_coeff_data(D: sp.csr_matrix) -> np.ndarray:
    """Per-stored-entry ``shell_coeff = 1 / |S_h(u)|``, in ``D``'s own CSR order.

    Aligned 1:1 with ``D.indices`` / ``D.data`` -- i.e. computed BEFORE any
    hub-splitting, width-sorting, ladder-quantizing, or padding, which is
    exactly the alignment ``make_plan`` requires of a coefficient plane
    (same order-of-operations as the archive bench's ``C1.data`` /
    ``C2.data``, sliced in original-CSR order during its own hub-split
    loop).

    The ``searchsorted`` maps each stored hop to its column in
    ``shell_counts``' compacted table; ``hop_values`` is sorted, so this is
    exact, and it means neither this function nor its caller has to carry a
    separate inverse-index array around.
    """
    hop_values, counts = shell_counts(D)
    u_of = row_of(D.indptr)
    col = np.searchsorted(hop_values, D.data)
    return (1.0 / counts[u_of, col]).astype(np.float32)


def degrees_from_D(D: sp.csr_matrix, degrees=None) -> np.ndarray:
    """Force-law degree: hop-1 count per row (or a passed-in override).

    NOT the same quantity as a CSR row's stored *width* (see
    ``sell_c_sigma.py``'s closing docstring note) -- this is specifically
    the count of ``D.data == 1`` entries per source row, matching
    ``fdge_jax.embedding.shell_force``'s degree convention exactly, so the
    two engines are numerically comparable given the same ``D``. Keeping it
    here rather than in the plan builder is the point of the split: "what
    counts as a degree" is a force-law question.
    """
    if degrees is None:
        u_of = row_of(D.indptr)
        return np.bincount(u_of[D.data == 1], minlength=D.shape[0]).astype(np.float64)
    return np.ascontiguousarray(degrees, dtype=np.float64)


# ---------------------------------------------------------------------------
# The force law itself
# ---------------------------------------------------------------------------
def shell_force(x, planes, params):
    """``Fa + Fr`` for every cell of a ``(R, k)`` tile. Pure, elementwise.

    Parameters
    ----------
    x      : ``(R, k)`` embedding distance ``||Z[v] - Z[u]||``.
    planes : ``(shell_coeff, h)``, each ``(R, k)`` -- ``1/|S_h(u)|`` and the
             pair's hop distance (or weighted distance, for
             ``WeightedShellForce``). Order matches the ``planes`` argument
             the plan was built with.
    params : dict of traced scalars ``k1, k2, k3, k4, h_shift``.

    Returns the force MAGNITUDE along ``u -> v``; the engine handles the
    projection onto the unit direction vector, the degree division, and the
    padding guards (see ``sell_c_sigma._step``).

    Pad cells arrive with both planes zeroed, so ``Fa = 0`` and
    ``Fr = -k3 * 0 * exp(...) = 0`` independently -- each term vanishes on
    its own, and the engine zeroes the contribution a second time via
    ``x == 0``. Keep that property if you edit this: a force law with a
    constant term would still be safe (the engine's guard catches it), but
    losing the per-term zeroing costs one layer of the double safety.
    """
    shell_coeff, h = planes
    Fa = params["k1"] * shell_coeff * x * jnp.exp(-params["k2"] * (h - params["h_shift"]))
    Fr = -params["k3"] * h * jnp.exp(-params["k4"] * x)
    return Fa + Fr


# ---------------------------------------------------------------------------
# Bindings: force law + a graph + the cache
# ---------------------------------------------------------------------------
class ShellForce:
    """The shell-averaged force law bound to a graph, with a per-``D`` cache.

    Wires ``shell_force`` (the law) to a ``PlanCache`` (the machinery) and
    exposes the ``ForceDirected.forces`` hook's calling convention. Drops
    into the composition root as::

        class ReferenceFDModel(ForceDirected):
            def __init__(self, *a, k1=0.999, ..., **kw):
                super().__init__(*a, **kw)
                self._sell = embedding.shell_force.ShellForce(k1=k1, ...)

            def forces(self, Z, D, row_start, row_end, **kw):
                return self._sell(Z, D, row_start, row_end, **kw)

    ``degrees`` may be omitted -- derived from ``D`` (hop-1 count per row).
    ``key=None`` falls back to a private, non-reproducible per-instance key
    stream (``drop.FallbackKeys``).

    ``k3`` default note: ``10.0`` matches the model's actual out-of-the-box
    constructor default. ``None`` is still a valid *explicit* per-call
    override (meaning "auto: use the node count", the legacy
    ``k3 = float(n) if k3 is None else k3`` fallback) but must never be the
    silent *default* -- that would make the default repulsion coefficient
    scale with graph size instead of being the constant 10.0 this package
    actually ships. See ``fdge_jax.embedding.shell_force.ShellForce``'s
    docstring for the full incident writeup; do not reintroduce it here.

    Layout tunables (``b_cells``, ``k_max``, ``ladder_base``) are passed
    straight through to ``PlanCache`` and only take effect for a ``D`` this
    instance hasn't seen yet -- the cache is keyed on ``D`` identity, not
    on these knobs. Construct a fresh ``ShellForce`` to sweep them against
    the same ``D``. (``k1..k4`` are *not* like this: they are traced, so
    sweeping them costs neither a re-derivation nor a recompile.)
    """

    def __init__(self, k1: float = 0.999, k2: float = 1.0,
                 k3: float | None = 10.0, k4: float = 0.01,
                 random_drop_rate: float = 0.5,
                 b_cells: int = 16_384, k_max: int = 256,
                 ladder_base: float = 1.5):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.random_drop_rate = random_drop_rate

        self._planner = PlanCache(shell_force, b_cells=b_cells, k_max=k_max,
                                  ladder_base=ladder_base)
        self._keys = FallbackKeys()

    # -- the two hooks a variant physics overrides ---------------------------
    def _plan_inputs(self, D, degrees):
        """``(matrix_to_plan_from, planes, degrees)`` for a cache miss.

        Hop-count physics: plan from ``D`` itself, and the ``h`` plane is
        simply ``D.data``, the hop count each stored pair carries.
        """
        Dc = _to_csr(D)
        planes = (shell_coeff_data(Dc), Dc.data)
        return Dc, planes, degrees_from_D(Dc, degrees)

    def _h_shift(self) -> float:
        """Attractive term's ``h`` baseline. 1.0 = "hop distance 1 is closest"."""
        return 1.0

    # -- call ----------------------------------------------------------------
    def __call__(self, Z, D, row_start: int, row_end: int,
                 k1: float | None = None, k2: float | None = None,
                 k3: float | None = None, k4: float | None = None,
                 random_drop_rate: float | None = None,
                 degrees=None, key=None, **kwargs):
        """Compute the dZ rows for nodes ``[row_start, row_end)``.

        Matches the ``ForceDirected.forces`` hook calling convention: the
        loop passes ``Z, D`` and (via kwargs) ``row_start`` / ``row_end`` /
        ``key`` plus loop bookkeeping (epoch, batch, ...) which is ignored.
        Per-call ``k1..k4`` / drop overrides fall back to the instance
        defaults.

        Batching note: this computes the kernel over the **full** graph
        every call, then slices ``[row_start:row_end]`` afterward. The
        bucketed plan's batches are grouped by width across the WHOLE
        graph, not by row range -- a row range slice has no clean
        correspondence to "some subset of rungs/batches," and even if it
        did, the plan's shapes are fixed at derivation time regardless of
        which rows a given epoch's batch loop is currently asking for. So
        ``batch_count > 1`` redoes the whole-graph computation on every
        batch call, the same accepted tradeoff ``fdge_jax``'s
        ``ShellForce`` documents -- not something to "fix" here, there is
        no cheaper correct alternative given fixed per-rung shapes.
        """
        der = self._planner.derive(D, lambda: self._plan_inputs(D, degrees))

        _k3 = self.k3 if k3 is None else k3
        if _k3 is None:
            _k3 = float(der["n"])       # legacy "auto: use the node count"
        params = {
            "k1": self.k1 if k1 is None else k1,
            "k2": self.k2 if k2 is None else k2,
            "k3": _k3,
            "k4": self.k4 if k4 is None else k4,
            "h_shift": self._h_shift(),
        }

        Zj = jnp.asarray(Z, dtype=jnp.float32)
        full = der["step"](Zj, der["plan"], der["inv_deg_ext"], params)
        out = full[row_start:row_end]

        rate = self.random_drop_rate if random_drop_rate is None else random_drop_rate
        return drop_steady_rate(out, self._keys() if key is None else key, rate)

    # -- instrumentation -----------------------------------------------------
    @property
    def n_derivations(self) -> int:
        """How many times the per-``D`` derivation actually ran (cache misses)."""
        return self._planner.n_derivations

    @property
    def stats(self):
        """Plan stats for the most recently derived ``D`` -- see ``PlanCache.stats``."""
        return self._planner.stats


class WeightedShellForce(ShellForce):
    """Same law, but ``h`` is a WEIGHTED distance instead of a hop count.

    Backs ``models.EuclideanDistanceModel``. Two matrices are in play and
    it matters which is which:

    * ``hops`` -- the ordinary unweighted hop matrix. Supplies the
      TOPOLOGY (which pairs are stored, hence the whole batch plan) and the
      shell grouping ``|S_h(u)|``, because shell membership is a topology
      notion and a weighted distance is not.
    * ``D`` (the argument passed to ``__call__``) -- the dense all-pairs
      weighted shortest-path matrix. Supplies only the ``h`` plane, gathered
      at ``hops``' own stored ``(u, v)`` positions, and the cache key.

    ``h_shift`` becomes ``h_min``, the closest weighted distance any two
    nodes actually achieve, since "distance 1" means nothing on an
    arbitrary Euclidean scale.

    That is the whole difference: two hook overrides, everything else --
    ``__call__``, the k1..k4 resolution, the drop, the fallback keys, the
    cache, the compiled kernel -- is inherited, one copy.

    Contract: ``bind_topology(hops, h_min)`` must be called before the
    first ``__call__``, and **again whenever a fresh ``D`` is produced**,
    in the same breath. That is what upholds ``PlanCache``'s invariant (see
    its docstring): the cache keys on ``D`` identity and cannot see that
    ``hops`` changed underneath it. ``models.EuclideanDistanceModel.
    augment_graph`` assigns ``self.D`` and calls this together, so the two
    can never drift apart.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hops = None
        self._h_min = None

    def bind_topology(self, hops, h_min: float) -> None:
        """Attach the hop matrix + weighted-distance baseline for the next ``D``."""
        self._hops = _to_csr(hops)
        self._h_min = float(h_min)

    def _plan_inputs(self, D, degrees):
        if self._hops is None or self._h_min is None:
            raise RuntimeError(
                "WeightedShellForce needs bind_topology(hops, h_min) before use -- "
                "call embed(G, ...) / augment_graph(G) rather than forces() directly.")
        hops = self._hops
        # The weighted distance for exactly the pairs `hops` stores, in
        # hops' own CSR order -- i.e. a valid coefficient plane.
        h_dist = np.asarray(D)[row_of(hops.indptr), hops.indices]
        planes = (shell_coeff_data(hops), h_dist)
        return hops, planes, degrees_from_D(hops, degrees)

    def _h_shift(self) -> float:
        return self._h_min
