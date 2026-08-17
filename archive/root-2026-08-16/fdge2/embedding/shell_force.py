"""embedding.shell_force -- the shell-averaged force law (T3.1 + T3.2).

Reference implementation of the "embed" stage's force law
(ARCHITECTURE.md). A faithful port of ``_scalar_force`` /
``_forces_204_hidx`` from
``fdge_numba/forcedirected_numba/model_204_shell.py`` onto the fdge2
three-stage ``D`` contract. The physics is *not* redesigned -- only the
seams the pipeline needs (RPD.md Sec 1: port the proven physics faithfully).

Physics (per node pair (u, v) with hop distance h = D[u, v] > 0 and
embedding distance x = ||Z[v] - Z[u]||):

    Fa(u,v) =  k1 * (1/|S_h(u)|) * x * exp(-k2 * (h - 1))     attractive
    Fr(u,v) = -k3 * h * exp(-k4 * x)                          repulsive

where S_h(u) is the shell of nodes exactly h hops from u. Per-node force
sums are divided by degree, then a per-node (whole-row) steady-rate random
drop is applied. Disconnected pairs carry h = n (node count), the legacy
sentinel convention (matches ``augment_graph``'s default ``unreachable=n``).

Researcher note -- to try a different force law, edit ``_scalar_force``
below. It is a scalar function of (x, h, shell_coeff, k1..k4), inlined into
the parallel kernel by Numba; nothing else needs to change.

--------------------------------------------------------------------------
The ``D`` contract this codes against (RPD.md Sec 3 -- ``D`` is a constant
input; this module must not know how it was produced):

* ``D`` is either a dense ``(n, n)`` int32 ``np.ndarray`` (is_sparse=False)
  or a ``scipy.sparse`` matrix wrapping the same values (is_sparse=True,
  the ``ForceDirected.augment_graph`` default).
* ``D[u, u] == 0``; ``D[u, v] == k`` = shortest-path hop distance for
  reachable pairs (k >= 1); ``D[u, v] == n`` (node count) for disconnected
  pairs. Symmetric.

Because hop-fill has no real sparsity (every reachable pair carries a
distance -- a documented known limitation, NOT something to fix here), the
sparse case is handled by densifying once: ``D.toarray()`` -> the same
dense-indexed kernel logic the reference already uses. This is deliberate;
a sparse-CSR-native kernel path is future work (T2.2/Q5), not this task.

Sparse ``D`` is detected by duck-typing (``hasattr(D, "toarray")``) so this
module never imports ``scipy`` -- import discipline (RPD.md Sec 5) allows
``embedding/`` only ``numpy``, ``numba``, ``fdge2.core``.

--------------------------------------------------------------------------
Caching design (the reason this module is more than one free function)
======================================================================
``forces`` is called once per row-batch, *every epoch*, always with the
**same** ``D`` object for a whole ``embed()`` call (see
``core.force_directed.ForceDirected.embed``). The quantities derived from
``D`` -- the densified hop matrix, the index-compressed hops, the shell
counts ``|S_h(u)|``, and node degrees -- are O(n^2)/O(n) to compute and
depend only on ``D``, not on ``Z`` or the row range. Recomputing them
every epoch would silently defeat the whole point of the row-streamed
architecture (an O(n^2) shell-count table rebuilt every epoch).

So the derivation is memoized. The cache is a **single slot keyed on
object identity** (``D is cached_D``), not a growing ``dict``:

* Object *identity* (``is``), not ``id(D)`` as a dict key: ``id()`` values
  are reused after garbage collection, so a fresh ``D`` in a later
  ``embed()`` call could collide with a freed one's ``id`` and return a
  stale cache hit. Comparing ``D is cached_D`` cannot alias. (Within one
  ``embed()`` call ``D`` is held alive by ``embed``'s local, so identity is
  stable there regardless.)
* A single slot (not a dict) means the cache holds a strong reference to at
  most one ``D`` at a time -- no unbounded growth, and the previous ``D``
  is released as soon as a new ``embed()`` call passes a different one.

The cache lives on a small stateful ``ShellForce`` instance. A concrete
model in ``fdge2/models.py`` holds one and delegates to it (see
``ShellForce`` docstring for the exact wiring). A module-level ``forces``
free function backed by a shared ``ShellForce`` singleton is also provided
so the composition-root snippet in API_DESIGN.md
(``embedding.shell_force.forces(Z, D, row_start, row_end, **kw)``) works
verbatim; instantiating your own ``ShellForce`` per model is cleaner (no
shared global) and is the recommended path.

--------------------------------------------------------------------------
Note on ``_shell_counts_from_hops`` (deliberate, documented duplication):
``graph_augmenting/hopfill.py`` has a ``get_shell_counts`` helper with the same
~10-line logic, but per ARCHITECTURE.md's "derived quantities live between
stages" rule and the RPD.md Sec 5 import boundary (``embedding/`` must NOT
import ``graph_augmenting/``), this module carries its own equivalent. This is
intentional -- do NOT "fix" it into a cross-package import.

Note on ``degrees``: the kernel divides each node's force sum by its
degree. Degrees can be passed in (e.g. from ``core.csr.graph_to_csr``,
which a model already has), but are also derivable from ``D`` alone --
node u's degree == the number of direct neighbors == the count of hop-1
entries in row u (``(D[u] == 1).sum()``). When ``degrees`` is not supplied
we derive it from ``D`` and cache it, so the embedding stage stays
self-contained (needs only ``D``).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Force law -- EDIT ME to try new force laws (researcher note, per reference)
# ---------------------------------------------------------------------------
@njit(inline="always")
def _scalar_force(x, h, shell_coeff, k1, k2, k3, k4):
    """Scalar force magnitude for one (u, v) pair. EDIT ME to try new laws.

    x           : embedding-space Euclidean distance ||Z[v] - Z[u]||
    h           : hop distance (>= 1; disconnected pairs carry h = n)
    shell_coeff : 1/|S_h(u)|, the shell-averaging factor
    Positive values attract (pull u toward v), negative repel.
    """
    Fa = k1 * shell_coeff * x * np.exp(-k2 * (h - 1.0))
    Fr = -k3 * h * np.exp(-k4 * x)
    return Fa + Fr


# ---------------------------------------------------------------------------
# Row-streamed parallel kernel (port of _forces_204_hidx)
# ---------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def _forces_shell(Z, hops_idx, h_of_idx, shell_counts, degrees,
                  k1, k2, k3, k4, row_start, row_end, out):
    """Accumulate the shell-averaged dZ rows for nodes [row_start, row_end).

    Hop values are index-compressed: ``hops_idx[u, v]`` indexes both
    ``shell_counts[u, .]`` and ``h_of_idx[.]``, where ``h_of_idx`` maps the
    index back to the numeric hop value (the sentinel index maps to
    h = n_nodes, the legacy convention).

    Memory guardrail (RPD.md Sec 3, IMPLEMENTATION.md): never materializes a
    (batch, n, d) pairwise-difference tensor -- forces are accumulated into
    a per-row ``acc`` (shape (d,)) and written straight into the
    preallocated ``out``. Peak *extra* memory is O(n*d).
    """
    n, d = Z.shape
    for r in prange(row_end - row_start):
        u = row_start + r
        hrow = hops_idx[u]
        crow = shell_counts[u]
        acc = np.zeros(d)
        for v in range(n):
            hi = hrow[v]
            if hi <= 0:
                continue
            s = 0.0
            for t in range(d):
                diff = Z[v, t] - Z[u, t]
                s += diff * diff
            x = np.sqrt(s)
            if x == 0.0:
                continue
            shell_coeff = 1.0 / crow[hi]
            f = _scalar_force(x, h_of_idx[hi], shell_coeff, k1, k2, k3, k4)
            scale = f / x
            for t in range(d):
                acc[t] += scale * (Z[v, t] - Z[u, t])
        deg = degrees[u]
        if deg > 0:
            for t in range(d):
                out[r, t] = acc[t] / deg
        else:
            for t in range(d):
                out[r, t] = 0.0


# ---------------------------------------------------------------------------
# Derived-quantity helpers (own copy, NOT imported from graph_augmenting/)
# ---------------------------------------------------------------------------
def _shell_counts_from_hops(hops_idx, n_bins):
    """counts[u, h] = |S_h(u)| = number of nodes exactly h (index) from u.

    Deliberate duplicate of ``graph_augmenting.hopfill.get_shell_counts`` -- kept
    here to respect the import boundary (see module docstring). Uses
    ``np.bincount`` (not ``np.add.reduceat``, which silently mishandles
    empty/degree-0 segments -- IMPLEMENTATION.md correctness pitfall).
    ``n_bins`` must be > the max index present in ``hops_idx``.
    """
    n = hops_idx.shape[0]
    counts = np.zeros((n, n_bins), dtype=np.int64)
    for u in range(n):
        binc = np.bincount(hops_idx[u], minlength=n_bins)
        counts[u, :] = binc[:n_bins]
    return counts


def _derive_from_D(D, degrees=None):
    """Compute the (cacheable) per-``D`` quantities the kernel needs.

    Returns a dict with the densified/compressed hop data, shell counts and
    float64 degrees. Faithful port of the setup done once in
    ``model_204_shell.FDModel.__init__``, moved here so it runs once per
    ``D`` (memoized by ``ShellForce``) instead of once per model.
    """
    Dd = D.toarray() if hasattr(D, "toarray") else np.asarray(D)
    Dd = np.ascontiguousarray(Dd, dtype=np.int32)
    n = Dd.shape[0]
    # sentinel (disconnected) hop value is n, the node count -- matches
    # augment_graph's default ``unreachable=n`` and the reference. Finite
    # hop distances in an n-node graph are always < n, so == n is exactly
    # the disconnected set.
    finite = Dd[Dd < n]
    maxhops = int(finite.max()) if finite.size else 0

    if (Dd == n).any():
        # remap sentinel n -> maxhops+1 for compact indexing; keep the
        # numeric value h = n inside the force law for legacy parity.
        sentinel_idx = maxhops + 1
        hops_idx = Dd.copy()
        hops_idx[hops_idx == n] = sentinel_idx
        h_of_idx = np.arange(sentinel_idx + 1, dtype=np.float64)
        h_of_idx[sentinel_idx] = float(n)
    else:
        sentinel_idx = maxhops
        hops_idx = Dd
        h_of_idx = np.arange(maxhops + 1, dtype=np.float64)

    hops_idx = np.ascontiguousarray(hops_idx, dtype=np.int32)
    shell_counts = _shell_counts_from_hops(hops_idx, sentinel_idx + 1)

    if degrees is None:
        # degree of u == number of direct (hop-1) neighbors in D
        deg = (Dd == 1).sum(axis=1).astype(np.float64)
    else:
        deg = np.ascontiguousarray(degrees, dtype=np.float64)

    return {
        "n": n,
        "hops_idx": hops_idx,
        "h_of_idx": h_of_idx,
        "shell_counts": shell_counts,
        "degrees": deg,
    }


# ---------------------------------------------------------------------------
# Random drop (port of model_204_shell.DropSteadyRate)
# ---------------------------------------------------------------------------
class DropSteadyRate:
    """Zero out whole dZ rows (nodes) with a fixed probability each call.

    Per-node (row-wise) drop, unlike a per-coordinate dropout (which would
    bias updates along coordinate axes). In fdge2 this is applied inside
    ``forces``/``ShellForce`` to the batch's output rows, playing the role
    ``forward()`` played in the reference (which applied it per batch too).
    """

    def __init__(self, drop_rate: float = 0.5, name: str = "steady-rate"):
        self.name = name
        self.drop_rate = float(drop_rate)

    def __call__(self, dZ: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.drop_rate <= 0.0:
            return dZ
        keep = rng.random(dZ.shape[0]) >= self.drop_rate
        dZ *= keep[:, None]
        return dZ


# ---------------------------------------------------------------------------
# ShellForce -- stateful callable holding the per-D cache
# ---------------------------------------------------------------------------
class ShellForce:
    """Shell-averaged force law with a per-``D`` derived-quantity cache.

    A concrete model in ``fdge2/models.py`` holds one instance and wires it
    onto the ``forces`` hook. Recommended wiring::

        from fdge2.core import ForceDirected
        from fdge2 import graph_building, graph_augmenting, embedding

        class ReferenceFDModel(ForceDirected):
            def __init__(self, *a, k1=0.999, k2=1.0, k3=10.0, k4=0.01,
                         random_drop_rate=0.5, **kw):
                super().__init__(*a, **kw)
                self._shell = embedding.shell_force.ShellForce(
                    k1=k1, k2=k2, k3=k3, k4=k4,
                    random_drop_rate=random_drop_rate)

            def make_graph(self, data, **kw):
                return graph_building.make_graph(data, **kw)
            def augment_graph(self, G, **kw):
                return graph_augmenting.hopfill.augment_graph(G, **kw)
            def forces(self, Z, D, row_start, row_end, **kw):
                return self._shell(Z, D, row_start, row_end,
                                   degrees=self.degrees, rng=self.rng, **kw)

    ``degrees`` may be omitted -- it is then derived from ``D`` (count of
    hop-1 entries per row). Pass ``rng=self.rng`` so the per-node random
    drop is tied to the model seed; if ``rng`` is ``None`` a private
    fallback generator is used (non-reproducible across processes).

    The k1..k4 / ``random_drop_rate`` defaults set at construction are used
    for any call that does not override them per-call.

    ``k3`` default note: ``10.0`` matches ``model_204_shell.FDModel``'s
    actual constructor default -- do not change this to ``None`` again.
    ``None`` is still a valid *explicit* value (meaning "auto: use the
    node count"), matching the legacy model's own
    ``self.k3 = float(n) if k3 is None else k3`` fallback, but the legacy
    model's out-of-the-box default is the literal constant ``10.0``, not
    ``None``-resolving-to-n. A prior version of this file defaulted ``k3``
    to ``None`` here, which silently made the *default* repulsion
    coefficient ~180x larger than the reference on a 1797-node graph
    (`k3` resolving to `n=1797` instead of `10.0`) -- caught by comparing
    `fdmap_new.py` against `fdge_numba/famap.py` output, not by the
    regression-parity test, which always pinned `k3=None` explicitly on
    both sides and so never exercised the *default*.
    """

    def __init__(self, k1: float = 0.999, k2: float = 1.0,
                 k3: float | None = 10.0, k4: float = 0.01,
                 random_drop_rate: float = 0.5):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.random_drop = DropSteadyRate(drop_rate=random_drop_rate)

        # single-slot identity cache (see module docstring)
        self._cache_D = None
        self._cache = None
        self._fallback_rng = None
        # instrumentation: how many times the O(n^2) derivation actually ran
        self.n_derivations = 0

    # -- cache -------------------------------------------------------------
    def _derived(self, D, degrees):
        """Return cached derived quantities for ``D``, computing once."""
        if self._cache is not None and self._cache_D is D:
            return self._cache
        self._cache = _derive_from_D(D, degrees=degrees)
        self._cache_D = D            # strong ref -> identity check is safe
        self.n_derivations += 1
        return self._cache

    def _rng(self, rng):
        if rng is not None:
            return rng
        if self._fallback_rng is None:
            self._fallback_rng = np.random.default_rng()
        return self._fallback_rng

    # -- call --------------------------------------------------------------
    def __call__(self, Z, D, row_start: int, row_end: int,
                 k1: float | None = None, k2: float | None = None,
                 k3: float | None = None, k4: float | None = None,
                 random_drop_rate: float | None = None,
                 degrees=None, rng=None, **kwargs) -> np.ndarray:
        """Compute the dZ rows for nodes [row_start, row_end).

        Matches the ``ForceDirected.forces`` hook calling convention: the
        loop passes ``Z, D`` and (via kwargs) ``row_start`` / ``row_end``
        plus loop bookkeeping (epoch, batch, ...) which is ignored.
        Per-call k1..k4 / drop overrides fall back to the instance defaults.
        """
        der = self._derived(D, degrees)

        _k1 = self.k1 if k1 is None else k1
        _k2 = self.k2 if k2 is None else k2
        _k4 = self.k4 if k4 is None else k4
        # k3 defaults to n (node count) when unset -- legacy convention
        _k3 = self.k3 if k3 is None else k3
        if _k3 is None:
            _k3 = float(der["n"])

        d = Z.shape[1]
        out = np.empty((row_end - row_start, d), dtype=np.float64)
        _forces_shell(Z, der["hops_idx"], der["h_of_idx"],
                      der["shell_counts"], der["degrees"],
                      _k1, _k2, _k3, _k4, row_start, row_end, out)

        drop = self.random_drop
        if random_drop_rate is not None:
            drop = DropSteadyRate(drop_rate=random_drop_rate)
        return drop(out, self._rng(rng))


# ---------------------------------------------------------------------------
# Module-level convenience: a shared-singleton free function
# ---------------------------------------------------------------------------
# Backs the exact composition-root snippet in API_DESIGN.md
# (``embedding.shell_force.forces(Z, D, row_start, row_end, **kw)``).
# Prefer instantiating your own ``ShellForce`` per model (no shared global);
# this exists for the one-liner wiring style.
_default_shell = ShellForce()


def forces(Z, D, row_start: int, row_end: int,
           k1: float = 0.999, k2: float = 1.0, k3: float | None = None,
           k4: float = 0.01, random_drop_rate: float = 0.5,
           degrees=None, rng=None, **kwargs) -> np.ndarray:
    """Shell-averaged force law (free-function form). See ``ShellForce``.

    Delegates to a module-level ``ShellForce`` singleton, so the per-``D``
    cache still holds across a whole ``embed()`` call. Per-call k1..k4 and
    ``random_drop_rate`` are honored (they override the singleton defaults).
    """
    return _default_shell(
        Z, D, row_start, row_end,
        k1=k1, k2=k2, k3=k3, k4=k4,
        random_drop_rate=random_drop_rate, degrees=degrees, rng=rng, **kwargs)
