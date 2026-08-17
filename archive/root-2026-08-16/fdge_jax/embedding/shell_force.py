"""embedding.shell_force -- the shell-averaged force law, JAX port (T3.1+T3.2).

Faithful port of ``fdge2/embedding/shell_force.py`` onto the fdge_jax
``HopMatrix`` contract (docs/DESIGN.md "embedding/shell_force.py -- the
hot-loop kernel"). The physics is *not* redesigned -- only the execution
engine changes: Numba's ``@njit(parallel=True)`` row kernel becomes a
``jax.jit``'d flat-edge-list + ``segment_sum`` kernel, the one shape
validated in ``sparse_elementwise_test.py``.

Physics (per node pair (u, v) with hop distance h = D[u, v] > 0 and
embedding distance x = ||Z[v] - Z[u]||):

    Fa(u,v) =  k1 * (1/|S_h(u)|) * x * exp(-k2 * (h - 1))     attractive
    Fr(u,v) = -k3 * h * exp(-k4 * x)                          repulsive

where S_h(u) is the shell of nodes exactly h hops from u. Per-node force
sums are divided by degree, then a per-node (whole-row) steady-rate random
drop is applied. Disconnected pairs carry whatever sentinel hop value the
augmentation policy used (``hopfill``'s default is h = n, the node count);
this module doesn't special-case it -- it's just another (large) h.

Researcher note -- to try a different force law, edit the two lines marked
FORCE LAW inside ``_step`` below. Unlike fdge2 (a separate
``@njit(inline="always")`` scalar function), JAX has no equivalent seam:
the scalar law is written directly as vectorized array expressions inside
the one jitted kernel, since the whole point of a fused XLA kernel is that
there's only one function to trace. Nothing else in this file needs to
change to try a new law.

--------------------------------------------------------------------------
The ``D`` contract this code works against (docs/DESIGN.md -- ``D`` is a
constant input; this module must not know or care how it was produced):

* ``D`` is either a ``fdge_jax.core.csr.HopMatrix`` or a dense ``(n, n)``
  ndarray (duck-typed via ``hasattr(D, "indptr")``).
* A stored/nonzero entry ``D[u, v] == h`` means hop distance ``h >= 1``.
  Self (``h = 0``) and any pair the augmentation policy didn't reach are
  simply absent -- never a stored 0. A dense ndarray is coerced to CSR
  once via ``np.nonzero`` (0 == absent, matching the contract exactly).

--------------------------------------------------------------------------
Caching design (why this is more than one free function)
==========================================================
``ShellForce.__call__`` runs once per row-batch, *every epoch*, always
against the **same** ``D`` object for a whole ``embed()`` call. Everything
derivable from ``D`` alone -- the flat edge arrays, the shell counts
``|S_h(u)|`` gathered per edge, and degrees -- does not depend on ``Z`` or
the row range, and is O(n) to O(nnz) to (re)compute. Recomputing it every
epoch would defeat the entire point of the compiled hot loop. So it's
memoized in a **single slot keyed on object identity** (``D is
cached_D``), not a growing dict -- see fdge2's docstring for the full
"why identity, not id()" argument (id() values get reused after GC across
separate ``embed()`` calls; a single slot also means at most one ``D`` is
kept alive by the cache at a time). The compiled ``jax.jit`` step function
is cached alongside the derived arrays, since it closes over ``n`` (static)
and is only valid for that ``D``'s edge-array shapes.

Why the per-edge ``shell_coeff`` precompute: ``shell_counts[u, h]`` is
naturally a 2D "gather by (row, hop-index)" table. Doing that gather once
in NumPy (not hot) and handing the jitted kernel a flat ``(nnz,)`` array
means the per-epoch kernel never touches a 2D table at all -- just flat
per-edge arrays, the natural shape for ``jax.jit``/XLA.

Why ``k1..k4`` are traced, not static ``jax.jit`` arguments: a researcher
sweeping hyperparameters calls the same ``ShellForce`` many times with
different k-values but the same ``D``. If k1..k4 were static args, every
distinct float would trigger a fresh XLA compilation (and static args must
be hashable Python scalars, awkward for a fitting loop). Traced args mean
only a genuinely new ``D`` (new edge-array shapes) recompiles.

Note on ``degrees``: same derivation as fdge2 -- count of hop-1 entries per
row (``(D.data == 1)`` per source), so the embedding stage stays
self-contained (needs only ``D``); may also be passed in explicitly (e.g.
from ``core.csr.graph_to_csr``, which a model already computed).
"""
from __future__ import annotations

import functools

import numpy as np
import jax
import jax.numpy as jnp

from fdge_jax.core.csr import HopMatrix, row_of


# ---------------------------------------------------------------------------
# The jitted per-epoch kernel (built once per D, see ShellForce._derived)
# ---------------------------------------------------------------------------
def _step(Z, u_of, v_of, h, shell_coeff, degrees_safe, degrees_zero,
          k1, k2, k3, k4, n):
    """One fused pass over every stored (u, v) pair -> full (n, d) dZ.

    Flat-edge-list + segment_sum shape (matches ``_jax_step`` in
    ``sparse_elementwise_test.py``): no (n, n, d) pairwise tensor is ever
    materialized, no Python loop over nodes -- one vectorized pass over
    ``nnz`` edges, row-reduced with ``jax.ops.segment_sum``.

    ``n`` is closed over via ``functools.partial`` (static -- fixes the
    kernel's shapes for this ``D``); ``k1..k4`` are regular traced
    arguments (see module docstring).

    ``indices_are_sorted=True`` on the ``segment_sum`` below is a real
    correctness-relevant perf fix, not a cosmetic hint: ``u_of`` always
    comes from ``core.csr.row_of(D.indptr)`` on a valid CSR ``indptr``
    (non-decreasing by construction), so it genuinely IS sorted, always --
    never pass this hint for an ``u_of`` from anywhere else without
    checking. Without it, XLA's GPU lowering falls back to a general
    atomic-scatter-add, and on GPUs without native hardware fp64 atomics
    (pre-Pascal, e.g. compute capability < 6.0), the ~n-way write
    contention on each output row (every stored pair sharing a source row
    lands on the same address) forces a compare-and-swap retry loop that
    serializes almost completely -- measured 50x on this project's dense
    hop-fill case (849ms -> 17ms per call, n=1797, nnz~3.2M, identical
    output). With the hint, XLA can instead do a genuine segmented
    reduction over the already-contiguous-per-row blocks -- no shared
    writes, no atomics, embarrassingly parallel like the rest of the
    kernel.
    """
    diff = Z[v_of] - Z[u_of]                          # (nnz, d)
    x = jnp.linalg.norm(diff, axis=-1)                 # (nnz,)
    x_safe = jnp.where(x == 0, 1.0, x)                 # avoid /0 below

    # ---- FORCE LAW: edit these two lines to try a new force law --------
    Fa = k1 * shell_coeff * x * jnp.exp(-k2 * (h - 1.0))     # attractive
    Fr = -k3 * h * jnp.exp(-k4 * x)                          # repulsive
    # ----------------------------------------------------------------------

    scale = jnp.where(x == 0, 0.0, (Fa + Fr) / x_safe)
    contrib = scale[:, None] * diff                    # (nnz, d)
    sums = jax.ops.segment_sum(contrib, u_of, num_segments=n,
                                indices_are_sorted=True)
    return jnp.where(degrees_zero[:, None], 0.0, sums / degrees_safe[:, None])


# ---------------------------------------------------------------------------
# Derived-quantity helpers (own copy -- embedding/ must not import
# graph_augmenting/, see module docstring / docs/DESIGN.md import table)
# ---------------------------------------------------------------------------
def _to_hopmatrix(D) -> HopMatrix:
    """Coerce a dense ndarray ``D`` to a ``HopMatrix``; pass a HopMatrix through.

    0 means "absent" in the dense contract (self, or unreached), exactly
    what ``HopMatrix`` never stores -- so ``np.nonzero`` is the whole
    conversion. Rows come out of ``np.nonzero`` in ascending (row-major)
    order already, so a plain bincount-cumsum builds a valid ``indptr``.
    """
    if hasattr(D, "indptr"):
        return D
    Dd = np.asarray(D)
    n = Dd.shape[0]
    rows, cols = np.nonzero(Dd)
    data = Dd[rows, cols]
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(rows, minlength=n))
    return HopMatrix(indptr, cols, data, n)


def _shell_counts_from_hops(D: HopMatrix) -> np.ndarray:
    """counts[u, h] = |S_h(u)| = number of stored neighbors at hop h from u.

    Deliberate duplicate of fdge2's ``_shell_counts_from_hops`` (own copy
    per the import boundary -- see module docstring). ``np.bincount`` per
    row, not ``np.add.reduceat`` (silently mishandles empty/degree-0
    segments). Cheap, one-time-per-``D`` NumPy work -- never runs inside
    the jitted kernel.
    """
    n = D.n
    n_bins = (int(D.data.max()) + 1) if D.data.size else 1
    counts = np.zeros((n, n_bins), dtype=np.int64)
    for u in range(n):
        row = D.data[D.indptr[u]:D.indptr[u + 1]]
        counts[u, :] = np.bincount(row, minlength=n_bins)
    return counts


def _derive_from_D(D, degrees=None) -> dict:
    """Compute + device_put the (cacheable) per-``D`` quantities the kernel needs.

    Returns flat ``(nnz,)`` edge arrays (``u_of``, ``v_of``, ``h``,
    ``shell_coeff``), ``(n,)`` degree arrays, and the compiled step
    function -- everything ``ShellForce.__call__`` needs to run an epoch
    without touching NumPy or ``D`` again. Memoized by ``ShellForce``
    (see module docstring), so this runs once per distinct ``D``.
    """
    D = _to_hopmatrix(D)
    n = D.n

    u_of = row_of(D.indptr)              # (nnz,) source node per stored pair
    v_of = D.indices.astype(np.int64)    # (nnz,) neighbor node per stored pair
    h = D.data.astype(np.float64)        # (nnz,) hop distance per stored pair

    shell_counts = _shell_counts_from_hops(D)          # (n, max_h + 1)
    shell_coeff = 1.0 / shell_counts[u_of, D.data.astype(np.int64)]  # (nnz,)

    if degrees is None:
        # degree of u == count of hop-1 entries in row u
        deg = np.bincount(u_of[D.data == 1], minlength=n).astype(np.float64)
    else:
        deg = np.ascontiguousarray(degrees, dtype=np.float64)
    degrees_safe = np.where(deg == 0, 1.0, deg)
    degrees_zero = deg == 0

    return {
        "n": n,
        "u_of": jax.device_put(u_of),
        "v_of": jax.device_put(v_of),
        "h": jax.device_put(h),
        "shell_coeff": jax.device_put(shell_coeff),
        "degrees_safe": jax.device_put(degrees_safe),
        "degrees_zero": jax.device_put(degrees_zero),
        "step": jax.jit(functools.partial(_step, n=n)),
    }


# ---------------------------------------------------------------------------
# Random drop (port of fdge2's DropSteadyRate, key-based per docs/DESIGN.md)
# ---------------------------------------------------------------------------
def drop_steady_rate(dZ, key, drop_rate: float):
    """Zero out whole ``dZ`` rows (nodes) with probability ``drop_rate``.

    Per-node (row-wise) drop, not per-coordinate (a per-coordinate drop
    would bias updates along coordinate axes). The ``if`` below runs at
    Python/trace level against a plain float, not a traced value -- fine,
    mirrors how fdge2's ``k3=None`` resolves before any njit/jit call.
    """
    if drop_rate <= 0.0:
        return dZ
    keep = jax.random.uniform(key, (dZ.shape[0],)) >= drop_rate
    return dZ * keep[:, None]


class DropSteadyRate:
    """Thin callable wrapper around ``drop_steady_rate``, for API parity
    with fdge2's class (not required by anything here -- ``ShellForce``
    uses the free function directly)."""

    def __init__(self, drop_rate: float = 0.5, name: str = "steady-rate"):
        self.name = name
        self.drop_rate = float(drop_rate)

    def __call__(self, dZ, key):
        return drop_steady_rate(dZ, key, self.drop_rate)


# ---------------------------------------------------------------------------
# ShellForce -- stateful callable holding the per-D cache
# ---------------------------------------------------------------------------
class ShellForce:
    """Shell-averaged force law with a per-``D`` derived-quantity + compiled-kernel cache.

    Recommended wiring (a concrete model in ``fdge_jax/models.py``)::

        class ReferenceFDModel(ForceDirected):
            def __init__(self, *a, k1=0.999, k2=1.0, k3=10.0, k4=0.01,
                         random_drop_rate=0.5, **kw):
                super().__init__(*a, **kw)
                self._shell = embedding.shell_force.ShellForce(
                    k1=k1, k2=k2, k3=k3, k4=k4,
                    random_drop_rate=random_drop_rate)

            def forces(self, Z, D, row_start, row_end, **kw):
                return self._shell(Z, D, row_start, row_end,
                                    degrees=self.degrees, **kw)

    ``ForceDirected.embed`` already threads ``key=batch_key`` through
    ``**kwargs`` every batch (docs/DESIGN.md), so it just flows through.

    ``degrees`` may be omitted -- derived from ``D`` (hop-1 count per row).
    ``key=`` replaces fdge2's ``rng=``; if ``key`` is ``None`` a private
    fallback key is used (split off a once-per-instance, OS-seeded root
    key) -- non-reproducible across processes, same spirit as fdge2's
    ``_fallback_rng``.

    ``k3`` default note (do NOT reintroduce this bug -- see fdge2's own
    docstring for the full incident writeup): ``10.0`` matches
    ``model_204_shell.FDModel``'s actual out-of-the-box constructor
    default. ``None`` is still a valid *explicit* per-call override
    (meaning "auto: use the node count", the legacy
    ``self.k3 = float(n) if k3 is None else k3`` fallback) but must never
    be the silent *default* -- that would make the default repulsion
    coefficient scale with graph size instead of being the constant 10.0
    fdge2 actually ships.
    """

    def __init__(self, k1: float = 0.999, k2: float = 1.0,
                 k3: float | None = 10.0, k4: float = 0.01,
                 random_drop_rate: float = 0.5):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.random_drop_rate = random_drop_rate

        # single-slot identity cache (see module docstring)
        self._cache_D = None
        self._cache = None
        # instrumentation: how many times the per-D derivation actually ran
        self.n_derivations = 0
        # lazily-created root key for the key=None fallback path
        self._fallback_root_key = None

    # -- cache -------------------------------------------------------------
    def _derived(self, D, degrees):
        """Return cached derived quantities for ``D``, computing once."""
        if self._cache is not None and self._cache_D is D:
            return self._cache
        self._cache = _derive_from_D(D, degrees=degrees)
        self._cache_D = D            # strong ref -> identity check is safe
        self.n_derivations += 1
        return self._cache

    def _fallback_key(self):
        """Non-reproducible fallback key when the caller doesn't pass one.

        Seeds a root key once (OS entropy via numpy's SeedSequence), then
        splits it on every call -- mirrors fdge2's ``_fallback_rng``
        (a single persistent ``np.random.default_rng()`` whose state
        advances across calls, rather than a fresh reseed each time).
        """
        if self._fallback_root_key is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
            self._fallback_root_key = jax.random.PRNGKey(seed)
        self._fallback_root_key, sub = jax.random.split(self._fallback_root_key)
        return sub

    # -- call --------------------------------------------------------------
    def __call__(self, Z, D, row_start: int, row_end: int,
                 k1: float | None = None, k2: float | None = None,
                 k3: float | None = None, k4: float | None = None,
                 random_drop_rate: float | None = None,
                 degrees=None, key=None, **kwargs):
        """Compute the dZ rows for nodes [row_start, row_end).

        Matches the ``ForceDirected.forces`` hook calling convention: the
        loop passes ``Z, D`` and (via kwargs) ``row_start`` / ``row_end`` /
        ``key`` plus loop bookkeeping (epoch, batch, ...) which is ignored.
        Per-call k1..k4 / drop overrides fall back to the instance defaults.

        Batching note: this computes the kernel over the **full** graph
        every call, then slices ``[row_start:row_end]`` afterward -- CSR
        row ranges have variable length (bad for jit's static shapes), and
        slicing the edge arrays per batch would defeat the point of one
        fused pass. So ``batch_count > 1`` redoes the whole-graph
        computation on every batch call; wasted work for that rare path,
        an accepted tradeoff (docs/DESIGN.md), not something to fix here.
        """
        der = self._derived(D, degrees)

        _k1 = self.k1 if k1 is None else k1
        _k2 = self.k2 if k2 is None else k2
        _k4 = self.k4 if k4 is None else k4
        # k3 defaults to n (node count) when explicitly None -- legacy convention
        _k3 = self.k3 if k3 is None else k3
        if _k3 is None:
            _k3 = float(der["n"])

        Zj = jnp.asarray(Z)
        full = der["step"](Zj, der["u_of"], der["v_of"], der["h"],
                            der["shell_coeff"], der["degrees_safe"],
                            der["degrees_zero"], _k1, _k2, _k3, _k4)
        out = full[row_start:row_end]

        rate = self.random_drop_rate if random_drop_rate is None else random_drop_rate
        if key is None:
            key = self._fallback_key()
        return drop_steady_rate(out, key, rate)


# ---------------------------------------------------------------------------
# Module-level convenience: a shared-singleton free function
# ---------------------------------------------------------------------------
# Backs the composition-root one-liner style
# (``embedding.shell_force.forces(Z, D, row_start, row_end, **kw)``).
# Prefer instantiating your own ShellForce per model (no shared global);
# this exists for quick scripts / tests.
_default_shell = ShellForce()


def forces(Z, D, row_start: int, row_end: int,
           k1: float = 0.999, k2: float = 1.0, k3: float | None = 10.0,
           k4: float = 0.01, random_drop_rate: float = 0.5,
           degrees=None, key=None, **kwargs):
    """Shell-averaged force law (free-function form). See ``ShellForce``.

    Delegates to a module-level ``ShellForce`` singleton, so the per-``D``
    cache still holds across a whole ``embed()`` call. Per-call k1..k4 and
    ``random_drop_rate`` are honored (they override the singleton
    defaults). ``k3`` defaults to ``10.0`` here (not ``None``) to match
    ``ShellForce``'s own default -- see the "k3 default note" in
    ``ShellForce``'s docstring for why a silent ``None`` default is a
    documented footgun this module deliberately avoids reintroducing, even
    at the free-function convenience layer.
    """
    return _default_shell(
        Z, D, row_start, row_end,
        k1=k1, k2=k2, k3=k3, k4=k4,
        random_drop_rate=random_drop_rate, degrees=degrees, key=key, **kwargs)
