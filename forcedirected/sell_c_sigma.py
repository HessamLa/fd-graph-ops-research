"""embedding.sell_c_sigma -- the SELL-C-sigma layout machinery. No physics.

This file knows how to turn a ``D`` into fixed-shape padded rectangular
batches and how to reduce them on device. It does **not** know what force
law it is evaluating: the per-cell formula arrives as a ``force_fn``
argument. If you came here to change the physics, you are in the wrong
file -- open ``shell_force.py``, which is small and contains nothing but
the force law and the quantities that define it.

Where the sibling ``fdge_jax.embedding.shell_force._step`` is a flat
``(nnz,)`` edge list reduced with ``jax.ops.segment_sum`` (a scatter-add),
this module bins the graph's rows into fixed-width, padded rectangular
batches (the "SELL-C-sigma" layout -- sliced ELLPACK with a cell-budget
slice size, ported from ``archive/fdmap_bucketed_bench_jax.py`` /
``dev_docs/fdmap_engine_design_notes.md``) and reduces each batch with a
plain, dense ``sum(axis=1)`` -- no scatter over the k axis, one predictable
shape per batch. The per-ROW write at the end of ``step`` is still a
scatter (``dZ.at[rows].add(..., mode="drop")``), and it is NOT
contention-free: a row wider than ``k_max`` becomes several virtual rows
that carry the SAME owner id, thus several values land on one address.
What the layout removes is the SCALE of that contention. ``segment_sum``
contends once per stored pair; this layout contends once per hub SPLIT,
which is ``ceil(width / k_max)`` and is a small number even on a
power-law graph (measured: 0 to 196 splits over n = 1k to 1M, BA). Only
*how the same sum gets computed* differs (see
``docs/DESIGN.md`` for why this wins on large power-law-degree graphs
specifically: XLA fuses the whole per-batch chain into ~2 memory passes,
whereas ``segment_sum`` lowers to a non-deterministic GPU scatter-add
whose write contention scales with how many stored pairs share a source
row -- exactly the pathology a power-law degree distribution maximizes).

--------------------------------------------------------------------------
The SELL-C-sigma layout, in one paragraph (full rationale: ``docs/DESIGN.md``
and ``dev_docs/fdmap_engine_design_notes.md`` Sec 3)
=====================================================
Host-side, once per distinct ``D`` (never inside the jitted kernel):

1. **Hub split** -- any row wider than ``k_max`` stored entries becomes
   ``ceil(width / k_max)`` virtual rows, each a contiguous slice of that
   row's CSR span, tagged with the real owner node id.
2. **Width sort** -- stable sort of virtual rows by width. Storage order
   (node ids) is untouched; only *processing* order changes.
3. **Ladder quantize** -- each virtual row's width is rounded UP to the
   nearest value on a geometric ladder (``build_ladder``), so a whole
   contiguous run of similarly-sized rows shares one padded width ``k``,
   bounding padding without needing one compiled shape per distinct width.
4. **Balanced cell-budget packing** -- rows of a given quantized width ``k``
   are packed into batches of ``R = ceil(rows_in_rung / nb)`` rows each,
   where ``nb`` is chosen so every batch in the rung is roughly full (see
   ``make_plan``'s docstring for exactly why "greedy fill, pad the last
   batch" is a measured regression here, not a stylistic preference).
5. **Pad** -- pad CELLS get neighbor = the row's own node id and EVERY
   coefficient plane zeroed. Pad ROWS get owner id = ``n`` (out of the
   valid ``[0, n)`` range, so ``.at[rows].add(..., mode="drop")`` silently
   discards their contribution) and ``deg_ext[n] = 0``
   (belt-and-suspenders: even if a pad row's write were *not* dropped, it
   would land scaled by zero). See ``_step``'s docstring for the full
   padding contract and why it is deliberately doubled up.

Per epoch, the jitted step is then a Python ``for rung in plan:`` loop
(unrolled at trace time -- ``plan`` is a fixed-shape/fixed-length tuple, so
this is exactly as static as any other part of the traced computation)
around a ``jax.lax.scan`` over that rung's stacked batches: gather row
centers (once per ROW) and neighbors (once per CELL), call ``force_fn`` on
the resulting ``(R, k)`` distance tile, and reduce with a plain
``sum(axis=1)`` -- see ``_step`` for the actual code.

--------------------------------------------------------------------------
Coefficient planes: the one thing that connects layout to physics
==================================================================
A "plane" is a ``(nnz,)`` array carrying one number per stored pair, in
``D``'s OWN pre-split/pre-sort CSR order. ``make_plan`` slices, reorders
and pads each plane exactly the way it slices ``D.indices``, and hands the
resulting ``(nb, R, k)`` tiles back to ``force_fn`` per batch. That is the
entire interface between this file and the force law: this file never
looks inside a plane, and the force law never sees the layout.

The hop-count physics in ``shell_force.py`` passes two planes (a shell
coefficient and ``h``); ``models.EuclideanDistanceModel`` passes the same
two but with a weighted Dijkstra distance in the ``h`` plane instead of a
hop count. Neither variant needed a line of this file to change, which is
the property the split exists to protect.

--------------------------------------------------------------------------
Why this file does NOT reuse ``fdge_jax.embedding.shell_force``
================================================================
Two independent reasons, both hard requirements, not style preferences:

* Import boundary: ``fdge_jax_sell_c_sigma`` must be fully self-contained --
  nothing under it may import the sibling ``fdge_jax`` package at runtime
  (only *tests* are allowed that cross-package import, for parity checks).
  So the small pieces this package shares in spirit with ``shell_force.py``
  (the ``D`` -> csr_matrix coercion, the per-row shell-count derivation, the
  steady-rate drop) are each a **deliberate, small, separately-maintained
  copy** here -- exactly the pattern ``graph_augmenting/``'s own modules
  already use for cross-boundary reuse (see e.g. ``hopfill.py``'s module
  docstring).
* The execution shape is genuinely different: ``shell_force`` never
  constructs anything except flat ``(nnz,)`` arrays; this module's whole
  point is the padded ``(nb, R, k)`` tensors above, which have no
  corresponding piece in ``shell_force`` to inherit from anyway.

--------------------------------------------------------------------------
The ``D`` contract this code works against (identical to ``shell_force``):

* ``D`` is either a ``scipy.sparse.csr_matrix`` or a dense ``(n, n)``
  ndarray (dispatched via ``sp.issparse(D)``).
* A stored/nonzero entry ``D[u, v] == h`` means hop distance ``h >= 1``.
  Self (``h = 0``) and any pair the augmentation policy didn't reach are
  simply absent -- never a stored 0.

--------------------------------------------------------------------------
Note on the plan builder's OWN notion of "degree": the batch-plan
preprocessing (hub-split / width-sort / pack) operates on ``D``'s raw CSR
row *width* -- the total count of stored entries per row, across every hop
distance the augmentation policy kept -- not the force law's ``deg(u)``
(hop-1 count only, derived over in ``shell_force.degrees_from_D``). These
are different numbers whenever ``D`` stores more than one hop per row
(bounded-radius ``sparse_hops`` with ``radius > 1``, or dense ``hopfill``).
Keep them straight: ``row_width`` decides how a row gets split/padded;
``degrees`` is what ``dZ`` gets divided by. Conflating them is the easiest
bug to reintroduce when touching this file.
"""
# CORRECTION 2026-08-25, to the module docstring above.
# The docstring said the batch reduce has "no scatter, no atomics, one
# predictable shape per batch". The clause "no scatter, no atomics" was
# FALSE for the kernel as a whole and it is now rewritten. Nothing is
# removed: the old wording is
#
#     "plain, dense ``sum(axis=1)`` -- no scatter, no atomics, one
#      predictable shape per batch."
#
# WHY IT CHANGED. `step` ends in `dZ.at[rows].add(F, mode="drop")`, which
# IS a scatter. A hub row that `make_plan` splits gives several virtual
# rows ONE owner id, thus several adds reach one address, and float32
# addition is not associative. The measured threshold is 3 addends inside
# one batch: 2 commute safely, 3 give distinct results. A reader who
# believed the old sentence would call the kernel deterministic and would
# then hunt a defect in any run that failed to reproduce.
# The claim about the K AXIS was and stays true; only the claim about the
# whole kernel was wrong. The rewrite states what the layout actually
# gives: contention once per hub SPLIT and not once per stored pair.
# EVIDENCE: `experiments/fdwalk/FINDINGS.md` lines 1100-1182, and
# `fodiwalk/dev-docs/CATALOG.md` section 15.
# ADOPTED AT: the unification of the copies into this package, 2026-08-25.
#
# Provenance: moved VERBATIM from `fodiwalk/core/sell_c_sigma.py` on
# 2026-08-25, to end the copy. `fodined/embedding/sell_c_sigma.py` and
# `fodiwalk/core/sell_c_sigma.py` are FORWARDERS to this file now, thus the
# repository holds ONE implementation of the algorithm. The copy was byte
# identical (sha256 cfdbe266...318d); the only edit after it is the public
# `to_csr` name above. Parity record: `sellcsigma/PARITY.md`.
#
# Provenance: moved VERBATIM from `fodined/embedding/sell_c_sigma.py` on
# 2026-08-19. Not one line changed: this file imports numpy, scipy and jax
# only, thus it had no import to rewrite. `core/plan_contract.py` is the new
# module that asserts the plane contract this file documents and does not
# check.
# Provenance: this file is a copy of
# `fdge_jax_sell_c_sigma/embedding/sell_c_sigma.py`. The code is identical.
# Some docstrings refer to documents that are not in this directory, for
# example `docs/DESIGN.md`. Refer to the origin package for these documents.
from __future__ import annotations

import functools

import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Ladder helper (host-side, NumPy only -- ported from
# archive/fdmap_bucketed_bench_jax.py's build_ladder, unchanged)
# ---------------------------------------------------------------------------
def build_ladder(k_max: int, base: float) -> np.ndarray:
    """Geometric width ladder ``{1, ceil(1*base), ceil(...*base), ..., k_max}``.

    Every virtual row's (post-hub-split) width gets quantized UP to the
    nearest value on this ladder (``make_plan``'s step 3), so a contiguous
    run of similarly-sized rows shares one padded batch shape. A finer
    ladder (``base`` closer to 1) means less padding but more distinct
    compiled shapes (each distinct ``k`` in the plan is its own
    ``lax.scan`` body with its own trace); ``base = 1.5`` is the measured
    sweet spot for both small and large embedding dimension (see
    ``dev_docs/fdmap_engine_design_notes.md`` Sec 5 -- 1.5 -> ~13% padding,
    2.0 -> ~25%, 1.3 -> ~7.4%, and the earlier "coarser ladder for large d"
    heuristic was measured wrong and retracted there).

    Not ``np.geomspace``: the ``max(ks[-1] + 1, ...)`` guard below forces
    progress when ``ceil(k * base) == k`` (i.e. at ``k = 1``, ``base < 2``),
    which produces a genuinely different sequence. Leave it alone.
    """
    ks = [1]
    while ks[-1] < k_max:
        ks.append(min(k_max, max(ks[-1] + 1, int(np.ceil(ks[-1] * base)))))
    return np.asarray(ks, dtype=np.int64)


def to_csr(D) -> sp.csr_matrix:
    """Coerce a dense ndarray ``D`` to a ``scipy.sparse.csr_matrix``; pass a
    sparse ``D`` through as CSR (a no-op if it's already CSR).

    0 means "absent" in the dense contract (self, or unreached) -- exactly
    scipy's own sparse convention, so ``sp.csr_matrix(Dd)`` is the whole
    conversion.
    """
    if sp.issparse(D):
        return D.tocsr()
    return sp.csr_matrix(np.asarray(D))


# `to_csr` is the public name. `_to_csr` stays as an alias: the callers of
# `fodined.embedding.shell_force` name it. A PLAIN module may not import a
# private name of another module (gate D6, `fodiwalk/tests/test_structure.py`),
# thus the shared package must offer a public one.
_to_csr = to_csr


# ---------------------------------------------------------------------------
# The plan builder (host-side, NumPy only, one-time per D)
# ---------------------------------------------------------------------------
def make_plan(D, planes, degrees=None,
              b_cells: int = 16_384, k_max: int = 256, ladder_base: float = 1.5):
    """Build the SELL-C-sigma batch plan for ``D`` (hub-split / width-sort /
    ladder-quantize / balanced cell-budget pack). Ports
    ``archive/fdmap_bucketed_bench_jax.py``'s ``make_plan`` onto ``D``'s own
    ``scipy.sparse.csr_matrix`` triple directly (``.indptr``/``.indices``/
    ``.data`` -- no separate adjacency structure to build), with the
    archive's fixed ``C1``/``C2`` pair generalized to an arbitrary
    ``planes`` sequence.

    Deviations from the archive script, all deliberate (see inline comments
    at each site below for the "why"):

    1. **``degrees`` is an explicit parameter**, not re-derived from CSR row
       width. The archive script's ``adj`` was a plain one-hop adjacency,
       so "row width" and "force-law degree" were the same number; here
       ``D`` may store multiple hops per row (bounded-radius
       ``sparse_hops`` with ``radius > 1``, or dense ``hopfill``), so the
       two must be kept separate -- see this module's closing docstring
       note. Required here rather than defaulted, because *what counts as a
       degree* is a force-law question and this file is force-law agnostic
       (``shell_force.degrees_from_D`` is the hop-count answer).
    2. **Isolated (degree-0-width) rows are never given a virtual row at
       all** -- exactly the archive script's own behavior (its hub-split
       loop is ``for cs in range(s, e, k_max)``, which is a no-op range
       when ``s == e``), preserved here explicitly rather than as an
       accidental byproduct, so an isolated node costs zero padding and
       ``dZ`` for it is the default zero from the caller's zero-initialized
       buffer -- never something this plan writes.
    3. **The per-slot fill step is vectorized**, not a per-virtual-row
       Python loop. The archive script flags its own equivalent loop with
       "# vectorizable if ever hot" -- for this project's 100k-500k node
       target, a pure-Python O(n) preprocessing loop *is* hot enough to
       matter (one-time-per-``D`` cost, but "one time" still means "once
       per benchmark run"), so it's vectorized here with a
       ``(virtual_rows_in_rung, k)`` boolean/gather formulation instead.
       Same algorithm, same output, just built with NumPy broadcasting
       instead of a Python ``for slot, v in enumerate(sel)`` loop.

    Parameters
    ----------
    D       : a ``scipy.sparse.csr_matrix`` (or dense ndarray, coerced).
    planes  : sequence of ``(D.nnz,)`` arrays, each carrying one coefficient
              per stored pair, **aligned 1:1 with ``D.indices`` / ``D.data``
              in ``D``'s own pre-split, pre-sort CSR order**. Each becomes
              one padded ``(nb, R, k)`` tile per rung, in the order given,
              and is handed straight to ``_step``'s ``force_fn``. This file
              never interprets them -- see the module docstring's
              "Coefficient planes" section. Cast to float32 here (this
              package is float32 by design, ``docs/DESIGN.md``).
    degrees : ``(n,)`` per-node divisor for the final ``dZ`` rows. Not
              derived here; see deviation 1.
    b_cells : cell budget per batch (peak transient ~ b_cells * d).
    k_max   : max row width; wider rows are hub-split.
    ladder_base : width-quantization ladder base (``build_ladder``).

    Returns
    -------
    plan : tuple of rungs; each rung is a tuple of stacked NumPy arrays
        ``(rows (nb, R) int32, nbrs (nb, R, k) int32, *plane_tiles)``, one
        ``(nb, R, k) float32`` tile per entry of ``planes``, same order.
        ``rows`` holds the owner node id per row slot (``n`` on pad rows,
        the sentinel ``.at[...].add(mode="drop")`` relies on downstream).
    deg_ext : ``(n + 1,) float32``, the ``degrees`` themselves with a
        trailing slot for pad rows (index ``n``) that holds 0. A real node
        of degree 0 also holds 0. ``step`` hands the row's value to the
        law as ``params["node_degree"]``; what a 0 means there is the
        LAW's decision (every law of `fodiwalk` gives exactly 0,
        which is what the old ``1 / deg`` array did). This file takes NO
        reciprocal and performs NO division (2026-09-09).
    stats : dict with ``cells`` (real, non-padded stored entries actually
        processed -- the throughput numerator), ``n_virtual`` (virtual row
        count after hub-splitting), ``n_split`` (rows that needed
        splitting), ``rungs`` (distinct quantized widths), ``pad_frac``
        (padded slots / total slots).

    The balanced-packing rule (``R = ceil(rows_in_rung / nb)``, not naive
    greedy ``R = b_cells // k`` with a possibly near-empty final batch) is
    load-bearing: ``dev_docs/fdmap_engine_design_notes.md`` Sec 3.2
    documents a measured regression from ~13% to ~40% padding at n >= 10k
    when this balancing step is dropped in favor of greedy packing. Do not
    "simplify" this away.
    """
    D = to_csr(D)
    n = D.shape[0]
    indptr = D.indptr
    indices = D.indices
    planes = tuple(np.ascontiguousarray(p, dtype=np.float32) for p in planes)
    for i, p in enumerate(planes):
        if p.shape != (D.nnz,):
            raise ValueError(
                f"planes[{i}] has shape {p.shape}, expected ({D.nnz},) -- every "
                f"plane must be aligned 1:1 with D.indices/D.data in D's own "
                f"pre-split CSR order")

    deg = np.ascontiguousarray(degrees, dtype=np.float64)   # force-law degree
    row_width = np.diff(indptr).astype(np.int64)            # CSR row width (all hops)

    # ---- 1) hub split -----------------------------------------------------
    # Vectorized fast path for the overwhelming majority of rows (width <=
    # k_max, one virtual row each, identical to the real row); a plain
    # Python loop only for the rare hub rows that actually need splitting
    # (dev_docs measures n_split growing 0 -> 196 over n = 1k -> 1M for BA
    # graphs, so this loop is short even at the largest sizes -- see
    # deviation 3 above for why the *other* fill step still needs this
    # treatment but this split-count loop does not).
    hub_mask = row_width > k_max
    nonhub_mask = (row_width > 0) & ~hub_mask       # deviation 2: skip width-0 rows
    nonhub_rows = np.nonzero(nonhub_mask)[0]
    hub_rows = np.nonzero(hub_mask)[0]

    owners_parts = [nonhub_rows.astype(np.int64)]
    starts_parts = [indptr[nonhub_rows].astype(np.int64)]
    lens_parts = [row_width[nonhub_rows]]
    for i in hub_rows:
        i = int(i)
        s, e = int(indptr[i]), int(indptr[i + 1])
        chunk_starts = np.arange(s, e, k_max, dtype=np.int64)
        chunk_lens = np.minimum(k_max, e - chunk_starts).astype(np.int64)
        owners_parts.append(np.full(chunk_starts.shape[0], i, dtype=np.int64))
        starts_parts.append(chunk_starts)
        lens_parts.append(chunk_lens)

    owners = np.concatenate(owners_parts)
    starts = np.concatenate(starts_parts)
    lens = np.concatenate(lens_parts)

    deg_ext = np.zeros(n + 1, dtype=np.float32)
    if owners.size == 0:
        # Degenerate D (no stored entries at all -- e.g. every node
        # isolated). Nothing to batch; the kernel's Python rung-loop over
        # an empty plan tuple is a legal zero-iteration trace.
        stats = dict(cells=0, n_virtual=0, n_split=0, rungs=0, pad_frac=0.0)
        return tuple(), deg_ext, stats

    # ---- 2) width sort + ladder quantize -----------------------------------
    order = np.argsort(lens, kind="stable")
    owners, starts, lens = owners[order], starts[order], lens[order]
    ladder = build_ladder(k_max, ladder_base)
    kq = ladder[np.searchsorted(ladder, lens)]

    # ---- 3) balanced cell-budget packing (per rung) + 4) pad ---------------
    rungs = []
    padded = 0
    for k in np.unique(kq):
        sel = np.nonzero(kq == k)[0]
        k = int(k)
        m = sel.size
        cap = max(1, b_cells // k)          # row cap implied by the cell budget
        nb = -(-m // cap)                   # batches needed
        R = -(-m // nb)                     # BALANCED rows/batch (<= cap) -- see
                                             # docstring: this is NOT `b_cells // k`
        total_slots = nb * R

        Os = owners[sel]
        Ss = starts[sel]
        Ls = lens[sel]

        # Vectorized fill (deviation 3): for every virtual row in this
        # rung, column j < L is a real stored entry at CSR offset S + j;
        # column j >= L is padding. Building the (m, k) tiles with one
        # broadcasted gather each replaces a per-row Python loop.
        col = np.arange(k)
        valid = col[None, :] < Ls[:, None]                      # (m, k) bool
        src = np.clip(Ss[:, None] + col[None, :], 0, indices.shape[0] - 1)
        nbrs_m = np.where(valid, indices[src], Os[:, None]).astype(np.int32)

        rows = np.full(total_slots, n, dtype=np.int32)           # pad rows -> dropped
        nbrs = np.zeros((total_slots, k), dtype=np.int32)
        rows[:m] = Os
        nbrs[:m] = nbrs_m

        tiles = []
        for p in planes:
            tile = np.zeros((total_slots, k), dtype=np.float32)  # pad cells -> 0
            tile[:m] = np.where(valid, p[src], 0.0)
            tiles.append(tile.reshape(nb, R, k))

        padded += total_slots * k - int(Ls.sum())
        rungs.append((rows.reshape(nb, R), nbrs.reshape(nb, R, k), *tiles))

    # The DEGREES themselves, not `1 / deg`. Taking the reciprocal is
    # force-law policy and it left this file on 2026-09-09; the law reads
    # `params["node_degree"]` and decides what to do with a 0. A degree of
    # 0 stays 0 here, and `deg_ext[n]` (the pad slot) stays 0 too.
    deg_ext[:n] = deg.astype(np.float32)

    cells = int(lens.sum())
    stats = dict(
        cells=cells,
        n_virtual=int(owners.size),
        n_split=int(hub_mask.sum()),
        rungs=len(rungs),
        pad_frac=padded / max(1, cells + padded),
    )
    return tuple(rungs), deg_ext, stats


# ---------------------------------------------------------------------------
# The jitted per-epoch kernel (built once per D, see PlanCache.derive)
# ---------------------------------------------------------------------------
def step(Z, plan, deg_ext, params, n, force_fn):
    """One fused pass over the whole padded plan -> full ``(n, d)`` dZ.

    Rectangular-batch shape (matches ``run``'s ``one_step`` body in
    ``archive/fdmap_bucketed_bench_jax.py``, adapted to running ONE pass
    per call instead of an inner ``N_STEPS`` loop -- this engine must be
    callable once per epoch from ``core.Fodined.embed``'s existing
    Python loop, not own a multi-epoch loop itself).

    Force law
    ---------
    ``force_fn(x, planes, params) -> (R, k)`` returns the force MAGNITUDE
    along the ``u -> v`` direction for every cell of the tile:

        x      : ``(R, k)`` embedding distance ``||Z[v] - Z[u]||``.
        planes : tuple of ``(R, k)`` coefficient tiles, in the order the
                 plan was built with (``make_plan``'s ``planes``).
        params : pytree of TRACED values (a plain dict). Traced, not
                 static, so a hyperparameter sweep against one ``D`` never
                 recompiles. This kernel ADDS one key before the call:

                 ``node_degree`` : ``(R, 1)`` the degree of the row, 0 on a
                 pad row. It broadcasts over the ``k`` axis.

    Everything around that call -- the ``/x`` projection onto ``Zdiff``, the
    ``x == 0`` guard, the drop of pad rows -- stays here, because it is
    layout, not physics. In particular the ``x == 0`` guard is protecting
    the PADDING contract (pad cells have ``x`` exactly 0 by construction),
    so a new force law cannot accidentally break it.

    THE DEGREE DIVISION IS NOT HERE, since 2026-09-09. ``dz_u = F_u /
    deg(u)`` is the averaging coefficient of the LAW: it decides whether a
    row converges, and the right denominator is the size of the set the law
    sums over. This kernel cannot know that set -- ``fdhop`` attracts only
    at ``h == 1`` and ``fdhop_all`` at every ``h`` -- so it stops deciding.
    It supplies the degree and nothing else. See
    ``fodiwalk/embed/forces.py`` and ``experiments/refactor-deg/``.

    ``n`` and ``force_fn`` are closed over via ``functools.partial``
    (static -- ``n`` fixes shapes for this ``D``, ``force_fn`` is a Python
    callable that must be baked into the trace).

    ``plan`` is a Python tuple of rungs (fixed length, fixed per-rung
    shapes for a given ``D`` -- see ``make_plan``), so the ``for ... in
    plan:`` loop below is unrolled at trace time, exactly like the archive
    bench's rung loop; there is no ``lax.fori_loop`` here at all (unlike
    the archive bench, which also loops over epochs/steps INSIDE the jit --
    this kernel only ever does one pass, epochs are the caller's loop).

    Padding contract (``dev_docs/fdmap_engine_design_notes.md`` Sec 2.2,
    adapted from the archive kernel's "exact zero on padding" contract):
    pad cells carry EVERY coefficient plane zeroed, so a force law built
    from those planes vanishes -- and their neighbor index is the row's own
    node id, so ``Zdiff = 0`` and ``x = 0`` too, which the guard below turns
    into an exact zero contribution regardless of what ``force_fn``
    returned. That is deliberate double safety: a force law with a constant
    term would still contribute nothing on padding. Pad ROWS carry owner id
    ``n`` (one past the last valid node), so ``dZ.at[rows].add(F,
    mode="drop")`` silently discards their entire contribution, and
    ``deg_ext[n] = 0`` neutralizes them a second way even before that
    drop, because a law reads a degree of 0 as "contribute nothing"
    (every force law of `fodiwalk.embed.forces`).
    """
    dZ = jnp.zeros_like(Z)
    for rung in plan:                                    # unrolled over rungs
        def body(dZ, batch):
            rows, nbrs, planes = batch[0], batch[1], batch[2:]
            Zc = Z[jnp.minimum(rows, n - 1)]             # (R, d) centers, ONE read/ROW
            Zj = Z[nbrs]                                 # (R, k, d) neighbors, one/CELL
            Zdiff = Zj - Zc[:, None, :]
            x = jnp.sqrt(jnp.sum(Zdiff * Zdiff, axis=-1))  # (R, k)
            x_safe = jnp.where(x == 0, 1.0, x)           # avoid /0 below

            # `node_degree` rides in `params` and NOT in a fourth argument:
            # it is a quantity of the law, like `k1`, and it changes per
            # batch. `dict(params, ...)` is built at TRACE time.
            p = dict(params, node_degree=deg_ext[rows][:, None])   # (R, 1)
            F_mag = force_fn(x, planes, p)               # (R, k) -- the physics
            scale = jnp.where(x == 0, 0.0, F_mag / x_safe)

            F = jnp.sum(Zdiff * scale[..., None], axis=1)  # (R, d) dense k-axis reduce
            dZ = dZ.at[rows].add(F, mode="drop")           # per-ROW write; id n dropped
            return dZ, None
        dZ, _ = jax.lax.scan(body, dZ, rung)
    return dZ


# `step` is the public name (defect D6: a cross-module import of a private
# name). `_step` stays as an alias: `PlanCache` below, the docstrings of
# three modules and the tests name it.
_step = step


# ---------------------------------------------------------------------------
# PlanCache -- per-D derivation + compile, memoized on object identity
# ---------------------------------------------------------------------------
class PlanCache:
    """Single-slot, identity-keyed cache of {plan, deg_ext, stats, step}.

    Everything derivable from ``D`` alone -- the batch plan, the padded
    coefficient tiles, the degrees, the compiled step function -- does not
    depend on ``Z`` or the row range, and would be pure waste to recompute
    every epoch. It is memoized here in a **single slot keyed on object
    identity** (``key is cached_key``), not a growing dict, for the reasons
    ``fdge_jax.embedding.shell_force``'s module docstring gives: ``id()``
    values get reused after GC across separate ``embed()`` calls, and a
    single slot bounds how many ``D``'s worth of plan the cache keeps
    alive. The compiled ``jax.jit`` step is cached alongside the plan since
    it closes over that plan's fixed shapes.

    Force-law agnostic: constructed with a ``force_fn`` and the layout
    tunables, so both the hop-count and the weighted-distance bindings in
    ``shell_force.py`` -- and any future force law -- share this one
    implementation, one ``jax.jit`` call site, one cache slot.

    THE INVARIANT, and it is on the caller
    --------------------------------------
    Identity on the key object cannot see that the coefficient *planes*
    changed. So:

        **the cache key object MUST be a fresh object whenever ANY plan
        input changes -- topology, planes, or degrees.**

    Deliberately not defended against here: hashing the ``(nnz,)`` plane
    arrays on every call would cost more than the plan it protects, and
    versioning them would leak plan bookkeeping into every caller. It holds
    naturally for both current callers -- ``ShellForce`` derives its planes
    from the key object itself, and ``WeightedShellForce``'s planes are
    built in the same breath as its key object (see
    ``models.EuclideanDistanceModel.augment_graph``, which assigns
    ``self.D``, ``self._hops`` and ``self._h_min`` together and hands the
    fresh ``self.D`` over as the key).

    Layout tunables are NOT part of the key either: changing ``b_cells`` /
    ``k_max`` / ``ladder_base`` only takes effect for a key this instance
    hasn't seen yet. Construct a fresh cache (or binding) to sweep them
    against the same ``D``.
    """

    def __init__(self, force_fn, b_cells: int = 16_384, k_max: int = 256,
                 ladder_base: float = 1.5):
        self.force_fn = force_fn
        self.b_cells = b_cells
        self.k_max = k_max
        self.ladder_base = ladder_base

        self._key = None
        self._derived = None
        # instrumentation: how many times the per-D derivation actually ran
        self.n_derivations = 0

    def derive(self, key, build_inputs):
        """Cached ``{n, plan, deg_ext, stats, step}`` for ``key``.

        ``build_inputs`` is a zero-argument thunk returning
        ``(D, planes, degrees)`` -- called ONLY on a cache miss, so
        whatever it costs (shell-count derivation, a weighted-distance
        gather) stays off the per-epoch path.

        Two arguments rather than one because the object the cache is keyed
        on is not always the matrix the plan is built from: the hop-count
        binding keys on ``D`` and plans from ``D``, but
        ``models.EuclideanDistanceModel`` keys on its weighted Dijkstra
        matrix while planning from the unweighted hop matrix. See the class
        docstring's invariant.
        """
        if self._derived is not None and self._key is key:
            return self._derived

        D, planes, degrees = build_inputs()
        D = to_csr(D)
        n = D.shape[0]
        plan_np, deg_ext_np, stats = make_plan(
            D, planes, degrees=degrees, b_cells=self.b_cells,
            k_max=self.k_max, ladder_base=self.ladder_base)

        self._derived = {
            "n": n,
            "plan": jax.tree_util.tree_map(jax.device_put, plan_np),
            "deg_ext": jax.device_put(deg_ext_np),
            "stats": stats,
            "step": jax.jit(functools.partial(
                _step, n=n, force_fn=self.force_fn)),
        }
        self._key = key              # strong ref -> identity check is safe
        self.n_derivations += 1
        return self._derived

    @property
    def stats(self):
        """Plan stats for the most recently derived key (``None`` before the
        first call). Fields: ``cells`` (real, non-padded stored entries --
        the throughput numerator for a Mcells/s figure), ``n_virtual``,
        ``n_split``, ``rungs``, ``pad_frac``. See ``make_plan``'s docstring
        for exact meanings. Exposed so callers (e.g.
        ``validation/bench_embedding_perf.py``) can report padding overhead
        and cells/s without reaching into the private cache dict.
        """
        return None if self._derived is None else self._derived["stats"]
