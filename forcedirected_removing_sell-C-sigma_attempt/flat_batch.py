"""forcedirected.flat_batch -- the flat batched kernel. No layout, no padded
tiles.

This module replaces `sell_c_sigma.py` (deleted 2026-09-02, owner's
decision). The reasoning: `fodiwalk` already processes a bounded batch of
ROWS that fits on the card, which is plain batch processing. Sorting rows
by width, packing them onto a geometric ladder of padded tile shapes, and
splitting hub rows across virtual rows earned nothing once that bound was
already in place. See `agentic-log/00.master-agent/prompts/001.dense-kernel.md`
for the decision and `forcedirected_old/sell_c_sigma.py` for the frozen
"before" copy this module is measured against.

THE SHAPE, and the one point a naive replacement gets wrong. Padding every
row in a batch out to the WIDEST row in that batch would be worse than the
tiled layout on a skewed-degree graph: one hub row would force every other
row in the batch to the hub's width. This module never does that. It keeps
one FLAT array of `(row, partner)` pairs for the whole batch and reduces it
with `jax.ops.segment_sum` -- a batch never pays for a row wider than it
needs, only for the true count of its own stored pairs.

That said, "flat" is not "pad-free". A stored-pair count is not fixed
across chunks or across separate benchmark runs in one process, and every
distinct count is a distinct compiled shape to JAX -- a fresh trace, not a
reused one. `_bucket` rounds the pair axis up to the next power of two so
runs with a similar pair count SHARE a compiled `step`, the same reason
`sell_c_sigma.build_ladder` rounded a row's width up to a rung. A power of
two, not a fixed step like `experiments/fodiwalk-streaming/bench_stream.py`'s
`PAD_TO = 1 << 16`: that script's batches are all one node count and a
fixed step suits a narrow range, but a chunk here can hold a few pairs (an
isolated test graph) or tens of millions (`com_youtube`), and a fixed step
would still be one shape per chunk over that range. A pad entry carries
`h = 0` (or `w = 0` for the fused law), which the law's own `live` test
zeroes, and its neighbour equal to its own row, so `Zdiff` is 0 too -- the
same double safety `sell_c_sigma`'s pad cells had (see `step`'s docstring
below).

THE MATH, unchanged from `sell_c_sigma.step` (see that file's own history
for where it came from). For one chunk's flat pairs:

    Zdiff  = Z[v] - Z[u]                    v = partner, u = row owner
    x      = sqrt(sum(Zdiff * Zdiff, -1))
    x_safe = where(x == 0, 1.0, x)
    F_mag  = force_fn(x, planes, params)    the law, unchanged
    scale  = where(x == 0, 0.0, F_mag / x_safe)
    F      = segment_sum(Zdiff * scale, u_local, num_segments=R)
    F      = F * inv_deg[u_rows]

`force_fn` still takes `(x, planes, params)`, `planes` a tuple of arrays
shaped like `x`, and `params` a dict of TRACED scalars -- traced, not
static, so a hyperparameter sweep against one `D` never recompiles. None of
that changed; only how the sum over a row's partners gets computed did.

WHAT DIFFERS FROM THE OLD KERNEL, on purpose:

* `segment_sum` needs a STATIC `num_segments` -- JAX cannot trace it. `R`
  (the row count of the chunk) is bound with `functools.partial` before
  `jax.jit`, exactly as `sell_c_sigma.step` bound `n` and `force_fn` the
  same way. `params` stays a plain (traced) argument.
* The result is `(R, d)`, the chunk's own rows only -- not `(n, d)` with
  every other row zero. `sell_c_sigma.step` wrote into a full `(n, d)` `dZ`
  with `dZ.at[rows].add(..., mode="drop")` because a hub split needed a
  scatter with a real drop path for pad rows. Nothing here needs to share
  one address between two pairs of DIFFERENT rows, so there is no scatter
  contention to protect against and no reason to allocate the whole graph's
  `dZ` on every chunk.
* Dtype: every plane is cast to `float32` here, matching
  `sell_c_sigma.make_plan`'s own cast (`D.data` itself is `float64`; the
  kernel has always run on the `float32` cast of it). Keeping that cast is
  what makes this kernel's numbers comparable to the old one -- an
  uncast `float64` plane would differ from the tiled kernel by more than
  the reduction-order gap this rewrite already introduces, for a reason
  that has nothing to do with the reduction order.

The `x == 0` guard protected the padding contract in the tiled kernel. Here
there is no padding INSIDE a real row's own pairs, so the guard now
protects two things only: a genuine self-pair or a coincident pair (kept,
it is cheap and such a pair is real), and a bucket-pad entry (`v = u`, so
`x = 0` there too, by construction).

`inv_deg` is `1 / degree`, degree 0 mapped to 0.0, same convention as the
old `inv_deg_ext` minus its trailing pad slot -- there is no pad ROW here,
so a plain `(n,)` array is enough; each chunk reads the `(R,)` slice of its
own rows. The trap is the same as before: a degree of 0 zeroes a row's
WHOLE force, including repulsion, and the row then never moves for the run
with nothing raised. `fodiwalk.embed.plan_contract.check_degrees` still
guards it, upstream of this module.

Names retired with `sell_c_sigma.py`, and why: `build_ladder` and
`PlanCache` served the tiled layout only (the ladder quantized a row's
width; `PlanCache` cached a tiled plan plus its compiled step) -- nothing
in this repository outside `sell_c_sigma.py` and its own tests used either
one. `to_csr` / `_to_csr` (CSR coercion) is folded into `build_batch` below,
inline, since the only caller left is this module; it is not re-exported,
since nothing outside this module needs it any more. `build_batch` widens
that coercion by one case the old `to_csr` did not need: `nbr_walk` builds
`D` as `augment_graph.rows.RowCSR`, a plain object with `.indptr` /
`.indices` / `.nnz` and no `scipy.sparse` base class, so `sp.issparse(D)`
is False for it. `embed.plan_contract.check` already reads such a carrier
duck-typed; `build_batch` does the same rather than forcing it through
`sp.csr_matrix(np.asarray(D))`, which does not know what to do with it.

Import discipline: numpy, scipy, jax, and this package's own `csr` module
only -- the same rule `sell_c_sigma.py` kept.
"""
from __future__ import annotations

import functools

import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp

from .csr import row_of


# The smallest bucket. Below this, a graph or a chunk pays a little
# wasted work on pad entries in exchange for one shared compiled shape
# instead of a fresh one every time a tiny case runs (a benchmark sweep or
# a pytest session builds many small models in one process).
PAD_FLOOR = 1024


def _bucket(m: int, floor: int = PAD_FLOOR) -> int:
    """The next power of two at or above `m`, floored at `floor`.

    Bounds the number of distinct compiled shapes to about
    `log2(max_pairs / floor)` over a run's whole range of chunk sizes,
    instead of one shape for every distinct pair count. `build_ladder`
    used the same idea (geometric rounding) for a row's width; this is
    that idea applied to the pair axis instead of the row axis.
    """
    m = max(int(m), 1)
    return max(floor, 1 << (m - 1).bit_length())


def build_batch(D, planes, row_start: int, row_end: int,
                 pad_floor: int = PAD_FLOOR):
    """The flat pair arrays of rows `[row_start, row_end)` of `D`.

    D       : a `scipy.sparse.csr_matrix` (or dense ndarray, coerced).
    planes  : sequence of `(D.nnz,)` arrays, each one value for each stored
              pair, aligned 1:1 with `D.indices` in `D`'s own CSR order --
              the same contract `sell_c_sigma.make_plan` read. Cast to
              float32 here (see the module docstring's dtype note).

    Returns `(u_glob, u_loc, v, tiles, R, m_real)`:
        u_glob  global row id, one per pair (repeated for every pair of
                that row), padded with the CHUNK's last real row's id.
        u_loc   the same row, as a LOCAL index in `[0, R)` -- the
                `segment_sum` target.
        v       global partner id, one per pair, padded with `u_glob`'s
                own pad value (so `Zdiff` is 0 there, same as `u_glob`).
        tiles   one `(m,) float32` array per plane, padded with 0.
        R       row count of the chunk (`row_end - row_start`), the static
                `num_segments` `step` needs.
        m_real  the true pair count, before the power-of-two pad. Kept for
                stats (`cells`), not used by `step`.
    """
    if sp.issparse(D):
        D = D.tocsr()
    elif not hasattr(D, "indptr"):
        # A dense ndarray -- `sell_c_sigma.to_csr`'s other accepted shape.
        D = sp.csr_matrix(np.asarray(D))
    # else: a duck-typed CSR carrier (`augment_graph.rows.RowCSR`, built by
    # the `nbr_walk` policy) -- used as is. This function touches only
    # `.indptr`, `.indices`, `.nnz`, exactly what `embed.plan_contract.check`
    # already relies on such a carrier to have; it is never coerced through
    # `np.asarray`, which does not know how to turn one into an array.
    nnz = D.nnz
    planes = tuple(np.ascontiguousarray(p, dtype=np.float32) for p in planes)
    for i, p in enumerate(planes):
        if p.shape != (nnz,):
            raise ValueError(
                f"planes[{i}] has shape {p.shape}, expected ({nnz},) -- "
                f"every plane must be aligned 1:1 with D.indices, in D's "
                f"own CSR order")

    indptr, indices = D.indptr, D.indices
    R = row_end - row_start
    lo, hi = int(indptr[row_start]), int(indptr[row_end])

    v = indices[lo:hi].astype(np.int32)
    u_loc = row_of(indptr[row_start:row_end + 1] - lo).astype(np.int32)
    u_glob = (u_loc + row_start).astype(np.int32)
    tiles = tuple(p[lo:hi] for p in planes)

    m_real = int(v.size)
    m = _bucket(m_real, pad_floor)
    pad = m - m_real
    if pad:
        # A pad entry is inert two ways at once: `v = u_glob` makes
        # `Zdiff = 0` (thus `x = 0`, caught by the guard in `step`), and
        # every plane is 0 there, which zeroes a law's `live` test on its
        # own -- the same double safety `sell_c_sigma`'s pad cells had.
        fill_loc = int(u_loc[-1]) if m_real else 0
        fill_glob = int(u_glob[-1]) if m_real else row_start
        u_loc = np.concatenate([u_loc, np.full(pad, fill_loc, np.int32)])
        u_glob = np.concatenate([u_glob, np.full(pad, fill_glob, np.int32)])
        v = np.concatenate([v, np.full(pad, fill_glob, np.int32)])
        tiles = tuple(np.concatenate([t, np.zeros(pad, np.float32)])
                      for t in tiles)

    return u_glob, u_loc, v, tiles, R, m_real


def step(Z, u_glob, u_loc, v, planes, inv_deg, params, *, R, force_fn):
    """`dZ` of one chunk's `R` rows, from its flat pair arrays.

    `R` and `force_fn` are closed over via `functools.partial` before
    `jax.jit` (static -- `R` fixes `segment_sum`'s shape, `force_fn` is a
    Python callable baked into the trace), exactly as `sell_c_sigma.step`
    closed over `n` and `force_fn`. `params` stays a plain argument, so it
    is traced and a parameter sweep against one chunk never recompiles.

    `inv_deg` is the `(R,)` slice of the caller's global `(n,)` degree
    divisor for this chunk's own rows -- see the module docstring.
    """
    Zc = Z[u_glob]                                       # (m, d), one gather per PAIR
    Zn = Z[v]
    Zdiff = Zn - Zc
    x = jnp.sqrt(jnp.sum(Zdiff * Zdiff, axis=-1))        # (m,)
    x_safe = jnp.where(x == 0, 1.0, x)                   # avoid /0 below

    F_mag = force_fn(x, planes, params)                  # (m,) -- the physics
    scale = jnp.where(x == 0, 0.0, F_mag / x_safe)

    F = jax.ops.segment_sum(Zdiff * scale[:, None], u_loc, num_segments=R)
    return F * inv_deg[:, None]                          # (R, d)


def make_step(force_fn, R: int):
    """The jitted `step` of one chunk: `R` and `force_fn` bound and traced
    once, then called every epoch with that chunk's own arrays."""
    return jax.jit(functools.partial(step, R=R, force_fn=force_fn))
