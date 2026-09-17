"""embed.forces -- THE FORCE LAWS, and the quantities they are defined in.

Everything that defines the physics is here and nowhere else: the laws, the
planes they read, and the registry that says which is which.

WHY THIS FILE IS IN `embed` AND NOT IN `misc`. The law and the plan are ONE
contract and not two components. `forcedirected.make_plan` takes a SEQUENCE
of planes, and `step` gives that sequence to `force_fn`, which unpacks it
POSITIONALLY:

    h, freq = planes                   # fdlinear
    w, = planes                        # fdlinear_fused

Nothing in the kernel knows the count, the order, or the meaning. Four
silent defects of the 2026-08-17..18 campaign came from exactly this seam:
a third plane given to a two-plane law, a two-plane law that read a
coefficient plane as `h`, a missing `freq` that fell back to different
physics, and a policy that emptied the `h = 1` rows and froze them. A run
gave a number, and not an error.

`FORCE_PLANES` below is the answer. It is the ONE statement of which planes
a law reads, and in what order. `embed/plan_contract.py` asserts it, and
`Fodiwalk.augment_graph` builds its plane list FROM it -- never from an
`if` chain.

    | law               | planes, in order          | attraction   |
    | ----------------- | ------------------------- | ------------ |
    | fdlinear          | (h, freq)                 | h <= 1 only  |
    | fdlinear_fused    | (w,)                      | w < 0        |
    | fdhop             | (h,)                      | h <= 1 only  |
    | fdhop2            | (h,)                      | h <= 1 only  |
    | fdhop_min         | (h, deg_le)               | h <= 1, deg  |
    | fdhop_all         | (h,)                      | EVERY h      |
    | fdhop_all_freq    | (h, freq)                 | EVERY h      |

REMOVED 2026-08-19 -- the shell-averaged laws. Three laws read a
per-stored-pair coefficient `1 / |S_h(u)|`, the size of the hop shell of
the source node. The project settled on the linear laws, thus those three
laws and the two host functions that made the coefficient are deleted from
this package. `experiments/fdwalk/` still holds them.

Provenance, verbatim moves:
  * This file was `fodiwalk/core/forces.py` until 2026-08-28. It moved
    here with `plan_contract.py`, and `fodiwalk/core/` was then empty and
    is deleted. No line of a body changed.
  * `degrees_from_D` from `fodined/embedding/shell_force.py`.
  * `fdlinear`, `fdlinear_fused`, `fuse` from
    `experiments/fdwalk/force_fdlinear.py`.
Only the imports and this docstring are new. No line of a body changed.

Import discipline: numpy, scipy.sparse, jax, and `forcedirected` for
`row_of`. Nothing else of the package.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import jax.numpy as jnp

from forcedirected import row_of


# ---------------------------------------------------------------------------
# Quantities the force laws are defined in terms of (host-side, one time
# for each D)
# ---------------------------------------------------------------------------
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
# The laws. `x` is the embedding distance, `planes` the tiles of the plan,
# `params` a dict of traced values. Each returns the force MAGNITUDE along
# `u -> v`; the kernel applies the direction.
# ---------------------------------------------------------------------------
# THE AVERAGING COEFFICIENT (2026-09-09). `dz_u = F_u / deg(u)`. The
# divisor decides whether a row converges, and the right denominator is the
# size of the set the LAW sums over -- so it is the law's, not the engine's.
# It sat in `forcedirected/sell_c_sigma.py` until 2026-09-09, one array for
# every law: right for `fdhop`, which attracts only at `h == 1`, and wrong
# for `fdhop_all`, which attracts at every `h` and sums ~7x more terms on
# cora. That is why `fdhop_all` went non-finite at `k1=0.999, k2=1.0`.
#
# The kernel now supplies `params["node_degree"]` and divides nothing.
# EVERY law must end with these two lines, written out:
#
#     deg = jnp.where(params["node_degree"] > 0, params["node_degree"], 1)
#     return (Fa + Fr) / deg
#
# `node_degree` is the `(R, 1)` degree of the row and it broadcasts over the
# `k` axis. The `jnp.where` is a GUARD AGAINST DIVISION BY ZERO and nothing
# more: a degree of 0 divides by 1, thus such a row keeps its whole sum
# instead of an average.
#
# WHICH ROWS CARRY 0, and why neither is reached. A PAD row carries 0, and
# the kernel drops it by owner id (`dZ.at[rows].add(F, mode="drop")`); its
# cells also hold `x == 0`, which the kernel zeroes separately. A REAL row
# with no `h == 1` entry would carry 0, and `plan_contract.check_degrees`
# (invariant I5) refuses the run before the first epoch. Turn that check
# off and such a row moves under an unaveraged force.
#
# CHANGED 2026-09-16, on the owner's instruction. The form before it was
# `jnp.where(d > 0, (Fa + Fr) / jnp.where(d > 0, d, 1.0), 0.0)`, which gave
# EXACTLY 0 on a degree of 0 and so froze the row -- what the `inv_deg_ext`
# array did before the divisor moved here. For every run I5 admits the two
# forms agree bit for bit.
#
# The two lines are INLINE IN EVERY LAW and not a shared helper. A law is
# read as one piece, thus its divisor is written where it applies and a
# reader never leaves the function to learn what the last line does. A law
# that omits them keeps the whole row sum and the row diverges in silence;
# `test_b3_every_law_divides_by_its_own_node_degree` is the gate.
#
# Division is linear, thus a per-CELL divide here equals a per-ROW divide
# after the sum, up to float32 rounding order; the measured difference per
# law is in `experiments/refactor-deg/PROGRESS.md`.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# UPDATE 2026-08-17 -- the plane count
# ---------------------------------------------------------------------------
# Nothing above is removed. What changes is the NUMBER of planes that the
# caller sends, and the reason is memory.
#
# Reason. The first form of `fdlinear` bound a shell-coefficient plane and
# never read it: the law has no such term. The caller built it anyway. At
# n = 1,134,890 and nnz = 23,163,843 that dead plane cost 92.7 MB of host
# array, a 109.5 MB packed tile, and about 460 MB of transient inside the
# shell-size table -- for a value the kernel drops.
#
# Thus `fdlinear` now unpacks TWO planes, `(h, freq)`. The physics is
# unchanged, line for line. Only the dead argument is gone.
#
# `fdlinear_fused` goes one step further, on the observation that `h` and
# `freq` never appear apart in this law:
#
#     w = -1.0      for h == 1     (a SENTINEL, see below)
#     w = h / freq  for h >= 2     (always > 0, since h >= 2 and freq >= 1)
#     w = 0.0       for a pad cell
#
# The sentinel is load-bearing. A plain `w = h / freq` cannot be decoded:
# `h=1, freq=3` and `h=2, freq=6` both give 0.333, thus the `h == 1` branch
# and the `h >= 2` branch become indistinguishable and the attraction
# fires on the wrong cells. The `h == 1` branch reads NEITHER `h` NOR
# `freq` -- its coefficient is the constant `kr` -- thus it needs one flag
# and no value, and the sign carries the flag at no cost. The three
# regions are disjoint: negative, positive, exactly zero.
#
# The cost of fusing: `h / freq` becomes a BUILD-time quantity. `k1`, `k4`
# and `kr` stay traced scalars and sweep for free, but a law of the form
# `h / freq**beta` could no longer sweep `beta` without rebuilding `D`.
# Take this path only when the law stops moving.


def fdlinear(x, planes, params):
    """`Fa + Fr` of the fdlinear law. `planes` is `(h, freq)`."""
    h, freq = planes
    live = h > 0                       # a pad cell has every plane at 0
    near = h <= 1

    ex = jnp.exp(params["sign"] * params["k4"] * x)
    Fa = jnp.where(near & live, params["k1"] * x, 0.0)
    coeff = jnp.where(near, params["kr"], h / jnp.maximum(freq, 1.0))
    Fr = jnp.where(live, -coeff * ex, 0.0)
    deg = jnp.where(params["node_degree"] > 0, params["node_degree"], 1)
    return (Fa + Fr) / deg             # the law's own averaging


def fdlinear_fused(x, planes, params):
    """The same law from ONE plane `w`. See the UPDATE note above."""
    w, = planes
    live = w != 0                      # a pad cell is exactly 0
    near = w < 0                       # the h == 1 sentinel

    ex = jnp.exp(params["sign"] * params["k4"] * x)
    Fa = jnp.where(near, params["k1"] * x, 0.0)
    coeff = jnp.where(near, params["kr"], w)
    Fr = jnp.where(live, -coeff * ex, 0.0)
    deg = jnp.where(params["node_degree"] > 0, params["node_degree"], 1)
    return (Fa + Fr) / deg             # the law's own averaging


def fdhop(x, planes, params):
    """`Fa + Fr` of the fdhop law. `planes` is `(h,)`.

        h == 1:  Fa = k1 * x,  Fr = -kr * h
        h >= 2:  Fa = 0,       Fr = -kr * h * exp(-k4 * x)

    Added 2026-09-03 on the user's specification, to test one question: what
    does the hop number alone do, with no `freq` term in the repulsion?

    UPDATED 2026-09-04, also on the user's specification: the FAR branch now
    decays with distance. The near branch does not -- at `h == 1` the
    repulsion stays the flat `-kr * h`.

    This is `fdlinear` without the frequency, with two differences that are
    deliberate and are worth stating because they change the numbers:

      * the far coefficient is `h`, not `h / freq`;
      * the decay is `exp(-k4 * x)` and NOT `exp(sign * k4 * x)`: the minus
        sign is written in, thus `sign` is unused here. This is the form of
        the ORIGINAL forcedirected work, `-k3 * h * exp(-k4 * x)`, with
        `k3 = kr`.

    `k4` ADDED 2026-09-04 on the user's instruction. Before that the decay
    was the literal `exp(-x)`, thus the run of 260904-071825 is reproduced
    at `k4 = 1.0` and NOT at the `Config` default of 0.01, which is 100x
    flatter.

    A pad cell carries `h = 0` and gives exactly 0 from both terms.
    """
    h, = planes
    live = h > 0                       # a pad cell has every plane at 0
    near = h <= 1

    Fa = jnp.where(near & live, params["k1"] * x, 0.0)
    decay = jnp.where(near, 1.0, jnp.exp(-params["k4"] * x))
    Fr = jnp.where(live, -params["kr"] * h * decay, 0.0)
    deg = jnp.where(params["node_degree"] > 0, params["node_degree"], 1)
    return (Fa + Fr) / deg             # the law's own averaging


def fdhop2(x, planes, params):
    """`Fa + Fr` of the fdhop2 law. `planes` is `(h,)`.

        Fr = -kr * h * exp(-k4 * x)   for EVERY h
        Fa = k1 * x                   for h == 1
        Fa = 0                        for h >= 2

    Added 2026-09-06 on the user's specification. It is `fdhop` with ONE
    change: the `h == 1` repulsion decays with distance as well. `fdhop`
    holds it flat at `-kr * h`, thus the two laws differ only on the
    neighbour rows, and any difference in a score comes from there.

    A pad cell carries `h = 0` and gives exactly 0 from both terms.
    """
    h, = planes
    live = h > 0                       # a pad cell has every plane at 0
    near = h <= 1

    Fa = jnp.where(near & live, params["k1"] * x, 0.0)
    Fr = jnp.where(live, -params["kr"] * h * jnp.exp(-params["k4"] * x), 0.0)
    deg = jnp.where(params["node_degree"] > 0, params["node_degree"], 1)
    return (Fa + Fr) / deg             # the law's own averaging


def fdhop_min(x, planes, params):
    """`Fa + Fr` of the fdhop_min law. `planes` is `(h, deg_le)`.

        h == 1 and deg(u) <= deg(v):  Fa = k1 * x,  Fr = -kr * h
        h >= 2:                       Fa = 0,       Fr = -kr * h * exp(-k4 * x)

    `fdhop` with a SUBSET of the neighbours. A neighbour counts only when
    the partner is at least as well connected as the row, thus a hub pulls
    on its equals and above and not on its leaves, while a leaf still pulls
    on its hub.

    **The case the specification does not name.** `h == 1` with
    `deg(u) > deg(v)` has no line. It is read here as "not considered": the
    pair contributes EXACTLY ZERO, no attraction and no repulsion. The
    other reading -- fall through to the far branch and repel -- would make
    a hub push its own leaves away, which is the opposite of what the
    subset is for. Say so if the other reading was meant; it is one line.

    `deg_le` is that mask, 1.0 where `deg(u) <= deg(v)`. It is built from
    `degrees_from_D`, the count of `h == 1` entries of the row, so under
    `walk_edges` it is the graph degree.

    A pad cell carries `h = 0` and gives exactly 0 from both terms.

    Specified by the user on 2026-09-07.
    """
    h, deg_le = planes
    live = h > 0                       # a pad cell has every plane at 0
    near = h <= 1
    keep = near & (deg_le > 0)         # a neighbour worth considering

    Fa = jnp.where(keep & live, params["k1"] * x, 0.0)
    Fr = jnp.where(live & keep, -params["kr"] * h,
                   jnp.where(live & ~near,
                             -params["kr"] * h * jnp.exp(-params["k4"] * x),
                             0.0))
    deg = jnp.where(params["node_degree"] > 0, params["node_degree"], 1)
    return (Fa + Fr) / deg             # the law's own averaging


def fdhop_all(x, planes, params):
    """`Fa + Fr` of the fdhop_all law. `planes` is `(h,)`.

        Fr = -kr * h * exp(-k4 * x)   for EVERY h
        Fa = k1 * x * exp(-k2 * h)    for EVERY h

    `fdhop2` with the attraction let loose. Every other law of this file
    attracts at `h == 1` and nowhere else; here a far pair pulls too, and
    `k2` sets how fast that pull dies with the hop number. `k2` large is
    `fdhop2` in the limit, since `exp(-k2 * h)` then vanishes for `h >= 2`
    while `exp(-k2)` merely rescales `k1` on the neighbours.

    The repulsion is `fdhop2`'s, unchanged.

    `k2` is the hop decay of the attraction. The name is not new: the
    shell-averaged law this project used before 2026-08-19 spelled the same
    quantity `exp(-k2 * (h - h_shift))`, and the frozen copy at
    `forcedirected/tests/reference/shell_force.py` still reads it that way.
    `Config.k2` defaults to 1.0, NOT to the 0.01 that `k4` defaults to; the
    two decay in different variables and a shared default would be a
    coincidence, not a convention.

    A pad cell carries `h = 0`, so `Fr` vanishes on its own. `Fa` does NOT:
    `exp(-k2 * 0)` is 1, thus the `live` guard is load-bearing here in a way
    it is not in `fdhop2`. Removing it would make every pad cell attract.

    Specified by the user on 2026-09-07.
    """
    h, = planes
    live = h > 0                       # a pad cell has every plane at 0

    Fa = jnp.where(live, params["k1"] * x * jnp.exp(-params["k2"] * h), 0.0)
    Fr = jnp.where(live, -params["kr"] * h * jnp.exp(-params["k4"] * x), 0.0)
    deg = jnp.where(params["node_degree"] > 0, params["node_degree"], 1)
    return (Fa + Fr) / deg             # the law's own averaging


def fdhop_all_freq(x, planes, params):
    """`Fa + Fr` of the fdhop_all_freq law. `planes` is `(h, freq)`.

        Fa =  freq * k1 * x * exp(-k2 * h)   for EVERY h
        Fr = -freq * kr * h * exp(-k4 * x)   for EVERY h

    `fdhop_all` with both terms weighted by how often the walks reached the
    partner. A pair the walks met many times pulls and pushes harder, in
    proportion.

    `freq` is the same plane `fdlinear` reads, and it is NOT small: on cora
    the far pairs average 18.6 and reach 771. Since `freq` multiplies the
    attraction, which is linear in `x`, it multiplies the row's spring
    constant by the same amount, and that constant must stay under 1.0 or
    the run goes to non-finite. `k1` is pegged at 1.0 by the specification,
    so `k2` is the only lever left -- exactly as it was for `fdhop_all`,
    only the number needed is larger.

    Both terms carry `freq`, so a pad cell is zero twice over: `freq = 0`
    there, and the `live` guard on `h` zeroes it again.

    Specified by the user on 2026-09-08, with `k1 = 1.0`.
    """
    h, freq = planes
    live = h > 0                       # a pad cell has every plane at 0

    Fa = jnp.where(live, freq * params["k1"] * x
                   * jnp.exp(-params["k2"] * h), 0.0)
    Fr = jnp.where(live, -freq * params["kr"] * h
                   * jnp.exp(-params["k4"] * x), 0.0)
    deg = jnp.where(params["node_degree"] > 0, params["node_degree"], 1)
    return (Fa + Fr) / deg             # the law's own averaging


def fuse(h, freq):
    """The `w` plane, on the host, from `D.data` and the `freq` data."""
    import numpy as np
    return np.where(h <= 1.0, -1.0,
                    h / np.maximum(freq, 1.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# THE REGISTRY. The one statement of which planes a law reads, in order.
# ---------------------------------------------------------------------------
# A plane name is not a decoration: `embed/plan_contract.py` reads it and
# asserts what the name promises (I1, I2, I4 of the PRD). `Fodiwalk` builds
# its plane list from this table and never from an `if` chain, thus a new
# law adds one row here and no branch anywhere.
FORCE_PLANES = {
    "fdlinear":       ("h", "freq"),
    "fdlinear_fused": ("w",),
    "fdhop":          ("h",),
    "fdhop2":         ("h",),
    "fdhop_min":      ("h", "deg_le"),
    "fdhop_all":      ("h",),
    "fdhop_all_freq": ("h", "freq"),
}

# The law of each name.
FORCE_FN = {
    "fdlinear":       fdlinear,
    "fdlinear_fused": fdlinear_fused,
    "fdhop":          fdhop,
    "fdhop2":         fdhop2,
    "fdhop_min":      fdhop_min,
    "fdhop_all":      fdhop_all,
    "fdhop_all_freq": fdhop_all_freq,
}

assert set(FORCE_PLANES) == set(FORCE_FN)   # one law, one plane tuple


def planes_of(law: str) -> tuple:
    """The plane names of `law`, in the order the law unpacks them."""
    if law not in FORCE_PLANES:
        raise ValueError(
            f"Unknown force law {law!r}. Known: {sorted(FORCE_PLANES)}")
    return FORCE_PLANES[law]


def force_fn(law: str):
    """The callable of `law`."""
    if law not in FORCE_FN:
        raise ValueError(
            f"Unknown force law {law!r}. Known: {sorted(FORCE_FN)}")
    return FORCE_FN[law]
