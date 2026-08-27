"""core.forces -- THE FORCE LAWS, and the quantities they are defined in.

Everything that defines the physics is here and nowhere else: the laws, the
planes they read, and the registry that says which is which.

WHY THIS FILE IS IN `core` AND NOT IN `misc`. The law and the plan are ONE
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
a law reads, and in what order. `core/plan_contract.py` asserts it, and
`Fodiwalk.augment_graph` builds its plane list FROM it -- never from an
`if` chain.

    | law               | planes, in order          | attraction   |
    | ----------------- | ------------------------- | ------------ |
    | fdlinear          | (h, freq)                 | h <= 1 only  |
    | fdlinear_fused    | (w,)                      | w < 0        |

REMOVED 2026-08-19 -- the shell-averaged laws. Three laws read a
per-stored-pair coefficient `1 / |S_h(u)|`, the size of the hop shell of
the source node. The project settled on the linear laws, thus those three
laws and the two host functions that made the coefficient are deleted from
this package. `experiments/fdwalk/` still holds them.

Provenance, verbatim moves:
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
# `params` a dict of traced scalars. Each returns the force MAGNITUDE along
# `u -> v`; the kernel applies the direction.
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
    return Fa + Fr


def fdlinear_fused(x, planes, params):
    """The same law from ONE plane `w`. See the UPDATE note above."""
    w, = planes
    live = w != 0                      # a pad cell is exactly 0
    near = w < 0                       # the h == 1 sentinel

    ex = jnp.exp(params["sign"] * params["k4"] * x)
    Fa = jnp.where(near, params["k1"] * x, 0.0)
    coeff = jnp.where(near, params["kr"], w)
    Fr = jnp.where(live, -coeff * ex, 0.0)
    return Fa + Fr


def fuse(h, freq):
    """The `w` plane, on the host, from `D.data` and the `freq` data."""
    import numpy as np
    return np.where(h <= 1.0, -1.0,
                    h / np.maximum(freq, 1.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# THE REGISTRY. The one statement of which planes a law reads, in order.
# ---------------------------------------------------------------------------
# A plane name is not a decoration: `core/plan_contract.py` reads it and
# asserts what the name promises (I1, I2, I4 of the PRD). `Fodiwalk` builds
# its plane list from this table and never from an `if` chain, thus a new
# law adds one row here and no branch anywhere.
FORCE_PLANES = {
    "fdlinear":       ("h", "freq"),
    "fdlinear_fused": ("w",),
}

# The law of each name.
FORCE_FN = {
    "fdlinear":       fdlinear,
    "fdlinear_fused": fdlinear_fused,
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
