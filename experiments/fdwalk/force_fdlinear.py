#!/bin/env python3
"""force_fdlinear.py -- the `fdlinear` force law.

The specification of 2026-08-17:

```
let z_uv = z_v - z_u

h == 1:  Fa = k * z_uv
         Fr = - exp(||z_uv||) * (z_uv) / ||z_uv||
h >= 2:  Fa = 0
         Fr = -(h / freq) * exp(||z_uv||) * (z_uv) / ||z_uv||
                                        # freq: how often v appeared
                                        # in the walks
```

The engine wants a MAGNITUDE along the direction `u -> v`, and it applies
the direction itself (`sell_c_sigma._step` multiplies by `diff / x`). Thus
`k * (z_v - z_u)` becomes `k * x`, where `x = ||z_v - z_u||`. The law is
therefore LINEAR in the distance, which is the name.

Two readings that the specification leaves open, and the decision for each:

1. **The sign inside the exponent.** `exp(z_v - z_u)` reads as `exp(+x)`,
   which GROWS with the distance: two nodes that are already far would push
   each other harder, without a limit, and no layout can settle. Every
   repulsion of this project, and of [1, 8, 9], falls with the distance.
   Thus the default is `exp(-k4 * x)`, and `--fdlinear-sign +1` gives the
   literal reading for anybody who wants to see it diverge.
2. **`freq`.** "How often node v appeared in the walks" has a per-PAIR
   reading (how often the walks OF ROW u reached v) and a per-NODE reading
   (how often v appeared in any walk). The per-pair reading is the default,
   because the plan carries one number for each stored entry, and because
   it is the multiplicity that node2vec optimizes [5, 6]. `--freq-mode
   node` gives the other one.

`freq` divides the repulsion, thus a node that a row reached MANY times is
pushed away LESS. That is the opposite direction from the far-pair bias of
roster item 1, which pushes the HUBS more. The two act on different sets: a
hub is reached often by many rows, thus `freq` weakens its repulsion, and
`deg^0.75` strengthens it. The experiment says which one wins.

There is no `shell_coeff` in this law. The specification writes
`Fa = k * (z_v - z_u)` with no division by the degree, thus the attraction
of an edge does not fall when the degree of the node grows. The engine
still divides the row sum by `deg1(u)` unless the caller passes
`degrees = 1`, and `--no-deg-norm` does that. Without the flag the law is
not the one that the specification asks for.

The padding contract holds: a pad cell arrives with every plane zeroed,
thus `h = 0` takes the `h == 1` branch, `Fa = k * 0 * x`... no: `Fa` uses
`x`, and a pad cell has `x = 0` exactly, thus `Fa = 0`. `Fr` at a pad cell
is `-kr * exp(0) = -kr`, which is NOT zero, thus this module zeroes it
explicitly with the `live` mask. The engine also drops the pad row a second
time.
"""
from __future__ import annotations

import jax.numpy as jnp


def fdlinear_3plane(x, planes, params):
    """`Fa + Fr` of the fdlinear law. `planes` is `(shell_coeff, h, freq)`.

    RENAMED 2026-08-17, from `fdlinear`. Nothing of the body is removed;
    the name moved so that the 2-plane form below can take it without
    shadowing this one. Kept callable, as the reference that every new
    form must reproduce number for number.
    """
    _shell, h, freq = planes
    live = h > 0                       # a pad cell has every plane at 0
    near = h <= 1

    # exp(-k4*x) by default; `sign = +1` gives the literal exp(+k4*x).
    ex = jnp.exp(params["sign"] * params["k4"] * x)

    Fa = jnp.where(near & live, params["k1"] * x, 0.0)

    # h == 1 has no `freq` in the specification, thus its coefficient is
    # the constant `kr`. h >= 2 carries `h / freq`.
    coeff = jnp.where(near, params["kr"],
                      h / jnp.maximum(freq, 1.0))
    Fr = jnp.where(live, -coeff * ex, 0.0)

    return Fa + Fr

# ---------------------------------------------------------------------------
# UPDATE 2026-08-17 -- the plane count
# ---------------------------------------------------------------------------
# Nothing above is removed. What changes is the NUMBER of planes that the
# caller sends, and the reason is memory.
#
# Reason. `fdlinear` above binds `_shell` and never reads it: the law has
# no `shell_coeff`, as the closing paragraph of the module docstring says.
# The caller built it anyway. At n = 1,134,890 and nnz = 23,163,843 that
# dead plane cost 92.7 MB of host array, a 109.5 MB packed tile, and about
# 460 MB of transient inside `shell_counts` -- for a value the kernel drops.
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
