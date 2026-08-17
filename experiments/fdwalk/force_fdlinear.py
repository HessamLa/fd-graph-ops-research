#!/bin/env python3
"""force_fdlinear.py -- the `fdlinear` force law.

The specification of 2026-08-17:

```
h == 1:  Fa = k * (z_v - z_u)
         Fr = - exp(z_v - z_u)
h >= 2:  Fa = 0
         Fr = -(h / freq) * exp(z_v - z_u)   # freq: how often v appeared
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


def fdlinear(x, planes, params):
    """`Fa + Fr` of the fdlinear law. `planes` is `(shell_coeff, h, freq)`."""
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
