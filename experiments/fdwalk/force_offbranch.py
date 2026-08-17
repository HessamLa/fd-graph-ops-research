#!/bin/env python3
"""force_offbranch.py -- the force law of the off-branch experiment.

The law of the package (`fodined/embedding/shell_force.py`) is:

    guard = (h <= 1)
    Fa = guard * k1 * shell_coeff * x * exp(-k2 * (h - h_shift))
    Fr = -k3 * h * exp(-k4 * x)
    F  = Fa + Fr

Thus the repulsion of EVERY pair falls with the distance `x`, through
`exp(-k4 * x)`. Two nodes that are already far apart push each other only a
little.

The law here changes that for the pairs with `h > 1`:

    h == 1:  F = Fa + Fr,  Fr = -k3 * h * exp(-k4 * x)      (unchanged)
    h  > 1:  F = Fr,       Fr = -k3 * h                     (NO decay)

A pair with `h > 1` therefore pushes with a CONSTANT magnitude, whatever
the distance between the two nodes is. The push does not weaken when the
embedding separates them.

What that means for the physics: the pairs at `h > 1` become a constant
outward pressure, and the edges at `h = 1` are the only thing that holds
the layout together. The balance is then between a spring that grows with
`x` (`Fa` is linear in `x`) and a constant push, thus an equilibrium exists
at a finite distance for each edge. In the law of the package the push
vanishes at a large `x`, thus the far pairs stop acting once the layout has
spread.

The padding contract holds. A pad cell arrives with every plane zeroed, thus
`h = 0`:
  * `h > 1` is false, thus the cell takes the `h == 1` branch,
  * `Fa = 0` because `shell_coeff = 0`,
  * `Fr = -k3 * 0 * exp(...) = 0`.
Both terms vanish on their own, as the package requires.
"""
from __future__ import annotations

import jax.numpy as jnp


def shell_force_v2(x, planes, params):
    """`Fa + Fr` for `h == 1`, and a constant `Fr` for `h > 1`."""
    shell_coeff, h = planes
    near = h <= 1

    Fa = jnp.where(near, 1.0, 0.0) * (
        params["k1"] * shell_coeff * x
        * jnp.exp(-params["k2"] * (h - params["h_shift"])))

    Fr_near = -params["k3"] * h * jnp.exp(-params["k4"] * x)
    Fr_far = -params["k3"] * h
    Fr = jnp.where(near, Fr_near, Fr_far)

    return Fa + Fr
