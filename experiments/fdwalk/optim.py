#!/bin/env python3
"""optim.py -- axis C of the experiment: how a step becomes a new position.

`Fodined.updateZ` is the seam. The base class does `Z = Z + lr * dZ`, and
every rule here replaces that one line. The state of a rule lives in a dict
that the caller keeps, thus the functions stay pure.

  plain     `Z = Z + lr * dZ`. The rule of fodined today.
  momentum  heavy ball, `m = beta * m + dZ` [13]. One state array.
  nesterov  the look-ahead form of the same [13]. One state array.
  adam      `m` and `v`, with the bias correction [11]. TWO state arrays,
            thus 1.16 GB more at 1.13M nodes and 128 dimensions. That is
            hypothesis H6, and it is arithmetic, not a guess.
  fa2       the local speed of ForceAtlas2 [10]. Each node gets its own step
            size, from its "swinging": a node that changes direction at
            every epoch is going too fast, thus its step decreases. It is
            the force-directed answer to the problem that Adam solves.

`dZ` here is a FORCE, and not the gradient of a loss. Thus a large `dZ`
means "this node must move far", and a rule that divides by the size of
`dZ` throws that information away. Adam does exactly that, and hypothesis
H5 says that it will cost quality.
"""
from __future__ import annotations

import jax.numpy as jnp


def step_plain(Z, dZ, lr, state, epoch):
    return Z + lr * dZ, state


def step_momentum(Z, dZ, lr, state, epoch, beta: float = 0.9):
    m = state.get("m")
    m = dZ if m is None else beta * m + dZ
    state["m"] = m
    return Z + lr * m, state


def step_nesterov(Z, dZ, lr, state, epoch, beta: float = 0.9):
    """The look-ahead form: the step uses `dZ` one more time.

    `Z + lr * (dZ + beta * m_new)`. The gradient is the one of the current
    position, thus this is the practical form, and not the exact original.
    """
    m = state.get("m")
    m = dZ if m is None else beta * m + dZ
    state["m"] = m
    return Z + lr * (dZ + beta * m), state


def step_adam(Z, dZ, lr, state, epoch, b1: float = 0.9, b2: float = 0.999,
              eps: float = 1e-8):
    m, v = state.get("m"), state.get("v")
    m = (1 - b1) * dZ if m is None else b1 * m + (1 - b1) * dZ
    v = (1 - b2) * dZ ** 2 if v is None else b2 * v + (1 - b2) * dZ ** 2
    state["m"], state["v"] = m, v
    t = epoch + 1
    mh = m / (1 - b1 ** t)
    vh = v / (1 - b2 ** t)
    return Z + lr * mh / (jnp.sqrt(vh) + eps), state


def step_fa2(Z, dZ, lr, state, epoch, k_s: float = 0.1, k_max: float = 10.0):
    """The local speed of ForceAtlas2 [10], for each node.

    swinging  = |dZ - dZ_previous|, thus how much the direction changes.
    traction  = |dZ + dZ_previous| / 2, thus how much it goes one way.
    speed     = k_s * traction / (1 + sqrt(swinging)), and a ceiling holds
                one node from a jump.

    A node that swings gets a small step, and a node that goes one way gets
    a large one. The size of the force stays in the step, thus this rule
    keeps the physics that Adam removes.
    """
    prev = state.get("prev")
    if prev is None:
        state["prev"] = dZ
        return Z + lr * dZ, state
    swing = jnp.linalg.norm(dZ - prev, axis=1, keepdims=True)
    tract = jnp.linalg.norm(dZ + prev, axis=1, keepdims=True) / 2.0
    speed = k_s * tract / (1.0 + jnp.sqrt(swing))
    speed = jnp.minimum(speed, k_max)
    state["prev"] = dZ
    return Z + lr * speed * dZ, state


RULES = {"plain": step_plain, "momentum": step_momentum,
         "nesterov": step_nesterov, "adam": step_adam, "fa2": step_fa2}

# How many arrays of the size of `Z` each rule holds. The report needs this
# number at 1M nodes, where it decides what fits on the card.
STATE_ARRAYS = {"plain": 0, "momentum": 1, "nesterov": 1, "adam": 2,
                "fa2": 1}
