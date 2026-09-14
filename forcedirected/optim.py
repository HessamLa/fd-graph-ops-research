#!/bin/env python3
"""optim.py -- axis C of the experiment: how a step becomes a new position.

`ForceDirected.updateZ` is the seam. The base class does `Z = Z + lr * dZ`, and
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
# Provenance: moved VERBATIM from `fodiwalk/misc/optim.py` on 2026-08-26,
# with the engine that dispatches through it: `ForceDirected.set_rule` is
# at the repository root now and a relative import cannot reach `fodiwalk`.
# `fodiwalk/misc/optim.py` was a forwarder and is deleted (2026-08-27).
# `fodiwalk.misc` still re-exports RULES and STATE_ARRAYS from here.
# Provenance: moved from `experiments/fdwalk/optim.py` on 2026-08-19.
# Verbatim: not one line of a body changed.
from __future__ import annotations

import jax
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


# ---------------------------------------------------------------------------
# UPDATE 2026-08-18 -- the roster that CLAUDE.md asks for
# ---------------------------------------------------------------------------
# Nothing above is removed. Three rules are added, because the instruction
# names five and only three of the five existed:
#
#   "No optimizer"  -> `plain`, above. Already present.
#   "Velocity"      -> `velocity`, below. NOT the same as `momentum` above.
#   "SGD"           -> `sgd`, below. See the note on what it means here.
#   "Adam"          -> `adam`, above. Already present.
#   "SQN"           -> `sqn`, below.


def step_velocity(Z, dZ, lr, state, epoch, eta: float = 0.3):
    """`v = eta*dZ + (1-eta)*v0` ; `Z = Z + lr*v` ; `v0 = v`.

    This is an EXPONENTIAL MOVING AVERAGE of the force, and it is NOT the
    `momentum` rule above. The difference is the steady-state gain, and it
    is not a detail:

        momentum  `m = beta*m + dZ`        -> m -> dZ/(1-beta),  gain 10 at
                                              beta = 0.9
        velocity  `v = eta*dZ + (1-eta)*v` -> v -> dZ,           gain 1

    Thus `momentum` multiplies the effective learning rate by `1/(1-beta)`
    and `velocity` does not. A learning rate tuned for one is wrong for the
    other, and the two must never be read as the same rule.

    One state array.
    """
    v = state.get("v")
    v = dZ if v is None else eta * dZ + (1.0 - eta) * v
    state["v"] = v
    return Z + lr * v, state


def step_sgd(Z, dZ, lr, state, epoch, frac: float = 0.5, seed: int = 0):
    """A stochastic ROW schedule: update a random `frac` of the rows.

    **This is an interpretation, and the instruction is ambiguous.** The
    list in `CLAUDE.md` names "No optimizer: Z = Z + lr*dZ" AND "SGD" as
    two different entries, thus "SGD" must mean something that `plain` does
    not. The engine is already full-batch over one plan, thus the only
    stochastic element available is WHICH ROWS move. That is the reading
    taken here, and it is the one with evidence behind it:
    `../drop_strategies/REPORT.md` drops half of the rows of each epoch and
    loses no AUC.

    The rule is unbiased over the epochs -- every row moves in expectation
    -- which is exactly the property that PLAN.md's axis E note says a
    non-symmetric `D` does NOT have. Thus this rule also serves as the
    control for that argument.

    Zero state arrays. The mask is drawn from `epoch`, thus a run repeats.
    """
    key = jax.random.PRNGKey(seed + epoch)
    keep = jax.random.bernoulli(key, frac, (Z.shape[0], 1))
    return Z + lr * jnp.where(keep, dZ, 0.0), state


def step_sqn(Z, dZ, lr, state, epoch, memory: int = 3, eps: float = 1e-10):
    """Stochastic Quasi-Newton, L-BFGS two-loop recursion.

    **The justification, which must be stated because it is not obvious.**
    `dZ` here is a FORCE and not the gradient of a loss, thus there is no
    objective for a curvature pair `(s, y)` to approximate -- and a
    quasi-Newton method exists only relative to an objective. The step is
    defensible for ONE reason: a force-directed layout IS the negative
    gradient of a layout energy `U`, thus `g = -dZ` is a gradient, of `U`,
    and the pairs `(s, y)` approximate the Hessian of `U`.

    **The reservation, and it is real.** That identity holds only when the
    force field is CONSERVATIVE. Ours is not exactly: the law's `1/deg(u)` is a
    per-ROW normalisation. The row sum of `u` is divided by `deg(u)` and
    the row sum of `v` by `deg(v)`, thus the force from `v` on `u` and from
    `u` on `v` differ, and a non-reciprocal field has no scalar potential.
    Thus `g = -dZ` is an APPROXIMATE gradient, and `sqn` here is a
    heuristic and not a method with a convergence proof. It is run because the
    instruction asks for it, and this docstring is the honest statement of
    what it is.

    **The curvature guard.** A pair with `s . y <= 0` is SKIPPED, which is
    the standard safeguard for a non-convex objective. Without it the
    approximated inverse Hessian stops being positive definite and the step
    can point uphill.

    **Memory, and it is the reason this may not reach 1.13M nodes.**
    `2 * memory` arrays of the shape of `Z`, plus the previous `Z` and `g`.
    At n = 1,134,890 and d = 64 one array is 290 MB, thus `memory = 3`
    costs 6 * 290 = 1.74 GB, and `memory = 5` costs 2.9 GB. That is
    hypothesis H6 measured rather than assumed.
    """
    g = -dZ                                   # the approximate gradient
    Zp, gp = state.get("Zp"), state.get("gp")
    S, Y = state.get("S", []), state.get("Y", [])

    if Zp is not None:
        s, y = Z - Zp, g - gp
        sy = jnp.vdot(s, y)
        if float(sy) > eps:                   # the curvature guard
            S.append(s); Y.append(y)
            if len(S) > memory:
                S.pop(0); Y.pop(0)
    state["Zp"], state["gp"], state["S"], state["Y"] = Z, g, S, Y

    if not S:
        return Z + lr * dZ, state             # no curvature yet: plain

    # two-loop recursion on `q = g`
    q = g
    alpha, rho = [], []
    for s, y in zip(reversed(S), reversed(Y)):
        r = 1.0 / jnp.vdot(y, s)
        a = r * jnp.vdot(s, q)
        q = q - a * y
        alpha.append(a); rho.append(r)
    s_l, y_l = S[-1], Y[-1]
    q = q * (jnp.vdot(s_l, y_l) / jnp.vdot(y_l, y_l))     # H0 scaling
    for (s, y), a, r in zip(zip(S, Y), reversed(alpha), reversed(rho)):
        b = r * jnp.vdot(y, q)
        q = q + s * (a - b)
    return Z - lr * q, state                  # q approximates H^-1 g


RULES = {"plain": step_plain, "momentum": step_momentum,
         "nesterov": step_nesterov, "adam": step_adam, "fa2": step_fa2,
         "velocity": step_velocity, "sgd": step_sgd, "sqn": step_sqn}

# How many arrays of the size of `Z` each rule holds. The report needs this
# number at 1M nodes, where it decides what fits on the card.
STATE_ARRAYS = {"plain": 0, "momentum": 1, "nesterov": 1, "adam": 2,
                "fa2": 1, "velocity": 1, "sgd": 0, "sqn": 8}
# `sqn` at the default memory = 3 holds 2*3 curvature arrays plus `Zp` and
# `gp`, thus 8. It is the only rule whose cost depends on a parameter.


def state_arrays(name: str, memory: int = 3) -> int:
    """How many arrays of the shape of `Z` a rule holds.

    `sqn` is the only rule whose cost depends on a parameter: `2*memory`
    curvature arrays, plus `Zp` and `gp`.
    """
    return 2 * memory + 2 if name == "sqn" else STATE_ARRAYS[name]
