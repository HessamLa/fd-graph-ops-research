#!/bin/env python3
"""evaluator.context -- the typed record of what produced a number.

A score is unquotable unless the row also says which method made it, at
which dimension, for how many epochs, at which EFFECTIVE learning rate,
and on which device (`METROLOGY.md` section 1). Today those facts travel
as free-text `tags`; nothing checks them and a wrong one is invisible.
`Context` is the typed record, and `validate()` fills the fields that can
be computed and never lets a caller type one of them directly.

THE EFFECTIVE-LEARNING-RATE LAW (`fodiwalk/dev-docs/CATALOG.md` 29.2).
Two update rules at the same NOMINAL learning rate do not run at the same
rate: `momentum` at `lr=0.1` and `plain` at `lr=1.0` share one effective
rate.

    effective lr = lr x DC gain

The DC gain, by `update_rule` (source of record: CATALOG 29.2 and 29.3):

    plain, sgd, velocity   1.0
    momentum, nesterov     1 / (1 - beta), beta default 0.9
    fa2                    10.0 (the speed cap k_max)
    adam, sqn              None -- adaptive, effective_lr stays None
    gen_momentum           alpha / (1 - beta), alpha default 1.0

Standing rule (project owner, 2026-08-29): no effective lr is ever 1.0.
1.0 is the stability edge. `validate()` writes a warning at or above it
and does not raise -- a caller may score a diverged run on purpose.

`beta` and `alpha` are not in the `Context` field list of PRD-v2 5.4; they
are added here because the gain law itself reads `context.beta` for
`momentum`/`nesterov`/`gen_momentum`. Both default to `None`, like every
other field.

Import discipline: `dataclasses` only.
"""
from __future__ import annotations

import dataclasses


# -- the gain table, source of record: CATALOG.md 29.2 and 29.3 -----------

_FIXED_GAIN = {"plain": 1.0, "sgd": 1.0, "velocity": 1.0, "fa2": 10.0}
_MOMENTUM_LIKE = {"momentum", "nesterov"}
_ADAPTIVE = {"adam", "sqn"}
_BETA_DEFAULT = 0.9
_ALPHA_DEFAULT = 1.0
_STABILITY_EDGE = 1.0

# Fields validate() computes; never counted as "missing" and never typed
# by the caller. `beta`/`alpha` are excluded too: they are gain-law
# knobs with a law default, not part of the row contract of
# METROLOGY.md section 1, so an un-set one is not a gap in the record.
_DERIVED_FIELDS = frozenset(
    {"n_dim", "dc_gain", "effective_lr", "regime", "beta", "alpha"})

# Dimension bands of METROLOGY.md section 5, keyed by the graph's PRIMARY
# class. PROVISIONAL: at n_dim 32 or 64, class H can mean R-quick (any
# turnaround run) or R-robust-H (an L5 claim against R-baseline-H); the
# table ties R-robust-H to class H specifically and R-quick to no class,
# so a class-only derivation picks R-robust-H. `level` is not read here.
_REGIME_S = {2: "R-low-S", 3: "R-low-S", 4: "R-low-S", 8: "R-low-S"}
_REGIME_H = {32: "R-robust-H", 64: "R-robust-H",
             128: "R-baseline-H", 256: "R-baseline-H"}


@dataclasses.dataclass(frozen=True)
class Context:
    """The measurement context of one score (PRD-v2 5.4). Every field
    defaults to `None`; a caller fills only what it knows.
    """

    method: str = None
    n_dim: int = None
    epochs: int = None
    seeds: object = None
    lr: float = None
    update_rule: str = None
    dc_gain: float = None
    effective_lr: float = None
    strategy: str = None
    device: str = None
    machine: str = None
    peak_rss_mb: float = None
    wall_s: float = None
    regime: str = None
    level: str = None
    tier: str = None
    campaign: str = None
    supersedes: str = None
    beta: float = None     # momentum/nesterov/gen_momentum decay; law default 0.9
    alpha: float = None    # gen_momentum's dZ gain; law default 1.0


def _dc_gain(update_rule, beta, alpha):
    """The gain for `update_rule`, or `None` for an adaptive or unknown rule."""
    if update_rule in _FIXED_GAIN:
        return _FIXED_GAIN[update_rule]
    if update_rule in _MOMENTUM_LIKE:
        b = _BETA_DEFAULT if beta is None else beta
        return 1.0 / (1.0 - b)
    if update_rule == "gen_momentum":
        b = _BETA_DEFAULT if beta is None else beta
        a = _ALPHA_DEFAULT if alpha is None else alpha
        return a / (1.0 - b)
    return None  # adam, sqn (adaptive), or a rule not in the table


def _derive_regime(n_dim, cls):
    """A dimension band from `n_dim` and the graph's primary class, or
    `None` when `cls` is absent or the dimension is not in the table.
    """
    if not cls or n_dim is None:
        return None
    primary = cls[0]
    if primary == "S":
        return _REGIME_S.get(n_dim)
    if primary == "H":
        return _REGIME_H.get(n_dim)
    return None


def validate(ctx, Z, cls=None):
    """Fill `n_dim`, `dc_gain`, `effective_lr`, `regime`; check `n_dim`.

    Returns `(new_ctx, missing, warnings)`. `new_ctx` is a new `Context`
    (this dataclass is frozen); `ctx` is not mutated. `missing` names
    every plain field still `None` after filling -- the four derived
    fields above are never listed there, since a caller never types them.
    `warnings` holds a note for each guard that passed without raising.

    `n_dim` is taken from `Z.shape[1]`. If `ctx.n_dim` was also given and
    disagrees, this raises with both values in the message: a record
    whose stated dimension is not the dimension of the array it scored is
    worse than a record with no dimension.

    `dc_gain` is looked up from `update_rule` (the table in this module's
    docstring); `effective_lr = lr * dc_gain` when both are known, else
    `None`. Neither is ever taken from a value the caller typed.

    `regime` is derived from `n_dim` and `cls` (the graph's `Profile.cls`
    tuple, PRD-v2 5.3) when not already given; `cls` is optional because
    `validate()` runs from a bare `Z` too.
    """
    z_dim = Z.shape[1]
    if ctx.n_dim is not None and ctx.n_dim != z_dim:
        raise ValueError(
            f"Context.validate: context n_dim={ctx.n_dim} does not match "
            f"Z.shape[1]={z_dim}.")

    dc_gain = _dc_gain(ctx.update_rule, ctx.beta, ctx.alpha)
    effective_lr = (ctx.lr * dc_gain
                     if ctx.lr is not None and dc_gain is not None
                     else None)
    regime = ctx.regime if ctx.regime is not None else _derive_regime(z_dim, cls)

    new_ctx = dataclasses.replace(
        ctx, n_dim=z_dim, dc_gain=dc_gain, effective_lr=effective_lr,
        regime=regime)

    warnings = []
    if effective_lr is not None and effective_lr >= _STABILITY_EDGE:
        warnings.append("effective lr at or above the stability edge")

    missing = [f.name for f in dataclasses.fields(new_ctx)
               if f.name not in _DERIVED_FIELDS
               and getattr(new_ctx, f.name) is None]

    return new_ctx, missing, warnings
