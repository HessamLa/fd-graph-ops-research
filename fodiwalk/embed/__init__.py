"""fodiwalk.embed -- STAGE 3. The force law, and the kernel that applies it.

`dev-docs/fodiwalk-module.md`: "This stage shall not do any graph analysis
or data preparation. It must only consume the data. Its main goal is to
apply the force function on the input data using the best implementation to
optimize resource utilization."

Stage 3 receives the DATA of `augment_graph.result.Augmentation` -- the
matrix `D`, whose `D.data` holds the `h` values, the `(nnz,)` `freq` array
aligned to `D.indices`, and the statistics -- and turns it into the planes,
the degree divisor, the batch PLAN and the jitted kernel steps that
`forcedirected` runs.

    forces.py         the laws, `FORCE_PLANES`, `FORCE_FN`, `degrees_from_D`
    plan_contract.py  the asserter that a law gets the values it expects
    planes.py         `ForceSpec`, `PLANE_BUILDERS`, `build_planes`
    degrees.py        `resolve_degrees`
    planner.py        `PlanSpec`, `PlanSet`, `build_plans`

EVERYTHING THAT KNOWS A FORCE LAW IS HERE. A plane, a degree divisor and a
force param are all defined by the law that reads them: `fdlinear` reads
`(h, freq)`, attraction lives at `h = 1`, thus the degree counts the `h = 1`
entries. `forces.py` and `plan_contract.py` were `fodiwalk/core/` and
`planes.py` and `degrees.py` were `fodiwalk/augment_graph/`; all four came
here on 2026-08-28 and `fodiwalk/core/` was deleted. Stage 2 had been
importing the law to ask it what to build, and the stages exchange DATA
only.

Usage, which is the wiring the model class holds::

    from .embed import (ForceSpec, PlanSpec, build_planes, resolve_degrees,
                        force_params, force_fn, build_plans, plan_contract)

    fspec = ForceSpec.from_config(cfg)
    pspec = PlanSpec.from_config(cfg)
    planes = build_planes(fspec.law, D, freq, cfg.pairs, cfg.policy)
    degrees = resolve_degrees(D, A, fspec, explicit=deg_from_A)
    if pspec.check_planes:
        plan_contract.check(fspec.law, planes, D)
        plan_contract.check_degrees(degrees, D)
    params = force_params(fspec)
    plan_set = build_plans(D, planes, degrees, pspec, force_fn(fspec.law))

Import discipline: numpy, scipy, jax, its own modules and the ROOT package
`forcedirected`. NO `augment_graph`, NO `make_graph`, NO `misc`, NO model.
Stage 2 imports nothing of this package either: `model.py` is where the two
stages meet, and it belongs to neither. `tests/test_structure.py` asserts
both directions.
"""
from __future__ import annotations

from .forces import (FORCE_PLANES, FORCE_FN, planes_of, force_fn,
                     fdlinear, fdlinear_fused, fuse, degrees_from_D)
from . import plan_contract
from .plan_contract import PlaneContractError
from .planes import ForceSpec, PLANE_BUILDERS, build_planes, force_params
from .degrees import resolve_degrees
from .planner import PlanSet, PlanSpec, build_plans

__all__ = [
    "FORCE_PLANES", "FORCE_FN", "planes_of", "force_fn",
    "fdlinear", "fdlinear_fused", "fuse", "degrees_from_D",
    "plan_contract", "PlaneContractError",
    "ForceSpec", "PLANE_BUILDERS", "build_planes", "force_params",
    "resolve_degrees",
    "PlanSpec", "PlanSet", "build_plans",
]
