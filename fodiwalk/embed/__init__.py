"""fodiwalk.embed -- STAGE 3, CONSUMPTION ONLY. The engine itself is `core`.

`dev-docs/fodiwalk-module.md`: "This stage shall not do any graph analysis
or data preparation. It must only consume the data. Its main goal is to
apply the force function on the input data using the best implementation to
optimize resource utilization."

Stage 3 takes what stage 2 already prepared -- the matrix `D`, its PLANES,
the DEGREE divisor and the force PARAMS -- and turns them into the batch
PLAN and the jitted kernel steps that `core.sell_c_sigma` runs. It builds
NO plane and resolves NO degree: `augment_graph.planes.build_planes` and
`augment_graph.degrees.resolve_degrees` do that, because a plane and a
degree are properties of the RECIPE (the pair policy plus the force law),
not of the kernel that later reads them (`dev-docs/CATALOG.md`, the RECIPE
entry, 2026-08-21).

    planner.py   `PlanSpec`, `PlanSet`, `build_plans`

Usage, which is the wiring the model class holds::

    from .augment_graph.planes import ForceSpec, build_planes, force_params
    from .augment_graph.degrees import resolve_degrees
    from .embed import PlanSpec, build_plans

    fspec = ForceSpec.from_config(cfg)
    pspec = PlanSpec.from_config(cfg)
    planes = build_planes(fspec.law, D, freq, cfg.pairs, cfg.policy)
    degrees = resolve_degrees(D, A, fspec, explicit=deg_from_A)
    if pspec.check_planes:
        plan_contract.check(fspec.law, planes, D)
        plan_contract.check_degrees(degrees, D)
    params = force_params(fspec)
    plan_set = build_plans(D, planes, degrees, pspec, force_fn(fspec.law))

Import discipline: `embed` imports `core`, numpy, scipy and jax. It imports
NO `augment_graph` and NO model, thus the dependency of the package runs
one way (`dev-docs/REFACTOR.md` section 3). `PlanSpec` reads CONFIGURATION,
and `core` may not; that is why this code is here and not in `core`.
"""
from __future__ import annotations

from .planner import PlanSet, PlanSpec, build_plans

__all__ = ["PlanSpec", "PlanSet", "build_plans"]
