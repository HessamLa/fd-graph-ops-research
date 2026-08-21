"""fodiwalk.core -- the engine, the layout machinery, and the physics.

Import discipline: every module here imports numpy, scipy, jax and other
`core` modules ONLY. `core` never imports `augment_graph`, `make_graph` or
`misc` at module level, thus the dependency of the package runs one way and
a cycle is a defect.

Public surface::

    from fodiwalk.core import ForceDirected, Callback_Base
    from fodiwalk.core import row_of, n_rows
    from fodiwalk.core import make_plan, PlanCache, build_ladder
    from fodiwalk.core import FORCE_PLANES, FORCE_FN, planes_of, force_fn
    from fodiwalk.core import plan_contract
"""
from __future__ import annotations

from .csr import row_of, n_rows
from .force_directed import ForceDirected, Callback_Base
from .sell_c_sigma import build_ladder, make_plan, PlanCache, _step
from .forces import (FORCE_PLANES, FORCE_FN, planes_of, force_fn,
                     fdlinear, fdlinear_fused, fuse, degrees_from_D)
from . import plan_contract
from .plan_contract import PlaneContractError

__all__ = [
    "row_of", "n_rows",
    "ForceDirected", "Callback_Base",
    "build_ladder", "make_plan", "PlanCache", "_step",
    "FORCE_PLANES", "FORCE_FN", "planes_of", "force_fn",
    "fdlinear", "fdlinear_fused", "fuse", "degrees_from_D",
    "plan_contract", "PlaneContractError",
]
