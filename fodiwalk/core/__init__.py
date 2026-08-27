"""fodiwalk.core -- the engine, the layout machinery, and the physics.

Import discipline: every module here imports numpy, scipy, jax and other
`core` modules ONLY. `core` never imports `augment_graph`, `make_graph` or
`misc` at module level, thus the dependency of the package runs one way and
a cycle is a defect.

THREE EXCEPTIONS, since 2026-08-26: `force_directed.py`, `sell_c_sigma.py`
and `csr.py` are FORWARDERS and import `forcedirected`, the ROOT package
that now holds the engine, the SELL-C-sigma kernel, the CSR helpers and the
update rules for this package and for `fodined` alike. It began 2026-08-25
as `sellcsigma`, which held the kernel alone; the kernel has exactly one
caller, thus 2026-08-26 absorbed it into the engine's own package.
`forcedirected` reads no package of this repository, thus the dependency
still runs one way and no cycle is possible. The structure gates read
`fodiwalk.*` imports only and are therefore silent on it; see CATALOG
section 23 and `forcedirected/PARITY.md`.

Public surface::

    from fodiwalk.core import ForceDirected, Callback_Base
    from fodiwalk.core import row_of, n_rows
    from fodiwalk.core import make_plan, PlanCache, build_ladder, step
    from fodiwalk.core import FORCE_PLANES, FORCE_FN, planes_of, force_fn
    from fodiwalk.core import plan_contract

`ForceDirected` was `ForceDirectedEmbedding` between 2026-08-21 and
2026-08-26 (it was renamed when `make_graph`/`augment_graph`/`fit` left it
-- it is a pure engine now, and the pipeline contract moved to
`fodiwalk.Fodiwalk_base`, one layer up). The 2026-08-26 rename back is a
REAL rename: the old name is gone and no alias is kept.
"""
from __future__ import annotations

from .csr import row_of, n_rows
from .force_directed import ForceDirected, Callback_Base
from .sell_c_sigma import build_ladder, make_plan, PlanCache, step, _step
from .forces import (FORCE_PLANES, FORCE_FN, planes_of, force_fn,
                     fdlinear, fdlinear_fused, fuse, degrees_from_D)
from . import plan_contract
from .plan_contract import PlaneContractError

__all__ = [
    "row_of", "n_rows",
    "ForceDirected", "Callback_Base",
    "build_ladder", "make_plan", "PlanCache", "step", "_step",
    "FORCE_PLANES", "FORCE_FN", "planes_of", "force_fn",
    "fdlinear", "fdlinear_fused", "fuse", "degrees_from_D",
    "plan_contract", "PlaneContractError",
]
