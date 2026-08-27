"""embedding.sell_c_sigma -- A FORWARDER. Do not edit the algorithm here.

The SELL-C-sigma layout and kernel live in ONE place now:

    forcedirected/sell_c_sigma.py

This module was one of two copies of that code. The copies never diverged
in the algorithm, but a copy that nobody diffs is a defect that waits, thus
2026-08-25 made a package and turned both copies into forwarders; 2026-08-26
ABSORBED that package into `forcedirected`, next to the one class that
calls the kernel. The implementation moved byte for byte. Nothing
was lost: the full module documentation -- the layout in one paragraph, the
coefficient-plane contract, the padding contract, the `D` contract, and the
note on the plan builder's own notion of degree -- moved WITH the code and
is unchanged there. Read it there. Parity evidence:
`forcedirected/PARITY.md`.

This file exists so `fodined`, `fdwalk/modular.py` and the four benchmark
scripts under `experiments/` keep their import path. It holds no logic.

`forcedirected` reads numpy, scipy and jax ONLY. This import therefore does
NOT make `fodined` depend on `fodiwalk`, and it must never be allowed to.

To change the layout or the kernel, edit `forcedirected/sell_c_sigma.py`.
"""
from __future__ import annotations

from forcedirected.sell_c_sigma import (
    build_ladder,
    make_plan,
    step,
    PlanCache,
    to_csr,
)

# The names this package's own modules import: `shell_force.py` names
# `_to_csr`, and `modular.py` and the benchmark scripts name `_step`. Bound
# by assignment and not by import, thus a private name never crosses a
# module boundary.
_step = step
_to_csr = to_csr

__all__ = [
    "build_ladder", "make_plan", "step", "_step", "PlanCache",
    "to_csr", "_to_csr",
]
