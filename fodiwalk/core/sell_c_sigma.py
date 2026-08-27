"""core.sell_c_sigma -- A FORWARDER. Do not edit the algorithm here.

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

`core/plan_contract.py` still asserts the plane contract that the moved
file documents and does not check. That did not move and does not change.

IMPORT DISCIPLINE. This file, `core/force_directed.py` and `core/csr.py`
are forwarders to `forcedirected`, the ROOT engine package. Every other
module of `core` imports numpy, scipy, jax and `core` only. `forcedirected`
reads NO package of this repository, thus the dependency runs one way and no
cycle is possible.

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

# `_step` is the old name of `step` and `core/__init__.py` re-exports it.
# Bound by assignment and not by import: gate D6
# (`tests/test_structure.py`) forbids a plain module to IMPORT a private
# name of another module, and that gate stays satisfied.
_step = step
_to_csr = to_csr

__all__ = [
    "build_ladder", "make_plan", "step", "_step", "PlanCache",
    "to_csr", "_to_csr",
]
