"""fodined.embedding -- the "embed" stage (see ../docs/DESIGN.md).

Copy of `fdge_jax_sell_c_sigma/embedding/__init__.py`. Only the module names
in the import discipline note and the example below are different. The
`graph_building` and `graph_augmenting` stage packages are not part of this
copy: `modular.py` does these two stages line by line.

Owns the force law: ``forces(Z, D, row_start, row_end, ...) -> dZ``. Three
files, one job each -- so that changing the physics never means reading the
layout machinery, which is the whole point of the split:

    shell_force.py    THE FORCE LAW. Fa/Fr, the shell counts and degrees
                      they're defined in terms of, and the thin classes
                      binding them to a graph. **Edit this one.**
    sell_c_sigma.py   Layout machinery, force-law agnostic: hub-split,
                      width-sort, ladder-quantize, pad, batch, the jitted
                      ``lax.scan`` kernel, the per-``D`` plan cache. It
                      takes the force law as an argument and never looks
                      inside a coefficient plane.
    drop.py           The steady-rate random drop regularizer (+ the
                      ``key=None`` fallback key stream it needs). Neither
                      layout nor physics.

The engine is the SELL-C-sigma bucketed/padded kernel -- same shell-averaged
physics as ``fdge_jax.embedding.shell_force``, but executed as fixed-shape,
padded rectangular batches reduced with a dense ``sum(axis=1)`` instead of a
flat edge-list ``segment_sum`` scatter (see ``sell_c_sigma.py``'s module
docstring and ``docs/DESIGN.md`` for the full rationale).

Import discipline (docs/DESIGN.md): this package imports ``numpy``,
``scipy.sparse``, ``jax``/``jax.numpy`` and ``fodined.core`` only -- never a
graph-building or graph-augmenting stage package, and never the sibling
``fdge_jax`` package. ``D`` arrives as a plain argument; this module must
not know or care how it was produced.

Public surface:

    from fodined.embedding import shell_force, sell_c_sigma, drop
    from fodined.embedding import (
        ShellForce, WeightedShellForce, drop_steady_rate)
"""
from __future__ import annotations

from . import drop, sell_c_sigma, shell_force
from .drop import drop_steady_rate, FallbackKeys
from .sell_c_sigma import PlanCache, build_ladder, make_plan
from .shell_force import ShellForce, WeightedShellForce

__all__ = [
    "drop", "sell_c_sigma", "shell_force",
    "ShellForce", "WeightedShellForce",
    "PlanCache", "build_ladder", "make_plan",
    "drop_steady_rate", "FallbackKeys",
]
