"""fdge_jax.embedding -- the "embed" stage (see ../docs/DESIGN.md).

Owns the force law: ``forces(Z, D, row_start, row_end, ...) -> dZ``. The
reference implementation is the shell-averaged force law in
``shell_force.py`` -- a JAX port of ``fdge2.embedding.shell_force`` (same
physics, ``jax.jit``'d execution engine).

Import discipline (docs/DESIGN.md): this package imports ``numpy``,
``jax``/``jax.numpy`` and ``fdge_jax.core`` only -- never ``numba``, never
``fdge_jax.graph_building`` or ``fdge_jax.graph_augmenting``. ``D`` arrives
as a plain argument; this module must not know or care how it was
produced.

Public surface:

    from fdge_jax.embedding import shell_force
    from fdge_jax.embedding.shell_force import ShellForce, forces, drop_steady_rate
"""
from __future__ import annotations

from . import shell_force
from .shell_force import ShellForce, DropSteadyRate, forces, drop_steady_rate

__all__ = ["shell_force", "ShellForce", "DropSteadyRate", "forces", "drop_steady_rate"]
