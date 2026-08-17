"""fdge2.embedding -- the "embed" stage (see ../docs/ARCHITECTURE.md).

Owns the force law: ``forces(Z, D, row_start, row_end, ...) -> dZ`` and the
scalar/vector force functions it calls. The reference implementation (T3.1)
is the shell-averaged force law in ``shell_force.py`` -- a faithful port of
``model_204_shell._forces_204_hidx`` onto the fdge2 ``D`` contract.

Import discipline (RPD.md Sec 5): this package imports ``numpy``, ``numba``
and ``fdge2.core`` only -- never ``fdge2.graph_building`` or
``fdge2.graph_augmenting``. ``D`` arrives as a plain argument; this module must
not know or care how it was produced. Sparse ``D`` is handled by
duck-typed densification (``hasattr(D, "toarray")``) so ``scipy`` is not
imported either.

Public surface:

    from fdge2.embedding import shell_force
    from fdge2.embedding.shell_force import ShellForce, forces, _scalar_force
"""
from __future__ import annotations

from . import shell_force
from .shell_force import ShellForce, DropSteadyRate, forces, _scalar_force

__all__ = ["shell_force", "ShellForce", "DropSteadyRate", "forces", "_scalar_force"]
