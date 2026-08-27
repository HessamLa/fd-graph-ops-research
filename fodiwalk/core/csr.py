"""core.csr -- A FORWARDER. Do not edit the helpers here.

`row_of` and `n_rows` live in ONE place now:

    forcedirected/csr.py

They moved 2026-08-26 with the engine that reads them: `ForceDirected` is a
ROOT package and `from .csr import n_rows` must resolve inside it. The
module documentation -- what `row_of` does and why `D` is a plain
`scipy.sparse.csr_matrix` -- went with the code and is unchanged there.

IMPORT DISCIPLINE: `forcedirected` reads numpy, scipy and jax only, thus
the dependency runs one way and no cycle is possible.

To change these helpers, edit `forcedirected/csr.py`.
"""
from __future__ import annotations

from forcedirected.csr import row_of, n_rows

__all__ = ["row_of", "n_rows"]
