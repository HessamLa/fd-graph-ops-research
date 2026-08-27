"""core.force_directed -- A FORWARDER. Do not edit the engine here.

The epoch loop, the batching, the callback events and the `Z` update live
in ONE place now:

    forcedirected/force_directed.py

The class is `ForceDirected` again. It was `ForceDirectedEmbedding` from
2026-08-21 to 2026-08-26; that name is GONE and no alias is kept here,
because two live names for one class is exactly what the rename removes. A
caller that still says `ForceDirectedEmbedding` gets an `ImportError` and
not a silent second name.

Why the engine left `core`: `fodined` uses the same engine and the same
kernel, and `fodined` must not be made to depend on `fodiwalk`. The engine
therefore sits at the repository ROOT, reads numpy, scipy and jax only, and
every dependency points AT it. Nothing was lost: the full module
documentation -- the chunk contract, I7, and the provenance of the three
changes of the original move -- went WITH the code and is unchanged there.

IMPORT DISCIPLINE. This file joins `core/sell_c_sigma.py` and `core/csr.py`
as a forwarder that reads a ROOT package. `forcedirected` reads NO package
of this repository, thus the dependency still runs one way and no cycle is
possible. See `dev-docs/CATALOG.md` and `forcedirected/PARITY.md`.

To change the engine, edit `forcedirected/force_directed.py`.
"""
from __future__ import annotations

from forcedirected.force_directed import ForceDirected, Callback_Base

__all__ = ["ForceDirected", "Callback_Base"]
