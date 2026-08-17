"""
graph_building: the "make graph" stage (see ../docs/ARCHITECTURE.md).

Self-contained implementation, not a wrapper. Graph-construction
strategies (MST, kNN, ...) live in `./strategies/` and register
themselves via `@register_strategy` (from `./registry.py`); shared
helpers live in `./building_blocks.py`. This is the same design
previously factored out as a standalone `graphmaking` package at the
repo root -- absorbed here so `fdge2` has no dependency on anything
outside its own tree (the repo-root `graphmaking/` package is unchanged
and still used by the legacy `fdge_numba` code; this is an independent
copy, not a shared one -- the two are free to diverge).

Per the fdge2 import discipline (RPD.md Sec 5): this module may depend on
`numpy`/`networkx`/`scipy`/`pynndescent` only -- never on
`fdge2.graph_augmenting` or `fdge2.embedding`. Its only contract is
"returns a graph"; it does not know or care what happens to that graph
downstream.
"""

from __future__ import annotations

from .registry import make_graph, register_strategy, available_strategies
from . import strategies  # noqa: F401  (side effect: registers all strategies)

__all__ = ["make_graph", "available_strategies", "register_strategy"]
