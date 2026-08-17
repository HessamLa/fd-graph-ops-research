"""fdge_jax -- force-directed graph embedding, JAX port of fdge2.

Same three-stage pipeline as fdge2 (see docs/DESIGN.md and
fdge2/docs/ARCHITECTURE.md):

    make_graph  ->  augment_graph  ->  embed
    (graph_building/)  (graph_augmenting/)   (embedding/)

JAX is enabled with float64 (x64) so numeric behavior matches fdge2's
float64 NumPy/Numba path by default -- a researcher who wants float32
speed can opt out by not importing this package's config side effect
(or by casting arrays down explicitly).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
