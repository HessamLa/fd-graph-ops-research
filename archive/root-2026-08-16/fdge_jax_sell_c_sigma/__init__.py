"""fdge_jax_sell_c_sigma -- force-directed graph embedding, JAX, SELL-C-sigma engine.

Same three-stage pipeline and same physics as ``fdge_jax`` (see
docs/DESIGN.md and fdge_jax/docs/DESIGN.md for the shared contract):

    make_graph  ->  augment_graph  ->  embed
    (graph_building/)  (graph_augmenting/)   (embedding/)

The only thing this sibling package changes is ``embedding/``'s execution
engine: instead of a flat edge-list + ``segment_sum`` kernel, it uses the
bucketed/padded SELL-C-sigma layout (host-side hub-split + width-sort +
ladder-quantize + cell-budget batching, see
``dev_docs/fdmap_engine_design_notes.md`` and
``archive/fdmap_bucketed_bench_jax.py``) for higher throughput on large,
power-law-degree graphs.

Unlike ``fdge_jax`` (float64 by default, for numeric parity with the
Numba ``fdge2`` sibling), this package leaves JAX's default float32
enabled -- the whole point of this engine is throughput, and the
bucketed kernel was designed and measured in float32
(``dev_docs/fdmap_engine_design_notes.md`` Sec 2, 9). A researcher who
needs float64 can still opt in with
``jax.config.update("jax_enable_x64", True)`` before importing this
package; nothing here hardcodes float32 at the JAX-config level, only in
``core.force_directed.ForceDirected``'s own array constructions.
"""
from __future__ import annotations
