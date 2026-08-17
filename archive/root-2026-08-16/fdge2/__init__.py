"""fdge2 -- force-directed graph embedding, three-stage pipeline.

Stages (see docs/ARCHITECTURE.md):
    make_graph  ->  augment_graph  ->  embed
    (graph_building/)  (graph_augmenting/)   (embedding/)

``core`` holds the ``ForceDirected`` base class and shared CSR utilities
that the three stage packages code against. Composition of a concrete
model (wiring one make_graph + one augment_graph + one forces together)
lives one level up in ``fdge2/models.py`` -- see core.force_directed's
module docstring for why.
"""
