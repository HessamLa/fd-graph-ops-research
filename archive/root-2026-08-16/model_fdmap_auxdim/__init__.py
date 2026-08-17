"""model_fdmap_auxdim -- force-directed embedding with annealed auxiliary dims.

A standalone research add-on sitting next to ``fdge_jax_sell_c_sigma``
(which it imports from, but is not part of). One model:
``EuclideanAuxDimModel``, an ``EuclideanDistanceModel`` that embeds in
``d + aux`` dimensions and shrinks the ``aux`` columns toward zero over
training, so the layout can route around obstructions early while the final
embedding is still genuinely ``d``-dimensional.

``idea.md`` in this directory is the full design record (mechanism,
rationale, known risks, validation plan, and the deferred Step 2 variant
that is deliberately NOT implemented). ``model.py`` implements Step 1 only.

    from model_fdmap_auxdim import EuclideanAuxDimModel, DECAY_SCHEDULES

    m = EuclideanAuxDimModel(d=2, aux=4, decay_schedule="cosine", seed=0)
    m.fit(data, epochs=500, strategy="mst")
    Z = m.get_embeddings()          # (n, 2) -- aux columns dropped
"""
from __future__ import annotations

from .model import (
    EuclideanAuxDimModel,
    DECAY_SCHEDULES,
    linear_decay,
    cosine_decay,
    linear_sine_decay,
    sigmoid_decay,
    exponential_decay,
)

__all__ = [
    "EuclideanAuxDimModel",
    "DECAY_SCHEDULES",
    "linear_decay",
    "cosine_decay",
    "linear_cosine_decay",
    "sigmoid_decay",
    "exponential_decay",
]
