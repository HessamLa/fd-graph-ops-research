"""embedding.drop -- the steady-rate random drop regularizer, and its keys.

Neither layout (``sell_c_sigma.py``) nor force law (``shell_force.py``):
this is a regularizer applied to ``dZ`` *after* the kernel has run, so it
lives on its own. It is applied per NODE (whole ``dZ`` row), not per
coordinate -- dropping a node means "this node doesn't move this step",
which is a meaningful perturbation of an n-body layout; dropping a single
coordinate of a node's displacement is not.

Own copy of ``fdge_jax.embedding.shell_force.drop_steady_rate`` -- see
``sell_c_sigma.py``'s module docstring for why this package duplicates a
handful of tiny helpers across the ``fdge_jax`` import boundary rather than
sharing them.
"""
from __future__ import annotations

import numpy as np
import jax


def drop_steady_rate(dZ, key, drop_rate: float):
    """Zero out whole ``dZ`` rows (nodes) with probability ``drop_rate``.

    The ``if`` below runs at Python/trace level against a plain float, not
    a traced value -- ``drop_rate = 0`` is a true no-op, not a multiply by
    a mask of ones.
    """
    if drop_rate <= 0.0:
        return dZ
    keep = jax.random.uniform(key, (dZ.shape[0],)) >= drop_rate
    return dZ * keep[:, None]


class FallbackKeys:
    """Lazily-seeded PRNG key stream for callers that pass ``key=None``.

    The force bindings accept ``key=None`` so a quick script doesn't have
    to thread a PRNG key through just to get a drop mask. That path is
    deliberately **not reproducible**: the root key is seeded once from OS
    entropy (``numpy.random.SeedSequence``) and split on every call, so two
    successive calls genuinely differ. Anything that needs reproducibility
    passes its own ``key``.

    One implementation, shared by every binding class -- this used to be a
    hand-copied ``_fallback_key`` method in two places, which is exactly
    the kind of duplication that drifts.
    """

    def __init__(self):
        self._root = None

    def __call__(self):
        if self._root is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
            self._root = jax.random.PRNGKey(seed)
        self._root, sub = jax.random.split(self._root)
        return sub
