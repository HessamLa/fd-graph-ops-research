"""embedding.drop -- the steady-rate random drop regularizer, and its keys.

Neither layout (``sell_c_sigma.py``) nor force law (``shell_force.py``):
this is a regularizer applied to ``dZ`` *after* the kernel has run, so it
lives on its own.

Two strategies, and they are NOT the same regularizer:

``random_rows`` zeroes a whole ``dZ`` row. The dropped nodes do not move.
``random_cells`` zeroes single cells. The node still moves, but in a
different direction than the force law computed.

Why the dropped rows are computed first
=======================================
The kernel computes every row, then this function discards about half. That
work looks wasted, but the plan has FIXED shapes for one ``jax.jit``
compile, and it groups rows by WIDTH, not by node id. A per-epoch row subset
therefore costs a recompile or a gather.
``experiments/drop_strategies/`` measures the alternatives. Read its
``REPORT.md`` before you change this file: the kernel is bandwidth-bound, so
less arithmetic does not imply less time, and the answer changes with the
size of the graph.

Own copy of ``fdge_jax.embedding.shell_force.drop_steady_rate`` -- see
``sell_c_sigma.py``'s module docstring for why this package duplicates a
handful of tiny helpers across the ``fdge_jax`` import boundary rather than
sharing them.
"""
# Provenance: moved VERBATIM from `fodined/embedding/drop.py` on 2026-08-19.
# Provenance: this file is a copy of `fdge_jax_sell_c_sigma/embedding/drop.py`.
# The code is identical. Some docstrings refer to documents that are not in
# this directory, for example `docs/DESIGN.md`. Refer to the origin package
# for these documents.
from __future__ import annotations

import numpy as np
import jax


def drop_steady_rate(dZ, key, drop_rate: float, strategy: str = 'random_rows'):
    """Zero whole rows (``random_rows``) or single cells (``random_cells``)
    of ``dZ`` with probability ``drop_rate``. See the module docstring: the
    two strategies are different regularizers.

    ``key`` must be NEW in each epoch. A repeated key repeats the mask, and
    the regularizer then does nothing.

    Two properties that are easy to miss:
     * The rate is an expectation, not a count: the dropped total differs
       per epoch. That is what "steady rate" means here.
     * Nothing is rescaled by ``1 / (1 - drop_rate)`` as dropout does, so a
       larger rate also shrinks the mean step. Rate and learning rate are
       coupled.

    The ``if`` below runs at Python/trace level against a plain float, not
    a traced value -- ``drop_rate = 0`` is a true no-op, not a multiply by
    a mask of ones.
    """
    if drop_rate <= 0.0:
        return dZ

    if strategy == 'random_rows':
        keep = jax.random.uniform(key, (dZ.shape[0],)) >= drop_rate
        dZ = dZ * keep[:, None]
    elif strategy == 'random_cells':
        keep = jax.random.uniform(key, dZ.shape) >= drop_rate
        dZ = dZ * keep
    else:
        # `raise "text"` raises TypeError instead: a str is not an exception.
        raise ValueError(
            f"Unknown drop strategy {strategy!r}. "
            f"Use 'random_rows' or 'random_cells'.")
    # dZ[keep,...]=0
    # dZ = dZ.at[keep].set(0)
    return dZ

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
