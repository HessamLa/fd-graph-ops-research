#!/bin/env python3
"""merge.py -- the ONE far-pair CSR merge of stage 2.

Three policies add long-range pairs to the matrix they built, and until
this module each one carried its own copy of the block. The copies were not
identical, thus a repair of one left the other two wrong.

THE ORDER OF THE COO TRIPLES IS PART OF THE RESULT. `sp.csr_matrix((data,
(row, col)))` SUMS a duplicate coordinate, and the sum order decides the
last bit of a float. The near entries go first, then the pairs in the
forward direction, then the pairs in the backward direction. Do not sort,
and do not group.

The far weight is the only real difference between the callers, thus it is
an ARGUMENT: one constant for each pair (`cap`, `buckets`, `nbr_walk`), or
a landmark distance for each pair (`landmarks`). `freq` always gets 1.0,
because a far pair was drawn and never observed.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def add_far_pairs(D, freq, far, weights):
    """The far pairs onto `D` and `freq`, in BOTH directions.

    `far` is `(m, 2)` node ids, `weights` is `(m,)` -- one weight for each
    pair, used for both directions. `freq` may be None, and then only the
    matrix comes back.

    `freq` keeps the sparsity of `D`, thus the two `.data` arrays still line
    up entry by entry after the build.
    """
    m = far.shape[0]
    near = D.tocoo()
    D_out = sp.csr_matrix(
        (np.concatenate([near.data, weights, weights]),
         (np.concatenate([near.row, far[:, 0], far[:, 1]]),
          np.concatenate([near.col, far[:, 1], far[:, 0]]))),
        shape=D.shape)
    if freq is None:
        return D_out, None
    counts = freq.tocoo()
    freq_out = sp.csr_matrix(
        (np.concatenate([counts.data, np.ones(2 * m)]),
         (np.concatenate([counts.row, far[:, 0], far[:, 1]]),
          np.concatenate([counts.col, far[:, 1], far[:, 0]]))),
        shape=freq.shape)
    return D_out, freq_out


def drop_pairs_of(far, near, n: int):
    """Drop a far pair that `near` holds in EITHER direction.

    `nbr_walk` only. `sample_far_pairs` rejects on the keys of `near` AS
    STORED, and `near` is DIRECTED there, thus a pair stored as `(v, u)`
    does not reject `(u, v)`. The CSR build then SUMS the two and the weight
    becomes 100 + the walk gap; the histogram showed entries at 101..119 on
    2026-08-17.

    Why this filter and not `sample_far_pairs(..., directed=True)`, which
    rejects both ways inside the loop: a changed rejection changes how many
    pairs one batch accepts, thus it changes the draw of the next batch and
    the far set of every recorded run. The two give the same PROPERTY and
    not the same pairs, and parity decides. See CATALOG.md 8.2. The filter
    runs AFTER the draw, and it must stay there.
    """
    stored = near.tocoo()
    have = np.sort(stored.row.astype(np.int64) * n
                   + stored.col.astype(np.int64))

    def absent(key):
        pos = np.searchsorted(have, key)
        pos[pos >= have.size] = 0
        return have[pos] != key

    forward = far[:, 0].astype(np.int64) * n + far[:, 1]
    backward = far[:, 1].astype(np.int64) * n + far[:, 0]
    return far[absent(forward) & absent(backward)]
