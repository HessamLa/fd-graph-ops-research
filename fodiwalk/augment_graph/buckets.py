#!/bin/env python3
"""buckets.py -- the augmentation policy of the off-branch experiment.

The policy:

    Keep EVERY original edge at `h = 1`.
    Add exactly `n * log10(n)` pairs with `h >= 2`, in the proportions
    50% at `h = 2`, 25% at `h = 3`, and 25% at `h >= 4`.

Thus the augmented graph holds `|E| + |V| * log10(|V|)` undirected pairs,
which is the size that the specification asks for.

Two consequences, and both are deliberate:

1. **No far pairs, and no sentinel.** The old policy adds random pairs at
   the weight 100. The size formula above leaves no room for them, thus
   this policy has none, and every `h` in `D` is a REAL hop distance or a
   real walk gap.
2. **The budget is global, and not per node.** The cap of the walk policy
   keeps `m` pairs for each node. This policy takes a fixed number of pairs
   from the whole graph. A hub can therefore hold many pairs, and a leaf
   can hold none beyond its edges.

A bucket that has fewer pairs than its share gives what it has, and the
caller reports the shortfall. It is not filled from another bucket: the
proportions are the experiment.
"""
# Provenance: moved from `experiments/fdwalk/buckets.py` on 2026-08-19.
# `bucket_sample` and `budget` are verbatim; `row_buckets` came from
# `experiments/fdwalk/walks.py`, body unchanged.
from __future__ import annotations

import numpy as np

FRACTIONS = (0.50, 0.25, 0.25)          # h == 2, h == 3, h >= 4


def bucket_sample(h, total: int, rng, fractions=FRACTIONS):
    """Indices into `h` of the pairs to keep. Returns `(idx, report)`.

    `h` holds the hop distance (or the walk gap) of every candidate pair
    with `h >= 2`. The pairs at `h = 1` are not candidates: the policy keeps
    all of them.
    """
    groups = (np.flatnonzero(h == 2), np.flatnonzero(h == 3),
              np.flatnonzero(h >= 4))
    names = ("h=2", "h=3", "h>=4")
    want = [int(round(total * f)) for f in fractions]

    keep, report = [], []
    for name, idx, w in zip(names, groups, want):
        if idx.size <= w:
            take = idx                    # the bucket is short
        else:
            take = rng.choice(idx, w, replace=False)
        keep.append(take)
        report.append({"bucket": name, "available": int(idx.size),
                       "asked": int(w), "taken": int(take.size)})
    return np.concatenate(keep) if keep else np.empty(0, np.int64), report


def budget(n: int):
    """`n * log10(n)`, the number of `h >= 2` pairs that the policy adds."""
    return int(n * np.log10(max(n, 10)))


# ---------------------------------------------------------------------------
# 2026-08-18 -- the same policy on the DIRECTED rows of `walk_rows`.
# Moved here from `experiments/fdwalk/walks.py`, body unchanged.
# ---------------------------------------------------------------------------
def row_buckets(stats, A, n: int, total: int, rng,
                fractions=(0.50, 0.25, 0.25)):
    """The bucket policy, on the DIRECTED rows of `walk_rows`.

    `buckets.bucket_sample` works on an undirected pair list. This is the
    same rule on directed rows, and it keeps the three properties that make
    the policy what it is:

    1. Every entry at `h = 1` stays. The budget applies to `h >= 2` only.
    2. The budget is GLOBAL, and not per row.
    3. A short bucket is not topped up from another.

    `walk_rows` gives `h` in `mn`, thus the strata read from `mn` directly
    and no hop distance has to be recomputed.
    """
    h = stats["mn"]
    near = h <= 1
    far = np.flatnonzero(h >= 2)
    groups = (far[h[far] == 2], far[h[far] == 3], far[h[far] >= 4])
    want = [int(round(total * f)) for f in fractions]
    keep, report = [np.flatnonzero(near)], []
    for name, idx, w in zip(("h=2", "h=3", "h>=4"), groups, want):
        take = idx if idx.size <= w else rng.choice(idx, w, replace=False)
        keep.append(take)
        report.append({"bucket": name, "available": int(idx.size),
                       "asked": int(w), "taken": int(take.size)})
    sel = np.sort(np.concatenate(keep))
    out = {k: (v[sel] if isinstance(v, np.ndarray) else v)
           for k, v in stats.items()}
    return out, report
