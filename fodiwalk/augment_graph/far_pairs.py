#!/bin/env python3
"""far_pairs.py -- the long-range pairs, and the degree bias that draws them.

A walk gives near pairs only. Without a long-range term the layout knows
the neighbourhood of a node and not its place in the graph, and the hop R2
falls. This module draws pairs that the walk did NOT find, and the caller
stores them at a large weight.

Provenance: `degree_table`, `_draw` and `sample_far_pairs` are a verbatim
move from `fodined/graph_augmentation.py`. TWO changes, and neither touches
the arithmetic:

  * ONE new parameter, `directed`, whose default keeps the old behaviour
    exactly. See `sample_far_pairs`.
  * the third parameter is named `near` and not `ball`. It is the matrix
    the sampler rejects against, and in this package that matrix is the
    near `D` of a WALK -- there is no ball here. The origin name described
    the k-hop ball of `fodined/modular.py`, which this package does not
    hold (`dev-docs/CATALOG.md` section 18).

Every function is pure, it prints nothing, and the generator arrives as an
argument.
"""
from __future__ import annotations

import numpy as np


def degree_table(A, alpha: float):
    """The cumulative weights `deg^alpha`, for a biased draw of a node.

    `alpha = 0` gives a uniform draw, and `alpha = 0.75` gives the
    distribution that word2vec and node2vec use for a negative sample
    [4, 5]. The difference is large on a graph with hubs: on com_youtube a
    node of degree 28,754 carries `28754^0.75 = 2208`, and a node of the
    average degree 5.27 carries `3.47`, thus the hub is 636 times more
    likely. A uniform draw makes them equal, thus a hub of that graph
    receives almost no repulsion, while its attraction is still divided by
    its degree, through `inv_deg_ext`.

    Returns `None` when `alpha == 0`, thus the caller keeps the fast path.
    The array is `n` float64, which is 9 MB at 1.13M nodes.
    """
    if alpha == 0.0:
        return None
    deg = np.diff(A.indptr).astype(np.float64)
    return np.cumsum(np.power(np.maximum(deg, 1.0), alpha))


def _draw(cum, n: int, size: int, rng):
    """`size` node ids, uniform when `cum is None`, and biased otherwise."""
    if cum is None:
        return rng.integers(0, n, size)
    return np.searchsorted(cum, rng.random(size) * cum[-1])


def sample_far_pairs(n: int, count: int, near, rng,
                     batch: int = 2_000_000, max_tries: int = 40,
                     cum=None, directed: bool = False):
    """`count` pairs that `near` does NOT hold.

    `directed` is the ONE addition of the move into this package, and the
    default `False` is the behaviour of every recorded run.

    The rejection test reads the keys of `near` as they are stored. A
    SYMMETRIC `near` holds both directions, thus a pair is rejected in
    either direction and `directed=False` is already correct. A DIRECTED
    `near` -- the `D` of `walk_rows` -- holds `(u, v)` and not `(v, u)`,
    thus a pair that the rows already hold as `(v, u)` passes the test,
    the CSR build SUMS the two entries, and the weight becomes
    `far_weight + the walk gap`. The histogram showed entries at 101..119
    on 2026-08-17. With `directed=True` the test looks at BOTH directions
    and the pair is rejected.

    The default stays `False` because a change of the rejection changes
    how many pairs one batch accepts, thus it changes the draw of the next
    batch, thus it changes the numbers of a recorded run. `True` is for a
    directed `near` only, where the old behaviour is a defect.

    `cum` comes from `degree_table`. With `None` the two nodes of a pair are
    drawn uniformly, which is what this function always did. With a table
    they are drawn in proportion to `deg^alpha`, thus a hub appears in many
    more far pairs. See `degree_table` for why that matters.

    The test needs no search. `near` already holds every pair the near
    policy kept, thus a pair that `near` does not hold is one the walks
    never joined, or one in another component. Both cases want the same
    large weight.

    The sampling is vectorized: a sorted array of the composite keys
    `u * n + v`, and one `np.searchsorted` for a whole batch. The older
    policy used a Python set, which needed 1.8 GB on a graph of a million
    nodes.

    `batch` bounds the memory. One draw holds about nine int64 arrays at the
    same time, thus a batch of the full `count` needs about 90 bytes for each
    pair: 2.2 GB for the 12.4M pairs that roadNet-CA asks for. The batch
    holds that peak near 180 MB, whatever `count` is.

    A dense `near` makes the rejection slow. The loop therefore stops after
    `max_tries` batches for each batch of work, and it returns what it has;
    the caller sees the count.
    """
    b = near.tocoo()
    keys = np.sort(b.row.astype(np.int64) * n + b.col.astype(np.int64))
    del b
    # The pairs that the loop accepted, as SORTED keys of `min(u, v)` and
    # `max(u, v)`. One key for a pair, thus a later batch cannot give the
    # same pair a second time, in either direction. A duplicate would become
    # one entry of `D` with the SUM of the two weights.
    acc = np.empty(0, np.int64)
    limit = max_tries * max(1, -(-count // batch))
    for _ in range(limit):
        if acc.size >= count:
            break
        draw = min((count - acc.size) * 2 + 1024, batch)
        u = _draw(cum, n, draw, rng)
        v = _draw(cum, n, draw, rng)
        ok = u != v
        u, v = u[ok], v[ok]
        k = u * n + v
        pos = np.searchsorted(keys, k)
        pos[pos >= keys.size] = 0
        hit = keys[pos] == k                     # `near` holds it: reject
        if directed:
            kr = v * n + u                       # the other direction
            pos = np.searchsorted(keys, kr)
            pos[pos >= keys.size] = 0
            hit = hit | (keys[pos] == kr)
        u, v = u[~hit], v[~hit]
        k = np.unique(np.minimum(u, v) * n + np.maximum(u, v))
        if acc.size:                             # not accepted before
            pos = np.searchsorted(acc, k)
            pos[pos >= acc.size] = 0
            k = k[acc[pos] != k]
        room = count - acc.size
        if k.size > room:
            # `np.unique` sorted the batch. A plain `k[:room]` would thus
            # keep the smallest node ids only, and the sample would not be
            # uniform.
            k = rng.choice(k, room, replace=False)
        acc = np.sort(np.concatenate([acc, k]))
    return np.column_stack([acc // n, acc % n])
