#!/bin/env python3
"""graph_augmentation.py -- the augmentation stage of fodined, as functions.

The augmentation takes a graph and it gives the matrix `D` that the embedding
engine consumes. `D` is a `scipy.sparse.csr_matrix`, and a stored entry
`D[u, v] = h >= 1` is a hop distance. An absent pair is a pair that the
policy did not keep.

The policy has three steps:

1. Take `count` random pairs of nodes that have NO edge.
2. Give each pair the hop distance of its two nodes in the original graph.
3. Put those pairs into the graph, beside the original edges, which keep the
   weight 1.0.

Every function here is pure: it reads only its arguments, it writes nothing
outside its result, and it prints nothing. The caller keeps the
configuration and the messages. `modular.py` is that caller.

The random generator arrives as an argument, and this module never makes one.
That is what keeps a run reproducible: `modular.py` uses ONE generator for
the augmentation and for the negative pairs of the link prediction, thus the
order of the draws is a part of the result.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path


def edge_set_of(A) -> set:
    """Both directions of every edge of `A`, as a set of `(u, v)` tuples.

    The two sampling loops ask "does this pair have an edge?" many times, and
    a Python set answers in constant time.

    The memory is the limit of this approach: a graph of a million nodes has
    millions of edges, and one tuple in a set costs about 130 bytes. See
    `experiments/large-graph-node2vec/REPORT.md`.
    """
    return set(zip(*(a.tolist() for a in A.nonzero())))


def sample_unconnected_pairs(n: int, count: int, edge_set: set, rng):
    """`count` pairs `(u, v)` that have no edge, and no pair two times.

    Rejection sampling. A graph of this class is sparse, thus a random pair
    almost never has an edge, and the loop rejects very rarely.

    `taken` holds both directions of each pair that the loop accepted. A pair
    that arrives two times would become one entry of `D` with the SUM of the
    two weights, which is not a hop distance.
    """
    taken = set()
    pairs = []
    while len(pairs) < count:
        u, v = (int(x) for x in rng.integers(0, n, size=2))
        if u == v or (u, v) in edge_set or (u, v) in taken:
            continue
        taken.add((u, v))
        taken.add((v, u))
        pairs.append((u, v))
    return np.asarray(pairs)


def source_block_size(n: int, budget_bytes: float = 200e6) -> int:
    """How many sources one call of `shortest_path` may ask for.

    `shortest_path` always returns a DENSE `(len(indices), n)` float64 array.
    The `indices` argument limits the number of ROWS, and it does not make
    the result sparse, because the value of an unreachable pair is `inf` and
    there is no zero to omit. Thus the block, and not the graph, must bound
    the memory: one block costs `block * n * 8` bytes.
    """
    return max(4, int(budget_bytes / (n * 8)))


def hop_distances(A, n: int, pairs, chunk: int | None = None):
    """The exact hop distance of every pair, over blocks of sources.

    Returns `(weights, sources, chunk)`. A pair whose two nodes are in
    different components gets `inf`; `replace_unreachable` gives it a number.

    One BFS runs for each DISTINCT source. The cost is therefore proportional
    to the number of distinct sources, and not to the number of pairs.
    """
    srcs = np.unique(pairs[:, 0])
    chunk = chunk or source_block_size(n)
    src_pos = np.searchsorted(srcs, pairs[:, 0])
    w = np.empty(pairs.shape[0])
    for s in range(0, srcs.size, chunk):
        blk = srcs[s:s + chunk]
        hop = shortest_path(A, method="D", unweighted=True, indices=blk)
        m = (src_pos >= s) & (src_pos < s + blk.size)
        w[m] = hop[src_pos[m] - s, pairs[m, 1]]
    return w, srcs, chunk


def replace_unreachable(w, value: float):
    """`inf` becomes `value`. Returns `(weights, how_many)`.

    A pair in a different component has no path. The force law needs a
    number, thus the caller gives a large one, and a large `h` makes the
    repulsion strong. `n` is the value that the origin package uses.
    """
    n_unreachable = int((~np.isfinite(w)).sum())
    return np.where(np.isfinite(w), w, float(value)), n_unreachable


def augmented_csr(A, pairs, w):
    """`D`: the edges of `A` at 1.0, plus each pair at its hop distance.

    Both directions of each new pair go in, thus `D` stays symmetric.
    """
    Ac = A.tocoo()
    return sp.csr_matrix(
        (np.concatenate([Ac.data, w, w]),
         (np.concatenate([Ac.row, pairs[:, 0], pairs[:, 1]]),
          np.concatenate([Ac.col, pairs[:, 1], pairs[:, 0]]))),
        shape=A.shape)


# ---------------------------------------------------------------------------
# The k-hop policy. It replaces the per-source BFS above.
# ---------------------------------------------------------------------------
def k_hop_ball(A, k: int, block: int = 2048):
    """Every pair within `k` hops, with its hop distance as the value.

    Returns a `csr_matrix` where a stored entry `(u, v) = h` means that `v`
    is exactly `h` hops from `u`, and `1 <= h <= k`. The diagonal is empty.

    Why this scales, and the per-source BFS does not: one BFS for each source
    costs `O(n * (V + E))` for the whole graph, and the sample of the older
    policy touches nearly every node as a source. This function instead takes
    `k` sparse products. It never computes a distance that it does not keep,
    thus the cost follows the SIZE OF THE RESULT and not the size of the
    graph.

    The rows are processed in blocks, thus the peak memory holds the ball of
    `block` nodes, and not of all of them.

    A warning about the result: on a small-world graph the ball grows very
    fast with `k`. A graph with an average degree of 5 has about 125 nodes in
    a 3-hop ball, but a hub reaches a large part of the graph. Measure
    `D.nnz` before a large `k`.
    """
    n = A.shape[0]
    Ab = A.astype(np.float32)
    Ab.data[:] = 1.0
    rows, cols, hops = [], [], []

    for s in range(0, n, block):
        e = min(s + block, n)
        F = Ab[s:e].copy()                       # exactly 1 hop from a row
        F.data[:] = 1.0
        F = _drop_self(F, s)
        seen = F.copy()
        _record(F, s, rows, cols, hops, 1)

        for h in range(2, k + 1):
            F = F @ Ab
            if F.nnz == 0:
                break
            F.data[:] = 1.0
            F = F - F.multiply(seen)             # keep only what is new
            F.eliminate_zeros()
            F = _drop_self(F, s)
            if F.nnz == 0:
                break
            _record(F, s, rows, cols, hops, h)
            seen = seen + F
            seen.data[:] = 1.0

    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float64)
    return sp.csr_matrix(
        (np.concatenate(hops).astype(np.float64),
         (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def _drop_self(F, row_offset: int):
    """A copy of `F` with the entries `(u, u)` removed.

    `F` holds a block of rows, thus row `i` of `F` is node `i + row_offset`.
    A product `F @ A` always returns a node to itself after two hops, and
    that entry is not a distance.
    """
    coo = F.tocoo()
    keep = (coo.row + row_offset) != coo.col
    return sp.csr_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])), shape=F.shape)


def _record(F, row_offset, rows, cols, hops, h):
    coo = F.tocoo()
    rows.append(coo.row + row_offset)
    cols.append(coo.col)
    hops.append(np.full(coo.row.size, h))


def degree_table(A, alpha: float):
    """The cumulative weights `deg^alpha`, for a biased draw of a node.

    `alpha = 0` gives a uniform draw, and `alpha = 0.75` gives the
    distribution that word2vec and node2vec use for a negative sample
    [4, 5]. The difference is large on a graph with hubs: on com_youtube a
    node of degree 28,754 carries `28754^0.75 = 2208`, and a node of the
    average degree 5.27 carries `3.47`, thus the hub is 636 times more
    likely. A uniform draw makes them equal, thus a hub of that graph
    receives almost no repulsion, while its attraction is divided by its
    degree two times (`shell_coeff` and `inv_deg_ext`).

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


def sample_far_pairs(n: int, count: int, ball, rng,
                     batch: int = 2_000_000, max_tries: int = 40,
                     cum=None):
    """`count` pairs that are MORE than `k` hops apart.

    `cum` comes from `degree_table`. With `None` the two nodes of a pair are
    drawn uniformly, which is what this function always did. With a table
    they are drawn in proportion to `deg^alpha`, thus a hub appears in many
    more far pairs. See `degree_table` for why that matters.

    The test needs no search. `ball` already holds every pair within `k`
    hops, thus a pair that `ball` does not hold is further away than `k`, or
    it is in another component. Both cases want the same large weight.

    The sampling is vectorized: a sorted array of the composite keys
    `u * n + v`, and one `np.searchsorted` for a whole batch. The older
    policy used a Python set, which needed 1.8 GB on a graph of a million
    nodes.

    `batch` bounds the memory. One draw holds about nine int64 arrays at the
    same time, thus a batch of the full `count` needs about 90 bytes for each
    pair: 2.2 GB for the 12.4M pairs that roadNet-CA asks for. The batch
    holds that peak near 180 MB, whatever `count` is.

    A dense ball makes the rejection slow. The loop therefore stops after
    `max_tries` batches for each batch of work, and it returns what it has;
    the caller sees the count.
    """
    b = ball.tocoo()
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
        near = keys[pos] == k                    # within the ball: reject
        u, v = u[~near], v[~near]
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


def augment_k_hop(A, n: int, k: int, far_count: int, far_weight: float, rng,
                  block: int = 2048):
    """The k-hop policy. Returns `(D, info)`.

    1. Every pair within `k` hops keeps its true hop distance.
    2. `far_count` random pairs that are further away get `far_weight`.

    No shortest-path search runs at any point.
    """
    ball = k_hop_ball(A, k, block=block)
    far = sample_far_pairs(n, far_count, ball, rng)
    if far.shape[0]:
        w = np.full(far.shape[0], float(far_weight))
        D = augmented_csr(ball, far, w)
    else:
        D = ball.copy()
    hop_counts = {int(h): int(c) for h, c in
                  zip(*np.unique(ball.data, return_counts=True))}
    return D, {"ball_nnz": int(ball.nnz), "hop_counts": hop_counts,
               "far_pairs": int(far.shape[0]), "pairs": far}


def augment(A, n: int, count: int, unreachable_w: float, edge_set: set, rng,
            chunk: int | None = None):
    """The three steps together. Returns `(D, info)`.

    `info` holds the numbers that a caller prints: `pairs`, `sources`,
    `chunk`, and `unreachable`.
    """
    pairs = sample_unconnected_pairs(n, count, edge_set, rng)
    w, srcs, chunk = hop_distances(A, n, pairs, chunk)
    w, n_unreachable = replace_unreachable(w, unreachable_w)
    return augmented_csr(A, pairs, w), {
        "pairs": pairs, "sources": srcs.size, "chunk": chunk,
        "unreachable": n_unreachable}
