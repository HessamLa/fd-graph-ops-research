"""
Generate a simple, connected, power-law graph (Barabasi-Albert model)
and return its adjacency matrix in sparse (scipy.sparse.csr_matrix) format.

Algorithm: standard Barabasi-Albert preferential attachment.
  - Start from a small connected seed graph (a path on m0 nodes).
  - Each new node attaches with m edges, chosen with probability
    proportional to existing node degree (preferential attachment).
  - Preferential sampling is done via a "repeated nodes" pool so each
    attachment step is O(m) amortized instead of O(n).

Guarantees:
  - Simple: no self-loops, no multi-edges (targets sampled without
    replacement within a single node's attachment step).
  - Connected: every node attaches to the existing graph as it's added.
  - Power-law degree distribution: asymptotic exponent gamma ~ 3.

Libraries: numpy (generation), scipy.sparse (adjacency assembly),
jax.experimental.sparse (optional conversion for downstream JAX use).
"""

import time
import numpy as np
from scipy.sparse import coo_matrix


def generate_ba_graph(n: int, m: int = 3, seed: int | None = 42, verbose: bool = True):
    """
    Generate a Barabasi-Albert graph.

    Parameters
    ----------
    n : int
        Number of nodes.
    m : int
        Number of edges each new node attaches with (m >= 1, m < n).
    seed : int or None
        RNG seed for reproducibility. None -> nondeterministic.
    verbose : bool
        Print basic progress/timing info.

    Returns
    -------
    row, col : np.ndarray
        Symmetric edge index arrays (undirected: both (i,j) and (j,i) present).
    """
    if m < 1 or m >= n:
        raise ValueError(f"require 1 <= m < n, got m={m}, n={n}")

    rng = np.random.default_rng(seed)

    # --- seed graph: a path on the first m0 = m+1 nodes, guarantees a
    # connected, simple starting point with every seed node degree >= 1 ---
    m0 = m + 1
    seed_row = np.arange(0, m0 - 1)
    seed_col = np.arange(1, m0)

    # Repeated-nodes pool, preallocated for the whole run: each node id
    # appears once per edge-endpoint it holds, so sampling uniformly from
    # the filled prefix of this array == sampling proportional to degree.
    # Preallocating avoids ever converting a growing Python list to a numpy
    # array inside the loop (that conversion was the O(n) per-iteration cost
    # that made the original version O(n^2) overall).
    final_edges = (m0 - 1) + m * (n - m0)
    pool = np.empty(2 * final_edges, dtype=np.int64)
    pool_len = 0
    pool[pool_len: pool_len + seed_row.size] = seed_row
    pool_len += seed_row.size
    pool[pool_len: pool_len + seed_col.size] = seed_col
    pool_len += seed_col.size

    # Preallocate output edge arrays too, instead of growing Python lists
    # of small arrays (also has per-call numpy overhead at n=100k).
    out_row = np.empty(2 * final_edges, dtype=np.int64)
    out_col = np.empty(2 * final_edges, dtype=np.int64)
    out_row[:pool_len] = pool[:pool_len]           # reuse seed edges as-is
    out_col[: seed_row.size] = seed_col
    out_col[seed_row.size:pool_len] = seed_row
    out_pos = pool_len

    t0 = time.time()
    oversample = 3  # draw extra candidates per round to make dedup cheap
    for new_node in range(m0, n):
        # Sample m distinct existing nodes weighted by degree, without
        # replacement (so this node's own edges stay simple). Draw a batch
        # of candidate indices at once and de-duplicate; loop only in the
        # rare case a batch doesn't yield m unique targets.
        targets = np.empty(m, dtype=np.int64)
        filled = 0
        while filled < m:
            need = m - filled
            idx = rng.integers(0, pool_len, size=need * oversample)
            cand = pool[idx]
            # dedup while preserving random draw order (np.unique sorts,
            # which would systematically bias toward low node IDs)
            _, first_occurrence = np.unique(cand, return_index=True)
            cand = cand[np.sort(first_occurrence)]
            take = cand[:need]
            targets[filled: filled + take.size] = take
            filled += take.size

        new_edges_row = np.full(m, new_node, dtype=np.int64)

        out_row[out_pos: out_pos + m] = new_edges_row
        out_col[out_pos: out_pos + m] = targets
        out_pos += m
        out_row[out_pos: out_pos + m] = targets
        out_col[out_pos: out_pos + m] = new_edges_row
        out_pos += m

        # update pool in place: new_node and each target gain one degree
        pool[pool_len: pool_len + m] = targets
        pool_len += m
        pool[pool_len: pool_len + m] = new_node
        pool_len += m

        if verbose and new_node % 20000 == 0:
            elapsed = time.time() - t0
            print(f"  node {new_node:>7}/{n}  ({elapsed:6.1f}s elapsed)")

    if verbose:
        print(f"done in {time.time() - t0:.1f}s, {out_pos // 2} undirected edges")

    return out_row, out_col


def build_sparse_adjacency(row, col, n, dtype=np.uint8):
    """Assemble a scipy CSR adjacency matrix from symmetric edge index arrays."""
    data = np.ones(row.shape[0], dtype=dtype)
    adj = coo_matrix((data, (row, col)), shape=(n, n))
    adj.sum_duplicates()   # should be no-ops if generation is correct; safety net
    adj.data[:] = 1        # re-binarize in case sum_duplicates created >1 entries
    return adj.tocsr()


def to_jax_bcoo(adj_csr):
    """
    Optional: convert a scipy CSR adjacency matrix to a JAX BCOO sparse matrix,
    for use downstream in fdge_jax / other JAX-based pipelines.
    """
    import jax.numpy as jnp
    from jax.experimental import sparse as jsparse

    adj_coo = adj_csr.tocoo()
    indices = np.stack([adj_coo.row, adj_coo.col], axis=1)
    bcoo = jsparse.BCOO(
        (jnp.asarray(adj_coo.data), jnp.asarray(indices)),
        shape=adj_coo.shape,
    )
    return bcoo


if __name__ == "__main__":
    N = 100_000
    M = 3
    SEED = 42

    print(f"generating BA graph: n={N}, m={M}, seed={SEED}")
    row, col = generate_ba_graph(N, m=M, seed=SEED, verbose=True)

    adj = build_sparse_adjacency(row, col, N)
    print(f"adjacency matrix: shape={adj.shape}, nnz={adj.nnz}, "
          f"density={adj.nnz / (N * N):.2e}")

    degrees = np.asarray(adj.sum(axis=1)).ravel()
    print(f"degree stats: min={degrees.min()}, max={degrees.max()}, "
          f"mean={degrees.mean():.2f}")

    # --- optional JAX conversion ---
    try:
        bcoo = to_jax_bcoo(adj)
        print(f"JAX BCOO adjacency: shape={bcoo.shape}, nse={bcoo.nse}")
    except ImportError:
        print("jax not installed; skipping BCOO conversion")