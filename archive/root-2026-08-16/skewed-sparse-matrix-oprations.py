# %%
# Using Jax+GPU
# mask is a sparse binary matrix, 
# shape nxn, its row sum is ceiling(n*e^(-r*20)), r=uniform distribution in [0, 1]
#
# C1 is a sparse matrix, float values
# shape nxn, if mask[i,j] is 0, then C1[i,j] = 0, else C1[i,j] = random float
# 
# C2 is similar to C1, but with different random values
#
# Z is dense nxd tensor, float values, randomly initialized
# 
# Zdiff = Z[:, None, :] - Z[None, :, :]  # shape nxnxd
# Zdiff_norm = jnp.linalg.norm(Zdiff, axis=-1)  # shape nxn
# Zdiff_unit = Zdiff / (Zdiff_norm[..., None] + 1e-8)  # shape nxnxd
#
# dZ is dense nxd tensor, float values
# We want to compute ddZ = Zdiff_unit @ (C1 @ Zdiff_norm - C2 @ exp(-Zdiff_norm))  # shape nxnxd, 
# where @ is matrix elementwise multiplication, and ddZ is nxnxd tensor
# Then we update dZ by summing ddZ row-wise, i.e., dZ[i] = sum_j ddZ[i,j,:]
#
# LOOP 10 times:
#   ddZ = Zdiff_unit @ (C1 @ Zdiff_norm - C2 @ exp(-Zdiff_norm))
#   dZ = reduce_sum ddZ along rows  # shape nxnxd -> nxd
#   cap dZ to be in [-1, 1]
#   Z += dZ * random learning rate
#
# Since C1 and C2 are sparse, we want to avoid computing the full nxnxd tensor ddZ, 
# and instead compute dZ directly using sparse matrix operations.
# 

# %%
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
import networkx as nx
import traceback
# %%
n = 10
d = 3
shape = (n, n, d)

# Build the sparsity pattern + values as plain COO triples first (row i,
# col j, per-edge d-vector), then construct the BCOO array ONCE at the
# end. Incrementally `.at[].set()`-ing into a sparse array row by row
# means rebuilding its whole index/data structure on every iteration
# (O(n) rebuilds of a growing array) -- building the index/value arrays
# in Python/NumPy and constructing the BCOO in one shot is both the
# standard idiom and avoids that.
rows, cols, vals = [], [], []

# Create mask nxn
# number of ones in each row is ceiling(n*e^(-r*20)), r=uniform distribution in [0, 1]
for i in range(n):
    r = np.random.uniform(0, 1)
    row_sum = int(np.ceil(n * np.exp(-r * 20)))  # in (0, n] for any r in [0,1] -> ceil in [1, n]

    # `row_sum` distinct column indices for this row (without
    # replacement -- a column can't be "one" twice).
    j_idx = np.random.choice(n, size=row_sum, replace=False)
    rows.append(np.full(row_sum, i))
    cols.append(j_idx)

rows = np.concatenate(rows)
cols = np.concatenate(cols)
mask_indices = np.stack([rows, cols], axis=1)  # (nnz, 2) -- (row, col) pairs where mask == 1

mask = jsparse.BCOO((np.ones(rows.shape[0]), mask_indices), shape=(n, n))
print(f"mask: shape={mask.shape}, nse={mask.nse} (density={mask.nse / (n * n):.3f})")

print(mask.todense())  # dense representation of the mask

# %%

G = nx.barabasi_albert_graph(100000, 2)
print(G.number_of_nodes())
print(G.number_of_edges())

# %%
constructor = [(200, 400, 0.8)]
G = nx.random_shell_graph(constructor)
nx.draw(G, node_size=1, width=0.5)

# %%
from generate_sparse_adjacency_matrix import \
        build_sparse_adjacency, to_jax_bcoo, generate_ba_graph

N = 1000_000
N = 25000
M = 2
SEED = 42

print(f"generating BA graph: n={N}, m={M}, seed={SEED}")
row, col = generate_ba_graph(N, m=M, seed=SEED, verbose=True)

adj = build_sparse_adjacency(row, col, N)
print(f"adjacency matrix: shape={adj.shape}, nnz={adj.nnz}, "
        f"density={adj.nnz / (N * N):.2e}")

degrees = np.asarray(adj.sum(axis=1)).ravel()
print(f"degree stats: min={degrees.min()}, max={degrees.max()}, "
        f"mean={degrees.mean():.2f}")
# print(f"adjacency matrix:\n{adj.todense()}")
# sort by degree
sorted_indices = np.argsort(degrees)
adj = adj[sorted_indices, :][:, sorted_indices]
# print(adj.todense())

# %%
d = 3
key = jax.random.key(SEED)
# Split into independent subkeys -- reusing one `key` across multiple
# jax.random calls (as the original Z/C1_flatten/C2_flatten draws did)
# gives IDENTICAL values every time, silently violating "C2 is similar
# to C1, but with different random values" from the spec at the top.
key_z, key_c1, key_c2 = jax.random.split(key, 3)
Z = jax.random.normal(key_z, shape=(N, d))

# C1 and C2 are sparse matrices with float values and the same sparsity
# pattern as the adjacency matrix -- reuse adj's own (row, col) index
# pairs (via the already-imported `to_jax_bcoo` helper) rather than
# rebuilding them, so C1/C2 are guaranteed aligned position-for-position
# with adj's stored entries.
adj_bcoo = to_jax_bcoo(adj)
C1_flatten = jax.random.normal(key_c1, shape=(adj.nnz,))
C2_flatten = jax.random.normal(key_c2, shape=(adj.nnz,))  # different key -> different values than C1
# unflatten C1 and C2 to the same shape as the adjacency matrix
C1 = jsparse.BCOO((C1_flatten, adj_bcoo.indices), shape=(N, N))
C2 = jsparse.BCOO((C2_flatten, adj_bcoo.indices), shape=(N, N))
print(f"C1: shape={C1.shape}, nse={C1.nse} (density={C1.nse / (N * N):.3f})")
print(f"C2: shape={C2.shape}, nse={C2.nse} (density={C2.nse / (N * N):.3f})")

from jax.experimental.sparse import bcoo_multiply_sparse
# print(C1.todense())
# smoke test for memory utilization
for batch_size in [1000, 5000, 10000, 15000, 20000, 25000]:
    print(f"batch size: {batch_size}")
    try:
        Zdiff = Z[None, :, :] - Z[:batch_size, None, :]  # shape (batch_size, N, d)
        Znorm = jnp.linalg.norm(Zdiff, axis=-1)  # shape (batch_size, N)
        print(f"Zdiff: shape={Zdiff.shape}, Znorm: shape={Znorm.shape}")
        C1_batch = jnp.multiply(C1[:batch_size, :], Znorm)  # shape (batch_size, N)
        C2_batch = jnp.multiply(C2[:batch_size, :], jnp.exp(-Znorm))  # shape (batch_size, N)
        H = Zdiff @ (C1_batch - C2_batch)[..., None]  # shape (batch_size, N, d)
        dZ = jnp.sum(H, axis=1)  # shape (batch_size, d)
    except Exception as e:
        print(f"batch size {batch_size} failed: {e}")
        traceback.print_exc()
        break

# %%

# Make Zdiff and Znorm sparse with the same sparsity pattern as the adjacency matrix 
Zdiff = Z[None, :, :] - Z[:, None, :]  # shape nxnxd
Znorm = jnp.linalg.norm(Zdiff, axis=-1)  # shape nx

# Now compute H
H = Zdiff*(C1 @ Znorm - C2 @ jnp.exp(-Znorm))[..., None]  # shape nxnxd
# %%

a = jnp.array([1, 4, 9])

print(a[None,:] - a[:2, None])  # shape 3x3
              