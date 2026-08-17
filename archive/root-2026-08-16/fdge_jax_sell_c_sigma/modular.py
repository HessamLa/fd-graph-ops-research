#!/bin/env python3
# In this file, we would like to implement fdge_jax_sell_c_sigma, but in a
# functional and modular way.
#
# - No main function is implemented here.
# - The script must run line by line and implement the exact functionality as
#   `fdge_jax_sell_c_sigma` module.
# - The only class from `fdge_jax_sell_c_sigma` imported here is
#  `ForceDirected` class in `./fdge_jax_sell_c_sigma/core/force_directed.py`.
# - Functions are implemented only when absolutely necessary. Some of the
#   helper or utility functions are borrowed or invoked from
#   `fdge_jax_sell_c_sigma` module, such as those in
#   `./fdge_jax_sell_c_sigma/embedding/shell_force.py`.
# - Here, only the graph embedding is implemented. We load an existing graph
#   and perform graph augmentation and embedding.
# - Don't delete the exsiting comments. You may add new comments though.
#
# ***************
# VERY IMPORTANT: Use a minimalistic approach. All the code must be
# implemented in a single file. Do not create any new files or modules.
# ***************
# VERY IMPORTANT: This is not a production grade code. Keep everything simple
# and minimalistic but functioning. Readability is the key.
# ***************
#
#---------------------------------------
# imports section
#---------------------------------------
# Run from the repo root (fdmap/), which is where `fdge_jax_sell_c_sigma`
# is importable from -- this file lives *inside* the package, so running it
# by path needs the root on sys.path:
#     PYTHONPATH=. .venv/bin/python fdge_jax_sell_c_sigma/modular.py
import functools
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

import jax
import jax.numpy as jnp

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

# The one class this file is allowed to import (see the header contract).
from fdge_jax_sell_c_sigma.core.force_directed import ForceDirected

# Everything else borrowed from the package is a plain function.
#   shell_force        -- THE force law (Fa + Fr), embedding/shell_force.py
#   shell_coeff_data   -- per-stored-pair 1/|S_h(u)| coefficient plane
#   degrees_from_D     -- force-law degree (count of hop-1 entries per row)
from fdge_jax_sell_c_sigma.embedding.shell_force import (
    shell_force, shell_coeff_data, degrees_from_D)
# SELL-C-sigma layout machinery: the host-side plan builder and the jitted
# per-epoch kernel. `_step` is "private" only in the sense that `PlanCache`
# normally owns it; importing it here is precisely what keeps this file on
# the SAME kernel as the package rather than a re-implementation of it
# (`PlanCache` itself is a class, so it is out of bounds for this file --
# and with a single, never-changing D there is nothing to cache anyway).
from fdge_jax_sell_c_sigma.embedding.sell_c_sigma import make_plan, _step
# The steady-rate random drop regularizer applied to dZ after the kernel.
from fdge_jax_sell_c_sigma.embedding.drop import drop_steady_rate

print(f"[modular] jax {jax.__version__} devices: {jax.devices()}")

#---------------------------------------
# Load graph and prepare the CSR
#---------------------------------------
# The edgelist of cora dataset is at
# `/home/h/gnn/fd-graph-embedding/fdmap/data_cache/cora/cora.cites`.
#
# Each line in this file contains `<node_a_id>\t<node_b_id>` representing an
# edge between node_a and node_b.
#
# Here are the first three lines of the edgelist
# ```
# 35	1033
# 35	103482
# 35	103515
# ```
EDGELIST = "/home/h/gnn/fd-graph-embedding/fdmap/data_cache/cora/cora.cites"

raw = np.loadtxt(EDGELIST, dtype=np.int64)                   # (n_edges, 2)
# Cora's node ids are arbitrary paper ids; the embedding is an (n, d) array
# indexed by row, so remap them onto a contiguous 0..n-1 range. `node_ids`
# keeps the original id of every row, in case a row needs naming later.
node_ids, flat = np.unique(raw.ravel(), return_inverse=True)
edges = flat.reshape(raw.shape)
edges = edges[edges[:, 0] != edges[:, 1]]                    # drop self-loops
n = node_ids.size

# Undirected: store both (u, v) and (v, u). Duplicate/parallel entries are
# merged by the CSR conversion but their weights SUM, so flatten the data
# back to a single 1.0 per stored pair afterwards.
rows = np.concatenate([edges[:, 0], edges[:, 1]])
cols = np.concatenate([edges[:, 1], edges[:, 0]])
A = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
A.data[:] = 1.0

# Both directions of every original edge -- used for the "is this pair
# already connected?" test in the two sampling loops below.
edge_set = set(zip(*(int_arr.tolist() for int_arr in A.nonzero())))

print(f"[modular] cora: n={n} nodes, {A.nnz // 2} undirected edges, "
      f"avg degree {A.nnz / n:.2f}")

#---------------------------------------
# Perform graph augmentation and prepare the augmented CSR
#---------------------------------------
# All existing edges have weight 1.0
# Basic graph augmentation. Randomly sample 1000 unconnected node pairs.
#
# Add an edge between each of the sampled node pairs and set the weight
# to the shortest path distance between the two nodes in the original graph.
# All the 1000 new edges will have weight > 1.0.
#
# If there is no path between the two nodes, set the weight to 100.0 (a large
# number).
N_NEW_EDGES = 1000
UNREACHABLE_W = 100.0
rng = np.random.default_rng(42)

# Rejection sampling: on a graph this sparse (avg degree ~4 out of 2708
# nodes) a uniformly random pair is almost always unconnected, so this loop
# rejects very rarely. `taken` keeps a pair from being drawn twice, which
# would otherwise create a duplicate entry whose weights SUM in the CSR.
taken = set()
new_pairs = []
while len(new_pairs) < N_NEW_EDGES:
    u, v = (int(x) for x in rng.integers(0, n, size=2))
    if u == v or (u, v) in edge_set or (u, v) in taken:
        continue
    taken.add((u, v))
    taken.add((v, u))
    new_pairs.append((u, v))
new_pairs = np.asarray(new_pairs)

# Hop distance for the sampled pairs: one unweighted BFS per DISTINCT source
# (~1000 rows of the all-pairs matrix), not the full (n, n) one.
srcs = np.unique(new_pairs[:, 0])
hop = shortest_path(A, method="D", unweighted=True, indices=srcs)
w_new = hop[np.searchsorted(srcs, new_pairs[:, 0]), new_pairs[:, 1]]
n_unreachable = int((~np.isfinite(w_new)).sum())
w_new[~np.isfinite(w_new)] = UNREACHABLE_W     # different component -> "far"

# The augmented D: original edges at weight 1.0 plus the sampled pairs at
# their hop distance, both directions. This is exactly the contract the
# embedding engine works against -- a csr_matrix whose stored entry
# D[u, v] = h >= 1 is a hop distance, and an absent pair is one the
# augmentation policy simply never reached.
Ac = A.tocoo()
D = sp.csr_matrix(
    (np.concatenate([Ac.data, w_new, w_new]),
     (np.concatenate([Ac.row, new_pairs[:, 0], new_pairs[:, 1]]),
      np.concatenate([Ac.col, new_pairs[:, 1], new_pairs[:, 0]]))),
    shape=(n, n))

print(f"[modular] augmented: {A.nnz} -> {D.nnz} stored entries "
      f"({N_NEW_EDGES} new pairs, {n_unreachable} of them unreachable), "
      f"hop weights {D.data.min():.0f}..{D.data.max():.0f}")

#---------------------------------------
# Perform graph embedding using SELL-C-sigma
#---------------------------------------
# Use the algorithm implemented in `fdge_jax_sell_c_sigma` module to perform
# graph embedding using JAX.
#
# - Instantiate an object or inherit a class from `ForceDirected`, depending
#   on how you plan to implement the `updateGradient` and/or `forces` methods.
# - Use the same hyperparameters and kernel as in `fdge_jax_sell_c_sigma`
#   module.
# - Set `n_dim=128` to get a 128-dimensional embedding.
# - The embedding will be performed for 2000 epochs, using GPU with JAX.
N_DIM = 128
EPOCHS = 2000
SEED = 42

# Verbatim from models.ReferenceFDModel / embedding.ShellForce's constructor
# defaults -- the "same hyperparameters" the header asks for. k3 is the
# literal 10.0, NOT None (None means "auto: use the node count", a far
# larger repulsion coefficient; see ShellForce's docstring).
K1, K2, K3, K4 = 0.999, 1.0, 10.0, 0.01
RANDOM_DROP_RATE = 0.5
B_CELLS, K_MAX, LADDER_BASE = 16_384, 256, 1.5
# ShellForce._h_shift(): hop distance 1 is the closest a stored pair can be.
# (WeightedShellForce's h_min baseline is for Euclidean edge weights; D here
# is a hop-count matrix, so the hop-count physics is the right one.)
H_SHIFT = 1.0


class SellCSigmaFD(ForceDirected):
    """`ForceDirected` with the two stage hooks this script needs.

    `make_graph` is deliberately left unimplemented (it raises, as in the
    base class): the graph is loaded from disk above, not built from raw
    feature vectors, so `embed(D)` is the entry point and stage 1 never
    runs.
    """

    def augment_graph(self, G, **kwargs):
        """Stage 2. D is already augmented above, so this only derives what
        the engine needs from it -- the same work `PlanCache.derive` does on
        a cache miss, done once here because there is only ever one D."""
        D = G
        planes = (shell_coeff_data(D), D.data)   # 1/|S_h(u)| and h, per pair
        plan, inv_deg_ext, self.plan_stats = make_plan(
            D, planes, degrees=degrees_from_D(D),
            b_cells=B_CELLS, k_max=K_MAX, ladder_base=LADDER_BASE)
        self.plan = jax.tree_util.tree_map(jax.device_put, plan)
        self.inv_deg_ext = jax.device_put(inv_deg_ext)
        # k1..k4/h_shift are traced scalars, so sweeping them never recompiles.
        self.params = dict(k1=K1, k2=K2, k3=K3, k4=K4, h_shift=H_SHIFT)
        self.step = jax.jit(functools.partial(
            _step, n=D.shape[0], force_fn=shell_force))
        return D

    def forces(self, Z, D, row_start, row_end, key=None, **kwargs):
        """Stage 3. One fused pass over the padded plan, then the row slice.

        The kernel always runs over the whole graph and the row range is
        sliced afterwards -- the plan's batches are grouped by row WIDTH
        across the whole graph, so a contiguous row range has no clean
        correspondence to a subset of them (same tradeoff ShellForce
        documents). With the default batch_count=1 it costs nothing.
        """
        if key is None:
            # embed()'s batch loop always passes one; this is for a direct
            # forces(...) call. ShellForce uses drop.FallbackKeys here, but
            # that is a class -- splitting self.key is the one-line
            # equivalent, and reproducible from the model's seed as a bonus.
            self.key, key = jax.random.split(self.key)
        full = self.step(jnp.asarray(Z, dtype=jnp.float32),
                         self.plan, self.inv_deg_ext, self.params)
        return drop_steady_rate(full[row_start:row_end], key, RANDOM_DROP_RATE)


fdobj = SellCSigmaFD(n_dim=N_DIM, verbosity=0, seed=SEED)

t0 = time.time()
fdobj.embed(D, epochs=EPOCHS)
Z = fdobj.get_embeddings()                       # (n, 128) numpy array
elapsed = time.time() - t0

print(f"[modular] plan: {fdobj.plan_stats}")
print(f"[modular] embedded {Z.shape} in {elapsed:.1f}s "
      f"({EPOCHS / elapsed:.1f} epochs/s), final ||dZ|| avg "
      f"{fdobj.Th(fdobj.dZ):.6f}, finite: {np.isfinite(Z).all()}")

#---------------------------------------
# Perform link prediction using the learned embedding
#---------------------------------------
# Analyze the embedding.
#
# - Fit a random forest classifier to predict whether two
#   input nodes are connected. Report accuracy, recall, precision,
#   F1-score, and AUC.
# - Use 80% of the edges for training and 20% for testing.

# Positives: every ORIGINAL undirected edge, once (upper triangle).
pos = np.column_stack(sp.triu(A, k=1).nonzero())

# Negatives: as many random unconnected pairs, so the classifier sees a
# balanced problem and accuracy stays meaningful.
neg = []
seen = set()
while len(neg) < pos.shape[0]:
    u, v = (int(x) for x in rng.integers(0, n, size=2))
    if u == v or (u, v) in edge_set or (u, v) in seen:
        continue
    seen.add((u, v))
    seen.add((v, u))
    neg.append((u, v))
neg = np.asarray(neg)

# Hadamard product of the two endpoints' embeddings -- the standard edge
# feature for embedding-based link prediction, and symmetric in (u, v) as
# an undirected edge feature must be.
pairs = np.vstack([pos, neg])
X = Z[pairs[:, 0]] * Z[pairs[:, 1]]
y = np.concatenate([np.ones(pos.shape[0]), np.zeros(neg.shape[0])])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

clf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print(f"[modular] link prediction on {X.shape[0]} pairs "
      f"({pos.shape[0]} edges + {neg.shape[0]} non-edges), "
      f"{X_train.shape[0]} train / {X_test.shape[0]} test")
print(f"  accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  precision: {precision_score(y_test, y_pred):.4f}")
print(f"  recall   : {recall_score(y_test, y_pred):.4f}")
print(f"  f1-score : {f1_score(y_test, y_pred):.4f}")
print(f"  auc      : {roc_auc_score(y_test, y_prob):.4f}")
