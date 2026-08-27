#!/bin/env python3
# In this file, we would like to implement fodined, but in a
# functional and modular way.
#
# - No main function is implemented here.
# - The script must run line by line and implement the exact functionality as
#   `fodined` module.
# - The only class from `fodined` imported here is
#  `Fodined` class in `./fodined/core/fodined.py`.
# - Functions are implemented only when absolutely necessary. Some of the
#   helper or utility functions are borrowed or invoked from
#   `fodined` module, such as those in
#   `./fodined/embedding/shell_force.py`.
# - Here, only the graph embedding is implemented. We load an existing graph
#   and perform graph augmentation and embedding.
# - Don't delete the exsiting comments. You may add new comments though.
#
# ***************
# VERY IMPORTANT: Use a minimalistic approach. All the code must be
# implemented in a single file. Do not create any new files or modules.
#   UPDATE 2026-08-14: the augmentation stage moved to
#   `./fodined/graph_augmentation.py`, on request. That file holds pure
#   functions only, and this file keeps the configuration, the order of the
#   steps, and every message. No other stage moved.
# ***************
# VERY IMPORTANT: This is not a production grade code. Keep everything simple
# and minimalistic but functioning. Readability is the key.
# ***************
#
#---------------------------------------
# imports section
#---------------------------------------
# Start this script from the repo root (fdmap/). Both of these commands
# work:
#     .venv/bin/python fodined/modular.py
#     .venv/bin/python -m fodined.modular
#
# This file is *inside* the package that it imports. Python puts only the
# directory of the script on sys.path, thus `import fodined` fails when you
# start the script by its path. The three lines below put the parent
# directory of `fodined` on sys.path, and the import finds the package.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functools
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path, breadth_first_order

# `DEVICE` selects the backend that JAX uses for the embedding.
#   'auto' -- JAX chooses. It takes the fastest backend that it finds.
#   'cpu' | 'gpu' | 'tpu' -- force that backend.
#
# This block is here, and not with the other constants below, because JAX
# starts its backend when you import it. A change after `import jax` does not
# move the arrays to a different device.
#
# A forced backend that is not available stops the script with a
# `RuntimeError`. That is better than a silent run on the CPU, which looks
# like a correct run but is 10 to 100 times slower.
DEVICE = 'gpu'          # 'auto' | 'cpu' | 'gpu' | 'tpu'
DEVICE = os.environ.get('FODINED_DEVICE', DEVICE)

# JAX calls the CUDA backend 'cuda', and not 'gpu'. The name 'gpu' makes JAX
# try 'cuda' AND 'rocm', and it then stops with a RuntimeError when 'rocm' is
# not in the build -- on this machine that happens although the CUDA card
# works. Thus 'gpu' becomes 'cuda' here. Set DEVICE = 'rocm' for an AMD card.
if DEVICE != 'auto':
    os.environ['JAX_PLATFORMS'] = {'gpu': 'cuda'}.get(DEVICE, DEVICE)

import jax
import jax.numpy as jnp

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score,
                             mean_absolute_error, mean_absolute_percentage_error)

# The one class this file is allowed to import (see the header contract).
from fodined.core.fodined import Fodined

# The augmentation stage, as pure functions. `augment` does the three steps
# together; the others are there for a caller that wants one of them alone.
from fodined.graph_augmentation import (
    augment, augment_k_hop, k_hop_ball, sample_far_pairs, edge_set_of,
    sample_unconnected_pairs, hop_distances, replace_unreachable,
    augmented_csr, source_block_size)

# The link prediction test, as pure functions. `link_prediction` does the
# steps together; the others are there for a caller that wants one of them
# alone.
from fodined.link_prediction import (
    link_prediction, sample_positives, sample_negatives, edge_features,
    classify)

# Everything else borrowed from the package is a plain function.
#   shell_force        -- THE force law (Fa + Fr), embedding/shell_force.py
#   shell_coeff_data   -- per-stored-pair 1/|S_h(u)| coefficient plane
#   degrees_from_D     -- force-law degree (count of hop-1 entries per row)
from fodined.embedding.shell_force import (
    shell_force, shell_coeff_data, degrees_from_D)
# SELL-C-sigma layout machinery: the host-side plan builder and the jitted
# per-epoch kernel. `_step` is "private" only in the sense that `PlanCache`
# normally owns it; importing it here is precisely what keeps this file on
# the SAME kernel as the package rather than a re-implementation of it
# (`PlanCache` itself is a class, so it is out of bounds for this file --
# and with a single, never-changing D there is nothing to cache anyway).
from fodined.embedding.sell_c_sigma import make_plan, _step
# The steady-rate random drop regularizer applied to dZ after the kernel.
from fodined.embedding.drop import drop_steady_rate

# `jax.default_backend()` reports what JAX really uses. Print it, because a
# request for 'gpu' and a run on the CPU must not look the same.
print(f"[modular] jax {jax.__version__}, DEVICE={DEVICE!r} -> "
      f"backend {jax.default_backend()}, devices: {jax.devices()}")

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
DATA = "/home/h/gnn/fd-graph-embedding/fdmap/data_cache"
EDGELIST = f"{DATA}/cora/cora.cites"

# `GRAPH` selects the dataset. Every branch below produces `raw`, which is an
# (m, 2) array of edges. The ids may be any integers, because the `np.unique`
# below maps them to 0..n-1. A tree gives the edges as (child, parent), and
# the truncation below needs that direction.
#
# `FODINED_GRAPH` in the environment overrides the value, thus one loop of
# the shell can run every graph without an edit of this file.
GRAPH = 'cora'      # 'cora' | 'pubmed' | 'wordnet' | 'ncbi_taxonomy'
                    # | 'com_youtube' | 'as_skitter' | 'roadnet_ca'
GRAPH = os.environ.get('FODINED_GRAPH', GRAPH)

# 0 keeps the whole graph. The two trees are far too large for the
# augmentation: NCBI has 2,937,016 nodes, and one BFS for each of them is
# many hours. See the truncation below.
MAX_NODES = 20_000
MAX_NODES = int(os.environ.get('FODINED_MAX_NODES', MAX_NODES))

# The three SNAP graphs have more than a million nodes each. Their files are
# `<src>\t<dst>` with `#` comment lines.
_SNAP = {'com_youtube': 'com_youtube/com-youtube.ungraph.txt',
         'as_skitter': 'as_skitter/as-skitter.txt',
         'roadnet_ca': 'roadnet_ca/roadNet-CA.txt'}

if GRAPH == 'cora':
    raw = np.loadtxt(EDGELIST, dtype=np.int64)               # (n_edges, 2)
elif GRAPH == 'pubmed':
    # `<edge_id>\tpaper:<src>\t|\tpaper:<dst>`, after two header lines.
    _src, _dst = [], []
    with open(f"{DATA}/pubmed/Pubmed-Diabetes/data/"
              f"Pubmed-Diabetes.DIRECTED.cites.tab") as _f:
        _f.readline(), _f.readline()
        for _line in _f:
            _p = _line.split("\t")
            _src.append(int(_p[1].split(":")[1]))
            _dst.append(int(_p[3].split(":")[1]))
    raw = np.column_stack([_src, _dst]).astype(np.int64)
elif GRAPH == 'wordnet':
    # The hypernym edges of the WordNet 3.0 nouns. Each line of `data.noun`
    # is one synset, and the pointer `@` (or `@i`) names its hypernym. The
    # byte offset of a synset is its id.
    _src, _dst = [], []
    with open(f"{DATA}/wordnet/dict/data.noun", encoding="latin-1") as _f:
        for _line in _f:
            if _line.startswith("  "):                       # licence header
                continue
            _parts = _line.partition("|")[0].split()
            _i = 3
            _i += 1 + 2 * int(_parts[_i], 16)                # skip the words
            _p_cnt = int(_parts[_i])
            _i += 1
            for _ in range(_p_cnt):
                if _parts[_i] in ("@", "@i"):
                    _src.append(int(_parts[0]))
                    _dst.append(int(_parts[_i + 1]))
                _i += 4                                      # sym off pos st
    raw = np.column_stack([_src, _dst]).astype(np.int64)
elif GRAPH == 'ncbi_taxonomy':
    # `tax_id | parent_tax_id | ...`. Each node has exactly one parent, thus
    # this file is a true tree.
    _src, _dst = [], []
    with open(f"{DATA}/ncbi_taxonomy/nodes.dmp") as _f:
        for _line in _f:
            _p = _line.split("\t|\t", 2)
            _a, _b = int(_p[0]), int(_p[1])
            if _a != _b:                                     # root: own parent
                _src.append(_a)
                _dst.append(_b)
    raw = np.column_stack([_src, _dst]).astype(np.int64)
elif GRAPH in _SNAP:
    # `pandas.read_csv` and not `np.loadtxt`: the C parser of pandas reads
    # 11M lines in seconds, and `np.loadtxt` needs minutes.
    import pandas as _pd
    raw = _pd.read_csv(f"{DATA}/{_SNAP[GRAPH]}", sep="\t", comment="#",
                       header=None, names=["u", "v"],
                       dtype=np.int64).to_numpy()
else:
    raise ValueError(f"Unknown GRAPH {GRAPH!r}")
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
A.sort_indices()

# Keep ONE subtree of about MAX_NODES nodes. The walk starts at a random node
# and it climbs to the parent until the subtree is large enough.
#
# A subtree keeps the DEPTH of the tree. A BFS from the node of the highest
# degree does not: NCBI has nodes with tens of thousands of children, thus a
# BFS returns a star of depth 1, every pair without an edge is then exactly 2
# hops apart, and every method reaches a perfect score on it. See
# `experiments/other-ge/REPORT.md`.
#
# A subtree needs a tree. roadNet-CA is not one: the walk below reads the
# edge list as "parent -> child", thus on a road network it returns 500,000
# nodes with 504 edges between them, and every score is then 1.0000 on an
# empty problem. A graph that is not a tree therefore takes a BFS ball, which
# is connected and which keeps the edges of the part that it takes.
_TREE_GRAPH = ('wordnet', 'ncbi_taxonomy')
if MAX_NODES and n > MAX_NODES and GRAPH not in _TREE_GRAPH:
    _order = breadth_first_order(A, int(np.random.default_rng(42).integers(0, n)),
                                 directed=False, return_predecessors=False)
    _keep = np.sort(_order[:MAX_NODES])
elif MAX_NODES and n > MAX_NODES:
    _parent = np.full(n, -1, dtype=np.int64)
    _parent[edges[:, 0]] = edges[:, 1]          # the first parent wins (DAG)
    _o = np.argsort(edges[:, 1], kind="stable")
    _child_of, _p_sorted = edges[_o, 0], edges[_o, 1]
    _lo = np.searchsorted(_p_sorted, np.arange(n))
    _hi = np.searchsorted(_p_sorted, np.arange(n), side="right")

    def _subtree(root, cap):
        """Node ids below `root`, at most `cap` of them."""
        out, frontier = [root], [root]
        while frontier and len(out) < cap:
            nxt = []
            for _u in frontier:
                for _k in _child_of[_lo[_u]:_hi[_u]]:
                    if len(out) >= cap:
                        break
                    out.append(int(_k))
                    nxt.append(int(_k))
            frontier = nxt
        return out

    _node = int(np.random.default_rng(42).integers(0, n))
    for _ in range(200):                        # guard against a long climb
        _got = _subtree(_node, MAX_NODES + 1)
        if len(_got) >= MAX_NODES // 2 or _parent[_node] < 0:
            break
        _node = int(_parent[_node])
    _keep = np.sort(np.asarray(_got[:MAX_NODES]))

# The renumbering, for both ways of choosing `_keep`.
if MAX_NODES and n > MAX_NODES:
    # `A[_keep][:, _keep]` is the short form, and it fails here: scipy indexes
    # the minor axis through a dense map of the columns, and the count of the
    # result then passes the int32 that it uses -- "negative dimensions are
    # not allowed" on roadNet-CA. This filter reads the entries one time and
    # it renumbers them, thus it costs O(nnz) and it holds no dense array.
    _mask = np.zeros(n, dtype=bool)
    _mask[_keep] = True
    _new_id = np.full(n, -1, dtype=np.int64)
    _new_id[_keep] = np.arange(_keep.size)
    _c = A.tocoo()
    _sel = _mask[_c.row] & _mask[_c.col]
    A = sp.csr_matrix(
        (_c.data[_sel], (_new_id[_c.row[_sel]], _new_id[_c.col[_sel]])),
        shape=(_keep.size, _keep.size))
    A.eliminate_zeros()
    n = A.shape[0]
    print(f"[modular] {GRAPH}: truncated to one subtree of {n} nodes")

# Both directions of every original edge -- the "is this pair already
# connected?" test of the 'sampled_pairs' policy. Only that policy needs it,
# and the set costs about 1.8 GB at a million nodes, thus the build waits for
# the dispatch below. The k_hop policy and the link prediction both test
# against the sparse matrix instead.
edge_set = None

print(f"[modular] {GRAPH}: n={n} nodes, {A.nnz // 2} undirected edges, "
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
N_NEW_EDGES = 100_000 # this is the best value for cora dataset.
N_NEW_EDGES = int(n*np.log10(n))  # ~10k for cora, ~1.5M for pubmed
# If there is no path between the two nodes, set the weight to `n`
UNREACHABLE_W = n
print(f"[modular] New edges: {N_NEW_EDGES}, Unreachable weight: {UNREACHABLE_W}")

# `AUGMENT` selects the policy.
#
# 'k_hop' (the policy for a large graph). For each node it takes the ball of
#   K_HOP hops, and every pair in that ball keeps its true hop distance.
#   Then N_NEW_EDGES random pairs that are FURTHER than K_HOP hops get the
#   weight FAR_WEIGHT.
#
#   It scales because it makes no shortest-path search. The ball comes from
#   K sparse products, thus the cost follows the size of the RESULT. And a
#   pair that the ball does not hold is further than K hops by definition,
#   thus the far pairs need no search either.
#
# 'sampled_pairs' (the older policy). It samples pairs first and then it asks
#   for the hop distance of each one, which needs one BFS for each distinct
#   source. On com_youtube that is about 114 hours, and the Python sets of
#   the sampler need more than 3 GB. See
#   `experiments/large-graph-node2vec/REPORT.md`.
AUGMENT = 'k_hop'        # 'k_hop' | 'sampled_pairs'
K_HOP = 2
FAR_WEIGHT = 100.0

# The value that means "this is not a path length". Each policy has its own:
# 'k_hop' writes FAR_WEIGHT for a far pair, and 'sampled_pairs' writes
# UNREACHABLE_W for a pair in another component. The hop regression below
# must remove it, or the model learns that sentinel and not a distance.
SENTINEL_W = FAR_WEIGHT if AUGMENT == 'k_hop' else UNREACHABLE_W
print(f"[modular] augmentation: {AUGMENT}"
      + (f", k={K_HOP}, far weight={FAR_WEIGHT}" if AUGMENT == 'k_hop' else ""))

rng = np.random.default_rng(42)

# Rejection sampling: on a graph this sparse (avg degree ~4 out of 2708
# nodes) a uniformly random pair is almost always unconnected, so this loop
# rejects very rarely. `taken` keeps a pair from being drawn twice, which
# would otherwise create a duplicate entry whose weights SUM in the CSR.
#
# Hop distance for the sampled pairs: one unweighted BFS per DISTINCT source.
# `shortest_path` always returns a DENSE ndarray of shape (len(srcs), n).
# The `indices` argument decreases the number of rows, but it does not make
# the result sparse: the value of an unreachable pair is `inf`, thus there is
# no zero to omit. If the sample touches all the nodes, this array is the
# full (n, n) matrix.
#
# The blocks are the reason this scales. One call for every source gives a
# dense (len(srcs), n) float64 array: at n = 20,000 with 20,000 sources that
# is 3.2 GB, and this machine has about 3 GB free. A block of CHUNK sources
# needs only CHUNK * n * 8 bytes, and the loop keeps only the cells that the
# sample asks for. The rule holds one block near 200 MB at any size.
# `experiments/bench_shortest_path_gemsec.py` measured that the blocks cost
# no extra time.
#
# The three steps now live in `graph_augmentation.py`. `augment` gets the
# SAME `rng` that the link prediction uses later, thus the order of the draws
# does not change and the result of a run does not change either.
t_aug = time.time()
if AUGMENT == 'k_hop':
    D, aug = augment_k_hop(A, n, K_HOP, N_NEW_EDGES, FAR_WEIGHT, rng)
    print(f"[modular] {K_HOP}-hop ball: {aug['ball_nnz']} stored pairs "
          f"{aug['hop_counts']}, "
          f"{aug['ball_nnz'] / n:.1f} per node ({time.time() - t_aug:.1f}s)")
    if aug['far_pairs'] < N_NEW_EDGES:
        # The rejection stops early when the ball covers nearly every pair.
        print(f"[modular] WARNING: only {aug['far_pairs']} pairs are further "
              f"than {K_HOP} hops, and {N_NEW_EDGES} were asked for")
else:
    edge_set = edge_set_of(A)
    D, aug = augment(A, n, N_NEW_EDGES, UNREACHABLE_W, edge_set, rng)
    #print the actual memory usage of `hop` matrix
    print(f"[modular] hop blocks: {aug['sources']} sources in blocks of "
          f"{aug['chunk']}, one block {aug['chunk'] * n * 8 / 1e6:.1f} MB vs "
          f"{aug['sources'] * n * 8 / 1e9:.2f} GB for one dense call "
          f"({time.time() - t_aug:.1f}s)")
new_pairs = aug["pairs"]

# The augmented D: original edges at weight 1.0 plus the sampled pairs at
# their hop distance, both directions. This is exactly the contract the
# embedding engine works against -- a csr_matrix whose stored entry
# D[u, v] = h >= 1 is a hop distance, and an absent pair is one the
# augmentation policy simply never reached.
print(f"[modular] augmented: {A.nnz} -> {D.nnz} stored entries "
      f"({aug.get('far_pairs', N_NEW_EDGES)} far pairs, "
      f"{aug.get('unreachable', 0)} of them unreachable), "
      f"hop weights {D.data.min():.0f}..{D.data.max():.0f}")

#---------------------------------------
# Perform graph embedding using SELL-C-sigma
#---------------------------------------
# Use the algorithm implemented in `fodined` module to perform
# graph embedding using JAX.
#
# - Instantiate an object or inherit a class from `Fodined`, depending
#   on how you plan to implement the `updateGradient` and/or `forces` methods.
# - Use the same hyperparameters and kernel as in `fodined`
#   module.
# - Set `n_dim=128` to get a 128-dimensional embedding.
# - The embedding will be performed for 2000 epochs, using GPU with JAX.
N_DIM = 128
EPOCHS = 2000
SEED = 42

# Verbatim from embedding.ShellForce's constructor defaults -- the "same
# hyperparameters" the header asks for. (The origin package's
# models.ReferenceFDModel passes the same values; models.py is not part of
# this directory, because modular.py does not use it.) k3 is the
# literal 10.0, NOT None (None means "auto: use the node count", a far
# larger repulsion coefficient; see ShellForce's docstring).
K1, K2, K3, K4 = 0.999, 1.0, 10.0, 0.01
RANDOM_DROP_RATE = 0.5
RANDOM_DROP_STRATEGY = 'random_rows' # 'random_rows' | 'random_cells'
B_CELLS, K_MAX, LADDER_BASE = 16_384, 256, 1.5
# ShellForce._h_shift(): hop distance 1 is the closest a stored pair can be.
# (WeightedShellForce's h_min baseline is for Euclidean edge weights; D here
# is a hop-count matrix, so the hop-count physics is the right one.)
H_SHIFT = 1.0


print(f"[modular] n_dim={N_DIM}, epochs={EPOCHS}")
print(f"[modular] K1, K2, K3, K4=[{K1, K2, K3, K4}]")
print(f"[modular] random_drop_rate={RANDOM_DROP_RATE}, drop strategy:{RANDOM_DROP_STRATEGY}")


class SellCSigmaFD(Fodined):
    """`Fodined` with the two stage hooks this script needs.

    `make_graph` is deliberately left unimplemented (it raises, as in the
    base class): the graph is loaded from disk above, not built from raw
    feature vectors, so `embed(D)` is the entry point and stage 1 never
    runs.
    """
    # def __init__(self, n_dim = None, lr = 1, beta = 0, epsilon = None, verbosity = 2, seed = None, **kwargs):
    #     super().__init__(n_dim, lr, beta, epsilon, verbosity, seed, **kwargs)
        

    def augment_graph(self, G, **kwargs):
        """Stage 2. D is already augmented above, so this only derives what
        the engine needs from it -- the same work `PlanCache.derive` does on
        a cache miss, done once here because there is only ever one D."""
        self.D = G
        planes = (shell_coeff_data(self.D), self.D.data)   # 1/|S_h(u)| and h, per pair
        plan, inv_deg_ext, self.plan_stats = make_plan(
            self.D, planes, degrees=degrees_from_D(self.D),
            b_cells=B_CELLS, k_max=K_MAX, ladder_base=LADDER_BASE)
        self.plan = jax.tree_util.tree_map(jax.device_put, plan)
        self.inv_deg_ext = jax.device_put(inv_deg_ext)
        # k1..k4/h_shift are traced scalars, so sweeping them never recompiles.
        self.params = dict(k1=K1, k2=K2, k3=K3, k4=K4, h_shift=H_SHIFT)
        self.step = jax.jit(functools.partial(
            _step, n=self.D.shape[0], force_fn=shell_force))
        return self.D

    # def updateZ(self, lr: float | None = None) -> None:
    #     """Apply the assembled step ``self.dZ`` to ``self.Z``.

    #     ``Z = Z + lr * dZ`` -- no in-place ``+=``, ``jnp.ndarray`` is
    #     immutable. Momentum is not applied here (it lives in
    #     ``updateGradient``); this is a thin override seam for a custom
    #     integrator if a researcher wants one.
    #     """
    #     if lr is None:
    #         lr = self.lr
    #     self.Z = self.Z + lr * self.dZ

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
        return drop_steady_rate(full[row_start:row_end], key, 
                                RANDOM_DROP_RATE, strategy=RANDOM_DROP_STRATEGY)


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

# This is a SAMPLE of the pairs, and not a census of them. A graph of a
# million nodes has millions of edges, thus all of them give a feature matrix
# of gigabytes and a forest that trains for hours. A balanced sample of this
# size measures the same property at a constant cost.
MAX_LP_PAIRS = 50_000            # positives and negatives together

# The steps live in `link_prediction.py`. It gets the SAME `rng` as the
# augmentation, thus the order of the draws does not change and the result of
# a run does not change either.
scores, lp = link_prediction(Z, A, n, MAX_LP_PAIRS, rng, SEED)

print(f"[modular] link prediction on {lp['pairs']} pairs "
      f"({lp['positives']} of {lp['edges']} edges + {lp['negatives']} "
      f"non-edges), cap {MAX_LP_PAIRS}, "
      f"{lp['train']} train / {lp['test']} test")
for name, value in scores.items():
    print(f"  {name:<9s}: {value:.4f}")


# for each d, sample the nodes that are d-hops away.
# Get the mean and std of the the d-hops away pairs.
D = fdobj.D
unique, counts = np.unique(D.data, return_counts=True)
print(f"[modular] augmented hop distances: {dict(zip(unique.astype(int), counts))}")

# For each hop distance d, take a sample of the stored pairs (u, v) that have
# D[u, v] == d. Then measure the Euclidean distance ||Z[u] - Z[v]|| of the two
# node embeddings, and give the mean and the standard deviation.
#
# This is the test of the physics. The attractive force decreases with d, thus
# the mean must INCREASE with d if the embedding kept the structure of the
# graph. A flat column, or a standard deviation that is larger than the
# difference between two adjacent rows, shows that the embedding did not
# separate the shells.
SAMPLE_PER_D = 200

# The source node of each stored entry. `D.indptr` gives the start of each
# row, thus `np.repeat` expands the row lengths into one source id for each
# entry. D is symmetric, thus each undirected pair is stored two times; a
# sample of the entries is still a sample of the pairs.
u_of = np.repeat(np.arange(n), np.diff(D.indptr))
v_of = D.indices

# The loop uses the values in `unique`, and not `range(1, D.max())`. These are
# the hop distances that D really stores, thus the loop does not print empty
# rows for a d that no pair has.
print(f"[modular] embedding distance vs hop distance "
      f"(up to {SAMPLE_PER_D} pairs for each d):")
print(f"  {'d':>5s} {'pairs':>10s} {'sampled':>8s} {'mean':>9s} {'mean norm':>9s}  {'std':>9s}")
dist = []
# for i, d in enumerate(unique):
#     at_d = np.nonzero(D.data == d)[0]
#     take = rng.choice(at_d, size=min(SAMPLE_PER_D, at_d.size), replace=False)
#     dist.append(np.linalg.norm(Z[u_of[take]] - Z[v_of[take]], axis=1))
#     tag = "  <- unreachable sentinel" if d == UNREACHABLE_W else ""
#     print(f"  {int(d):>5d} {at_d.size:>10d} {take.size:>8d} "
#           f"{dist[i].mean():>9.3f} {dist[i].mean()/dist[0].mean():>9.3f} "
#           f"{dist[i].std():>9.3f}{tag}")


#---------------------------------------
# Shortest path distance approaximation using (1) random forest and (2) MLP
#---------------------------------------
# The table above shows that the mean embedding distance increases with the
# hop distance. This section asks the harder question: can a model READ the
# hop distance back out of the two node embeddings?
#
# The feature of a pair is the elementwise absolute difference
# |Z[u] - Z[v]|. It is symmetric in (u, v), as an undirected pair needs. The
# Euclidean distance of the table above is a function of this vector, thus
# the model gets all the information of that table, and more.
#
# Two groups of rows are removed before the split:
#  * The unreachable pairs, which have the weight `n`. That value is a
#    sentinel of the augmentation, and not a path length. A model that learns
#    it learns nothing about a path.
#  * One direction of each pair. D stores (u, v) and (v, u), and the feature
#    is symmetric, thus the two rows are IDENTICAL. If one row goes to the
#    train set, and the other goes to the test set, the test score is not
#    honest. `u_of < v_of` keeps one row for each pair.
sp_mask = (u_of < v_of) & (D.data != SENTINEL_W) & \
            (D.data > 1) # only get the non-neighbor pairs
# keep no more than 2000 samples
sp_idx = np.flatnonzero(sp_mask)               # bool to indices
k = min(2000, sp_idx.size)
if k < sp_idx.size:
    sp_idx = rng.choice(sp_idx, k, replace=False)   # shuffle + take k
sp_mask = np.zeros_like(sp_mask)               # indices back t bool
sp_mask[sp_idx] = True

# NOTE for the 'k_hop' policy: the ball holds the hop values 1..K_HOP only,
# thus this mask leaves the values 2..K_HOP. With K_HOP = 3 the target has
# TWO values, and the regression is then nearly a binary decision. The task
# is not broken; the policy simply does not produce a range of distances.
# Raise K_HOP for a wider range, and watch `ball_nnz` grow.
sp_y = D.data[sp_mask]
# sp_X = np.abs(Z[u_of[sp_mask]] - Z[v_of[sp_mask]])
# sp_X = (Z[u_of[sp_mask]] + Z[v_of[sp_mask]])/2
sp_X = np.linalg.norm(Z[u_of[sp_mask]] - Z[v_of[sp_mask]], axis=1)[:,None]


X_tr, X_te, y_tr, y_te = train_test_split(
    sp_X, sp_y, test_size=0.2, random_state=SEED)
print(f"[modular] hop-distance regression on {sp_X.shape[0]} unique pairs "
      f"(hops {sp_y.min():.0f}..{sp_y.max():.0f}), "
      f"{X_tr.shape[0]} train / {X_te.shape[0]} test")

# The MLP uses gradient descent, thus it needs the features on a common
# scale. The random forest splits one feature at a time, thus the scale does
# not change its result, and it uses the raw features.
scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

# `exact` is the fraction of pairs where the rounded prediction is the true
# hop count. A hop distance is an integer, thus this is easier to read than
# the MAE. The baseline gives the mean of the train set to every pair: a
# model that cannot do better than the baseline has learned nothing.
def report(name, y_pred, seconds=None):
    print(f"  {name:>14s} "
          f"{mean_absolute_error(y_te, y_pred):>8.3f} "
          f"{mean_absolute_percentage_error(y_te, y_pred):>8.3f} " # mean relative error
          f"{np.sqrt(mean_squared_error(y_te, y_pred)):>8.3f} "
          f"{r2_score(y_te, y_pred):>8.3f} "
          f"{np.mean(np.rint(y_pred) == y_te):>8.1%}"
          f"{'' if seconds is None else f'  ({seconds:.1f}s)'}")

print(f"  {'model':>14s} {'MAE':>8s} {'MRE':>8s} {'RMSE':>8s} {'R2':>8s} {'exact':>8s}")
report("mean baseline", np.full(y_te.shape, y_tr.mean()))

for name, model, xa, xb in [
    ("random forest", RandomForestRegressor(
        n_estimators=200, random_state=SEED, n_jobs=-1), X_tr, X_te),
    ("MLP", MLPRegressor(
        hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True,
        random_state=SEED), X_tr_s, X_te_s),
]:
    t0 = time.time()
    model.fit(xa, y_tr)
    report(name, model.predict(xb), time.time() - t0)