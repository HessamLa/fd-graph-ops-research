#!/bin/env python3
# fdwalk, in the modular style of `fodined/modular.py`.
#
# - No main function. The script runs line by line.
# - It keeps the configuration, the ORDER of the steps, and every message.
#   Every reusable function is imported.
# - Only the AUGMENTATION differs from `fodined/modular.py`: random walks
#   make the pairs, and a walk statistic makes the weight. The force law,
#   the layout, the engine and the epoch loop are the same objects.
# - The evaluation is the `evaluator` package. Nothing is measured here.
#
# Start it from the repo root (fdmap/):
#     .venv/bin/python fdwalk/modular.py
#     FDWALK_GRAPH=pubmed .venv/bin/python fdwalk/modular.py
#
# THE ORDER OF THE GENERATOR IS PART OF THE RESULT. ONE
# `np.random.default_rng(SEED)` goes to the augmentation, then to the link
# prediction, then to the hop sample, in that order. A second generator, or
# another order, gives other pairs and other numbers -- and the difference
# reads as a defect of the physics. This is why `rng=` is passed to every
# `evaluator` call instead of letting each task seed itself.
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                     # `walks`, `weights`
sys.path.insert(0, os.path.dirname(HERE))    # `fodined`, `evaluator`

import functools
import resource
import time

import numpy as np
import scipy.sparse as sp


def rss_mb():
    """Peak resident memory of this process, in MB. `CLAUDE.md` asks for
    it, and for the runtime of each stage, in every experiment."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

# The backend must be chosen before `import jax`: JAX starts its backend at
# the import, thus a change after it does not move the arrays.
DEVICE = 'gpu'                               # 'auto' | 'cpu' | 'gpu' | 'tpu'
DEVICE = os.environ.get('FDWALK_DEVICE', DEVICE)
if DEVICE != 'auto':
    os.environ['JAX_PLATFORMS'] = {'gpu': 'cuda'}.get(DEVICE, DEVICE)

import jax
import jax.numpy as jnp

# The engine, the force law and the layout come from `fodined` unchanged.
from fodined.core.fodined import Fodined
from fodined.embedding.shell_force import (
    shell_force, shell_coeff_data, degrees_from_D)
from fodined.embedding.sell_c_sigma import make_plan, _step
from fodined.embedding.drop import drop_steady_rate
from fodined.graph_augmentation import sample_far_pairs, degree_table

# The augmentation. Both files are VERBATIM copies of
# `experiments/fdwalk/`, thus `diff` is the parity argument and this
# script cannot drift from the harness that made the recorded numbers.
# `experiments/fdwalk/` itself HAS drifted since G2 was recorded; see
# `PAIR_BLOCK` below.
import walks as W
import weights as WT

# The measurement. `evaluator` owns every score and this file owns none.
import evaluator as ev

print(f"[fdwalk] jax {jax.__version__}, DEVICE={DEVICE!r} -> "
      f"backend {jax.default_backend()}, devices: {jax.devices()}")

# ---------------------------------------
# Load graph
# ---------------------------------------
# `evaluator.load_graph` reads the registry that `fodiwalk.make_graph`
# owns, thus the script and the measurement hold ONE graph object and the
# graph is read one time.
GRAPH = 'cora'          # 'cora' | 'pubmed' | 'com_youtube' | 'roadnet_ca' | ...
GRAPH = os.environ.get('FDWALK_GRAPH', GRAPH)
MAX_NODES = 0           # 0 keeps the whole graph
MAX_NODES = int(os.environ.get('FDWALK_MAX_NODES', MAX_NODES))
SEED = int(os.environ.get('FDWALK_SEED', 42))

t_load = time.time()
A, n, graph_info = ev.load_graph(GRAPH, MAX_NODES, SEED)
print(f"[fdwalk] {GRAPH}: n={n} nodes, {A.nnz // 2} undirected edges, "
      f"avg degree {A.nnz / n:.2f}, max degree {int(np.diff(A.indptr).max())} "
      f"({time.time() - t_load:.1f}s)")

# ---------------------------------------
# Perform graph augmentation and prepare the augmented CSR
# ---------------------------------------
# The k-hop ball of `fodined/modular.py` follows the degree of the hubs:
# 2.5 G pairs at k=2 on com_youtube, and no budget reaches that. A walk has
# a BUDGET -- `N_WALKS` walks of `WALK_LEN` steps from each node, and `CAP`
# pairs kept for each node -- thus the size of `D` is a number we choose.
#
# The three steps:
#   1. `walk_pair_stats` -- two nodes inside one WINDOW of one walk make a
#      pair, and it collects the smallest gap, the sum and the count.
#   2. `cap_per_node` -- at most `CAP` pairs for each node, the most
#      visited. A pair survives if EITHER endpoint keeps it.
#   3. `weights.RULES[WEIGHT]` -- a walk statistic becomes the weight `h`.
#      `min_gap` is the smallest step gap, an upper bound of the hop
#      distance that a walk of `t` steps proves.
#
# `h` is an INTEGER in `[1, WINDOW]`, and 1 means adjacency. A float weight
# gives every pair its own shell, `degrees_from_D` then returns 0 for every
# row, and the whole force vanishes with no error raised.
#
# Then `N_FAR` random pairs that `D` does not hold take `FAR_WEIGHT`. They
# carry the only long-range term the layout gets.
WEIGHT = 'min_gap'      # 'min_gap' | 'flat' | 'mean_gap' | 'pmi'
WEIGHT = os.environ.get('FDWALK_WEIGHT', WEIGHT)
N_WALKS, WALK_LEN, WINDOW, CAP = 10, 20, 5, 16
PRUNE_MAX, PRUNE_FACTOR = 4_000_000, 4

# `PAIR_BLOCK` is how many start nodes walk at one time, and it is NOT only
# a memory knob: `uniform_walks` takes one `rng.random(block)` for each
# STEP, thus the partition decides which numbers of the stream reach which
# walk. Two block sizes give two different walk sets from ONE seed.
#
# It matters only when `n * N_WALKS` passes it. Cora (27,080 starts) is one
# block at any value here and reproduces at all of them; PubMed (197,170)
# is not.
#
# 100,000 is the value the recorded G1 and G2 tables were made with, thus
# this script reproduces them. **`experiments/fdwalk/walks.py` holds 50,000
# today**, and at that value the harness no longer reproduces its own G2
# table: `dnnz` 654,190 against the recorded 653,866. The drift is in that
# file and not in this one. See the entry of 2026-08-24 in
# `experiments/fdwalk/FINDINGS.md`.
PAIR_BLOCK = 100_000
PAIR_BLOCK = int(os.environ.get('FDWALK_PAIR_BLOCK', PAIR_BLOCK))
FAR_WEIGHT, FAR_BIAS = 100.0, 0.0
N_FAR = int(n * np.log10(max(n, 10)))

print(f"[fdwalk] augmentation: walk/{WEIGHT}, {N_WALKS} walks x {WALK_LEN} "
      f"steps, window {WINDOW}, cap {CAP} for each node, "
      f"{N_FAR} far pairs at weight {FAR_WEIGHT:.0f}")

rng = np.random.default_rng(SEED)

t_aug = time.time()
stats = W.walk_pair_stats(A, n, N_WALKS, WALK_LEN, WINDOW, rng, cap=CAP,
                          block=PAIR_BLOCK, prune_max=PRUNE_MAX,
                          prune_factor=PRUNE_FACTOR)
raw_pairs, unique_pairs = int(stats["raw"]), int(stats["key"].size)

keep = W.cap_per_node(stats["key"], stats["cnt"], n, CAP)
stats = {k: (v[keep] if isinstance(v, np.ndarray) else v)
         for k, v in stats.items()}
h = WT.RULES[WEIGHT](stats, WINDOW, n)
near = W.to_csr(stats["key"], h, n)

# The far pairs. `sample_far_pairs` rejects every pair that `near` holds,
# thus it needs no search: a pair the walks did not store is far by
# construction. `degree_table` gives `None` at `FAR_BIAS = 0`, which is the
# uniform draw.
far = sample_far_pairs(n, N_FAR, near, rng, cum=degree_table(A, FAR_BIAS))
if far.shape[0]:
    fw = np.full(far.shape[0], FAR_WEIGHT)
    c = near.tocoo()
    D = sp.csr_matrix(
        (np.concatenate([c.data, fw, fw]),
         (np.concatenate([c.row, far[:, 0], far[:, 1]]),
          np.concatenate([c.col, far[:, 1], far[:, 0]]))), shape=(n, n))
else:
    D = near
t_aug = time.time() - t_aug

print(f"[fdwalk] pairs: {raw_pairs} raw -> {unique_pairs} unique -> "
      f"{stats['key'].size} after the cap ({stats['prunes']} prunes of the "
      f"accumulator)")
print(f"[fdwalk] augmented: {A.nnz} -> {D.nnz} stored entries "
      f"({far.shape[0]} far pairs), weights {D.data.min():.0f}.."
      f"{D.data.max():.0f} ({t_aug:.1f}s)")
_w, _c = np.unique(np.rint(D.data).astype(int), return_counts=True)
print(f"[fdwalk] weight histogram: {dict(zip(_w.tolist(), _c.tolist()))}")

# ---------------------------------------
# Perform graph embedding using SELL-C-sigma
# ---------------------------------------
# Identical to `fodined/modular.py`: the same force law, the same plan
# builder, the same jitted kernel, the same constants. Only `D` arrived by
# another route.
N_DIM = 128
EPOCHS = 2000
LR = 1.0

K1, K2, K3, K4 = 0.999, 1.0, 10.0, 0.01
RANDOM_DROP_RATE = 0.5
RANDOM_DROP_STRATEGY = 'random_rows'         # 'random_rows' | 'random_cells'
B_CELLS, K_MAX, LADDER_BASE = 16_384, 256, 1.5
H_SHIFT = 1.0

print(f"[fdwalk] embedding: n_dim={N_DIM}, epochs={EPOCHS}, lr={LR}, "
      f"K1..K4=[{K1, K2, K3, K4}], drop {RANDOM_DROP_RATE} "
      f"({RANDOM_DROP_STRATEGY})")


class SellCSigmaFD(Fodined):
    """`Fodined` with the two stage hooks this script needs.

    Copied from `fodined/modular.py`. `make_graph` stays unimplemented:
    the graph is loaded from disk above, thus `embed(D)` is the entry
    point and stage 1 never runs.
    """

    def augment_graph(self, G, **kwargs):
        """Stage 2. `D` is already augmented above, thus this only derives
        what the engine needs from it, one time."""
        self.D = G
        planes = (shell_coeff_data(self.D), self.D.data)   # 1/|S_h(u)| and h
        plan, inv_deg_ext, self.plan_stats = make_plan(
            self.D, planes, degrees=degrees_from_D(self.D),
            b_cells=B_CELLS, k_max=K_MAX, ladder_base=LADDER_BASE)
        self.plan = jax.tree_util.tree_map(jax.device_put, plan)
        self.inv_deg_ext = jax.device_put(inv_deg_ext)
        # k1..k4/h_shift are traced scalars, thus a sweep never recompiles.
        self.params = dict(k1=K1, k2=K2, k3=K3, k4=K4, h_shift=H_SHIFT)
        self.step = jax.jit(functools.partial(
            _step, n=self.D.shape[0], force_fn=shell_force))
        return self.D

    def forces(self, Z, D, row_start, row_end, key=None, **kwargs):
        """Stage 3. One fused pass over the padded plan, then the row slice.

        The kernel runs over the whole graph and the row range is sliced
        after it: the plan groups rows by WIDTH across the whole graph,
        thus a contiguous row range has no clean correspondence to a subset
        of the batches. At the default `batch_count=1` it costs nothing.
        """
        if key is None:
            self.key, key = jax.random.split(self.key)
        full = self.step(jnp.asarray(Z, dtype=jnp.float32),
                         self.plan, self.inv_deg_ext, self.params)
        return drop_steady_rate(full[row_start:row_end], key,
                                RANDOM_DROP_RATE, strategy=RANDOM_DROP_STRATEGY)


fdobj = SellCSigmaFD(n_dim=N_DIM, lr=LR, verbosity=0, seed=SEED)

t0 = time.time()
fdobj.embed(D, epochs=EPOCHS)
Z = fdobj.get_embeddings()                   # (n, N_DIM) numpy array
t_embed = time.time() - t0

print(f"[fdwalk] plan: {fdobj.plan_stats}")
print(f"[fdwalk] embedded {Z.shape} in {t_embed:.1f}s "
      f"({EPOCHS / t_embed:.1f} epochs/s), final ||dZ|| avg "
      f"{fdobj.Th(fdobj.dZ):.6f}, finite: {np.isfinite(Z).all()}, "
      f"peak RSS {rss_mb():.0f} MB")

# ---------------------------------------
# Evaluate the embedding
# ---------------------------------------
# `evaluator` holds every protocol of this repository under a NAME, and a
# record carries the name. Nothing is measured in this file.
#
#   `fodined`        link prediction: 50,000 pairs, a random forest of 200
#                    trees over the Hadamard product.
#   `fodiwalk_dist`  the hop distance from ONE feature, the embedding
#                    distance. This is the number that compares with
#                    node2vec.
#   `fodiwalk_vec`   the same sample, read from `N_DIM` features. A model
#                    with 128 features can win because it has more of them,
#                    thus the two rows must never share a table.
LP_PROTOCOL = 'fodined'
HOP_PROTOCOLS = {"distance": "fodiwalk_dist", "vector": "fodiwalk_vec"}

graph, emb = (A, n, graph_info), (Z, {"source": "<memory>"})

lp = ev.link_prediction(graph, emb, protocol=LP_PROTOCOL, seed=SEED, rng=rng)
print(f"[fdwalk] link prediction ({LP_PROTOCOL}) on {lp.sizes['pairs']} "
      f"pairs ({lp.sizes['positives']} of {lp.sizes['edges']} edges + "
      f"{lp.sizes['negatives']} non-edges), {lp.sizes['train']} train / "
      f"{lp.sizes['test']} test ({lp.seconds:.1f}s)")
for name, value in lp.scores.items():
    print(f"  {name:<9s}: {value:.4f}")

# ONE hop sample, read two ways. The generator state is restored before the
# second protocol, thus both features score the SAME pairs -- which is what
# the recorded runs do, and what makes the two rows comparable to each
# other.
hop_state = rng.bit_generator.state
hop = {}
for feature, protocol in HOP_PROTOCOLS.items():
    rng.bit_generator.state = hop_state
    hop[feature] = ev.dist_approx(graph, emb, protocol=protocol, seed=SEED,
                                  rng=rng)
    s = hop[feature].sizes
    print(f"[fdwalk] hop distances ({protocol}) for {s['n_pairs']} pairs, "
          f"hops {s['hop_min']:.0f}..{s['hop_max']:.0f}, backend "
          f"{s['backend']} ({hop[feature].seconds:.1f}s)")
    print(f"  {feature:>14s} {'MAE':>8s} {'MRE':>8s} {'RMSE':>8s} "
          f"{'R2':>8s} {'exact':>8s}")
    for model, sc in hop[feature].scores.items():
        print(f"  {model:>14s} {sc['mae']:8.3f} {sc['mre']:8.3f} "
              f"{sc['rmse']:8.3f} {sc['r2']:8.3f} {sc['exact'] * 100:7.1f}%")

# One machine-readable line, in the fields of
# `experiments/fdwalk/results/*.tsv`, thus a run diffs against the record.
print("[fdwalk] RESULT\t" + "\t".join(f"{k}={v}" for k, v in [
    ("graph", GRAPH), ("pairs", "walk"), ("weight", WEIGHT),
    ("optim", "plain"), ("seed", SEED), ("n", n), ("dnnz", D.nnz),
    ("t_aug", f"{t_aug:.1f}"), ("t_embed", f"{t_embed:.1f}"),
    ("dz", f"{float(fdobj.Th(fdobj.dZ)):.4f}"),
    ("acc", f"{lp.scores['accuracy']:.4f}"),
    ("f1_score", f"{lp.scores['f1_score']:.4f}"),
    ("auc", f"{lp.scores['auc']:.4f}"),
    ("r2_dist", f"{hop['distance'].scores['mlp']['r2']:.3f}"),
    ("mae_dist", f"{hop['distance'].scores['mlp']['mae']:.3f}"),
    ("r2_vec", f"{hop['vector'].scores['mlp']['r2']:.3f}"),
    ("mae_vec", f"{hop['vector'].scores['mlp']['mae']:.3f}"),
    ("rss", f"{rss_mb():.0f}"),
]))
