#!/bin/env python3
"""bench_fdwalk.py -- one variant of the fdwalk experiment, on one graph.

It reads PLAN.md: axis A is `--weight`, axis B is `--pairs`, and axis C is
`--optim`. It prints the same lines that `fodined/modular.py` prints, thus
the two are comparable line by line.

    .venv/bin/python experiments/fdwalk/bench_fdwalk.py \\
        --graph cora --pairs walk --weight min_gap --optim plain --seed 42

The embedding engine, the force law, and the link prediction come from the
`fodined` package. Only the augmentation and the update rule are new.
"""
from __future__ import annotations

import argparse
import functools
import os
import resource
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

# `sys.path[0]` is the directory of this script, thus `import fodined` fails
# without the root of the repository. `HERE/../..` is that root.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

# The backend must be chosen before `import jax`.
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--graph", default="cora")
_ap.add_argument("--max-nodes", type=int, default=0)
_ap.add_argument("--pairs", default="walk",
                 choices=("walk", "walk_edges", "ball", "nbr_walk"),
                 help="nbr_walk = the specification of 2026-08-17: row u "
                      "holds EVERY neighbour of u at h=1, plus every node "
                      "that a walk FROM u reached, at h = the first step. "
                      "No cap, no window, and D is not symmetric.")
_ap.add_argument("--edge-rule", default="both", choices=("both", "low_deg"),
                 help="both = every edge in both rows; low_deg = an edge "
                      "enters the row of u only when deg(v) >= deg(u), thus "
                      "it costs one entry and not two")
_ap.add_argument("--deg-source", default="auto", choices=("auto", "D", "A"),
                 help="what `degrees_from_D` should be. D = count the h=1 "
                      "entries of D (the package default). A = the true "
                      "degree of the graph. `auto` picks A for low_deg, "
                      "because a hub can then hold no h=1 entry and "
                      "inv_deg_ext would zero every force of its row.")
_ap.add_argument("--weight", default="min_gap",
                 choices=("flat", "min_gap", "mean_gap", "pmi"))
_ap.add_argument("--optim", default="plain",
                 choices=("plain", "momentum", "nesterov", "adam", "fa2"))
_ap.add_argument("--seed", type=int, default=42)
_ap.add_argument("--dim", type=int, default=128)
_ap.add_argument("--epochs", type=int, default=2000)
_ap.add_argument("--lr", type=float, default=1.0)
_ap.add_argument("--walks", type=int, default=10, help="walks from each node")
_ap.add_argument("--len", type=int, default=20, dest="walk_len")
_ap.add_argument("--window", type=int, default=5)
_ap.add_argument("--cap", type=int, default=16, help="pairs kept per node")
_ap.add_argument("--k", type=int, default=3, help="the ball, for --pairs ball")
_ap.add_argument("--far", type=int, default=0, help="0 = n*log10(n)")
_ap.add_argument("--prune-max", type=int, default=4_000_000,
                 help="cut the pair accumulator when it passes this")
_ap.add_argument("--prune-factor", type=int, default=4,
                 help="the prune keeps this many times `cap` for each node. "
                      "The accumulator is thus bounded at "
                      "`prune_factor * cap * n` pairs, and NOT at "
                      "`prune_max`. G3 died three times on that difference.")
_ap.add_argument("--far-weight", type=float, default=100.0)
_ap.add_argument("--far-bias", type=float, default=0.0,
                 help="draw the far pairs in proportion to deg^ALPHA. "
                      "0 = uniform (what fodined always did); 0.75 = the "
                      "negative-sample distribution of word2vec/node2vec")
_ap.add_argument("--bucket-total", type=int, default=0,
                 help="0 = n*log10(n), the size that the policy asks for")
_ap.add_argument("--landmarks", type=int, default=0,
                 help="0 = one constant for every far pair; N = N exact BFS")
_ap.add_argument("--far-max", type=int, default=32,
                 help="cap of the landmark distance, to hold shell_counts small")
_ap.add_argument("--far-scale", type=float, default=1.0,
                 help="multiply the landmark distance. It keeps the ORDER of "
                      "the far pairs and it moves them out of the attraction "
                      "band: Fa falls with exp(-k2*(h-1)), thus a weight near "
                      "4 still attracts and a weight near 40 does not")
_ap.add_argument("--chunks", type=int, default=1,
                 help="axis D1: split the plan into this many row chunks")
_ap.add_argument("--chunk-host", action="store_true",
                 help="keep the chunk plans in the host memory, and move "
                      "one to the device for each use")
_ap.add_argument("--k1", type=float, default=0.999, help="attraction gain")
_ap.add_argument("--k3", type=float, default=10.0, help="repulsion gain (v1/v2)")
_ap.add_argument("--k4", type=float, default=0.01,
                 help="decay of the repulsion with the distance")
_ap.add_argument("--kr", type=float, default=1.0,
                 help="fdlinear: the repulsion constant of an h=1 pair")
_ap.add_argument("--fdlinear-sign", type=float, default=-1.0,
                 help="fdlinear: -1 gives exp(-k4*x), which decays; "
                      "+1 gives the literal exp(+k4*x), which diverges")
_ap.add_argument("--freq-mode", default="pair", choices=("pair", "node"),
                 help="fdlinear: `freq` is the count of the PAIR (how often "
                      "the walks of row u reached v) or of the NODE (how "
                      "often v appeared in any walk)")
_ap.add_argument("--no-deg-norm", action="store_true",
                 help="pass degrees=1 to make_plan, thus the engine does NOT "
                      "divide the row sum by deg1(u). fdlinear needs this to "
                      "be the law that its specification writes.")
_ap.add_argument("--fuse-planes", action="store_true",
                 help="fdlinear: carry ONE plane w = h/freq (-1 at h=1) "
                      "instead of the pair. Same law, ~20%% less memory; "
                      "h/freq becomes build-time. See force_fdlinear.py.")
_ap.add_argument("--force", default="v1", choices=("v1", "v2", "fdlinear"),
                 help="v1 = the law of the package; v2 = the off-branch law, "
                      "with a constant repulsion for h > 1")
_ap.add_argument("--policy", default="cap", choices=("cap", "buckets"),
                 help="cap = m pairs for each node, plus far pairs; "
                      "buckets = every edge, plus n*log10(n) pairs at "
                      "50%% h=2, 25%% h=3, 25%% h>=4, and NO far pairs")
_ap.add_argument("--cache-d", default="",
                 help="a .npz path. The augmentation writes it, and a later "
                      "run reads it. The augmentation of com_youtube needs "
                      "10 minutes, thus a retry of the embedding alone must "
                      "not pay it again.")
_ap.add_argument("--device", default="gpu")
_ap.add_argument("--lp-pairs", type=int, default=50_000)
_ap.add_argument("--hop-sources", type=int, default=200)
_ap.add_argument("--hop-pairs", type=int, default=20_000)
_ap.add_argument("--hop-min", type=int, default=2,
                 help="drop the pairs nearer than this, as ../other-ge does")
args = _ap.parse_args()

if args.device != "auto":
    os.environ["JAX_PLATFORMS"] = {"gpu": "cuda"}.get(args.device, args.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             mean_absolute_percentage_error)

from fodined.core.fodined import Fodined
from fodined.embedding.shell_force import (
    shell_force, shell_coeff_data, degrees_from_D)
from fodined.embedding.sell_c_sigma import make_plan, _step
from fodined.embedding.drop import drop_steady_rate
from fodined.graph_augmentation import (k_hop_ball, sample_far_pairs,
                                        degree_table)
from fodined.link_prediction import link_prediction

import datasets
import optim as optimizers
import buckets as BK
import force_fdlinear
import force_offbranch
import landmarks as LM
import walks as W
import weights as WT

# The force constants of `fodined/modular.py`. They do not change here: this
# experiment moves the augmentation, thus the physics must stay fixed.
FORCE_FN = {"v1": shell_force, "v2": force_offbranch.shell_force_v2,
            "fdlinear": (force_fdlinear.fdlinear_fused if args.fuse_planes
                         else force_fdlinear.fdlinear)}[args.force]

K1, K2, K3, K4 = args.k1, 1.0, args.k3, args.k4
RANDOM_DROP_RATE = 0.5
RANDOM_DROP_STRATEGY = "random_rows"
B_CELLS, K_MAX, LADDER_BASE = 16_384, 256, 1.5
H_SHIFT = 1.0

TAG = (f"{args.graph}/{args.pairs}/{args.weight}/{args.optim}"
       f"/seed{args.seed}")


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def log(msg):
    print(f"[fdwalk] {msg}", flush=True)


# ---------------------------------------------------------------------------
# The augmentation
# ---------------------------------------------------------------------------
def ball_stats(A, n, k, window):
    """The k-hop ball, in the same form as the statistics of a walk.

    `cnt` becomes `window + 1 - hop`, thus the cap keeps the NEAREST pairs,
    which is the best that a ball can do. This is the control of H4: the
    same budget `M`, and a different way to choose the pairs.
    """
    ball = k_hop_ball(A, k)
    c = sp.triu(ball, k=1).tocoo()
    key = c.row.astype(np.int64) * n + c.col.astype(np.int64)
    hop = c.data.astype(np.int32)
    return {"key": key, "mn": hop, "sm": hop.astype(np.int64),
            "cnt": np.maximum(window + 1 - hop, 1).astype(np.int64),
            "raw": int(ball.nnz)}


def take(stats, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else v)
            for k, v in stats.items()}


def build_D(A, n, rng):
    """The augmented matrix, and the numbers that the log needs."""
    info = {}
    t0 = time.time()

    if args.pairs == "nbr_walk":
        # 2026-08-18: `nbr_walk` returns before the `--policy buckets`
        # block below, thus a run that asked for `buckets` would get `cap`
        # behaviour AND would record `policy=buckets` in its RESULT line.
        # A mislabelled result is worse than a refused run, thus this stops.
        if args.policy == "buckets":
            _ap.error("--policy buckets is not implemented for --pairs "
                      "nbr_walk: the nbr_walk path builds rows directly and "
                      "never reaches the bucket sampler. Use --pairs walk or "
                      "walk_edges, or implement the row-wise bucket rule.")
        # The specification of 2026-08-17. Row u = every neighbour of u,
        # plus every node that a walk from u reached.
        st = W.walk_rows(A, n, args.walks, args.walk_len, rng)
        st = (W.with_neighbours_low_deg(st, A, n) if args.edge_rule == "low_deg"
              else W.with_all_neighbours(st, A, n))
        info["prunes"] = 0
        info["raw_pairs"] = int(st["raw"])
        info["unique_pairs"] = int(st["key"].size)
        info["capped_pairs"] = int(st["key"].size)
        info["t_pairs"] = time.time() - t0
        h = st["mn"].astype(np.float64)
        near = W.to_csr_directed(st["key"], h, n)
        info["near_nnz"] = int(near.nnz)
        info["h1_entries"] = int((near.data == 1).sum())
        info["edges_of_A"] = int(A.nnz)
        # `freq`, aligned to the SAME sparsity, thus the two `.data` arrays
        # match entry by entry after the CSR build.
        if args.freq_mode == "node":
            visits = np.bincount(st["key"] % n, weights=st["cnt"],
                                 minlength=n)
            fq = visits[st["key"] % n]
        else:
            fq = st["cnt"].astype(np.float64)
        freq = W.to_csr_directed(st["key"], fq.astype(np.float64), n)
        info["far_pairs"] = 0
        info["far_asked"] = 0
        if args.far > 0:
            # The test of 2026-08-17: is the low R2 of nbr_walk the missing
            # long-range term? Add far pairs at h = far_weight, in BOTH
            # directions, and the freq of a far pair is 1.
            cum = degree_table(A, args.far_bias)
            far = sample_far_pairs(n, args.far, near, rng, cum=cum)
            # `sample_far_pairs` rejects on the DIRECTED key of `near`, and
            # `near` is directed here, thus a pair stored as (v, u) does not
            # reject (u, v). The CSR build then SUMS the two, and the weight
            # becomes 100 + the walk gap. The histogram showed entries at
            # 101..119. Drop any far pair that `near` already holds, in
            # either direction.
            if far.shape[0]:
                nk = near.tocoo()
                have = np.sort(nk.row.astype(np.int64) * n
                               + nk.col.astype(np.int64))

                def _absent(k):
                    pos = np.searchsorted(have, k)
                    pos[pos >= have.size] = 0
                    return have[pos] != k

                k1 = far[:, 0].astype(np.int64) * n + far[:, 1]
                k2 = far[:, 1].astype(np.int64) * n + far[:, 0]
                far = far[_absent(k1) & _absent(k2)]
                del nk, have
            if far.shape[0]:
                c = near.tocoo(); fc = freq.tocoo()
                fw = np.full(2 * far.shape[0], args.far_weight)
                rr = np.concatenate([c.row, far[:, 0], far[:, 1]])
                cc = np.concatenate([c.col, far[:, 1], far[:, 0]])
                near = sp.csr_matrix((np.concatenate([c.data, fw]), (rr, cc)),
                                     shape=(n, n))
                freq = sp.csr_matrix(
                    (np.concatenate([fc.data, np.ones(2 * far.shape[0])]),
                     (rr, cc)), shape=(n, n))
                info["far_pairs"] = int(far.shape[0])
                info["far_asked"] = int(args.far)
                info["near_nnz"] = int(near.nnz)
        info["t_aug"] = time.time() - t0
        return near, dict(st, freq=freq), info

    if args.pairs == "ball":
        stats = ball_stats(A, n, args.k, args.window)
    else:
        stats = W.walk_pair_stats(A, n, args.walks, args.walk_len,
                                  args.window, rng, cap=args.cap,
                                  prune_max=args.prune_max,
                                  prune_factor=args.prune_factor)
    info["prunes"] = int(stats.get("prunes", 0))
    info["raw_pairs"] = int(stats["raw"])
    info["unique_pairs"] = int(stats["key"].size)
    info["t_pairs"] = time.time() - t0

    keep = W.cap_per_node(stats["key"], stats["cnt"], n, args.cap)
    stats = take(stats, keep)
    info["capped_pairs"] = int(stats["key"].size)

    h = WT.RULES[args.weight](stats, args.window, n)

    if args.policy == "buckets":
        # Every ORIGINAL edge stays at h = 1, and the walk pairs at h = 1
        # would repeat them, thus only the pairs at h >= 2 are candidates.
        far2 = h >= 2
        idx, info["buckets"] = BK.bucket_sample(
            h[far2], args.bucket_total or BK.budget(n), rng)
        sel = np.flatnonzero(far2)[idx]
        stats = take(stats, sel)
        h = h[sel]
        e = sp.triu(A, k=1).tocoo()
        ekey = e.row.astype(np.int64) * n + e.col.astype(np.int64)
        stats["key"] = np.concatenate([stats["key"], ekey])
        stats["mn"] = np.concatenate([stats["mn"],
                                      np.ones(ekey.size, dtype=np.int32)])
        stats["sm"] = np.concatenate([stats["sm"],
                                      np.ones(ekey.size, dtype=np.int32)])
        stats["cnt"] = np.concatenate([stats["cnt"],
                                       np.ones(ekey.size, dtype=np.int32)])
        h = np.concatenate([h, np.ones(ekey.size)])
        info["edges_added"] = int(ekey.size)
        info["near_nnz"] = 0
        near = W.to_csr(stats["key"], h, n)
        info["near_nnz"] = int(near.nnz)
        info["far_pairs"] = 0
        info["far_asked"] = 0
        info["t_aug"] = time.time() - t0
        return near, stats, info

    if args.pairs == "walk_edges":
        # The first-order edges always go in, at the distance 1. A walk can
        # miss the edge of a low-degree node, and that edge is the pair that
        # we trust most [7].
        # EVERY array of `stats` grows, and not only the key. The H2 check
        # below builds a matrix from `key` and `mn` together, thus a key
        # without its statistic stops the run with "all index and data
        # arrays must have the same length".
        e = sp.triu(A, k=1).tocoo()
        ekey = e.row.astype(np.int64) * n + e.col.astype(np.int64)
        new = ~np.isin(ekey, stats["key"])
        add = int(new.sum())
        stats["key"] = np.concatenate([stats["key"], ekey[new]])
        stats["mn"] = np.concatenate([stats["mn"],
                                      np.ones(add, dtype=np.int32)])
        stats["sm"] = np.concatenate([stats["sm"],
                                      np.ones(add, dtype=np.int64)])
        stats["cnt"] = np.concatenate([stats["cnt"],
                                       np.ones(add, dtype=np.int64)])
        h = np.concatenate([h, np.ones(add)])
        info["edges_added"] = add

    near = W.to_csr(stats["key"], h, n)
    info["near_nnz"] = int(near.nnz)
    # `freq` for fdlinear, on the SAME sparsity as `near`, thus the two
    # `.data` arrays line up entry by entry after the CSR build.
    fq_near = W.to_csr(stats["key"], stats["cnt"].astype(np.float64), n)

    n_far = args.far or int(n * np.log10(max(n, 10)))
    cum = degree_table(A, args.far_bias)
    far = sample_far_pairs(n, n_far, near, rng, cum=cum)
    if cum is not None:
        deg = np.diff(A.indptr)
        info["far_mean_deg"] = float(deg[far.ravel()].mean()) if far.size else 0.0
    info["far_pairs"] = int(far.shape[0])
    info["far_asked"] = n_far
    freq_csr = fq_near
    if far.shape[0]:
        fc = fq_near.tocoo()
        freq_csr = sp.csr_matrix(
            (np.concatenate([fc.data, np.ones(2 * far.shape[0])]),
             (np.concatenate([fc.row, far[:, 0], far[:, 1]]),
              np.concatenate([fc.col, far[:, 1], far[:, 0]]))), shape=(n, n))
        if args.landmarks:
            # A real distance for the far pairs, and not one constant. See
            # `landmarks.py` for why the constant costs the R2.
            t1 = time.time()
            lm = LM.pick(A, n, args.landmarks, rng)
            table = LM.distances(A, n, lm)
            info["t_landmark"] = time.time() - t1
            info["landmark_mb"] = table.nbytes / 1e6
            fw = LM.pair_distance(table, far[:, 0], far[:, 1],
                                  args.far_max, args.far_weight / args.far_scale)
            fw = fw * args.far_scale
            info["far_unreached"] = int((fw == args.far_weight).sum())
            del table
        else:
            fw = np.full(far.shape[0], args.far_weight)
        c = near.tocoo()
        D = sp.csr_matrix(
            (np.concatenate([c.data, fw, fw]),
             (np.concatenate([c.row, far[:, 0], far[:, 1]]),
              np.concatenate([c.col, far[:, 1], far[:, 0]]))), shape=(n, n))
    else:
        D = near
    info["t_aug"] = time.time() - t0
    return D, dict(stats, freq=freq_csr), info


# ---------------------------------------------------------------------------
# The embedding
# ---------------------------------------------------------------------------
class FDWalk(Fodined):
    """`Fodined` with the plan of `modular.py` and a chosen update rule."""

    freq = None
    fused = None
    no_deg_norm = False
    deg_from_A = None

    def set_rule(self, name, lr, chunks=1, resident=True):
        self.rule = optimizers.RULES[name]
        self.opt_state = {}
        self.lr = lr
        self.n_chunks = max(1, chunks)
        self.resident = resident

    def augment_graph(self, G, **kwargs):
        """Stage 2. One plan, or `chunks` plans of a row range each.

        Axis D1. The whole plan of a graph of a million nodes does not fit
        beside `Z` and `dZ` on a 2 GB card. A chunk holds the rows
        `[a, b)` only, thus the device holds ONE chunk at a time.

        Why a row range is the correct cut, and not any set of pairs: the
        kernel writes `dZ.at[rows].add(...)`, and a row range gives
        DISJOINT rows to each chunk, thus nothing must be summed across the
        chunks. `Fodined.embed` already slices `dZ` by the same row range
        for its batches, thus the chunk and the batch are the same object
        and the engine needs no change.

        The three global quantities stay global, and they must:
        `shell_coeff_data` and `degrees_from_D` read the WHOLE `D`, and a
        chunk only slices the result. A degree that counts one chunk is not
        the degree of the node.
        """
        self.D = G
        n = self.D.shape[0]
        # `shell_coeff` goes to the plan ONLY for a law that reads it.
        # 2026-08-17: `fdlinear` never did, and building it cost an nnz
        # array, a padded tile, and the transients of `shell_counts`.
        if self.fused is not None:
            planes = (self.fused,)
        elif self.freq is not None:
            planes = (self.D.data, self.freq)
        else:
            planes = (shell_coeff_data(self.D), self.D.data)
        if self.no_deg_norm:
            degrees = np.ones(n, dtype=np.int64)
        elif self.deg_from_A is not None:
            # The true degree of the graph. `low_deg` can leave a hub with
            # no entry at h=1, and `degrees_from_D` would then give 0, which
            # `inv_deg_ext` turns into 0.0 and the row never moves.
            degrees = self.deg_from_A
        else:
            degrees = degrees_from_D(self.D)
        self.params = dict(k1=K1, k2=K2, k3=K3, k4=K4, h_shift=H_SHIFT,
                           kr=args.kr, sign=args.fdlinear_sign)

        self.chunk_rows = max(1, (n + self.n_chunks - 1) // self.n_chunks)
        self.plans, self.steps, cells = [], [], 0
        for a in range(0, n, self.chunk_rows):
            b = min(a + self.chunk_rows, n)
            lo, hi = int(self.D.indptr[a]), int(self.D.indptr[b])
            # The rows outside `[a, b)` become empty, and `make_plan` gives
            # an isolated row no virtual row at all, thus an empty row costs
            # nothing in the plan of another chunk.
            indptr = np.zeros(n + 1, dtype=self.D.indptr.dtype)
            indptr[a + 1:b + 1] = self.D.indptr[a + 1:b + 1] - lo
            indptr[b + 1:] = indptr[b]
            Dc = sp.csr_matrix(
                (self.D.data[lo:hi], self.D.indices[lo:hi], indptr),
                shape=self.D.shape)
            plan, inv_deg_ext, stats = make_plan(
                Dc, tuple(p[lo:hi] for p in planes), degrees=degrees,
                b_cells=B_CELLS, k_max=K_MAX, ladder_base=LADDER_BASE)
            cells += stats["cells"]
            # `resident` keeps the plan on the device, which is the fast
            # form and the one that needs the memory. Otherwise the plan
            # stays in the host memory and it moves for each use.
            self.plans.append(jax.tree_util.tree_map(jax.device_put, plan)
                              if self.resident else plan)
            self.steps.append(jax.jit(functools.partial(
                _step, n=n, force_fn=FORCE_FN)))
            self.plan_stats = stats
        self.plan_stats = dict(stats, cells=cells, chunks=len(self.plans),
                               rows_per_chunk=self.chunk_rows)
        self.inv_deg_ext = jax.device_put(inv_deg_ext)
        return self.D

    def forces(self, Z, D, row_start, row_end, key=None, **kwargs):
        if key is None:
            self.key, key = jax.random.split(self.key)
        i = min(row_start // self.chunk_rows, len(self.plans) - 1)
        plan = self.plans[i]
        if not self.resident:
            plan = jax.tree_util.tree_map(jax.device_put, plan)
        full = self.steps[i](jnp.asarray(Z, dtype=jnp.float32),
                             plan, self.inv_deg_ext, self.params)
        out = drop_steady_rate(full[row_start:row_end], key,
                               RANDOM_DROP_RATE, strategy=RANDOM_DROP_STRATEGY)
        del full, plan
        return out

    def updateZ(self, lr=None):
        """Axis C. The base class does `Z = Z + lr * dZ`; this dispatches."""
        lr = self.lr if lr is None else lr
        self.Z, self.opt_state = self.rule(
            self.Z, self.dZ, lr, self.opt_state, self.epoch)


# ---------------------------------------------------------------------------
# The evaluation
# ---------------------------------------------------------------------------
def hop_sample(A, n, rng, n_sources, n_pairs, gap_csr=None):
    """Exact hop distances for a sample of pairs. Returns `(u, v, d, h2)`.

    `shortest_path` gives a DENSE `(len(indices), n)` array, thus the
    sources go in blocks: one block costs `block * n * 8` bytes. A full
    matrix at 1.13M nodes is 10 TB, and one block near 200 MB is the rule
    that `../bench_shortest_path_gemsec.py` measured.

    `gap_csr` holds the walk gap of every stored pair. While a row of the
    BFS is in the memory, this function reads EVERY stored partner of that
    row and it compares the gap to the true distance. That is the
    measurement of H2, and it gives thousands of pairs. A comparison
    against the random sample above gives only the pairs that both sets
    hold, which was 140 of 17,769 in the first run.
    """
    src = rng.choice(n, size=min(n_sources, n), replace=False)
    per = max(1, n_pairs // src.size)
    block = max(1, int(200e6 / (n * 8)))
    us, vs, ds = [], [], []
    gaps, trues = [], []
    for s in range(0, src.size, block):
        blk = src[s:s + block]
        hop = shortest_path(A, method="D", unweighted=True, indices=blk)
        for i, u in enumerate(blk):
            tgt = rng.choice(n, size=min(per * 2, n), replace=False)
            d = hop[i, tgt]
            ok = np.isfinite(d) & (d > 0)
            tgt, d = tgt[ok][:per], d[ok][:per]
            us.append(np.full(tgt.size, u))
            vs.append(tgt)
            ds.append(d)
            if gap_csr is not None:
                lo, hi = gap_csr.indptr[u], gap_csr.indptr[u + 1]
                part = gap_csr.indices[lo:hi]
                g = gap_csr.data[lo:hi]
                t = hop[i, part]
                fin = np.isfinite(t)
                gaps.append(g[fin])
                trues.append(t[fin])
        del hop
    h2 = None
    if gaps and sum(x.size for x in gaps):
        g = np.concatenate(gaps)
        t = np.concatenate(trues)
        h2 = {"pairs": int(g.size), "exact": float(np.mean(g == t)),
              "over": float(np.mean(g > t)), "under": float(np.mean(g < t)),
              "mae": float(np.mean(np.abs(g - t)))}
    return (np.concatenate(us), np.concatenate(vs),
            np.concatenate(ds).astype(np.float64), h2)


def task_hop(Z, u, v, d, seed, feature="vector"):
    """Can a model read the hop distance out of two embeddings?

    Two feature sets, because this repository holds two protocols and a
    number must be comparable to the baseline that it claims to beat:

      'vector'   `|Z[u] - Z[v]|`, thus `n_dim` features. This is the
                 protocol of `fodined/modular.py`.
      'distance' the Euclidean distance alone, thus ONE feature. This is
                 the protocol of `../other-ge/bench_other_ge.py`, and the
                 node2vec and Poincaré numbers use it.

    A model with 128 features can win only because it has more of them, thus
    the two rows must not be mixed.
    """
    if feature == "distance":
        X = np.linalg.norm(Z[u] - Z[v], axis=1)[:, None]
    else:
        X = np.abs(Z[u] - Z[v])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, d, test_size=0.2, random_state=seed)
    rows = []

    def score(name, pred, secs=None):
        rows.append((name, mean_absolute_error(y_te, pred),
                     mean_absolute_percentage_error(y_te, pred),
                     np.sqrt(mean_squared_error(y_te, pred)),
                     r2_score(y_te, pred),
                     float(np.mean(np.rint(pred) == y_te)), secs))

    score("mean baseline", np.full(y_te.size, y_tr.mean()))
    t = time.time()
    rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    score("random forest", rf.predict(X_te), time.time() - t)
    t = time.time()
    sc = StandardScaler().fit(X_tr)
    mlp = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=300,
                       random_state=seed)
    mlp.fit(sc.transform(X_tr), y_tr)
    score("MLP", mlp.predict(sc.transform(X_te)), time.time() - t)
    return rows


# H2 is measured inside `hop_sample`, where a row of the BFS is already in
# the memory. The gap of a walk is an UPPER BOUND of the hop distance: a
# walk that goes from `u` to `v` in `t` steps proves a path of `t` steps.
# Thus "under" must be 0.0, and any other value is a defect of this code.


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
log(f"variant {TAG}")
log(f"jax {jax.__version__}, backend {jax.default_backend()}, "
    f"devices {jax.devices()}")

t_load = time.time()
A, n = datasets.load(args.graph, args.max_nodes, args.seed)
log(f"{args.graph}: n={n} nodes, {A.nnz // 2} undirected edges, "
    f"avg degree {A.nnz / n:.2f}, max degree {int(np.diff(A.indptr).max())} "
    f"({time.time() - t_load:.1f}s)")

rng = np.random.default_rng(args.seed)
if args.cache_d and os.path.exists(args.cache_d):
    z = np.load(args.cache_d, allow_pickle=True)
    D = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
    stats = {"key": z["key"], "mn": z["mn"]}
    info = z["info"].item()
    log(f"D read from {args.cache_d}, thus the augmentation did not run again")
else:
    D, stats, info = build_D(A, n, rng)
    if args.cache_d:
        np.savez(args.cache_d, data=D.data, indices=D.indices,
                 indptr=D.indptr, key=stats["key"], mn=stats["mn"],
                 info=np.array(info, dtype=object))
        log(f"D written to {args.cache_d}")

log(f"pairs: {info['raw_pairs']} raw -> {info['unique_pairs']} unique -> "
    f"{info['capped_pairs']} after the cap of {args.cap} for each node "
    f"({info['t_pairs']:.1f}s, {info['prunes']} prunes of the accumulator)")
if "edges_added" in info:
    log(f"first-order edges added: {info['edges_added']}")
log(f"D: {info['near_nnz']} near + {2 * info['far_pairs']} far entries "
    f"= {D.nnz}, {D.nnz / n:.1f} for each node, weights "
    f"{D.data.min():.2f}..{D.data.max():.2f} ({info['t_aug']:.1f}s, "
    f"RSS {rss_mb():.0f} MB)")
uw, cw = np.unique(np.rint(D.data).astype(int), return_counts=True)
if "far_mean_deg" in info:
    log(f"far pairs drawn at deg^{args.far_bias}: mean degree of an endpoint "
        f"{info['far_mean_deg']:.2f} (the graph mean is {A.nnz / n:.2f})")
if "h1_entries" in info:
    log(f"neighbours: {info['h1_entries']} entries at h=1, and A has "
        f"{info['edges_of_A']} directed edges "
        f"({100 * info['h1_entries'] / max(info['edges_of_A'], 1):.1f}%)")
log(f"weight histogram: {dict(zip(uw.tolist(), cw.tolist()))}")
if "buckets" in info:
    for b in info["buckets"]:
        log(f"  bucket {b['bucket']:>5s}: {b['available']} available, "
            f"{b['asked']} asked, {b['taken']} taken"
            + ("  SHORT" if b["taken"] < b["asked"] else ""))
if args.landmarks:
    log(f"landmarks: {args.landmarks} exact BFS, table {info['landmark_mb']:.0f} MB "
        f"({info['t_landmark']:.1f}s), {info['far_unreached']} far pairs "
        f"reached by no landmark keep the constant {args.far_weight:.0f}")
if info["far_pairs"] < info["far_asked"]:
    log(f"WARNING: only {info['far_pairs']} far pairs of "
        f"{info['far_asked']} asked for")

log(f"embedding: dim={args.dim}, epochs={args.epochs}, optim={args.optim} "
    f"({optimizers.STATE_ARRAYS[args.optim]} state arrays), lr={args.lr}, "
    f"chunks={args.chunks}"
    + (" on the host" if args.chunk_host else ""))
fd = FDWalk(n_dim=args.dim, verbosity=0, seed=args.seed)
fd.no_deg_norm = args.no_deg_norm
_use_A = args.deg_source == "A" or (args.deg_source == "auto"
                                    and args.edge_rule == "low_deg")
fd.deg_from_A = np.maximum(np.diff(A.indptr), 1).astype(np.int64) if _use_A else None
if _use_A:
    log("degrees for the force law come from A, and not from the h=1 count of D")
# The `freq` plane goes to the plan ONLY for a force law that reads it.
# `shell_force` of the package unpacks exactly two planes, thus a third one
# stops it with "too many values to unpack".
fd.freq = (stats["freq"].data
           if args.force == "fdlinear" and stats.get("freq") is not None
           else None)
if args.fuse_planes:
    if fd.freq is None:
        _ap.error("--fuse-planes needs --force fdlinear and a freq plane")
    # The degree MUST come from the intact `h`, thus it is read here,
    # before `h` and `freq` collapse into `w`. `degrees_from_D` counts the
    # h == 1 entries of a row, and `w` keeps only their SIGN.
    if fd.deg_from_A is None:
        fd.deg_from_A = degrees_from_D(D)
    fd.fused = force_fdlinear.fuse(D.data, fd.freq)
    fd.freq = None
    stats["freq"] = None                    # free the CSR before the plan
    log(f"fused plane: 1 plane of {fd.fused.nbytes / 1e6:.1f} MB replaces "
        f"h + freq + shell_coeff")
fd.set_rule(args.optim, args.lr, args.chunks, not args.chunk_host)
t0 = time.time()
fd.embed(D, epochs=args.epochs, batch_count=args.chunks)
Z = fd.get_embeddings()
t_embed = time.time() - t0
log(f"plan: {fd.plan_stats}")
log(f"embedded {Z.shape} in {t_embed:.1f}s ({args.epochs / t_embed:.1f} "
    f"epochs/s), final ||dZ|| avg {fd.Th(fd.dZ):.6f}, "
    f"finite: {bool(np.isfinite(Z).all())}, RSS {rss_mb():.0f} MB")

t0 = time.time()
scores, lp = link_prediction(Z, A, n, args.lp_pairs, rng, args.seed)
log(f"link prediction on {lp['pairs']} pairs ({lp['positives']} of "
    f"{lp['edges']} edges + {lp['negatives']} non-edges), "
    f"{lp['train']} train / {lp['test']} test ({time.time() - t0:.1f}s)")
for name, value in scores.items():
    print(f"  {name:<9s}: {value:.4f}")

t0 = time.time()
gap_csr = W.to_csr(stats["key"], stats["mn"].astype(np.float64), n)
u, v, d, h2 = hop_sample(A, n, rng, args.hop_sources, args.hop_pairs, gap_csr)
# The pairs at hop 1 go out. A neighbour is the easy case, and the baseline
# of `../other-ge/` removes it too, thus the numbers stay comparable.
keep = d >= args.hop_min
u, v, d = u[keep], v[keep], d[keep]
log(f"hop distances for {u.size} pairs from {args.hop_sources} sources "
    f"({time.time() - t0:.1f}s), hops {int(d.min())}..{int(d.max())}")
best = {}
for feature in ("distance", "vector"):
    print(f"  {feature:>14s} {'MAE':>8s} {'MRE':>8s} {'RMSE':>8s} {'R2':>8s} "
          f"{'exact':>8s}")
    for name, mae, mre, rmse, r2, ex, secs in task_hop(
            Z, u, v, d, args.seed, feature):
        print(f"  {name:>14s} {mae:8.3f} {mre:8.3f} {rmse:8.3f} {r2:8.3f} "
              f"{ex * 100:7.1f}%" + (f"  ({secs:.1f}s)" if secs else ""))
        if name == "MLP":
            best[feature] = (r2, mae)

if h2:
    log(f"H2, the walk gap against the true hop distance on {h2['pairs']} "
        f"stored pairs: exact {h2['exact'] * 100:.1f}%, "
        f"over {h2['over'] * 100:.1f}%, under {h2['under'] * 100:.1f}%, "
        f"MAE {h2['mae']:.3f}")

log(f"TOTAL peak RSS {rss_mb():.0f} MB, variant {TAG}")

# One machine-readable line for each run. `run_g1.sh` collects them into a
# table, thus no number is copied by hand.
print("[fdwalk] RESULT\t" + "\t".join(f"{k}={v}" for k, v in [
    ("graph", args.graph), ("pairs", args.pairs), ("weight", args.weight),
    ("optim", args.optim), ("seed", args.seed), ("n", n),
    ("dnnz", D.nnz), ("t_aug", f"{info['t_aug']:.1f}"),
    ("t_embed", f"{t_embed:.1f}"), ("dz", f"{float(fd.Th(fd.dZ)):.4f}"),
    ("acc", f"{scores['accuracy']:.4f}"), ("f1", f"{scores['f1-score']:.4f}"),
    ("auc", f"{scores['auc']:.4f}"),
    ("r2_dist", f"{best['distance'][0]:.3f}"),
    ("mae_dist", f"{best['distance'][1]:.3f}"),
    ("r2_vec", f"{best['vector'][0]:.3f}"),
    ("mae_vec", f"{best['vector'][1]:.3f}"),
    ("h2_exact", f"{h2['exact']:.3f}" if h2 else "na"),
    ("h2_pairs", h2["pairs"] if h2 else 0),
    ("lm", args.landmarks),
    ("force", args.force),
    ("far_bias", args.far_bias),
    ("kr", args.kr),
    ("degnorm", 0 if args.no_deg_norm else 1),
    ("edge_rule", args.edge_rule),
    ("policy", args.policy),
    ("rss", f"{rss_mb():.0f}")]), flush=True)
