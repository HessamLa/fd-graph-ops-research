#!/bin/env python3
"""bench_fodiwalk.py -- the baseline of the `fodiwalk` package, on one graph.

The baseline is two decisions, and `verify_baseline.py` proves both:

    the walk   the uniform walk of DeepWalk, which IS node2vec at p = q = 1.
               `make_walker(A, n, 1, 1)` gives `uniform_walks` itself, and
               that function is bit-identical to the one the node2vec
               baseline of `experiments/other-ge/` drives.
    the law    `fdlinear`, which reads the planes `(h, freq)`.

It reports, for each stage, the wall time and the peak resident memory, and
then the three measurements: link prediction (accuracy, F1, AUC), the hop
distance regression (R2 on the distance feature), and the final ||dZ||.

    .venv/bin/python experiments/fodiwalk/bench_fodiwalk.py --graph cora

The structure follows `experiments/fdwalk/bench_fdwalk.py`, and the RESULT
line keeps that file's `key=value` shape so one collector reads both. What
is different: the augmentation, the plan and the embedding are ONE object
here, and not 400 lines of script.
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--graph", default="cora")
ap.add_argument("--max-nodes", type=int, default=0)
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--epochs", type=int, default=2000)
ap.add_argument("--lr", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--optim", default="plain")
# the walk. The defaults are the package defaults, thus this IS the baseline.
ap.add_argument("--pairs", default="walk",
                choices=["walk", "walk_edges", "nbr_walk"],
                help="every policy is walk-based; h is the walk gap")
ap.add_argument("--weight", default="min_gap",
                choices=["flat", "min_gap", "mean_gap", "pmi"])
ap.add_argument("--far", type=int, default=0,
                help="long-range pairs; 0 = n*log10(n)")
ap.add_argument("--walks", type=int, default=10, help="walks from each node")
ap.add_argument("--len", type=int, default=20, dest="walk_len")
ap.add_argument("--window", type=int, default=5)
ap.add_argument("--cap", type=int, default=16, help="pairs kept for each node")
ap.add_argument("--p", type=float, default=1.0, help="1.0 = the uniform walk")
ap.add_argument("--q", type=float, default=1.0)
# the law
ap.add_argument("--k1", type=float, default=0.999, help="attraction gain")
ap.add_argument("--k4", type=float, default=0.01, help="decay of the repulsion")
ap.add_argument("--kr", type=float, default=1.0, help="repulsion at h = 1")
ap.add_argument("--far-bias", type=float, default=0.0)
ap.add_argument("--far-weight", type=float, default=100.0)
ap.add_argument("--chunks", type=int, default=1)
ap.add_argument("--chunk-host", action="store_true")
ap.add_argument("--device", default="gpu")
# the evaluation
ap.add_argument("--lp-pairs", type=int, default=50_000)
ap.add_argument("--hop-sources", type=int, default=200)
ap.add_argument("--hop-pairs", type=int, default=20_000)
ap.add_argument("--hop-min", type=int, default=2,
                help="drop the pairs nearer than this, as ../other-ge does")
ap.add_argument("--out", default="", help="append the RESULT line to this file")
args = ap.parse_args()

# The backend must be chosen before `import jax`, thus before `fodiwalk`.
if args.device != "auto":
    os.environ["JAX_PLATFORMS"] = {"gpu": "cuda"}.get(args.device, args.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import numpy as np

from fodiwalk import Fodiwalk
from forcedirected import Callback_Base
from fodiwalk.embed.forces import planes_of
from fodiwalk.augment_graph import pairs as PR
from fodiwalk.make_graph import load
from forcedirected import optim as optimizers
from fodiwalk.misc.evaluation import link_prediction, hop_sample, task_hop


def rss_mb():
    """Peak resident memory of this process, in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def log(msg):
    print(f"[fodiwalk] {msg}", flush=True)


TAG = f"{args.graph}/{args.pairs}/fdlinear/{args.optim}/seed{args.seed}"
log(f"baseline {TAG}")

import jax                                   # after the backend is chosen
log(f"jax {jax.__version__}, backend {jax.default_backend()}, "
    f"devices {jax.devices()}")

# ---------------------------------------------------------------------------
# 1. the graph
# ---------------------------------------------------------------------------
t0 = time.time()
A, n = load(args.graph, args.max_nodes, args.seed)
t_load = time.time() - t0
log(f"{args.graph}: n={n} nodes, {A.nnz // 2} undirected edges, avg degree "
    f"{A.nnz / n:.2f}, max degree {int(np.diff(A.indptr).max())} "
    f"({t_load:.1f}s, RSS {rss_mb():.0f} MB)")

# ---------------------------------------------------------------------------
# 2. the model. Every knob below is a `Config` field.
# ---------------------------------------------------------------------------
fw = Fodiwalk(n_dim=args.dim, lr=args.lr, seed=args.seed, verbosity=0,
              optim=args.optim,
              pairs=args.pairs, weight=args.weight, force="fdlinear",
              walks=args.walks, walk_len=args.walk_len, window=args.window,
              cap=args.cap, p=args.p, q=args.q,
              far=args.far,
              k1=args.k1, k4=args.k4, kr=args.kr,
              far_bias=args.far_bias, far_weight=args.far_weight,
              chunks=args.chunks, chunk_host=args.chunk_host)
log(f"walk: {args.walks} walks x {args.walk_len} steps, window "
    f"{args.window}, cap {args.cap}, p={args.p} q={args.q} "
    f"({'uniform, the node2vec walk at p=q=1' if args.p == args.q == 1.0 else 'second order'})")
log(f"pairs: {args.pairs}, weight rule: {args.weight}")
log(f"law: {fw.law}, planes {planes_of(fw.law)}, k1={args.k1} "
    f"k4={args.k4} kr={args.kr}")

# ---------------------------------------------------------------------------
# 3. + 4. the augmentation and the embedding, in ONE `embed` call
#
# The augmentation must run EXACTLY ONCE. `embed` calls `augment_graph`
# itself, thus a call here to time the stage would run the walks a second
# time, on an `rng` the first call already advanced -- a different `D`
# under the numbers this script prints. A callback times the seam instead:
# `train_begin` fires after the augmentation and before the first epoch.
# ---------------------------------------------------------------------------
class Stopwatch(Callback_Base):
    """The clock at the augmentation/embedding seam, and the report of the
    augmentation, which is complete when `train_begin` fires."""

    t_aug_end = None
    rss_aug = None

    def on_train_begin(self, model, **kw):
        self.t_aug_end = time.time()
        self.rss_aug = rss_mb()
        D, info = model.D, model.info
        log(f"pairs: {info['raw_pairs']} raw -> {info['unique_pairs']} "
            f"unique -> {info['capped_pairs']} after the cap "
            f"({info['t_pairs']:.1f}s)")
        log(f"D: {info['near_nnz']} near + {2 * info['far_pairs']} far = "
            f"{D.nnz} entries, {D.nnz / n:.1f} for each node, weights "
            f"{D.data.min():.0f}..{D.data.max():.0f} "
            f"({self.t_aug_end - t_start:.1f}s, RSS {self.rss_aug:.0f} MB)")
        uw, cw = np.unique(np.rint(D.data).astype(int), return_counts=True)
        log(f"weight histogram: {dict(zip(uw.tolist(), cw.tolist()))}")
        log(f"plan: {model.plan_stats}")
        log(f"embedding: dim={args.dim}, epochs={args.epochs}, "
            f"optim={args.optim} "
            f"({optimizers.state_arrays(args.optim)} state arrays), "
            f"lr={args.lr}, chunks={args.chunks}")


watch = Stopwatch()
fw.attach_callback(watch)
t_start = time.time()
fw.embed(A, epochs=args.epochs, batch_count=args.chunks)
Z = fw.get_embeddings()
t_end = time.time()

D = fw.D                                   # the ONE matrix the run used
t_aug = watch.t_aug_end - t_start
t_embed = t_end - watch.t_aug_end
rss_aug = watch.rss_aug
dz = float(fw.Th(fw.dZ))
rss_embed = rss_mb()
log(f"embedded {Z.shape} in {t_embed:.1f}s ({args.epochs / t_embed:.1f} "
    f"epochs/s), final ||dZ|| avg {dz:.6f}, finite {not fw.diverged}, "
    f"RSS {rss_embed:.0f} MB")


def result(**extra):
    """One machine-readable line, in the shape `bench_fdwalk.py` prints."""
    fields = [("graph", args.graph), ("pairs", args.pairs),
              ("weight", args.weight),
              ("force", "fdlinear"), ("optim", args.optim),
              ("seed", args.seed), ("n", n), ("dnnz", int(D.nnz)),
              ("dim", args.dim), ("epochs", args.epochs), ("lr", args.lr),
              ("t_load", f"{t_load:.1f}"), ("t_aug", f"{t_aug:.1f}"),
              ("t_embed", f"{t_embed:.1f}")]
    fields += list(extra.items())
    fields += [("rss", f"{rss_mb():.0f}")]
    line = "[fodiwalk] RESULT\t" + "\t".join(f"{k}={v}" for k, v in fields)
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


# I7. A diverged run RECORDS the divergence and stops cleanly. It never
# reaches sklearn, which would raise "Input X contains NaN" and carry away
# the time, the memory and the RESULT line with it.
if fw.diverged:
    log(f"DIVERGED: Z holds {fw.n_nonfinite} non-finite values at epoch "
        f"{fw.diverged_epoch}. The evaluation is skipped.")
    result(dz="nan", acc="nan", f1="nan", auc="nan", r2_dist="nan",
           diverged=1)
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# 5. link prediction. `fw.rng` and not a fresh generator: the augmentation
#    already advanced it, and the order of the draws is part of a run.
# ---------------------------------------------------------------------------
t0 = time.time()
scores, lp = link_prediction(Z, A, n, args.lp_pairs, fw.rng, args.seed)
t_lp = time.time() - t0
log(f"link prediction on {lp['pairs']} pairs ({lp['positives']} of "
    f"{lp['edges']} edges + {lp['negatives']} non-edges), {lp['train']} "
    f"train / {lp['test']} test ({t_lp:.1f}s)")
for name, value in scores.items():
    print(f"  {name:<9s}: {value:.4f}")

# ---------------------------------------------------------------------------
# 6. the hop-distance regression
# ---------------------------------------------------------------------------
t0 = time.time()
# `gap_csr` measures H2, the walk gap against the true hop distance. Every
# policy is walk-based, thus every run has one.
gap_csr = PR.to_csr(fw.stats["key"], fw.stats["mn"].astype(np.float64), n)
u, v, d, h2 = hop_sample(A, n, fw.rng, args.hop_sources, args.hop_pairs,
                         gap_csr)
keep = d >= args.hop_min          # a neighbour is the easy case; ../other-ge
u, v, d = u[keep], v[keep], d[keep]          # drops it too, thus comparable
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
t_hop = time.time() - t0

if h2:
    log(f"H2, the walk gap against the true hop distance on {h2['pairs']} "
        f"stored pairs: exact {h2['exact'] * 100:.1f}%, over "
        f"{h2['over'] * 100:.1f}%, under {h2['under'] * 100:.1f}%, "
        f"MAE {h2['mae']:.3f}")

# ---------------------------------------------------------------------------
# 7. the report
# ---------------------------------------------------------------------------
log("")
log(f"{'stage':<18s} {'time s':>9s} {'peak RSS MB':>12s}")
log(f"{'load':<18s} {t_load:9.1f} {'':>12s}")
log(f"{'augment + plan':<18s} {t_aug:9.1f} {rss_aug:12.0f}")
log(f"{'embed':<18s} {t_embed:9.1f} {rss_embed:12.0f}")
log(f"{'link prediction':<18s} {t_lp:9.1f} {'':>12s}")
log(f"{'hop regression':<18s} {t_hop:9.1f} {'':>12s}")
log(f"{'TOTAL':<18s} {t_load + t_aug + t_embed + t_lp + t_hop:9.1f} "
    f"{rss_mb():12.0f}")
log("")
log(f"accuracy {scores['accuracy']:.4f}  F1 {scores['f1_score']:.4f}  "
    f"AUC {scores['auc']:.4f}  hop R2 (distance) {best['distance'][0]:.3f}  "
    f"||dZ|| {dz:.6f}")

result(dz=f"{dz:.4f}", acc=f"{scores['accuracy']:.4f}",
       f1=f"{scores['f1_score']:.4f}", auc=f"{scores['auc']:.4f}",
       r2_dist=f"{best['distance'][0]:.3f}",
       mae_dist=f"{best['distance'][1]:.3f}",
       r2_vec=f"{best['vector'][0]:.3f}",
       h2_exact=f"{h2['exact']:.3f}" if h2 else "na", diverged=0)
