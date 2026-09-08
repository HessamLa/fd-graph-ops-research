#!/bin/env python3
"""bench_n2v1m.py -- fodiwalk on com_youtube, scored under protocol n2v1m.

This is the A12 comparison script. It builds the fodiwalk baseline
embedding with the SAME walk settings node2vec used (5 walks x 20 steps,
seed 42, first-order uniform walk p=q=1), then scores it through
`evaluator.link_prediction` / `evaluator.dist_approx` under protocol
`n2v1m` -- the protocol `bench_node2vec_1M.py` used for the recorded
reference numbers (`experiments/large-graph-node2vec/`).

`bench_fodiwalk.py`'s own scoring (`fodiwalk.misc.evaluation`, protocol
`fodiwalk_dist`/`fodiwalk_vec`) is NOT used here. That protocol draws
negatives and hop pairs a different way (see `evaluator/config.py`) and a
score under it is not comparable to a node2vec number.

`--lp-max-pairs` defaults to 50,000 to match the archived node2vec run:
`bench_node2vec_1M.py` ran with `--lp-pairs 25000` (POSITIVES), and its
own code doubles that to `max_pairs=50000` before calling `evaluator`
(`bench_node2vec_1M.py:220`). That is an explicit override of the bare
`n2v1m` protocol default of 80,000 -- the recorded 0.9750 accuracy was
produced at 50,000 pairs, not 80,000, so this script matches 50,000.

Epochs are a SWEEP, not one run. One force-relaxation call runs to the
highest requested checkpoint; a callback on `on_epoch_end` snapshots `Z`
at each requested epoch count, so the sweep pays the graph augmentation
once and each checkpoint is scored on its own snapshot.

    .venv/bin/python experiments/fodiwalk/bench_n2v1m.py --checkpoints 50
    .venv/bin/python experiments/fodiwalk/bench_n2v1m.py --checkpoints 200,1000,2000
"""
from __future__ import annotations

import argparse
import os
import resource
import subprocess
import sys
import threading
import time

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--graph", default="com_youtube")
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--walks", type=int, default=5)
ap.add_argument("--walk-len", type=int, default=20)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--checkpoints", default="2000",
                help="comma-separated epoch counts to snapshot and score")
ap.add_argument("--lp-max-pairs", type=int, default=50_000,
                help="total link-prediction pairs; matches node2vec's "
                     "--lp-pairs 25000 (positives), doubled by "
                     "bench_node2vec_1M.py to max_pairs=50000")
ap.add_argument("--chunks", type=int, default=1)
ap.add_argument("--chunk-host", action="store_true")
ap.add_argument("--device", default="gpu")
ap.add_argument("--out", default="", help="append RESULT lines to this file")
args = ap.parse_args()

# The backend must be chosen before `import jax`, thus before `fodiwalk`.
if args.device != "auto":
    os.environ["JAX_PLATFORMS"] = {"gpu": "cuda"}.get(args.device, args.device)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from fodiwalk import Fodiwalk
from forcedirected import Callback_Base
from fodiwalk.make_graph import load
import evaluator as ev


def rss_mb():
    """Peak resident memory of this process so far, ru_maxrss cross-check."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def log(msg):
    print(f"[n2v1m] {msg}", flush=True)


# ---- RSS sampler thread: /proc/self/statm, the pattern of fodiwalk/tests/mem.py.
# ru_maxrss never falls and only marks a high-water mark across the whole
# process life; the statm sample is the number that answers "peak now".
def _statm_rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2 ** 20


_rss_peak = [0.0]
_stop_rss = [False]


def _rss_sampler():
    while not _stop_rss[0]:
        _rss_peak[0] = max(_rss_peak[0], _statm_rss_mb())
        time.sleep(0.02)


threading.Thread(target=_rss_sampler, daemon=True).start()

# ---- GPU memory sampler thread: nvidia-smi, polled.
_gpu_peak = [0.0]
_stop_gpu = [False]


def _gpu_used_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0.0


def _gpu_sampler():
    while not _stop_gpu[0]:
        _gpu_peak[0] = max(_gpu_peak[0], _gpu_used_mb())
        time.sleep(0.2)


threading.Thread(target=_gpu_sampler, daemon=True).start()

checkpoints = sorted(int(x) for x in args.checkpoints.split(","))
max_epoch = checkpoints[-1]

log(f"baseline com_youtube/nbr_walk/fdlinear/n2v1m dim={args.dim} "
    f"walks={args.walks} x {args.walk_len} seed={args.seed} "
    f"checkpoints={checkpoints}")

import jax  # after the backend is chosen

log(f"jax {jax.__version__}, backend {jax.default_backend()}, "
    f"devices {jax.devices()}")

t0 = time.time()
A, n = load(args.graph, 0, args.seed)
t_load = time.time() - t0
log(f"{args.graph}: n={n} nodes, {A.nnz // 2} undirected edges "
    f"({t_load:.1f}s, RSS {rss_mb():.0f} MB)")

fw = Fodiwalk(n_dim=args.dim, lr=1.0, seed=args.seed, verbosity=0,
              optim="plain", pairs="nbr_walk", weight="min_gap",
              force="fdlinear", walks=args.walks, walk_len=args.walk_len,
              p=1.0, q=1.0, far=0,
              k1=0.999, k4=0.01, kr=1.0, far_bias=0.0, far_weight=100.0,
              chunks=args.chunks, chunk_host=args.chunk_host)
# `nbr_walk` is a DIRECTED, per-row policy: row u holds every node a walk
# FROM u reached, at the first step that reached it. Its budget is
# walks * walk_len for each row -- it has NO window and NO cap, so those
# knobs are not passed here (fodiwalk/dev-docs/fodiwalk-module.md and
# fodiwalk/augment_graph/policy_nbr_walk.py:row_stats confirm this; passing
# them would do nothing and mislead a reader of this log).
log(f"walk: {args.walks} walks x {args.walk_len} steps, uniform p=q=1 "
    f"(the node2vec walk); pairs=nbr_walk (directed, no window, no cap), "
    f"weight=min_gap, force=fdlinear")


class Checkpointer(Callback_Base):
    """The seam between augmentation and embedding, and one snapshot of
    `Z` for each requested epoch count, taken without stopping training."""

    t_aug_end = None
    rss_aug = None
    _t_train_begin = None

    def __init__(self):
        self.snapshots = {}  # epoch -> (Z copy, wall seconds since train_begin)

    def on_train_begin(self, model, **kw):
        self._t_train_begin = time.time()
        self.t_aug_end = self._t_train_begin
        self.rss_aug = rss_mb()
        D, info = model.D, model.info
        log(f"pairs: {info['raw_pairs']} raw -> {info['unique_pairs']} "
            f"unique -> {info['capped_pairs']} capped ({info['t_pairs']:.1f}s)")
        log(f"D: {D.nnz} entries, {D.nnz / n:.1f} for each node "
            f"(augmentation done, RSS {self.rss_aug:.0f} MB)")
        log(f"plan: {model.plan_stats}")

    def on_epoch_end(self, model, epoch=None, **kw):
        if epoch in checkpoints:
            wall = time.time() - self._t_train_begin
            self.snapshots[epoch] = (model.get_embeddings().copy(), wall)
            log(f"checkpoint epoch={epoch} wall_since_train_begin={wall:.1f}s "
                f"RSS {rss_mb():.0f} MB")


cp = Checkpointer()
fw.attach_callback(cp)
t_embed_start = time.time()
fw.embed(A, epochs=max_epoch, batch_count=args.chunks)

_stop_rss[0] = True
_stop_gpu[0] = True

D = fw.D
t_aug = cp.t_aug_end - t_embed_start
log(f"augmentation {t_aug:.1f}s, D.nnz={D.nnz}, peak RSS during run "
    f"{_rss_peak[0]:.0f} MB (statm), {rss_mb():.0f} MB (ru_maxrss), "
    f"peak GPU {_gpu_peak[0]:.0f} MB")

if fw.diverged:
    log(f"DIVERGED: Z holds {fw.n_nonfinite} non-finite values at epoch "
        f"{fw.diverged_epoch}. Evaluation is skipped.")
    sys.exit(0)


def result(**extra):
    fields = [("graph", args.graph), ("dim", args.dim),
              ("walks", args.walks), ("walk_len", args.walk_len),
              ("seed", args.seed), ("n", n), ("dnnz", int(D.nnz))]
    fields += list(extra.items())
    line = "[n2v1m] RESULT\t" + "\t".join(f"{k}={v}" for k, v in fields)
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


for epoch in checkpoints:
    Z, t_since_begin = cp.snapshots[epoch]
    log(f"--- scoring checkpoint epoch={epoch}, evaluator protocol n2v1m ---")

    t0 = time.time()
    lp = ev.link_prediction(A, Z, protocol="n2v1m", seed=args.seed,
                            max_pairs=args.lp_max_pairs)
    t_lp = time.time() - t0
    log(f"lp epoch={epoch}: {lp.sizes['pairs']} pairs "
        f"({lp.sizes['positives']} pos / {lp.sizes['negatives']} neg), "
        f"protocol_modified={lp.cfg['protocol_modified']} ({t_lp:.1f}s)")
    for k, v in lp.scores.items():
        log(f"  {k:<10s}: {v:.4f}")
    if lp.warnings:
        log(f"  lp warnings: {lp.warnings}")

    t0 = time.time()
    da = ev.dist_approx(A, Z, protocol="n2v1m", seed=args.seed)
    t_da = time.time() - t0
    log(f"da epoch={epoch}: {da.sizes['n_pairs']} pairs, "
        f"{da.cfg['n_sources']} sources, "
        f"protocol_modified={da.cfg['protocol_modified']} ({t_da:.1f}s)")
    for model_name, s in da.scores.items():
        log(f"  {model_name:>10s} mae={s['mae']:.3f} mre={s['mre']:.3f} "
            f"rmse={s['rmse']:.3f} r2={s['r2']:.3f} exact={s['exact']:.3f}")
    if da.warnings:
        log(f"  da warnings: {da.warnings}")

    t_total = t_load + t_aug + t_since_begin + t_lp + t_da
    result(epoch=epoch, t_load=f"{t_load:.1f}", t_aug=f"{t_aug:.1f}",
          t_embed=f"{t_since_begin:.1f}", t_lp=f"{t_lp:.1f}",
          t_da=f"{t_da:.1f}", t_total=f"{t_total:.1f}",
          acc=f"{lp.scores['accuracy']:.4f}",
          prec=f"{lp.scores['precision']:.4f}",
          rec=f"{lp.scores['recall']:.4f}",
          f1_score=f"{lp.scores['f1_score']:.4f}", auc=f"{lp.scores['auc']:.4f}",
          mlp_r2=f"{da.scores['mlp']['r2']:.4f}",
          mlp_mae=f"{da.scores['mlp']['mae']:.4f}",
          peak_rss=f"{_rss_peak[0]:.0f}", peak_gpu=f"{_gpu_peak[0]:.0f}")
