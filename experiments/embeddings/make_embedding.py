"""Generate ONE embedding and store it under `data_cache/embeddings/`.

Two methods, one node order. Both read the graph through
`fodiwalk.make_graph.load`, thus row `i` of `Z` is the same node for
`fodiwalk` and for `node2vec` and the two embeddings compare directly.

    .venv/bin/python experiments/embeddings/make_embedding.py \
        --method fodiwalk --graph cora --dim 128 \
        --pairs nbr_walk --optim plain --score

The layout and the JSON fields are defined in
`data_cache/embeddings/README.md`. This file is the code that writes them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "large-graph-node2vec"))

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--method", required=True,
                choices=("fodiwalk", "node2vec", "deepwalk"))
ap.add_argument("--graph", default="cora")
ap.add_argument("--max-nodes", type=int, default=0)
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--epochs", type=int, default=200)
ap.add_argument("--seed", type=int, default=42)
# fodiwalk
ap.add_argument("--pairs", default="nbr_walk",
                choices=("walk", "walk_edges", "nbr_walk"))
ap.add_argument("--weight", default="min_gap")
ap.add_argument("--force", default="fdlinear")
ap.add_argument("--optim", default="plain")
ap.add_argument("--lr", type=float, default=0.999)
ap.add_argument("--k1", type=float, default=0.999, help="attraction gain")
ap.add_argument("--k2", type=float, default=1.0,
                help="hop decay of the attraction. `fdhop_all` only")
ap.add_argument("--k4", type=float, default=0.01,
                help="decay of the repulsion. `fdhop` wants 1.0; the "
                     "Config default 0.01 is 100x flatter")
ap.add_argument("--kr", type=float, default=1.0, help="repulsion at h = 1")
ap.add_argument("--sign", type=float, default=-1.0, help="fdlinear only")
ap.add_argument("--device", default="cpu")
ap.add_argument("--chunks", type=int, default=1)
# both
ap.add_argument("--walks", type=int, default=10)
ap.add_argument("--walk-len", type=int, default=20)
# node2vec
ap.add_argument("--window", type=int, default=5)
ap.add_argument("--n2v-epochs", type=int, default=5)
ap.add_argument("--p", type=float, default=1.0,
                help="return parameter of the second-order walk. 1.0 gives "
                     "a first-order uniform walk. Used by node2vec AND by "
                     "fodiwalk, whose `walk`/`walk_edges` policies call the "
                     "same `make_walker`")
ap.add_argument("--q", type=float, default=1.0,
                help="in-out parameter. q < 1 sends the walk away from "
                     "where it came (DFS-like); q > 1 keeps it near "
                     "(BFS-like). 1.0 gives a uniform walk")
ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
# scoring and output
ap.add_argument("--score", action="store_true")
ap.add_argument("--protocol", default="n2v1m")
ap.add_argument("--lp-max-pairs", type=int, default=50_000)
ap.add_argument("--store", default=os.path.join(ROOT, "data_cache", "embeddings"))
ap.add_argument("--log", default=os.path.join(ROOT, "experiments", "embeddings",
                                              "runs.log"))
args = ap.parse_args()

os.environ.setdefault("JAX_PLATFORMS",
                      {"gpu": "cuda"}.get(args.device, args.device))


def log(msg):
    print(f"[emb] {msg}", flush=True)


def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2 ** 20


# `peak_rss_mb` was ONE reading taken after the embedding, so it recorded
# the resident size at that instant and not the peak. Augmentation is the
# hungriest phase and it is already over by then: a com_youtube run
# measured 5.34 GB with `ps` while writing 1392 MB into its config. Every
# record made before 2026-09-07 UNDERSTATES its memory. A sampler thread
# fixes it, the same way `experiments/fodiwalk-streaming/bench_stream.py`
# does.
_peak = [0.0]
_stop = [False]


def _sample():
    while not _stop[0]:
        _peak[0] = max(_peak[0], rss_mb())
        time.sleep(0.02)


threading.Thread(target=_sample, daemon=True).start()


def run_log(status, note):
    """One line per attempt, so a failure leaves a record and not a gap."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.method == "fodiwalk":
        settings = (f"pairs={args.pairs}\toptim={args.optim}\t"
                    f"force={args.force}\tepochs={args.epochs}\t"
                    f"lr={args.lr}")
    else:
        # node2vec has no pairs policy, force law or force-directed rule.
        settings = (f"walks={args.walks}x{args.walk_len}\t"
                    f"window={args.window}\tepochs={args.n2v_epochs}\t"
                    f"p={args.p}\tq={args.q}")
    line = (f"{stamp}\t{status}\tmethod={args.method}\tgraph={args.graph}\t"
            f"dim={args.dim}\t{settings}\tseed={args.seed}\t{note}\n")
    with open(args.log, "a") as f:
        f.write(line)
    log(f"{status}: {note}")


# hash_spec "1": the fields that define the run.
HASH_FIELDS_1 = ("dataset.name", "n_dim", "method.name", "method.version",
                 "walker", "force", "optimizer", "strategy", "run")

# hash_spec "2", from 2026-09-04. `force.params` now holds the scalars the
# law reads, and `method.git_commit` separates two runs of the same settings
# against different code -- the defect that made both `fdhop` runs of
# 2026-09-04 hash to eb56e1fb.
HASH_FIELDS_2 = HASH_FIELDS_1 + ("method.git_commit",)
HASH_SPECS = {"1": HASH_FIELDS_1, "2": HASH_FIELDS_2}
HASH_SPEC = "2"


def fingerprint(cfg, fields=HASH_FIELDS_2):
    """First 8 hex characters of the sha256 of the named config fields."""
    picked = {}
    for path in fields:
        node = cfg
        for key in path.split("."):
            node = node[key]
        picked[path] = node
    blob = json.dumps(picked, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def git_state():
    def sh(*c):
        return subprocess.run(c, cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    return sh("git", "rev-parse", "--short", "HEAD"), bool(
        sh("git", "status", "--porcelain"))


def versions():
    import jax, jaxlib, scipy
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "jax": jax.__version__,
            "jaxlib": jaxlib.__version__}


# ------------------------------------------------------------------ methods
def embed_fodiwalk(A, n):
    """Stage 2 and stage 3 of the package, with the settings of `args`."""
    from fodiwalk import Fodiwalk
    fw = Fodiwalk(n_dim=args.dim, lr=args.lr, seed=args.seed, verbosity=0,
                  pairs=args.pairs, weight=args.weight, force=args.force,
                  optim=args.optim, walks=args.walks, walk_len=args.walk_len,
                  window=args.window, k1=args.k1, k4=args.k4, kr=args.kr,
                  fdlinear_sign=args.sign, p=args.p, q=args.q,
                  k2=args.k2)
    fw.embed(A, epochs=args.epochs, batch_count=args.chunks)
    Z = np.asarray(fw.get_embeddings(), dtype=np.float32)
    diverged = bool(getattr(fw, "diverged", False))
    return Z, diverged, {"strategy": "precomputed", "batch": 0}


def _write_biased_walks(A, n, path):
    """Second-order walks of `p`/`q`, one walk per line. For node2vec."""
    from fodiwalk.augment_graph.walks import make_walker
    walker = make_walker(A, n, args.p, args.q)
    rng = np.random.default_rng(args.seed)
    with open(path, "w") as f:
        for _ in range(args.walks):
            for s0 in range(0, n, 100_000):
                cur = np.arange(s0, min(s0 + 100_000, n))
                w = walker(cur, args.walk_len, rng)
                f.write("\n".join(" ".join(map(str, row)) for row in w))
                f.write("\n")


def embed_word2vec(A, n):
    """DeepWalk or node2vec. They differ in the WALK and in the OBJECTIVE.

        deepwalk   first-order uniform walks, hierarchical softmax
        node2vec   second-order walks of (p, q), negative sampling

    At `p = q = 1` a second-order walk IS a uniform walk, thus the two then
    differ in the objective ALONE. The `p` and `q` of the run are recorded,
    so a reader can tell which case a record is.
    """
    import tempfile
    from bench_node2vec_1M import write_walks
    from gensim.models import Word2Vec

    deepwalk = args.method == "deepwalk"
    biased = (not deepwalk) and (args.p != 1.0 or args.q != 1.0)
    with tempfile.TemporaryDirectory(prefix="walks_", dir="/tmp") as tmp:
        wp = os.path.join(tmp, "walks.txt")
        if biased:
            _write_biased_walks(A, n, wp)
        else:
            write_walks(A, n, wp, args.walks, args.walk_len, args.seed)
        model = Word2Vec(corpus_file=wp, vector_size=args.dim,
                         window=args.window, min_count=0, sg=1,
                         hs=1 if deepwalk else 0,
                         negative=0 if deepwalk else 5,
                         workers=args.workers, epochs=args.n2v_epochs,
                         seed=args.seed)
    Z = np.zeros((n, args.dim), dtype=np.float32)
    for i in range(n):
        k = str(i)
        if k in model.wv:
            Z[i] = model.wv[k]
    del model
    return Z, False, {"strategy": "hierarchical_softmax" if deepwalk
                      else "negative_sampling", "batch": 0}


# ------------------------------------------------------------------ the run
def main():
    from fodiwalk.make_graph import load

    t0 = time.perf_counter()
    A, n = load(args.graph, args.max_nodes, args.seed)
    t_load = time.perf_counter() - t0
    log(f"{args.graph}: n={n:,}, {A.nnz // 2:,} undirected edges, "
        f"avg degree {A.nnz / n:.2f} (load {t_load:.1f}s)")

    t0 = time.perf_counter()
    if args.method == "fodiwalk":
        Z, diverged, strategy = embed_fodiwalk(A, n)
    else:
        Z, diverged, strategy = embed_word2vec(A, n)
    t_embed = time.perf_counter() - t0
    peak = max(_peak[0], rss_mb())
    log(f"embedded in {t_embed:.1f}s, Z {Z.shape} {Z.dtype}, "
        f"peak RSS {peak:.0f} MB")

    if diverged or not np.isfinite(Z).all():
        raise RuntimeError(
            f"Z holds non-finite values (diverged={diverged}). Nothing is "
            f"stored: a non-finite embedding cannot be scored or reused.")

    metrics = None
    if args.score:
        import evaluator
        lp = evaluator.link_prediction(A, Z, protocol=args.protocol,
                                       seed=args.seed,
                                       max_pairs=args.lp_max_pairs)
        da = evaluator.dist_approx(A, Z, protocol=args.protocol,
                                   seed=args.seed)
        metrics = {"protocol": args.protocol,
                   "protocol_modified": bool(lp.protocol_modified
                                             or da.protocol_modified),
                   "lp_max_pairs": args.lp_max_pairs}
        # One name for the harmonic mean. `evaluator` spells it `f1`
        # and the older code spells it `f1-score`; the store spells it
        # `f1_score` and nothing else.
        rename = {"f1": "f1_score", "f1-score": "f1_score"}
        metrics.update({rename.get(k, k): float(v)
                        for k, v in lp.scores.items()})
        for m, sc in da.scores.items():
            metrics[f"{m}_r2"] = float(sc["r2"])
            metrics[f"{m}_mae"] = float(sc["mae"])
        log("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                             if isinstance(v, float)))

    # The scalars each law READS. A law is not charged for one it ignores:
    # `fdhop` does not read `sign`, thus a different `sign` is the same run.
    LAW_PARAMS = {"fdlinear": ("k1", "k4", "kr", "sign"),
                  "fdlinear_fused": ("k1", "k4", "kr", "sign"),
                  "fdhop": ("k1", "k4", "kr"),
                  "fdhop2": ("k1", "k4", "kr"),
                  "fdhop_min": ("k1", "k4", "kr"),
                  "fdhop_all": ("k1", "k2", "k4", "kr")}
    scalars = {"k1": args.k1, "k2": args.k2, "k4": args.k4,
               "kr": args.kr, "sign": args.sign}
    fparams = {k: scalars[k] for k in LAW_PARAMS.get(args.force, ())}

    _peak[0] = max(_peak[0], rss_mb())
    peak = _peak[0]                      # the sampled peak, scoring included
    commit, dirty = git_state()
    dc_gain = {"plain": 1.0, "sgd": 1.0, "velocity": 1.0}.get(args.optim, 0.0)
    cfg = {
        "schema_version": "1",
        "hash_spec": HASH_SPEC,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "datetime": time.strftime("%y%m%d-%H%M%S", time.gmtime()),
        "dataset": {"name": args.graph, "loader": "fodiwalk.make_graph.load",
                    "n": int(n), "edges": int(A.nnz // 2),
                    "avg_degree": round(float(A.nnz / n), 4),
                    "max_nodes": args.max_nodes},
        "n_dim": args.dim,
        "method": {
            # node2vec carries its search bias in its NAME: the whole of
            # what separates it from deepwalk is (p, q), and a directory
            # that hides them cannot be told apart at a glance.
            "name": ("fodiwalk_precomp" if args.method == "fodiwalk"
                     else f"node2vec_{args.p}_{args.q}"
                     if args.method == "node2vec" else args.method),
            "entry": "experiments/embeddings/make_embedding.py",
            "package": "fodiwalk" if args.method == "fodiwalk" else "gensim",
            "version": "00.03" if args.method == "fodiwalk" else "",
            "git_commit": commit, "git_dirty": dirty},
        "walker": {"name": ((args.pairs if args.p == 1.0 and args.q == 1.0
                             else f"{args.pairs}_pq")
                            if args.method == "fodiwalk" else
                            "uniform" if args.method == "deepwalk" or
                            (args.p == 1.0 and args.q == 1.0)
                            else "node2vec_pq"),
                   "module": ("fodiwalk/augment_graph/policies.py"
                              if args.method == "fodiwalk"
                              else "experiments/large-graph-node2vec/"
                                   "bench_node2vec_1M.py:write_walks"),
                   "walks": args.walks, "walk_len": args.walk_len,
                   "window": args.window,
                   # `deepwalk` has no search bias by definition; every
                   # other method walks with these.
                   "p": None if args.method == "deepwalk" else args.p,
                   "q": None if args.method == "deepwalk" else args.q,
                   "weight": args.weight if args.method == "fodiwalk" else ""},
        "force": ({"law": args.force, "module": "fodiwalk/embed/forces.py",
                   "params": fparams} if args.method == "fodiwalk"
                  else {"law": "", "module": "", "params": {}}),
        "optimizer": ({"rule": args.optim, "module": "forcedirected/optim.py",
                       "lr": args.lr, "lr_decay": "const",
                       "dc_gain": dc_gain,
                       "effective_lr": round(args.lr * dc_gain, 6),
                       "params": {}} if args.method == "fodiwalk"
                      else {"rule": "word2vec_sgd", "module": "gensim",
                            "lr": 0.025, "lr_decay": "linear",
                            "dc_gain": 1.0, "effective_lr": 0.025,
                            "params": {"sg": 1,
                                       "hs": 1 if args.method == "deepwalk" else 0,
                                       "negative": 0 if args.method == "deepwalk"
                                       else 5}}),
        "strategy": strategy,
        "run": {"epochs": (args.epochs if args.method == "fodiwalk"
                           else args.n2v_epochs),
                "seed": args.seed,
                "device": args.device if args.method == "fodiwalk" else "cpu"},
        "environment": dict(versions(), peak_rss_mb=round(peak, 1),
                            seconds=round(t_load + t_embed, 1)),
        "agent": {"model": "claude-opus-5", "session_ref": "6bad1b",
                  "transcript_uuid": "fd084f99-dab1-4ce5-a2b5-7c74e42c28d8"},
        "notes": "",
    }

    h = fingerprint(cfg, HASH_SPECS[HASH_SPEC])
    name = f"{cfg['datetime']}-{cfg['method']['name']}-{h}"
    out = os.path.join(args.store, args.graph, str(args.dim), name)
    os.makedirs(out, exist_ok=True)
    zp = os.path.join(out, "Z.npy")
    np.save(zp, Z)
    cfg["outputs"] = {"file": "Z.npy", "shape": list(Z.shape),
                      "dtype": str(Z.dtype), "bytes": os.path.getsize(zp),
                      "sha256": hashlib.sha256(
                          open(zp, "rb").read()).hexdigest()}
    if metrics:
        cfg["metrics"] = metrics
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    run_log("OK", f"dir={os.path.relpath(out, ROOT)}")
    log(f"stored {os.path.relpath(out, ROOT)}")


try:
    main()
except Exception as exc:
    run_log("FAIL", f"{type(exc).__name__}: {exc}".replace("\n", " ")[:300])
    traceback.print_exc()
    sys.exit(1)
