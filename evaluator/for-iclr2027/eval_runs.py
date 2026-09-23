"""Score the ICLR 2027 embeddings and write `evaluation.json` into each run folder.

Usage:
    .venv/bin/python data_cache/evaluator/for-iclr2027/eval_runs.py [--list] [DIR ...]

With no DIR, every run folder under experiments/for-iclr2027/embeddings that
has `Z.npy` and `config.json` but no `evaluation.json` is scored. A folder
that already has `evaluation.json` (with all its checkpoints) is skipped, so the script is safe to run
twice.

One evaluation seed (EVAL_SEED) for every run, whatever the embedding seed.
Thus all runs of one graph are scored on the same LP split, the same hop
pairs and the same kNN queries, and a spread across embedding seeds is not
mixed with a spread across evaluation samples. The runner's `--score`
numbers in `config.json` use the embedding seed as the evaluation seed, so
they differ from these.

Protocol `n2v1m` with max_pairs=50_000, the same as the store records, so
`protocol_modified` is True.
"""
import json, os, subprocess, sys, time, traceback, warnings
from pathlib import Path

import numpy as np

REPO = Path("/home/h/gnn/fd-graph-embedding/fdmap")
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

import evaluator as ev
from evaluator import hops as hops_mod, metrics as met, pairs as pr, rank, stress

STORE = REPO / "experiments/for-iclr2027/embeddings"

SCHEMA = "2"   # a file with an older schema is scored again
EVAL_SEED = 42
PROTOCOL = "n2v1m"
LP_MAX_PAIRS = 50_000
N_HOP_PAIRS = 20_000
N_QUERIES = 1_000
KNN_K = 10

# Above BIG_N nodes, hop pairs come from BIG_SOURCES sources. An unbounded
# sample makes `hops.choose_backend` build a PLL index over the whole graph,
# which does not fit. kNN also goes approximate above metrics.RECON_MAX_N.
BIG_N = 200_000
BIG_SOURCES = 200
BIG_QUERIES = 500


def git_head():
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "evaluator"],
                               capture_output=True, text=True).stdout.strip() != ""
        return out, dirty
    except Exception:
        return None, None


def fixture(name, max_nodes, seed):
    """Graph, hop-pair sample and kNN queries for one graph. Drawn once."""
    A, n, info = ev.load_graph(name, max_nodes=max_nodes, seed=seed)
    rng = np.random.default_rng(EVAL_SEED)
    big = n > BIG_N
    src = rng.choice(n, size=BIG_SOURCES, replace=False) if big else None
    p = pr.pairs_for_hops(A, n, N_HOP_PAIRS, rng, src, "reject")
    h = hops_mod.hop_distance(A, n, p, "auto", rng)
    # hops.UNREACHABLE is inf: drop it before the min-hop test.
    keep = np.isfinite(h) & (h >= 2)
    q = rng.choice(n, size=min(BIG_QUERIES if big else N_QUERIES, n), replace=False)
    return dict(A=A, n=n, info=info, pairs=p[keep], hop=h[keep].astype(np.float64),
                queries=q, dropped=int((~keep).sum()),
                sources=(0 if src is None else BIG_SOURCES))


def neighbour_recall(A, Z, queries, k):
    """Share of true graph neighbours found among a node's k nearest rows."""
    idx, _, path = met.knn(Z, queries, k)
    hits = tot = 0
    for row, u in enumerate(queries):
        nbr = set(A.indices[A.indptr[u]:A.indptr[u + 1]].tolist())
        if nbr:
            hits += len(nbr & set(idx[row].tolist()))
            tot += min(len(nbr), k)
    return (hits / tot if tot else float("nan")), path


def metrics(Z, fx):
    """Every score for one Z on the graph's fixed sample."""
    g = (fx["A"], fx["n"], fx["info"])
    t0 = time.time()

    lp = ev.link_prediction(g, Z, protocol=PROTOCOL, seed=EVAL_SEED,
                            max_pairs=LP_MAX_PAIRS)
    da = ev.dist_approx(g, Z, protocol=PROTOCOL, seed=EVAL_SEED)

    x = met.distance(Z, fx["pairs"][:, 0], fx["pairs"][:, 1], "euclidean")
    y = fx["hop"]
    tau_b = rank.kendall_tau_b(x, y)
    tie_y, tie_x = rank.tie_frac(y), rank.tie_frac(x)
    s1, s_star = stress.stress1(x, y)
    rec, path = neighbour_recall(fx["A"], Z, fx["queries"], KNN_K)

    return {
        "protocol_modified": bool(lp.protocol_modified or da.protocol_modified),
        "link_prediction": lp.scores,
        "dist_approx": da.scores,
        "rank": {"spearman": rank.spearman(x, y),
                 "kendall_tau_b": tau_b,
                 # Somers' D from tau_b; rank.somers_d is O(n^2), 160 s per run.
                 "somers_d": float(tau_b * np.sqrt((1 - tie_y) / (1 - tie_x))),
                 "tie_frac_hop": tie_y, "tie_frac_dist": tie_x,
                 "tau_b_max": rank.tau_b_max(y)},
        "stress": {"stress1": s1, "scale": s_star,
                   "distortion": stress.distortion(x, y, s_star)},
        "knn": {f"neighbour_recall_at_{KNN_K}": rec, "path": path,
                "queries": int(fx["queries"].size)},
        "sample": {"n": int(fx["n"]), "hop_pairs": int(y.size), "min_hop": 2,
                   "hop_sources": fx["sources"] or "all",
                   "dropped_unreachable_or_adjacent": fx["dropped"],
                   "da_backend": da.sizes.get("backend")},
        "seconds": round(time.time() - t0, 1),
    }


def ckpt_names(cfg):
    """Checkpoint files other than the final Z.npy, e.g. Z_ep050.npy."""
    return [c for c in cfg.get("outputs", {}).get("checkpoints") or [] if c != "Z.npy"]


def score(run, fx, head):
    """Top level: the final Z. `checkpoints`: one block per saved epoch."""
    cfg = json.loads((run / "config.json").read_text())
    out = {
        "schema_version": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run": run.name,
        "epoch": cfg.get("run", {}).get("epochs"),
        "z_sha256": cfg.get("outputs", {}).get("sha256"),
        "evaluator": {"git_commit": head[0], "evaluator_dirty": head[1],
                      "script": "data_cache/evaluator/for-iclr2027/eval_runs.py",
                      "eval_seed": EVAL_SEED, "protocol": PROTOCOL,
                      "lp_max_pairs": LP_MAX_PAIRS},
    }
    out.update(metrics(np.load(run / "Z.npy"), fx))
    names = ckpt_names(cfg)
    if names:
        # Key "epoch_050" from "Z_ep050.npy". With a decaying lr, epoch 50 of
        # a 200-epoch run is not the same as a 50-epoch run.
        out["checkpoints"] = {"epoch_" + c[len("Z_ep"):-len(".npy")]:
                              metrics(np.load(run / c), fx) for c in names}
    return out


# Runner times the large-graph embeddings (paper section 5.6). While one
# runs, a large graph must not be loaded: the machine has 15 GB, the desktop
# holds about 10 GB and a com_youtube fodiwalk run peaks at 5.3 GB. Small
# and medium graphs are allowed then, at one thread (Runner, 2026-09-19).
LARGE = ("com_youtube", "roadnet_ca", "ncbi_taxonomy", "as_skitter")
LARGE_EMBED = r"make_embedding.py.*--graph (com_youtube|roadnet_ca|ncbi_taxonomy|as_skitter)"


def large_embed_running():
    return subprocess.run(["pgrep", "-f", LARGE_EMBED], capture_output=True).returncode == 0


def to_py(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def pending(args):
    if args:
        return [Path(a).resolve() for a in args]
    runs = sorted(p.parent for p in STORE.glob("*/*/*/Z.npy"))
    return [r for r in runs if (r / "config.json").exists() and todo(r)]


def todo(run):
    """True if evaluation.json is missing or lacks a listed checkpoint."""
    ev_path = run / "evaluation.json"
    if not ev_path.exists():
        return True
    ev = json.loads(ev_path.read_text())
    if ev.get("schema_version") != SCHEMA:
        return True
    names = ckpt_names(json.loads((run / "config.json").read_text()))
    return bool(names) and "checkpoints" not in ev


def main():
    argv = sys.argv[1:]
    listing = "--list" in argv
    runs = pending([a for a in argv if a != "--list"])
    if listing:
        for r in runs:
            print(r.relative_to(STORE.resolve()) if r.is_relative_to(STORE.resolve()) else r)
        print(f"{len(runs)} pending")
        return
    head = git_head()
    fx, key_now, failed, skipped = None, None, 0, 0
    # Group by graph: one graph in memory at a time.
    runs.sort(key=lambda r: (json.loads((r / "config.json").read_text())["dataset"]["name"], r.name))
    for i, run in enumerate(runs, 1):
        cfg = json.loads((run / "config.json").read_text())
        ds = cfg["dataset"]
        if ds["name"] in LARGE and large_embed_running():
            print(f"[{i}/{len(runs)}] SKIP {ds['name']} {run.name}: "
                  f"a large-graph embedding is in flight", flush=True)
            skipped += 1
            continue
        mx = ds.get("max_nodes", 0) or 0
        # A truncated graph depends on the load seed: use the run's seed then.
        gseed = cfg["run"]["seed"] if mx else EVAL_SEED
        key = (ds["name"], mx, gseed)
        # Start and end times: to find an overlap with a timed large-graph run.
        t_start = time.strftime("%H:%M:%SZ", time.gmtime())
        try:
            if key != key_now:
                fx = None
                fx = fixture(ds["name"], mx, gseed)
                key_now = key
            # A cut graph (max_nodes > 0) is a BFS ball picked by the run seed.
            # A wrong load gives a wrong score with no error, so check its size.
            if fx["n"] != ds["n"] or fx["A"].nnz // 2 != ds["edges"]:
                raise ValueError(f"graph mismatch: loaded n={fx['n']} "
                                 f"edges={fx['A'].nnz // 2}, config n={ds['n']} "
                                 f"edges={ds['edges']} (max_nodes={mx}, seed={gseed})")
            out = score(run, fx, head)
            tmp = run / "evaluation.json.tmp"
            tmp.write_text(json.dumps(out, indent=2, default=to_py) + "\n")
            os.replace(tmp, run / "evaluation.json")
            print(f"[{i}/{len(runs)}] OK   {t_start}-{time.strftime('%H:%M:%SZ', time.gmtime())} {ds['name']} {run.name} ckpts={len(out.get('checkpoints', {}))}", flush=True)
        except Exception as exc:
            failed += 1
            traceback.print_exc()
            print(f"[{i}/{len(runs)}] FAIL {ds['name']} {run.name} {type(exc).__name__}: {exc}",
                  flush=True)
    print(f"done: {len(runs) - failed - skipped} written, {failed} failed, "
          f"{skipped} skipped (large graph, embedding in flight)")


main()
