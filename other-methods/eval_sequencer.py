#!/bin/env python3
"""Evaluate every stored embedding, preprocessing each graph ONCE per
experiment, and write the result next to the embedding.

Structure (the point of it):

    for experiment in {spearman_rho, recall_at_10, community, lp_hop}:
        for graph in {cora, citeseer, pubmed}:
            load the graph once
            G_data = experiment.preprocess(graph)      # done ONCE per graph
            for embedding of the graph:
                Z = load(embedding)
                res = experiment.evaluate(Z, G_data)
                merge res into <embedding_dir>/eval.json

So the expensive per-graph setup -- sampling the hop pairs, running
Louvain, listing the true neighbours -- happens one time per graph, not
once per embedding. And every embedding of one graph is scored against the
SAME setup, which is what makes the numbers comparable across methods.

Evaluation uses a FIXED eval seed (42), NOT the embedding seed: the hop
pairs, the query nodes and the k-means start are the same for every
embedding, so the spread across embedding seeds is method spread. (The LP
and hop-R2 values come from `config.json`, where `evaluator` wrote them at
embed time.)

Results live in `<dir>/eval.json` beside `Z.npy` and `config.json`, keyed
to the Z sha256. The run is INCREMENTAL: an experiment already recorded for
a Z (same sha) is skipped, so a later pass only scores new embeddings.

Usage:  eval_sequencer.py [graph ...]      (default: all three)
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import networkx as nx
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, "/home/h/gm/fd-graph-ops-research")
from evaluator.metrics import knn
from evaluator.pairs import pairs_for_hops
from evaluator.config import PROTOCOLS
from fodiwalk.make_graph import load

STORE = "/home/h/gm/fd-graph-ops-research/data_cache/embeddings"
DA = PROTOCOLS["fodiwalk_dist"].da
EVAL_SEED = 42
GRAPHS = ("cora", "citeseer", "pubmed")


# -- experiments: each is (preprocess(graph)->G_data, evaluate(Z,G_data)->dict) --

def pre_rho(A, n):
    rng = np.random.default_rng(EVAL_SEED)
    src = rng.choice(n, size=min(DA.n_sources, n), replace=False)
    pr, h = pairs_for_hops(A, n, DA.n_pairs, rng, src, DA.pair_draw)
    keep = np.isfinite(h) & (h >= DA.min_hop)
    return {"pairs": pr[keep], "h": h[keep]}


def ev_rho(Z, gd):
    d = np.linalg.norm(Z[gd["pairs"][:, 0]] - Z[gd["pairs"][:, 1]], axis=1)
    r, _ = spearmanr(d, gd["h"])
    return {"rho": float(r)}


def pre_recall(A, n):
    deg = np.diff(A.indptr)
    q = np.flatnonzero(deg > 0)
    truth = [A.indices[A.indptr[u]:A.indptr[u + 1]] for u in q]
    return {"A": A, "q": q, "truth": truth}


def ev_recall(Z, gd):
    idx, _d, _p = knn(Z, gd["q"], k=10)
    hits = denom = 0
    for row, true in enumerate(gd["truth"]):
        hits += np.intersect1d(idx[row], true, assume_unique=False).size
        denom += min(true.size, 10)
    return {"recall_at_10": hits / denom}


def pre_community(A, n):
    G = nx.from_scipy_sparse_array(A)
    comms = nx.algorithms.community.louvain_communities(G, seed=EVAL_SEED)
    lab = np.empty(n, dtype=np.int64)
    for cid, members in enumerate(comms):
        lab[list(members)] = cid
    return {"labels": lab, "k": len(comms)}


def ev_community(Z, gd):
    km = KMeans(n_clusters=gd["k"], random_state=EVAL_SEED, n_init=10).fit(Z)
    return {"nmi": float(normalized_mutual_info_score(gd["labels"], km.labels_)),
            "ari": float(adjusted_rand_score(gd["labels"], km.labels_))}


# lp_hop reads config.json (evaluator already scored it); no graph needed.
def ev_lp_hop(cfg):
    m = cfg.get("metrics", {})
    return {"hop_r2": m.get("rf_r2"), "lp_auc": m.get("auc"),
            "lp_f1": m.get("f1_score")}


EXPERIMENTS = {
    "spearman_rho": (pre_rho, ev_rho, ("rho",)),
    "recall_at_10": (pre_recall, ev_recall, ("recall_at_10",)),
    "community": (pre_community, ev_community, ("nmi", "ari")),
}


def read_eval(d):
    p = os.path.join(d, "eval.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def write_eval(d, e):
    json.dump(e, open(os.path.join(d, "eval.json"), "w"), indent=1)


def main():
    want = sys.argv[1:] or list(GRAPHS)
    dirs = {g: sorted(glob.glob(os.path.join(STORE, g, "128", "*"))) for g in want}

    # lp_hop first: cheap, no graph load
    for g in want:
        for d in dirs[g]:
            cfgp = os.path.join(d, "config.json")
            if not os.path.exists(cfgp):
                continue
            cfg = json.load(open(cfgp))
            e = read_eval(d)
            sha = cfg.get("outputs", {}).get("sha256")
            if e.get("z_sha256") != sha:
                e = {"z_sha256": sha}
            if "lp_auc" not in e:
                e.update(ev_lp_hop(cfg))
                write_eval(d, e)

    for name, (pre, ev, keys) in EXPERIMENTS.items():
        for g in want:
            todo = []
            for d in dirs[g]:
                if not os.path.exists(os.path.join(d, "config.json")):
                    continue
                cfg = json.load(open(os.path.join(d, "config.json")))
                sha = cfg.get("outputs", {}).get("sha256")
                e = read_eval(d)
                if e.get("z_sha256") == sha and all(k in e for k in keys):
                    continue                      # already done for this Z
                todo.append((d, sha))
            if not todo:
                print(f"[{name}] {g}: all cached, skip", flush=True)
                continue
            A, n = load(g)                        # preprocess ONCE per graph
            gd = pre(A, n)
            print(f"[{name}] {g}: preprocessed once, scoring {len(todo)} "
                  f"embeddings", flush=True)
            for d, sha in todo:
                Z = np.load(os.path.join(d, "Z.npy"))
                res = ev(Z, gd)
                e = read_eval(d)
                if e.get("z_sha256") != sha:
                    e = {"z_sha256": sha}
                e.update(res)
                write_eval(d, e)
            print(f"[{name}] {g}: wrote {len(todo)} eval.json", flush=True)
    print("SEQUENCER DONE", flush=True)


if __name__ == "__main__":
    main()
