#!/bin/env python3
"""Score stored embeddings and print per-dataset comparison tables.

One place for the seven columns of the comparison report: Spearman rho,
hop R2, recall@10, LP AUC, LP f1, community NMI, community ARI. It reads a
list of run directories (each holds `Z.npy` and `config.json`), groups by
(graph, method), and prints mean +/- std over seeds -- the format of
`evaluator/reports/260917-iclr2027-preliminary.md`.

Metric definitions are copied verbatim from this session's scripts, not
invented here:
  rho        `agentic-log/recall-and-rho/extra_metrics.py` -- Spearman of
             embedding distance vs hop distance, hop >= 2, on the SAME
             pairs the `fodiwalk_dist` protocol scores hop R2 on.
  recall@10  micro average, denominator min(degree, 10)
             (`evaluator/reports/260904-store-scorecard.md:296`).
  NMI, ARI   `agentic-log/recall-and-rho/community_nmi.py` -- Louvain on A
             (networkx, seed 42, once per graph) vs k-means on Z at the
             Louvain community count.
  hop R2, LP AUC, LP f1  read from the stored `metrics` (evaluator
             `fodiwalk_dist`, the value `make_embedding.py --score` wrote).

Usage:  score_report.py <run_dirs.txt> [<report.md>]
A run-dirs file is one directory path per line. Without a report path the
tables go to stdout.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

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

DA_CFG = PROTOCOLS["fodiwalk_dist"].da
LOUVAIN_SEED = 42

_graph = {}
_louvain = {}


def graph(name):
    if name not in _graph:
        _graph[name] = load(name)
    return _graph[name]


def rho(name, Z, seed):
    A, n = graph(name)
    rng = np.random.default_rng(seed)
    src = rng.choice(n, size=min(DA_CFG.n_sources, n), replace=False)
    pr, h = pairs_for_hops(A, n, DA_CFG.n_pairs, rng, src, DA_CFG.pair_draw)
    keep = np.isfinite(h) & (h >= DA_CFG.min_hop)
    pr, h = pr[keep], h[keep]
    d = np.linalg.norm(Z[pr[:, 0]] - Z[pr[:, 1]], axis=1)
    r, _ = spearmanr(d, h)
    return float(r)


def recall_at_10(name, Z):
    A, n = graph(name)
    deg = np.diff(A.indptr)
    q = np.flatnonzero(deg > 0)
    idx, _d, _p = knn(Z, q, k=10)
    hits = denom = 0
    for row, u in enumerate(q):
        true = A.indices[A.indptr[u]:A.indptr[u + 1]]
        hits += np.intersect1d(idx[row], true, assume_unique=False).size
        denom += min(true.size, 10)
    return hits / denom


def louvain_labels(name):
    if name in _louvain:
        return _louvain[name]
    A, n = graph(name)
    G = nx.from_scipy_sparse_array(A)
    comms = nx.algorithms.community.louvain_communities(G, seed=LOUVAIN_SEED)
    lab = np.empty(n, dtype=np.int64)
    for cid, members in enumerate(comms):
        lab[list(members)] = cid
    _louvain[name] = (lab, len(comms))
    return _louvain[name]


def community(name, Z, seed):
    lab, k = louvain_labels(name)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Z)
    return (normalized_mutual_info_score(lab, km.labels_),
            adjusted_rand_score(lab, km.labels_))


def score_dir(d):
    cfg = json.load(open(f"{d}/config.json"))
    Z = np.load(f"{d}/Z.npy")
    g = cfg["dataset"]["name"]
    seed = cfg["run"]["seed"]
    method = cfg["method"]["name"]
    m = cfg.get("metrics", {})
    nmi, ari = community(g, Z, seed)
    return {"graph": g, "method": method, "seed": seed,
            "rho": rho(g, Z, seed), "hop_r2": m.get("rf_r2", float("nan")),
            "recall_at_10": recall_at_10(g, Z),
            "lp_auc": m.get("auc", float("nan")),
            "lp_f1": m.get("f1_score", float("nan")),
            "nmi": nmi, "ari": ari,
            "seconds": cfg.get("environment", {}).get("seconds", 0.0),
            "peak_rss_mb": cfg.get("environment", {}).get("peak_rss_mb", 0.0)}


COLS = [("rho", "Spearman rho"), ("hop_r2", "hop R2 (rf)"),
        ("recall_at_10", "recall@10"), ("lp_auc", "LP AUC"),
        ("lp_f1", "LP f1_score"), ("nmi", "NMI"), ("ari", "ARI")]
GRAPH_ORDER = ["cora", "citeseer", "pubmed"]


def cell(vals):
    v = np.array(vals, dtype=float)
    if np.isnan(v).all():
        return "TBD"
    return f"{np.nanmean(v):.4f} ± {np.nanstd(v):.4f}"


def tables(rows):
    by_gm = defaultdict(list)
    for r in rows:
        by_gm[(r["graph"], r["method"])].append(r)
    graphs = [g for g in GRAPH_ORDER if any(k[0] == g for k in by_gm)]
    graphs += sorted({k[0] for k in by_gm} - set(graphs))
    out = []
    for g in graphs:
        out.append(f"\n**{g}, dim 128**\n")
        out.append("| method | seeds | " + " | ".join(h for _, h in COLS) + " |")
        out.append("|---" * (len(COLS) + 2) + "|")
        methods = sorted({k[1] for k in by_gm if k[0] == g},
                         key=lambda mth: -np.nanmean(
                             [r["rho"] for r in by_gm[(g, mth)]]))
        for mth in methods:
            rs = by_gm[(g, mth)]
            cells = [cell([r[k] for r in rs]) for k, _ in COLS]
            out.append(f"| {mth} | {len(rs)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    run_dirs_file = sys.argv[1]
    rows = []
    for line in open(run_dirs_file):
        d = line.strip()
        if d:
            rows.append(score_dir(d))
    md = tables(rows)
    if len(sys.argv) > 2:
        json.dump(rows, open(sys.argv[2].replace(".md", ".json"), "w"), indent=2)
        open(sys.argv[2], "w").write(md)
        print(f"wrote {sys.argv[2]}", file=sys.stderr)
    print(md)


if __name__ == "__main__":
    main()
