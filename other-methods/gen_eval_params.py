#!/bin/env python3
"""Write the evaluation parameters that produced the comparison tables.

Pulls the link-prediction and hop-regression settings LIVE from the
`fodiwalk_dist` protocol (so they cannot drift from what actually ran) and
records the sequencer's own settings for rho, recall@10 and community.
Writes `other-methods/results/eval_params.json`.
"""
import dataclasses
import json
import os
import sys

sys.path.insert(0, "/home/h/gm/fd-graph-ops-research")
from evaluator.config import PROTOCOLS, RECON_MAX_N

P = PROTOCOLS["fodiwalk_dist"]
OUT = "/home/h/gm/fd-graph-ops-research/other-methods/results/eval_params.json"

params = {
    "protocol": "fodiwalk_dist",
    "dim": 128,
    "node_numbering": "fodiwalk.make_graph.load (one order for every method)",
    "seeds": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52],
    "eval_seed_note": ("LP and hop-regression were scored at embed time with "
                       "the EMBEDDING seed (stored in config.json); rho, "
                       "recall@10 and community use a FIXED eval seed 42."),

    "link_prediction": {
        **dataclasses.asdict(P.lp),
        "train_test_split": f"{1 - P.lp.test_size:.0%}/{P.lp.test_size:.0%}",
        "classifier": f"random forest, {P.lp.n_estimators} trees",
        "features": P.lp.feature,
        "negatives": f"{P.lp.neg_ratio}:1 (balanced), drawn as {P.lp.neg_draw}",
        "positives": f"graph edges, {P.lp.pos_draw}, capped at {P.lp.max_pairs}",
        "held_out": ("NO -- the embedding saw every edge, so this is edge "
                     "reconstruction, not held-out link prediction"),
        "scores": ["accuracy", "precision", "recall", "f1_score", "auc"],
    },

    "hop_regression": {
        **dataclasses.asdict(P.da),
        "train_test_split": f"{1 - P.da.test_size:.0%}/{P.da.test_size:.0%}",
        "pairs": f"{P.da.n_pairs} from {P.da.n_sources} BFS sources, "
                 f"min_hop {P.da.min_hop}, draw {P.da.pair_draw}",
        "feature": f"one scalar: {P.da.metric} distance in Z",
        "target": "true BFS hop distance",
        "reported_metric": "hop R2 = random forest R2 on the test split",
    },

    "spearman_rho": {
        "definition": "Spearman between euclidean Z-distance and hop distance",
        "pairs": f"{P.da.n_pairs} from {P.da.n_sources} BFS sources, "
                 f"min_hop {P.da.min_hop} -- the SAME sample as hop_regression",
        "eval_seed": 42,
    },

    "recall_at_10": {
        "definition": "share of a node's true neighbours among its 10 "
                      "euclidean-nearest nodes",
        "k": 10,
        "queries": "every node of degree > 0",
        "average": "micro, denominator sum(min(degree, 10))",
        "knn": f"exact brute-force (all graphs are below RECON_MAX_N={RECON_MAX_N})",
        "self_excluded": True,
    },

    "community": {
        "reference": "Louvain on A (networkx louvain_communities)",
        "louvain_seed": 42,
        "louvain_resolution": 1.0,
        "louvain_k": {"cora": 105, "citeseer": 471, "pubmed": 45},
        "clusters": "k-means on Z, k = Louvain community count",
        "kmeans_n_init": 10,
        "kmeans_random_state": 42,
        "scores": ["NMI (normalized_mutual_info_score)",
                   "ARI (adjusted_rand_score)"],
    },

    "machine": "8 cores, 7 GB RAM, no GPU used",
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(params, open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
print(json.dumps(params, indent=1))
