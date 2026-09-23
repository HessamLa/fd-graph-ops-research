#!/bin/env python3
"""Collect every stored run for the comparison methods into one run list.

Scans the whole dim-128 store, keeps the records whose method label is in
the comparison set (so stray fodiwalk cells -- nbr_walk, plain, fdlinear,
nesterov -- are left out), dedups by (graph, label, seed), and writes the
list plus a per-(graph,label) seed census.

Usage:  collect_store.py <out_runlist.txt>
"""
import glob
import json
import os
import sys
from collections import Counter

ROOT = "/home/h/gm/fd-graph-ops-research"
STORE = os.path.join(ROOT, "data_cache", "embeddings")
GRAPHS = ("cora", "citeseer", "pubmed")

# the exact labels the report shows
TARGETS = {
    "fodiwalk fdhop walk_edges sqn", "deepwalk", "node2vec",
    "landmark_mds", "laplacian_eigenmaps", "prone", "randne", "line",
    "netmf", "grarep", "hope", "tforce2vec", "rforce2vec", "force2vec",
}


def label(cfg):
    name = cfg["method"]["name"]
    if name == "fodiwalk_precomp":
        return (f"fodiwalk {cfg.get('force', {}).get('law', '')} "
                f"{cfg.get('walker', {}).get('name', '')} "
                f"{cfg.get('optimizer', {}).get('rule', '')}".strip())
    if name.startswith("node2vec"):
        return "node2vec"
    return name


def main(out):
    best = {}
    for g in GRAPHS:
        for d in glob.glob(os.path.join(STORE, g, "128", "*")):
            cfgp = os.path.join(d, "config.json")
            if not os.path.exists(cfgp):
                continue
            cfg = json.load(open(cfgp))
            lab = label(cfg)
            if lab not in TARGETS:
                continue
            key = (cfg["dataset"]["name"], lab, cfg["run"]["seed"])
            if key not in best or os.path.basename(d) > os.path.basename(best[key]):
                best[key] = d
    with open(out, "w") as f:
        for d in sorted(best.values()):
            f.write(d + "\n")
    print(f"{len(best)} records -> {out}")
    census = Counter((k[0], k[1]) for k in best)
    for (g, m), n in sorted(census.items()):
        print(f"  {g:9s} {m:32s} {n}")


if __name__ == "__main__":
    main(sys.argv[1])
