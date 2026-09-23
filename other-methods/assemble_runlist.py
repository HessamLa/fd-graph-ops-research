#!/bin/env python3
"""Build the final run list for the comparison report.

The report puts the new methods next to three anchors already in the store:
deepwalk, node2vec, and fodiwalk's strongest cell this session (fdhop
walk_edges with the sqn optimiser). This scans the store for those anchors,
adds the new-method runs from run_all's list, drops duplicate
(graph, label, seed) records (keeping the newest), and writes one list.

Usage:  assemble_runlist.py <run_all_runlist.txt> <out_final.txt>
"""
import glob
import json
import os
import sys

ROOT = "/home/h/gm/fd-graph-ops-research"
STORE = os.path.join(ROOT, "data_cache", "embeddings")
GRAPHS = ("cora", "citeseer", "pubmed")


def label(cfg):
    name = cfg["method"]["name"]
    if name == "fodiwalk_precomp":
        return (f"fodiwalk {cfg.get('force', {}).get('law', '')} "
                f"{cfg.get('walker', {}).get('name', '')} "
                f"{cfg.get('optimizer', {}).get('rule', '')}".strip())
    if name.startswith("node2vec"):
        return "node2vec"
    return name


def is_anchor(cfg):
    """The baseline rows to pull from the store."""
    name = cfg["method"]["name"]
    if name in ("deepwalk",) or name.startswith("node2vec"):
        return True
    # fodiwalk anchor: fdhop, walk_edges, sqn -- the best cell this session
    return (name == "fodiwalk_precomp"
            and cfg.get("force", {}).get("law") == "fdhop"
            and cfg.get("walker", {}).get("name") == "walk_edges"
            and cfg.get("optimizer", {}).get("rule") == "sqn")


def main(new_list, out):
    dirs = []
    with open(new_list) as f:
        dirs += [ln.strip() for ln in f if ln.strip()]
    for g in GRAPHS:
        for d in glob.glob(os.path.join(STORE, g, "128", "*")):
            cfg_path = os.path.join(d, "config.json")
            if not os.path.exists(cfg_path):
                continue
            cfg = json.load(open(cfg_path))
            if is_anchor(cfg):
                dirs.append(d)

    # dedup by (graph, label, seed); keep the newest directory name
    best = {}
    for d in dirs:
        cfg = json.load(open(os.path.join(d, "config.json")))
        key = (cfg["dataset"]["name"], label(cfg), cfg["run"]["seed"])
        if key not in best or os.path.basename(d) > os.path.basename(best[key]):
            best[key] = d
    with open(out, "w") as f:
        for d in sorted(best.values()):
            f.write(d + "\n")
    print(f"{len(best)} unique (graph,label,seed) records -> {out}")
    # a short census
    from collections import Counter
    c = Counter((k[0], k[1]) for k in best)
    for (g, m), n in sorted(c.items()):
        print(f"  {g:9s} {m:35s} {n}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
