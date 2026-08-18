#!/bin/env python3
"""grid_table.py -- one markdown table from a set of RESULT lines.

`CLAUDE.md` asks that every experiment report peak memory and the runtime
of each stage, thus those columns are NOT optional here and the script
fails loudly if a log lacks them.

    .venv/bin/python experiments/fdwalk/grid_table.py <dir> <prefix> [group_keys]
"""
from __future__ import annotations
import sys, glob, os, re
from collections import defaultdict

COLS = [("auc", "AUC"), ("acc", "accuracy"), ("f1", "F1"),
        ("r2_dist", "hop R2"), ("dz", "final dZ"),
        ("dnnz", "D.nnz"), ("t_aug", "aug s"), ("t_embed", "embed s"),
        ("rss", "peak MB")]


def rows(d, prefix):
    out = []
    for f in sorted(glob.glob(os.path.join(d, prefix + "*.log"))):
        for ln in open(f):
            if "\tRESULT\t" in ln or ln.startswith("[fdwalk] RESULT"):
                kv = dict(p.split("=", 1) for p in ln.strip().split("\t")
                          if "=" in p)
                kv["_name"] = os.path.basename(f)[:-4]
                out.append(kv)
    return out


def agg(vals):
    import statistics as st
    v = [float(x) for x in vals]
    if len(v) == 1:
        return f"{v[0]:.4g}"
    return f"{st.mean(v):.4g} ±{st.pstdev(v):.2g}"


def main():
    d, prefix = sys.argv[1], sys.argv[2]
    keys = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    rs = rows(d, prefix)
    if not rs:
        print(f"no RESULT lines under {d}/{prefix}*"); return
    groups = defaultdict(list)
    for r in rs:
        g = (tuple(r.get(k, "-") for k in keys) if keys
             else (re.sub(r"_s\d+$", "", r["_name"]),))
        groups[g].append(r)
    head = (keys or ["variant"]) + ["seeds"] + [c[1] for c in COLS]
    print("| " + " | ".join(head) + " |")
    print("| " + " | ".join("---" for _ in head) + " |")
    for g, rr in sorted(groups.items()):
        cells = list(g) + [str(len(rr))]
        for k, _ in COLS:
            if k not in rr[0]:
                cells.append("MISSING"); continue
            cells.append(agg([r[k] for r in rr]))
        print("| " + " | ".join(cells) + " |")


main()
