#!/bin/env python3
"""summarize.py -- the RESULT lines of a grid into a markdown table.

    .venv/bin/python experiments/fdwalk/summarize.py results/g1_cora.tsv

No number goes into FINDINGS.md by hand. This script reads the logs and it
gives the mean and the standard deviation over the seeds. PLAN.md section 5
says that a difference smaller than that standard deviation is not a result.
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

COLS = [("dnnz", "D.nnz", "{:.0f}"), ("t_aug", "aug s", "{:.1f}"),
        ("t_embed", "embed s", "{:.1f}"), ("auc", "AUC", "{:.4f}"),
        ("acc", "accuracy", "{:.4f}"),
        ("r2_dist", "R2 dist", "{:.3f}"), ("r2_vec", "R2 vec", "{:.3f}"),
        ("h2_exact", "H2 exact", "{:.3f}"), ("dz", "final dZ", "{:.3f}")]


def read(path):
    runs = []
    for line in open(path):
        if "RESULT" not in line:
            continue
        row = {}
        for field in line.split("RESULT", 1)[1].strip().split("\t"):
            if "=" in field:
                k, v = field.split("=", 1)
                row[k] = v
        runs.append(row)
    return runs


def main(path):
    runs = read(path)
    groups = defaultdict(list)
    for r in runs:
        groups[(r["pairs"], r["weight"], r["optim"])].append(r)

    head = "| pairs | weight | optim | seeds | " + \
        " | ".join(c[1] for c in COLS) + " |"
    print(head)
    print("| --- " * (4 + len(COLS)) + "|")
    for key in sorted(groups):
        rows = groups[key]
        cells = []
        for name, _, fmt in COLS:
            try:
                v = np.array([float(r[name]) for r in rows if r.get(name) not in
                              (None, "na")])
            except ValueError:
                cells.append("na")
                continue
            if v.size == 0:
                cells.append("na")
            elif v.size == 1:
                cells.append(fmt.format(v[0]))
            else:
                cells.append(fmt.format(v.mean()) + " ±" +
                             fmt.format(v.std()).lstrip("0"))
        print(f"| {key[0]} | {key[1]} | {key[2]} | {len(rows)} | "
              + " | ".join(cells) + " |")
    print(f"\n{len(runs)} runs, {len(groups)} variants.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "experiments/fdwalk/results/g1_cora.tsv")
