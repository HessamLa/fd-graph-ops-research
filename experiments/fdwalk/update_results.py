#!/bin/env python3
"""update_results.py -- regenerate RESULTS.md from every log on disk.

Run it at any time; it reads what has finished and ignores what has not.
`CLAUDE.md` requires peak memory and the runtime of each stage in every
report, thus those columns are always emitted and a missing one prints
`--` rather than being dropped.

    .venv/bin/python experiments/fdwalk/update_results.py
"""
from __future__ import annotations
import glob, os, re, statistics as st, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "RESULTS.md")

# (title, glob, how to make the row label from the file stem)
GRIDS = [
    ("Grid A -- optimizer x learning rate",
     "grid_cora/A_lrscan_*.log",
     lambda s: tuple(s.replace("A_lrscan_", "").rsplit("_lr", 1))),
    ("Grid B -- cap x buckets X symmetric x directed",
     "grid_cora/B_*.log",
     lambda s: (re.sub(r"^B_|_s\d+$", "", s),)),
    ("Grid C -- buckets: far pairs, and the budget",
     "grid_cora/C_*.log",
     lambda s: (re.sub(r"^C_|_s\d+$", "", s),)),
    ("Grid D -- the second-order walk (p, q)",
     "grid_cora/D_*.log",
     lambda s: (re.sub(r"^D_|_s\d+$", "", s),)),
    ("LR ladder -- constant, 0.999 -> 0.9 -> 0.1",
     "lrladder/const_*.log",
     lambda s: tuple(s.replace("const_", "").rsplit("_lr", 1))),
    ("LR ladder -- linear decay, from 0.999 and 0.9",
     "lrladder/decay_*.log",
     lambda s: tuple(s.replace("decay_", "").rsplit("_lr", 1))),
    # 2026-08-18: PubMed carries only the Cora survivors (survivors.py),
    # thus its Grid A has no fixed point list and is read from disk as-is.
    ("PubMed A -- optimizer, Cora survivors only",
     "grid_pubmed/A_*.log",
     lambda s: (re.sub(r"^A_", "", s),)),
    ("PubMed B -- cap x buckets X symmetric x directed",
     "grid_pubmed/B_*.log",
     lambda s: (re.sub(r"^B_|_s\d+$", "", s),)),
    ("PubMed C -- buckets: far pairs, and the budget",
     "grid_pubmed/C_*.log",
     lambda s: (re.sub(r"^C_|_s\d+$", "", s),)),
    ("PubMed D -- the second-order walk (p, q)",
     "grid_pubmed/D_*.log",
     lambda s: (re.sub(r"^D_|_s\d+$", "", s),)),
    # com_youtube, 1,134,890 nodes. Added 2026-08-18, BEFORE that stage
    # starts, so the reporting session never has to guess the directory.
    # Until the first run lands these sections print "no completed run yet".
    ("com_youtube A -- optimizer",
     "grid_youtube/A_*.log",
     lambda s: tuple(s.replace("A_", "").rsplit("_lr", 1))),
    ("com_youtube B -- cap x buckets X symmetric x directed",
     "grid_youtube/B_*.log",
     lambda s: (re.sub(r"^B_|_s\d+$", "", s),)),
    ("com_youtube C -- buckets: far pairs",
     "grid_youtube/C_*.log",
     lambda s: (re.sub(r"^C_|_s\d+$", "", s),)),
    ("com_youtube D -- the second-order walk (p, q)",
     "grid_youtube/D_*.log",
     lambda s: (re.sub(r"^D_|_s\d+$", "", s),)),
]
COLS = [("t_aug", "aug s"), ("t_embed", "embed s"), ("rss", "peak MB"),
        ("auc", "AUC"), ("acc", "accuracy"), ("f1", "F1"),
        ("r2_dist", "hop R2"), ("dz", "final dZ"), ("dnnz", "D.nnz")]


def parse(path):
    for ln in open(path, errors="ignore"):
        if "RESULT\t" in ln:
            return dict(p.split("=", 1) for p in ln.strip().split("\t")
                        if "=" in p)
    return None


def cell(vals):
    if not vals:
        return "--"
    if any(v == "nan" for v in vals):
        return "**diverged**"
    try:
        f = [float(v) for v in vals]
    except ValueError:
        return "/".join(sorted(set(vals)))
    if len(f) == 1:
        return f"{f[0]:.4g}"
    return f"{st.mean(f):.4g} ±{st.pstdev(f):.2g}"


def main():
    out = ["# fdwalk -- results, regenerated automatically",
           "",
           f"Regenerated {time.strftime('%Y-%m-%dT%H:%M:%S%z')} by "
           "`update_results.py`. It reads every log on disk, thus a grid "
           "that is still running shows the points that have finished.",
           "",
           "Peak memory and the runtime of each stage are in every table, "
           "as `CLAUDE.md` requires. **On Cora they are indicative only**: "
           "a fixed ~1 GB of JAX and Python dominates the peak, and `D.nnz` "
           "is the better memory proxy at this size.",
           "",
           "A row with several seeds shows `mean ±pstdev`. A configuration "
           "that produced a non-finite `Z` shows **diverged**.", ""]
    for title, pat, label in GRIDS:
        rows = defaultdict(list)
        for f in sorted(glob.glob(os.path.join(RES, pat))):
            r = parse(f)
            if r:
                rows[label(os.path.basename(f)[:-4])].append(r)
        out.append(f"## {title}")
        out.append("")
        if not rows:
            out += ["_no completed run yet._", ""]
            continue
        nk = max(len(k) for k in rows)
        head = [f"key{i+1}" for i in range(nk)] + ["runs"] + [c[1] for c in COLS]
        out.append("| " + " | ".join(head) + " |")
        out.append("| " + " | ".join("---" for _ in head) + " |")
        for k in sorted(rows):
            rr = rows[k]
            line = list(k) + ["--"] * (nk - len(k)) + [str(len(rr))]
            for key, _ in COLS:
                line.append(cell([r[key] for r in rr if key in r]))
            out.append("| " + " | ".join(line) + " |")
        out.append("")
    done = sum(1 for _, p, _ in GRIDS
               for f in glob.glob(os.path.join(RES, p)) if parse(f))
    out.append(f"_{done} completed runs on disk._")
    open(OUT, "w").write("\n".join(out) + "\n")
    print(f"wrote {OUT} ({done} runs)")


main()
