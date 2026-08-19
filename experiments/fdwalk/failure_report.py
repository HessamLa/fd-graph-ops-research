#!/bin/env python3
"""failure_report.py -- why each com_youtube run died, for later analysis.

`/usr/bin/time -v` writes a .time file beside every log. A run killed by
the OOM killer leaves "Command terminated by signal 9" there together with
the peak RSS it reached, thus the evidence survives even though the process
left no traceback. This collects it.

    .venv/bin/python experiments/fdwalk/failure_report.py
"""
from __future__ import annotations
import glob, os, re, csv

D = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "results", "grid_youtube")


def last_stage(log):
    """The last thing the run reported before it died."""
    keep = ""
    for ln in open(log, errors="ignore"):
        if ln.startswith("[fdwalk]"):
            keep = ln.strip()[9:]
    return keep[:110]


def main():
    rows = []
    for t in sorted(glob.glob(os.path.join(D, "*.time"))):
        name = os.path.basename(t)[:-5]
        txt = open(t, errors="ignore").read()
        sig = re.search(r"Command terminated by signal (\d+)", txt)
        rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt)
        wall = re.search(r"Elapsed \(wall clock\).*?:\s*([\d:.]+)", txt)
        log = os.path.join(D, name + ".log")
        ok = os.path.exists(log) and "RESULT" in open(log, errors="ignore").read()
        rows.append({
            "config": name,
            "outcome": "ok" if ok else (f"killed sig{sig.group(1)}" if sig
                                        else "failed"),
            "peak_rss_mb": int(rss.group(1)) // 1024 if rss else "",
            "wall": wall.group(1) if wall else "",
            "died_after": "" if ok else last_stage(log) if os.path.exists(log) else "",
        })
    if not rows:
        print("no .time files yet"); return
    w = max(len(r["config"]) for r in rows)
    print(f"{'config':<{w}}  {'outcome':<14} {'peak MB':>8} {'wall':>9}  died after")
    for r in rows:
        print(f"{r['config']:<{w}}  {r['outcome']:<14} {r['peak_rss_mb']:>8} "
              f"{r['wall']:>9}  {r['died_after']}")
    out = os.path.join(D, "failures.csv")
    with open(out, "w", newline="") as f:
        c = csv.DictWriter(f, fieldnames=list(rows[0]))
        c.writeheader(); c.writerows(rows)
    print(f"\n-> {out}")


main()
