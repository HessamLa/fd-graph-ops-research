#!/bin/env python3
"""survivors.py -- the (optim, lr) points that did NOT diverge.

Instruction of the user, 2026-08-18: carry to the next graph only the
parameter values that do not diverge. This reads the learning-rate ladder
and prints one `optim lr` pair per line, for the driver to consume.

A rule that matters for the ladder: it descends 0.999 -> 0.9 -> 0.1 and
STOPS at the first rate that survives, thus a rung that was never run is
not a survivor and not a failure -- it is absent, and this script must not
invent it.

    .venv/bin/python experiments/fdwalk/survivors.py [dir]
"""
from __future__ import annotations
import glob, os, re, sys

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "lrladder")

best = {}
for f in sorted(glob.glob(os.path.join(d, "const_*.log"))):
    stem = os.path.basename(f)[:-4].replace("const_", "")
    optim, lr = stem.rsplit("_lr", 1)
    txt = open(f, errors="ignore").read()
    if "RESULT" not in txt:
        continue
    if "diverged=1" in txt:
        continue
    # the ladder tries the LARGEST rate first, thus the first survivor is
    # the one the instruction asks to keep.
    if optim not in best or float(lr) > float(best[optim]):
        best[optim] = lr
for o, lr in sorted(best.items()):
    print(o, lr)
