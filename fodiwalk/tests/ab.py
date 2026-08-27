"""ab.py -- the A/B record for switching fodiwalk onto the shared packages.

`fodiwalk` is being changed to use `forcedirected/` for the engine and
`evaluator/` for the scoring, in place of its own copies. That is a code
move, so THE NUMBERS MUST NOT CHANGE.

`golden.py` already covers the graph work and a short run. It does not
cover SCORING, which is what `evaluator/` replaces. This file covers the
whole path: load a graph, build the pairs, run the embedding, score it.

Use::

    .venv/bin/python -m fodiwalk.tests.ab --write   # record side A
    .venv/bin/python -m fodiwalk.tests.ab --check   # compare side B to it

`--write` refuses to overwrite unless `--force` is given. Re-recording
after a change compares the change against itself, which proves nothing.

Runs on the CPU. A row wider than `k_max` splits into virtual rows that
share an owner id, and the add-into-place on a GPU then happens in a free
order (`forcedirected/PARITY.md` section 5). On the CPU the order is
fixed, so the same run repeats exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

BASELINE = pathlib.Path(__file__).with_name("ab_baseline.json")

# One case for each scoring path that the switch can disturb. `feature`
# picks how a pair becomes a row of numbers: `distance` is one column,
# `vector` is one column for each dimension.
CASES = {
    "nbr_walk_d32_e40": dict(pairs="nbr_walk", weight="min_gap",
                             dim=32, epochs=40, lr=1.0, optim="plain"),
    "walk_d32_e40":     dict(pairs="walk", weight="min_gap",
                             dim=32, epochs=40, lr=1.0, optim="plain"),
    "nbr_walk_adam":    dict(pairs="nbr_walk", weight="min_gap",
                             dim=32, epochs=40, lr=0.1, optim="adam"),
    "walk_buckets":     dict(pairs="walk", policy="buckets", far=2000,
                             dim=32, epochs=40, lr=1.0, optim="plain"),
}

# The fields a run reports. `t_embed` is wall time and it is NOT compared.
SCORES = ("acc", "f1", "auc", "r2_dist", "mae_dist", "r2_vec",
          "dz", "dnnz", "lp_pairs", "hop_pairs", "diverged")


def _num(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), 10)
    return x


def one(name: str, cfg: dict) -> dict:
    """One whole run: graph, pairs, embedding, both scoring tasks."""
    from fodiwalk.make_graph import load
    from fodiwalk.tests.harness import run

    A, n = load("cora")
    cfg = dict(cfg)
    out = run(A, n, seed=42, lp_pairs=20_000, hop_sources=100,
              hop_pairs=8_000, **cfg)
    rec = {k: _num(out[k]) for k in SCORES if k in out}
    rec["plan"] = {k: _num(v) for k, v in out["plan"].items()}
    return rec


def build() -> dict:
    return {"dataset": "cora",
            "cases": {k: one(k, c) for k, c in CASES.items()}}


def compare(new, ref, path="", rtol=1e-4) -> list:
    """Every field that differs. Scores get `rtol`; counts must be exact."""
    bad = []
    if isinstance(ref, dict):
        for k in sorted(set(ref) | set(new)):
            if k not in ref:
                bad.append(f"{path}.{k}: ABSENT -> {new[k]!r}")
            elif k not in new:
                bad.append(f"{path}.{k}: {ref[k]!r} -> ABSENT")
            else:
                bad += compare(new[k], ref[k], f"{path}.{k}", rtol)
        return bad
    if isinstance(ref, float) and isinstance(new, (int, float)):
        if abs(new - ref) <= rtol * max(abs(ref), 1e-12):
            return bad
    if new != ref:
        bad.append(f"{path}: {ref!r} -> {new!r}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    snap = build()
    if a.write:
        if BASELINE.exists() and not a.force:
            print(f"{BASELINE} exists. --force to overwrite. Re-recording "
                  f"after a change compares the change against itself.")
            return 2
        BASELINE.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
        print(f"wrote {BASELINE}")
        for name, rec in snap["cases"].items():
            print(f"  {name:18s} acc={rec['acc']:.4f} auc={rec['auc']:.4f} "
                  f"r2_dist={rec['r2_dist']:.4f} dnnz={rec['dnnz']}")
        return 0

    ref = json.loads(BASELINE.read_text())
    bad = compare(snap, ref)
    if bad:
        print(f"A/B MISMATCH, {len(bad)} fields:")
        for line in bad[:60]:
            print("  " + line)
        return 1
    print("A/B OK -- every score matches side A")
    for name, rec in snap["cases"].items():
        print(f"  {name:18s} acc={rec['acc']:.4f} auc={rec['auc']:.4f} "
              f"r2_dist={rec['r2_dist']:.4f} dnnz={rec['dnnz']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
