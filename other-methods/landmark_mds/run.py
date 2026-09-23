#!/bin/env python3
"""Landmark shortest-path MDS -- the graph-distance reference baseline.

`evaluator/recommended-comparisons.md` (method 8) asks for this as an
"explicitly specified experimental baseline, not a claim that one particular
landmark implementation is the canonical Isomap algorithm." So it is written
here, to that specification, not taken from a package.

Method (de Silva & Tenenbaum, landmark MDS):
  1. Pick `L` landmark nodes at random (the seed picks them).
  2. Exact shortest-path (hop) distances from every landmark to every node,
     `scipy.sparse.csgraph.dijkstra` unweighted -> a dense (L, n) array.
  3. Classical MDS on the L x L landmark block gives landmark coordinates.
  4. Project every node from its distances to the landmarks (the standard
     landmark out-of-sample formula).

Unlike every other method here, this baseline reads the TRUE shortest-path
distances directly, so it is the reference for how well distance geometry
CAN be held in `dim` coordinates, not a learned embedding.

Two graph facts, both recorded in the run `notes`:
  disconnected  an unreachable node has an infinite hop distance. It is
                capped at (max finite distance + 1), so the pair is far but
                finite. A different cap moves those rows only.
  isolated      a degree-0 node is unreachable from every landmark, so it
                sits at the far cap from all of them -- one shared far point.

`L = max(2*dim, 200)` so the L x L block has more than `dim` usable axes.

Run:  .venv/bin/python other-methods/landmark_mds/run.py \
          --graph cora --seed 42 --dim 128
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.sparse.csgraph import dijkstra

ROOT = "/home/h/gm/fd-graph-ops-research"
sys.path.insert(0, os.path.join(ROOT, "other-methods"))
from common.store import load_graph, save_scored


def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2 ** 20


def embed(A, n, dim, seed):
    rng = np.random.default_rng(seed)
    n_land = max(2 * dim, 200)
    n_land = min(n_land, n)
    land = rng.choice(n, size=n_land, replace=False)

    # (L, n) exact hop distances. `unweighted` = BFS on the 0/1 graph.
    D = dijkstra(A, directed=False, indices=land, unweighted=True)
    finite = np.isfinite(D)
    cap = D[finite].max() + 1 if finite.any() else 1.0
    D = np.where(finite, D, cap)
    Dll = D[:, land]                       # L x L landmark block

    # classical MDS on the landmark block
    Dll2 = Dll ** 2
    J = np.eye(n_land) - 1.0 / n_land
    B = -0.5 * J @ Dll2 @ J
    w, U = np.linalg.eigh(B)               # ascending
    order = np.argsort(w)[::-1]
    w, U = w[order], U[:, order]
    pos = w > 1e-9
    d = min(dim, int(pos.sum()))
    w_d, U_d = w[:d], U[:, :d]
    sqrt_w = np.sqrt(w_d)

    # project every node from its landmark distances (landmark MDS formula)
    mean_col = Dll2.mean(axis=0)           # length L
    Dn2 = D ** 2                           # L x n
    L_pinv = (U_d / sqrt_w).T              # d x L  == diag(1/sqrt_w) U_d^T
    Z = (-0.5 * L_pinv @ (Dn2 - mean_col[:, None])).T   # n x d
    if d < dim:                            # pad if the block gave < dim axes
        Z = np.pad(Z, ((0, 0), (0, dim - d)))
    return np.ascontiguousarray(Z, dtype=np.float32), n_land, float(cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dim", type=int, default=128)
    args = ap.parse_args()

    A, n = load_graph(args.graph, seed=args.seed)
    t0 = time.perf_counter()
    Z, n_land, cap = embed(A, n, args.dim, args.seed)
    secs = time.perf_counter() - t0
    save_scored(Z, graph=args.graph, method="landmark_mds", seed=args.seed,
                A=A, n=n, seconds=secs, peak_rss_mb=rss_mb(),
                params={"package": "scipy (own landmark-MDS)",
                        "landmarks": n_land, "unreachable_cap": cap},
                notes=f"{n_land} random landmarks; unreachable hop capped at {cap}")


if __name__ == "__main__":
    main()
