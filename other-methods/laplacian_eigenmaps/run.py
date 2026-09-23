#!/bin/env python3
"""Laplacian Eigenmaps, from scikit-learn.

`sklearn.manifold.SpectralEmbedding` with `affinity="precomputed"` on the
graph's own adjacency. It returns the eigenvectors of the normalized graph
Laplacian, which is the Belkin-Niyogi (2003) embedding. Published code, no
new algorithm here.

Two graph facts the method must survive, both named in
`evaluator/recommended-comparisons.md` (method 7):
  disconnected components  every citation graph here has more than one.
                           `SpectralEmbedding` warns and embeds the whole
                           graph; kept.
  isolated (degree-0) nodes  citeseer has 48. The normalized Laplacian
                           divides by degree, so a degree-0 row is a divide
                           by zero. Fix: add a tiny self-loop to EVERY node
                           (`A += eps*I`, eps=1e-6) before the solve. This
                           is standard Laplacian regularization; it moves
                           no connected node and it gives an isolated node a
                           finite, self-only row instead of a NaN. Recorded
                           in the run `notes`.

The embedding is (up to eigenvector sign and solver start vector) a
property of the graph, not of the seed. Three seeds are still run: the
solver start vector and the evaluation's k-means vary, so the seed spread
here measures solver and scoring noise, near zero, not method noise.

Run:  .venv/bin/python other-methods/laplacian_eigenmaps/run.py \
          --graph cora --seed 42 --dim 128
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

ROOT = "/home/h/gm/fd-graph-ops-research"
sys.path.insert(0, os.path.join(ROOT, "other-methods"))
from common.store import load_graph, save_scored

EPS = 1e-6


def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2 ** 20


def embed(A, n, dim, seed):
    from sklearn.manifold import SpectralEmbedding
    Areg = A + EPS * sp.eye(n, format="csr")   # self-loops save degree-0 rows
    model = SpectralEmbedding(n_components=dim, affinity="precomputed",
                              random_state=seed, eigen_solver="arpack")
    Z = model.fit_transform(Areg)
    return np.ascontiguousarray(Z, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dim", type=int, default=128)
    args = ap.parse_args()

    A, n = load_graph(args.graph, seed=args.seed)
    n_iso = int((np.diff(A.indptr) == 0).sum())
    t0 = time.perf_counter()
    Z = embed(A, n, args.dim, args.seed)
    secs = time.perf_counter() - t0
    save_scored(Z, graph=args.graph, method="laplacian_eigenmaps",
                seed=args.seed, A=A, n=n, seconds=secs, peak_rss_mb=rss_mb(),
                params={"package": "scikit-learn", "affinity": "precomputed",
                        "eigen_solver": "arpack", "self_loop_eps": EPS},
                notes=f"{n_iso} isolated nodes regularized with eps*I self-loops")


if __name__ == "__main__":
    main()
