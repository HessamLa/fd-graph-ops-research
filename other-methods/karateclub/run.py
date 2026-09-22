#!/bin/env python3
"""Factorization, proximity and random-projection baselines, from published
packages: karateclub and nodevectors.

`evaluator/recommended-comparisons.md` names ProNE (method 5), NetSMF (6,
the sparse cousin of NetMF), LINE (4) and GraRep (optional). This runner
serves them from published code, not a rewrite:

  prone   nodevectors.ProNE        (factorization + spectral propagation)
  netmf   karateclub.NetMF         (matrix factorization of the PMI matrix)
  grarep  karateclub.GraRep        (k-step transition factorization)
  line    karateclub First+Second order LINE, concatenated to `dim`
  hope    karateclub.HOPE          (higher-order proximity, Katz)
  randne  karateclub.RandNE        (random projection of A powers)

karateclub pins an ancient numpy, so everything installs into
`.venv-karate` (`--no-deps` for karateclub and nodevectors) beside a modern
stack. This runner runs in THAT interpreter and writes only `Z.npy` +
`meta.json` to a handoff dir; scoring runs later in the shared `.venv`
(where `evaluator` lives) through `common/store_handoff.py`.

NODE ALIGNMENT: both backends read the `fodiwalk.make_graph.load`
adjacency with nodes `0..n-1` and return rows in that order (karateclub
from a networkx graph, nodevectors from a `csrgraph` built on the same
CSR). So row `i` of `Z` is the same node as for every other method.

MEMORY on this 7 GB box: `netmf`, `grarep`, `hope` form a dense `n x n`
matrix and OOM at pubmed's 19,717 nodes -- the caller runs them on cora and
citeseer only. `prone`, `line`, `randne` stay sparse/streamed and run on
all three.

Emit:  .venv-karate/bin/python run.py --emit --method prone \
           --graph cora --seed 42 --dim 128 --out /tmp/handoff
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2 ** 20


def load_A(adj_npz):
    """A as scipy CSR from a pre-saved `.npz`. The shared `.venv` writes it
    with `fodiwalk.make_graph.load`, so the numbering is fodiwalk's; this
    venv only reads it (scipy), thus it needs no fodiwalk and no jax."""
    import scipy.sparse as sp
    A = sp.csr_matrix(sp.load_npz(adj_npz))
    return A, int(A.shape[0])


def embed_karateclub(method, A, n, dim, seed):
    import networkx as nx
    G = nx.from_scipy_sparse_array(A)
    G.add_nodes_from(range(n))             # keep isolated nodes present
    if method == "netmf":
        from karateclub import NetMF
        model = NetMF(dimensions=dim, seed=seed)
    elif method == "grarep":
        from karateclub import GraRep
        order = 4
        model = GraRep(dimensions=dim // order, order=order, seed=seed)
    elif method == "hope":
        from karateclub import HOPE
        model = HOPE(dimensions=dim, seed=seed)
    elif method == "randne":
        from karateclub import RandNE
        model = RandNE(dimensions=dim, seed=seed)
    elif method == "line":
        # LINE's own paper concatenates first- and second-order halves.
        from karateclub import FirstOrderLINE, SecondOrderLINE
        half = dim // 2
        m1 = FirstOrderLINE(dimensions=half, seed=seed)
        m2 = SecondOrderLINE(dimensions=half, seed=seed)
        m1.fit(G)
        m2.fit(G)
        return np.hstack([m1.get_embedding(), m2.get_embedding()])
    else:
        raise ValueError(method)
    model.fit(G)
    return np.asarray(model.get_embedding())


def embed_nodevectors(method, A, n, dim, seed):
    import csrgraph as cg
    if method == "prone":
        from nodevectors import ProNE
        model = ProNE(n_components=dim, verbose=False)
    else:
        raise ValueError(method)
    G = cg.csrgraph(A, nodenames=list(range(n)))
    return np.asarray(model.fit_transform(G))


KARATE = {"netmf", "grarep", "hope", "randne", "line"}
NODEVEC = {"prone"}


def emit(method, graph, seed, dim, out, adj_npz):
    A, n = load_A(adj_npz)
    t0 = time.perf_counter()
    if method in KARATE:
        Z = embed_karateclub(method, A, n, dim, seed)
    elif method in NODEVEC:
        Z = embed_nodevectors(method, A, n, dim, seed)
    else:
        raise ValueError(method)
    secs = time.perf_counter() - t0
    Z = np.asarray(Z, dtype=np.float32)
    if Z.shape[0] != n:
        raise RuntimeError(f"{method}: Z has {Z.shape[0]} rows, graph has {n}")
    if Z.shape[1] != dim:                  # pad/trim to exactly dim
        Z = (np.pad(Z, ((0, 0), (0, dim - Z.shape[1]))) if Z.shape[1] < dim
             else Z[:, :dim])
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "Z.npy"), Z)
    json.dump({"method": method, "graph": graph, "seed": seed, "dim": dim,
               "seconds": secs, "peak_rss_mb": rss_mb(), "n": n},
              open(os.path.join(out, "meta.json"), "w"))
    print(f"[{method}] emitted {graph} seed={seed} Z{Z.shape} in {secs:.1f}s "
          f"peak {rss_mb():.0f}MB -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--method", required=True,
                    choices=("prone", "netmf", "grarep", "hope", "randne", "line"))
    ap.add_argument("--graph", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adj", required=True, help="pre-saved A.npz (fodiwalk numbering)")
    args = ap.parse_args()
    emit(args.method, args.graph, args.seed, args.dim, args.out, args.adj)
