#!/bin/env python3
"""Force2Vec family, from the official HipGraph/Force2Vec C++ code.

`evaluator/recommended-comparisons.md` (methods 1-3) asks for the Force2Vec
family and names the exact options of the published binary: option 1 is the
quadratic all-pairs Force2Vec, option 5 is tForce2Vec (t-distribution with
negative sampling), option 7 is rForce2Vec (sigmoid with a semi-random
walk). This wrapper runs that binary; it writes no embedding algorithm.

NODE ALIGNMENT, the thing that makes the numbers comparable. The binary
reads a Matrix Market `.mtx` file with its own node ids and writes an
`.embd` file keyed on those ids. So this wrapper builds the `.mtx` FROM the
`fodiwalk.make_graph.load` adjacency, 1-indexed (`fodiwalk id + 1`), and
reads the `.embd` back into `Z[id-1]`. Thus row `i` of `Z` is the same node
here as for every other method, and the shared scorer needs no special
case. Feeding the repo's bundled `cora.mtx` instead would embed a DIFFERENT
node numbering and every score would be wrong with nothing to warn.

The binary exposes no seed flag AND it is deterministic (two runs give a
byte-identical `.embd`). So rerunning cannot give error bars. To get real
run-to-run variance, the `--seed` here PERMUTES the node order: relabel
nodes by the seed, embed, map back. The embedding differs because the
binary's initialisation and sampling order follow the node numbering, while
the graph is the same. This measures the method's stability, the same thing
an embedding seed measures for the other methods.

`--seed 42` is the identity permutation, so it reproduces the original
single-run record; 43 and 44 are random permutations.

Run:  .venv/bin/python other-methods/force2vec/run.py \
          --graph cora --option 5 --dim 128 --iter 1200 --seed 43
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = "/home/h/gm/fd-graph-ops-research"
HERE = os.path.dirname(os.path.abspath(__file__))
BINARY = os.path.join(HERE, "src", "bin", "Force2Vec")
sys.path.insert(0, os.path.join(ROOT, "other-methods"))
from common.store import load_graph, save_scored

# option -> the report's method label (recommended-comparisons.md methods 1-3)
OPTION_NAME = {1: "force2vec", 5: "tforce2vec", 7: "rforce2vec"}


def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2 ** 20


def write_mtx(A, n, path):
    """Upper-triangle edges of A, 1-indexed, Matrix Market pattern
    symmetric. One line per undirected edge (`i < j`)."""
    U = A.tocoo()
    m = U.row < U.col
    r, c = U.row[m] + 1, U.col[m] + 1
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
        f.write(f"{n} {n} {r.size}\n")
        np.savetxt(f, np.c_[r, c], fmt="%d")


def read_embd(path, n, dim):
    """`.embd` -> Z aligned to fodiwalk ids. Line `id x1..xd`, id 1-indexed.
    A node the binary dropped (an isolated node can be absent) stays zero."""
    Z = np.zeros((n, dim), dtype=np.float32)
    with open(path) as f:
        header = f.readline().split()
        file_dim = int(header[1])
        if file_dim != dim:
            raise RuntimeError(f"{path}: dim {file_dim} != requested {dim}")
        for line in f:
            parts = line.split()
            if not parts:
                continue
            i = int(parts[0]) - 1
            Z[i] = np.asarray(parts[1:1 + dim], dtype=np.float32)
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--option", type=int, required=True, choices=(1, 5, 7))
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--iter", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--nsamples", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42,
                    help="node-permutation seed (binary is deterministic); "
                         "42 is identity")
    args = ap.parse_args()
    if not os.path.exists(BINARY):
        sys.exit(f"binary missing: {BINARY} -- run `make` in src/ first")

    A, n = load_graph(args.graph, seed=42)
    # a node permutation is the only source of run-to-run variance for this
    # deterministic binary. seed 42 = identity, so it reproduces the
    # original record.
    if args.seed == 42:
        perm = np.arange(n)
    else:
        perm = np.random.default_rng(args.seed).permutation(n)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    Aperm = A[perm][:, perm]

    with tempfile.TemporaryDirectory(prefix="f2v_", dir="/tmp") as tmp:
        mtx = os.path.join(tmp, f"{args.graph}.mtx")
        write_mtx(Aperm, n, mtx)
        t0 = time.perf_counter()
        cmd = [BINARY, "-input", mtx, "-output", tmp + "/",
               "-iter", str(args.iter), "-dim", str(args.dim),
               "-batch", str(args.batch), "-nsamples", str(args.nsamples),
               "-lr", str(args.lr), "-threads", str(args.threads),
               "-option", str(args.option)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        secs = time.perf_counter() - t0
        if r.returncode != 0:
            sys.exit(f"Force2Vec failed:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
        embd = [f for f in os.listdir(tmp) if f.endswith(".embd")]
        if not embd:
            sys.exit(f"no .embd produced:\n{r.stdout[-800:]}")
        Zperm = read_embd(os.path.join(tmp, embd[0]), n, args.dim)
    Z = Zperm[inv]                             # back to original node order

    method = OPTION_NAME[args.option]
    save_scored(Z, graph=args.graph, method=method, seed=args.seed, A=A, n=n,
                seconds=secs, peak_rss_mb=rss_mb(),
                params={"package": "HipGraph/Force2Vec", "option": args.option,
                        "iter": args.iter, "batch": args.batch,
                        "nsamples": args.nsamples, "lr": args.lr},
                notes=(f"official binary option {args.option}; deterministic, "
                       f"no seed flag -- seed {args.seed} is a node permutation"))


if __name__ == "__main__":
    main()
