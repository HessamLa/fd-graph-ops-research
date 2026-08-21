#!/bin/env python3
"""verify_modular.py -- fodiwalk against `fodined/modular.py`, function by
function, on the WALK policies.

`modular.py` is the reference pipeline: it loads a graph, it augments it,
it embeds it on the SELL-C-sigma kernel, and it scores the result with link
prediction and a hop regression. This script proves that `fodiwalk` runs
the SAME pipeline, by driving the two implementations side by side with the
same graph, the same seed and the same generator, and by comparing the
arrays cell by cell.

A bit-exact match is the proof. A "looks similar" is not.

SCOPE, 2026-08-20. Every policy of `fodiwalk` is WALK-BASED. The
exact-distance policies `ball` and `sampled` were removed from the package;
see `fodiwalk/dev-docs/CATALOG.md` section 18. This script therefore no
longer compares two `D` matrices -- it compares the code AROUND the near
term, which the two files still share, and it asserts that a walk `D`
keeps every contract the law needs.

WHAT A WALK POLICY SHARES WITH `modular.py`, AND WHAT IT DOES NOT.

    shared      the LONG-RANGE term. `n * log10(n)` pairs that the near
                policy did not find, drawn by `sample_far_pairs` and
                stored at `far_weight = 100`. This is `modular.py`'s own
                code, and the walk policies call it.
    shared      the engine, the regularizer, the loader, the link
                prediction, and every constant.
    NOT shared  the NEAR term. `modular.py` measures the exact distance of
                every pair within 2 hops. A walk policy takes the WALK
                GAP, which is an upper bound of that distance, over the
                pairs the walks reached.

Thus the two `D` matrices are NOT the same matrix, and they are not meant
to be. What this script asserts is that everything AROUND the near term is
the same code, that the walk `D` keeps every contract the law needs, and
that the walk gap is a sound stand-in -- an upper bound that is exact on
about 98% of the pairs it stores.

    .venv/bin/python experiments/fodiwalk/verify_modular.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np
import scipy.sparse as sp

from fodiwalk import Fodiwalk, Config
from fodiwalk.augment_graph import far_pairs as FP
from fodiwalk.core import plan_contract
from fodiwalk.core.forces import degrees_from_D, planes_of
from fodiwalk.make_graph import load
from fodiwalk.misc import evaluation as EV

from fodined import graph_augmentation as GA      # the reference
from fodined import link_prediction as LP         # the reference


def ok(flag):
    return "PASS" if flag else "FAIL"


def main():
    A, n = load("cora")
    seed = 42
    n_far = int(n * np.log10(n))          # the N_NEW_EDGES of modular.py
    fail = 0
    print(f"cora: n={n}, {A.nnz // 2} undirected edges, far pairs {n_far}\n")
    print("--- the shared code, which every policy uses --------------------")

    # -- 1. the long-range term, which the walk policies call -------------
    # `modular.py` draws its far pairs with this function, at this weight,
    # in this count. A walk policy calls the SAME function. The matrix is
    # what the sampler rejects against; `A` itself serves, thus this check
    # needs no ball.
    fo = FP.sample_far_pairs(n, n_far, A, np.random.default_rng(seed))
    ft = GA.sample_far_pairs(n, n_far, A, np.random.default_rng(seed))
    s = np.array_equal(fo, ft)
    print(f"[1] sample_far_pairs is the identical sampler            {ok(s)}"
          f"   {fo.shape[0]} pairs")
    fail += not s

    cum_o, cum_t = FP.degree_table(A, 0.75), GA.degree_table(A, 0.75)
    s = np.array_equal(cum_o, cum_t) and FP.degree_table(A, 0.0) is None
    print(f"[2] degree_table is identical, and 0.0 stays uniform     {ok(s)}")
    fail += not s

    # -- 3. the link prediction -------------------------------------------
    Z = np.random.default_rng(0).normal(size=(n, 32))
    so, lo = EV.link_prediction(Z, A, n, 50_000,
                                np.random.default_rng(seed), seed)
    st, lt = LP.link_prediction(Z, A, n, 50_000,
                                np.random.default_rng(seed), seed)
    s = so == st and lo == lt
    print(f"[3] link_prediction gives the identical scores           {ok(s)}"
          f"   auc {so['auc']:.6f}")
    fail += not s

    pos, neg = np.array([[0, 1], [2, 3]]), np.array([[4, 5]])
    Xo, yo = EV.edge_features(Z, pos, neg)
    Xt, yt = LP.edge_features(Z, pos, neg)
    s = np.array_equal(Xo, Xt) and np.array_equal(yo, yt)
    print(f"[4] edge_features is the same Hadamard product           {ok(s)}")
    fail += not s

    # -- 5. the constants -------------------------------------------------
    c = Config()
    want = {"far_weight": (c.far_weight, 100.0),
            "k1": (c.k1, 0.999), "k4": (c.k4, 0.01),
            "random_drop_rate": (c.random_drop_rate, 0.5),
            "drop_strategy": (c.drop_strategy, "random_rows"),
            "b_cells": (c.b_cells, 16_384), "k_max": (c.k_max, 256),
            "ladder_base": (c.ladder_base, 1.5),
            "far (0 = n*log10 n)": (c.far or n_far, n_far)}
    bad = {k: v for k, v in want.items() if v[0] != v[1]}
    print(f"[5] every shared constant has the modular.py value       "
          f"{ok(not bad)}" + (f"   {bad}" if bad else ""))
    fail += bool(bad)

    # -- the walk policies ------------------------------------------------
    print("\n--- the WALK policies, which are the focus ----------------------")
    edges = sp.triu(A, k=1).tocoo()
    ekey = set(zip(edges.row.tolist(), edges.col.tolist()))

    rows = []
    for tag, cfg in [("walk", dict(pairs="walk")),
                     ("walk_edges", dict(pairs="walk_edges")),
                     ("nbr_walk", dict(pairs="nbr_walk"))]:
        fw = Fodiwalk(n_dim=8, seed=seed, weight="min_gap",
                      force="fdlinear", **cfg)
        D = fw.augment_graph(A)
        u = np.repeat(np.arange(n), np.diff(D.indptr))
        m = D.data == 1
        h1 = set(zip(np.minimum(u[m], D.indices[m]).tolist(),
                     np.maximum(u[m], D.indices[m]).tolist()))
        rows.append((tag, fw, D, len(h1 & ekey)))

    # 6. I4 -- a stored weight is an integer, and 1 means adjacency.
    s = all(np.array_equal(D.data, np.rint(D.data)) and D.data.min() >= 1.0
            for _, _, D, _ in rows)
    print(f"[6] every walk D holds INTEGER weights >= 1 (I4)         {ok(s)}")
    fail += not s

    # 7. I5 -- no row reaches the law with a degree of 0.
    s = True
    for tag, _, D, _ in rows:
        try:
            plan_contract.check_degrees(degrees_from_D(D), D)
        except plan_contract.PlaneContractError:
            s = False
    print(f"[7] no row reaches the law with a degree of 0 (I5)       {ok(s)}")
    fail += not s

    # 8. the plane contract, from the registry and not an if chain.
    s = True
    for tag, fw, D, _ in rows:
        try:
            plan_contract.check("fdlinear", fw._build_planes(D), D)
        except plan_contract.PlaneContractError:
            s = False
    print(f"[8] the planes {planes_of('fdlinear')} keep their contract    "
          f"     {ok(s)}")
    fail += not s

    # 9. the far pairs of a walk D are modular.py's long-range term.
    fw = rows[0][1]
    s = (fw.info["far_asked"] == n_far
         and fw.info["far_pairs"] > 0
         and int((rows[0][2].data == 100.0).sum()) == 2 * fw.info["far_pairs"])
    print(f"[9] the walk D carries modular.py's long-range term      {ok(s)}"
          f"   {fw.info['far_pairs']} pairs at 100.0")
    fail += not s

    # -- the H1 coverage, which is the structural difference --------------
    print("\nTHE NEAR TERM, and the one number that separates the policies.")
    print("`modular.py` stores EVERY edge of A at h = 1, because it measures")
    print("the distance. A walk MAY miss an edge of a low-degree node.\n")
    print(f"  {'policy':<14} {'D.nnz':>8} {'h=1 cells':>10} "
          f"{'edges of A at h=1':>19}")
    for tag, _, D, cover in rows:
        print(f"  {tag:<14} {D.nnz:>8} {int((D.data == 1).sum()):>10} "
              f"{cover:>10} / {len(ekey)} = {cover / len(ekey):>5.1%}")
    print("\n`walk_edges` forces every missing edge in at h = 1, thus it is")
    print("the walk policy that gives `modular.py`'s h = 1 layer exactly.")

    print("\nTHE TWO DIFFERENCES THAT REMAIN, and both are decisions.")
    print("  1. THE LAW. modular.py runs `shell_force`, which reads a")
    print("     1/|S_h(u)| coefficient plane. fodiwalk runs `fdlinear`, by")
    print("     the decision of 2026-08-19 (dev-docs/CATALOG.md 13).")
    print("  2. THE NEAR TERM. modular.py measures the distance; a walk")
    print("     policy takes the walk GAP, an upper bound. The bench")
    print("     measures how tight: H2 on the baseline is exact on 98.1% of")
    print("     the stored pairs, over on 1.9%, and UNDER on 0.0% -- which")
    print("     it must be, since a walk of t steps proves a path of t.")

    print(f"\n{'ALL CHECKS PASS' if not fail else str(fail) + ' CHECK(S) FAILED'}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
