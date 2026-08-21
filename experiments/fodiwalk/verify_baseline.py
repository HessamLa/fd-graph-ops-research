#!/bin/env python3
"""verify_baseline.py -- the two claims the baseline of fodiwalk rests on.

    1. The walk IS the walk of node2vec.
    2. The force law IS `fdlinear`.

Claim 1 is not a statement about intent. `experiments/other-ge/bench_other_ge.py`
holds the node2vec baseline of this repository, and its `uniform_walks` is
the walk that node2vec takes at `p = q = 1` -- the DeepWalk walk. This
script drives that function and `fodiwalk`'s own with the SAME graph, the
same start nodes and the same seed, and it compares the two arrays cell by
cell. A bit-exact match is the proof; anything less is not.

The second-order case is checked too: `make_walker(A, n, p, q)` gives the
rejection sampler above `p = q = 1`, thus `p` and `q` reach the walk.

    .venv/bin/python experiments/fodiwalk/verify_baseline.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "other-ge"))

import numpy as np

from fodiwalk import Fodiwalk, Config
from fodiwalk.augment_graph import walks as W
from fodiwalk.core import forces
from fodiwalk.make_graph import load

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "bench_other_ge", os.path.join(ROOT, "experiments", "other-ge",
                                   "bench_other_ge.py"))
_n2v = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_n2v)          # the node2vec baseline of the repo


def ok(flag):
    return "PASS" if flag else "FAIL"


def main():
    A, n = load("cora")
    n_walks, walk_len, seed = 10, 20, 42
    fail = 0

    # -- 1. the walk ------------------------------------------------------
    theirs = _n2v.uniform_walks(A, n, n_walks, walk_len, seed)
    starts = np.tile(np.arange(n, dtype=np.int64), n_walks)
    ours = W.uniform_walks(A, starts, walk_len, np.random.default_rng(seed))
    same = ours.shape == theirs.shape and np.array_equal(ours, theirs)
    print(f"[1a] fodiwalk.uniform_walks == other-ge.uniform_walks   {ok(same)}"
          f"   {theirs.shape} cells, bit-exact")
    fail += not same

    walker = W.make_walker(A, n, 1.0, 1.0)
    is_self = getattr(walker, "func", None) is W.uniform_walks
    print(f"[1b] make_walker(p=1, q=1) IS uniform_walks             {ok(is_self)}")
    fail += not is_self

    drv = walker(starts, walk_len, np.random.default_rng(seed))
    same2 = np.array_equal(drv, theirs)
    print(f"[1c] the walker the baseline uses gives those walks     {ok(same2)}")
    fail += not same2

    w2 = W.make_walker(A, n, 2.0, 1.0)
    second = getattr(w2, "func", None) is W.node2vec_walks
    print(f"[1d] p != 1 gives the second-order node2vec walk        {ok(second)}")
    fail += not second

    # -- 2. the force law -------------------------------------------------
    cfg = Config()
    is_fdl = cfg.force == "fdlinear"
    print(f"[2a] Config().force is 'fdlinear'                       {ok(is_fdl)}"
          f"   (got {cfg.force!r})")
    fail += not is_fdl

    fw = Fodiwalk(n_dim=8, seed=seed)
    bound = fw.force_fn is forces.fdlinear and fw.law == "fdlinear"
    print(f"[2b] the model binds `fdlinear` itself                  {ok(bound)}")
    fail += not bound

    planes = forces.planes_of("fdlinear")
    right = planes == ("h", "freq")
    print(f"[2c] its planes are {planes}                    {ok(right)}")
    fail += not right

    only = sorted(forces.FORCE_FN) == ["fdlinear", "fdlinear_fused"]
    print(f"[2d] no other law is registered                         {ok(only)}"
          f"   {sorted(forces.FORCE_FN)}")
    fail += not only

    # -- the parameters, which are NOT a claim but a difference to state --
    print(f"\nWalk budget. fodiwalk baseline: walks={cfg.walks}, "
          f"len={cfg.walk_len}, window={cfg.window}, p={cfg.p}, q={cfg.q}.")
    print("other-ge node2vec defaults:   walks=10, len=40, window=5, "
          "p=1.0, q=1.0.")
    print("The walk is the same function. The LENGTH differs, thus a "
          "comparison must set it.")

    print(f"\n{'ALL CHECKS PASS' if not fail else str(fail) + ' CHECK(S) FAILED'}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
