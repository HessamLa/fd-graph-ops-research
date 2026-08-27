#!/bin/env python3
"""check_api.py -- gate G3. The public surface of the package did not move.

A refactor that splits a module must not move a NAME. This file records the
surface as it is BEFORE the split, and it fails when the split changes it.

    .venv/bin/python -m fodiwalk.tests.check_api     # 0 = pass
    from fodiwalk.tests.check_api import check       # for a pytest test

Every list below is LITERAL, and that is the point of the file. A check
that reads `dataclasses.fields(Config)` compares Config to itself and it
passes whatever the refactor does. A hardcoded list is the record.

The surface is `dev-docs/REFACTOR.md` section 5.1. Recorded 2026-08-20
against the tree of the tag `fodiwalk-pre-refactor` (89abaf2); the working
tree Config was identical to the tag at that time.

Every failure is COLLECTED and printed together. One report shows the whole
damage of a refactor, and not the first item of it.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # G3 needs no GPU

import numpy as np
import scipy.sparse as sp

# --- the record ------------------------------------------------------------
# The 39 fields of Config, in order, with their default values. REFACTOR.md
# section 5.1 says "the 41 of today"; the tree and the tag both hold 39,
# thus the count of the document is wrong and this list is the tree.
CONFIG_FIELDS = [
    ("pairs", "walk"), ("edge_rule", "both"), ("policy", "cap"),
    ("walks", 10), ("walk_len", 20), ("window", 5), ("cap", 16),
    ("row_cap", 0), ("p", 1.0), ("q", 1.0), ("prune_max", 4_000_000),
    ("prune_factor", 4), ("weight", "min_gap"), ("freq_mode", "pair"),
    ("far", 0), ("far_weight", 100.0), ("far_bias", 0.0),
    ("far_with_buckets", False), ("bucket_total", 0), ("landmarks", 0),
    ("far_max", 32), ("far_scale", 1.0), ("force", "fdlinear"),
    ("fuse_planes", False), ("no_deg_norm", False), ("deg_source", "auto"),
    ("k1", 0.999), ("k4", 0.01), ("kr", 1.0), ("fdlinear_sign", -1.0),
    ("random_drop_rate", 0.5), ("drop_strategy", "random_rows"),
    ("b_cells", 16_384), ("k_max", 256), ("ladder_base", 1.5),
    ("chunks", 1), ("chunk_host", False), ("check_planes", True),
    ("check_padding", True),
]

# The keyword names of `Fodiwalk.__init__`, beside `**cfg`.
INIT_KEYWORDS = ["n_dim", "lr", "seed", "verbosity", "optim", "lr_decay",
                 "eta", "sgd_frac", "sqn_memory"]

METHODS = ["make_graph", "graph_walk", "set_D", "augment_graph", "forces",
           "embed", "get_embeddings", "attach_callback", "set_rule", "Th",
           # `golden.py` calls these two. They may become thin wrappers of
           # a stage package (`embed/` first, `augment_graph/` since
           # 2026-08-21), and they must stay callable on the model.
           "_build_planes", "_build_degrees"]

# `fit` does NOT exist anywhere in the hierarchy (REMOVED 2026-08-21):
# `make_graph` is a stub, thus a `fit` that chained it into `embed`
# promised a stage that was never real. See `check` below.

ATTRIBUTES = ["cfg", "law", "rng", "D", "stats", "freq", "info",
              "plan_stats", "plans", "steps", "inv_deg_ext", "chunk_rows",
              "resident", "params", "diverged", "dZ", "Z"]

# The KEY SETS after a default run. A key that goes away breaks a log or a
# RESULT line; a key that arrives is new information and it is allowed.
INFO_KEYS = ["capped_pairs", "far_asked", "far_pairs", "near_nnz", "prunes",
             "raw_pairs", "t_aug", "t_pairs", "unique_pairs", "walk_order"]
PLAN_KEYS = ["cells", "chunks", "n_split", "n_virtual", "pad_frac",
             "rows_per_chunk", "rungs"]
STATS_KEYS = ["cnt", "freq", "key", "mn", "prunes", "raw", "sm"]


def tiny_graph(n: int = 60):
    """The `tiny` fixture of `conftest.py`, built here.

    A copy, and not an import: this file must run under `python -m` and a
    fixture needs pytest. Node 0 is the hub, and the chords make it more
    than a star. Keep the two builders the same.
    """
    rng = np.random.default_rng(0)
    rows, cols = [], []
    for v in range(1, n):                      # a star, thus it is connected
        rows += [0, v]
        cols += [v, 0]
    for _ in range(120):                       # and some random chords
        u, v = rng.integers(1, n, 2)
        if u != v:
            rows += [int(u), int(v)]
            cols += [int(v), int(u)]
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    A.sum_duplicates()
    A.data[:] = 1.0
    A.sort_indices()
    return A, n


def check(verbose: bool = True) -> list:
    """Every item of REFACTOR.md 5.1. Returns the list of failures."""
    bad = []

    def want(flag, msg):
        if not flag:
            bad.append(msg)

    # -- 1. the import ------------------------------------------------------
    try:
        from fodiwalk import Fodiwalk, Config
    except Exception as e:
        return [f"`from fodiwalk import Fodiwalk, Config` raises {e!r}"]

    # -- 2. the fields of Config, and their defaults ------------------------
    import dataclasses
    got = {f.name: f.default for f in dataclasses.fields(Config)}
    recorded = dict(CONFIG_FIELDS)
    for name in sorted(set(recorded) - set(got)):
        bad.append(f"Config lost the field `{name}`")
    for name in sorted(set(got) - set(recorded)):
        bad.append(f"Config gained the field `{name}` (record it here)")
    cfg = Config()
    for name, value in CONFIG_FIELDS:
        if name in got and getattr(cfg, name, object()) != value:
            bad.append(f"Config.{name} defaults to "
                       f"{getattr(cfg, name)!r} and the record is {value!r}")

    # -- 3. the signature of the constructor --------------------------------
    import inspect
    par = inspect.signature(Fodiwalk.__init__).parameters
    for name in INIT_KEYWORDS:
        want(name in par, f"Fodiwalk.__init__ lost the keyword `{name}`")
    want(any(p.kind is p.VAR_KEYWORD for p in par.values()),
         "Fodiwalk.__init__ lost `**cfg`, thus a Config field cannot pass")

    # An unknown name must RAISE. A silent drop gives the run a default and
    # the number of the run is then not the number of the command line.
    try:
        Fodiwalk(n_dim=4, seed=0, no_such_option=1)
        bad.append("an unknown kwarg does not raise TypeError")
    except TypeError:
        pass
    except Exception as e:
        bad.append(f"an unknown kwarg raises {type(e).__name__}, not TypeError")

    # -- 4. the methods -----------------------------------------------------
    for name in METHODS:
        want(callable(getattr(Fodiwalk, name, None)),
             f"Fodiwalk lost the method `{name}`")

    # -- 5. one short run, then the attributes and the key sets -------------
    A, n = tiny_graph()
    try:
        fw = Fodiwalk(n_dim=8, seed=0, lr=1.0)
        fw.embed(A, epochs=5)
    except Exception as e:
        bad.append(f"embed on the tiny graph raises {type(e).__name__}: {e}")
        return _report(bad, verbose)

    for name in ATTRIBUTES:
        want(hasattr(fw, name), f"the model lost the attribute `{name}`")

    for tag, keys, held in [("info", INFO_KEYS, fw.info),
                            ("plan_stats", PLAN_KEYS, fw.plan_stats),
                            ("stats", STATS_KEYS, fw.stats)]:
        for k in sorted(set(keys) - set(held or {})):
            bad.append(f"`{tag}` lost the key `{k}`")

    # `fit` was removed entirely (2026-08-21): it must not exist at all,
    # on Fodiwalk or on any class of its hierarchy.
    want(not hasattr(Fodiwalk, "fit"), "Fodiwalk has a `fit` method; it "
         "was removed entirely and must not come back")

    # The two seams `golden.py` drives, on a real D.
    try:
        want(len(fw._build_planes(fw.D)) == 2,
             "_build_planes(D) does not give the two planes of fdlinear")
        want(np.asarray(fw._build_degrees(fw.D, A)).shape == (n,),
             "_build_degrees(D, A) does not give one degree for each row")
    except Exception as e:
        bad.append(f"_build_planes/_build_degrees raise {type(e).__name__}: {e}")

    return _report(bad, verbose)


def _report(bad, verbose):
    if verbose:
        if bad:
            print(f"[check_api] {len(bad)} FAILURE(S):")
            for msg in bad:
                print(f"  - {msg}")
        else:
            print(f"[check_api] API OK: {len(CONFIG_FIELDS)} Config fields, "
                  f"{len(METHODS)} methods, {len(ATTRIBUTES)} attributes, "
                  f"{len(INFO_KEYS) + len(PLAN_KEYS) + len(STATS_KEYS)} keys")
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if check() else 0)
