#!/bin/env python3
"""evaluator.config -- the task settings, and the named protocols.

Four files of this repository measure an embedding, each a different way:
another classifier size, another pair budget, another feature width,
another source of the ground truth. A number of one is NOT comparable to a
number of another. This module ends the silence: each recorded baseline is
a NAMED PROTOCOL, and each record carries the name.

`fodiwalk/misc/evaluation.task_hop` states the rule:

    A model with 128 features can win only because it has more of them,
    thus the two rows must not be mixed. ... A number is comparable to the
    baseline that used the SAME settings, and to no other.

`LPCfg` and `DACfg` are FLAT and frozen. Flat, because `tests/test_api.py`
holds every field name and default literally, thus a rename breaks a test
and not a recorded number. Frozen, because a task that mutates its own
configuration cannot report which settings produced its score. `override`
returns a new object and says it is no longer the baseline.

A protocol records a baseline INCLUDING its poor choices -- the empty
target range of `fodined`, its one-column feature. A repair here makes a
recorded number unreproducible.

Provenance of each protocol, verified against the code and not the
docstrings:

  `otherge`   `experiments/other-ge/bench_other_ge.py`, the functions
              `task_link_prediction` (line 386) and `task_sp_regression`
              (line 415).
  `n2v1m`     `experiments/large-graph-node2vec/bench_node2vec_1M.py`,
              `main` (line 130).
  `fodined`   `fodined/modular.py` (lines 538 and 610-668) with
              `fodined/link_prediction.py`.
  `fodiwalk_dist`, `fodiwalk_vec`
              `fodiwalk/misc/evaluation.py` (`hop_sample` line 142,
              `task_hop` line 193) with the campaign defaults of
              `experiments/fodiwalk/bench_fodiwalk.py` (lines 63-66).
              ONE baseline, TWO names: the callers sweep both feature
              widths. The note above `PROTOCOLS` says why.
  `default`   no baseline. The dataclass defaults.

Import discipline: `dataclasses` only. This module is the bottom of the
tree; every other module of the package imports it, and it imports nothing
of the package.
"""
from __future__ import annotations

import dataclasses


# ---------------------------------------------------------------------------
# The two task configurations
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class LPCfg:
    """Link prediction: does the geometry hold the adjacency?

    A classifier reads a feature of the two vectors of a pair and it must
    say whether the pair has an edge. Every field below differs between at
    least two recorded baselines, or it names the choice that a baseline
    made silently.
    """

    max_pairs: int = 50_000        # positives AND negatives together
    test_size: float = 0.2
    neg_ratio: float = 1.0         # negatives for each positive
    feature: str = "hadamard"      # hadamard | l1 | concat | avg
    model: str = "rf"              # rf | logreg
    n_estimators: int = 200
    pos_draw: str = "over_cap"     # over_cap | always -- see the note below
    neg_draw: str = "reject"       # reject | far_pairs -- see the note below


@dataclasses.dataclass(frozen=True)
class DACfg:
    """Distance approximation: does the geometry hold the HOP DISTANCE?

    A regressor reads the two vectors of a pair and it must say how many
    hops apart the two nodes are. `feature` is the field that makes two
    rows incomparable: `distance` gives ONE column and `vector` gives
    `n_dim` columns.
    """

    n_pairs: int = 20_000
    n_sources: int = 0             # 0 = every node is a candidate source
    min_hop: int = 2               # a neighbour is the easy case; drop it
    feature: str = "distance"      # distance (1 col) | vector (n_dim cols)
    metric: str = "euclidean"
    hops: str = "auto"             # auto | pll | bfs | landmark
    pair_draw: str = "reject"      # reject | bfs_rows -- see the note below
    models: tuple = ("baseline", "rf", "mlp")
    test_size: float = 0.2
    rf_estimators: int = 100
    rf_min_leaf: int = 25
    mlp_hidden: tuple = (64, 32)
    mlp_early_stop: bool = True


# `pos_draw` is a legacy compatibility knob and it exists for one reason:
# `bench_other_ge.py:395` calls `rng.choice` only when the edge count is
# over the cap, and `bench_node2vec_1M.py:198` calls it always. The two
# consume the generator differently, thus the NEGATIVES that follow are
# different pairs. Exact parity needs both behaviours.
#
# `neg_draw` is the THIRD knob of the same kind, and it was added on
# 2026-08-24 to repair a row that was wrong. `otherge` and `n2v1m` reject
# non-edges with `bench_other_ge.sample_non_edges`. `fodined` and
# `fodiwalk` do NOT: both `sample_negatives`
# (`fodined/link_prediction.py:80`, `fodiwalk/misc/evaluation.py:80`)
# delegate to `graph_augmentation.sample_far_pairs`, which collapses the
# direction with `np.unique`, sorts the batch, and spends a SECOND
# `rng.choice` when the batch overfills. Thus it returns other pairs AND it
# leaves the generator in another state, and every draw after it differs.
#
# Until this field existed, the `fodined` and `fodiwalk_*` rows carried the
# `otherge` sampler and could not reproduce the files they are named for.
# Measured on cora with one embedding: accuracy 0.9744 against 0.9777, and
# the hop R2 that followed the shifted generator 0.616 against 0.630.
# `evaluator/tests/reference/fodined_lp.py` has the same defect on purpose
# -- its docstring calls the substitution "ONE CHANGE" -- thus that frozen
# copy is a reference for the rejection sampler and NOT for `fodined`.
#
# `pair_draw` is the same kind of knob, for the hop pairs. `otherge`,
# `n2v1m` and `fodined` reject every pair that has an edge
# (`bench_other_ge.py:366`, `bench_node2vec_1M.py:107`). `fodiwalk` does
# not: `hop_sample` (`fodiwalk/misc/evaluation.py:142`) holds one BFS row
# in the memory and it draws `2 * n_pairs // n_sources` random targets of
# THAT row, then it keeps the first `n_pairs // n_sources` reachable ones.
# The two draws give different pairs from the same `(n_sources, n_pairs)`,
# thus the parity tests P6a and P6b need both. The first form of PRD block B1 did not name
# this field; A10 added it and B1, B3 and B10 now carry it.


# ---------------------------------------------------------------------------
# The protocols
# ---------------------------------------------------------------------------
# The `fodiwalk` baseline is a SWEEP and not one run: every caller loops
# over BOTH feature widths and it reports both rows --
# `fodiwalk/tests/harness.py:63`, `experiments/fodiwalk/bench_fodiwalk.py:245`
# and `experiments/fdwalk/bench_fdwalk.py:856` all run
#
#     for feature in ("distance", "vector"):
#         ... task_hop(Z, u, v, d, seed, feature) ...
#
# thus TWO recorded rows carry the one name. One name for two rows makes
# `protocol_modified` lie: a user who asks for the other width would get
# the flag that means "not comparable to a recorded baseline", when the
# run reproduces a recorded baseline exactly. Two names keep the flag
# honest, and I4 keeps the two rows out of one table.
#
# The DEFAULT of `task_hop` is not the protocol, and it never was. Every
# caller passes the width explicitly. The default was `"vector"` and it
# is now `"distance"`; the change is behaviour-neutral for that reason.
# The parity tests P6a and P6b read the frozen copy at
# `tests/reference/fodiwalk_eval.py`, which pins the older default, and
# not the live file.
#
# The shared cells are written ONE time. Six rows of repeated literals
# are how a table drifts.
_FODIWALK_LP = LPCfg(neg_draw="far_pairs")   # 50,000 pairs, 200 trees
_FODIWALK_DA = DACfg(n_sources=200, feature="distance", hops="bfs",
                     pair_draw="bfs_rows", rf_min_leaf=1,
                     mlp_hidden=(256, 128), mlp_early_stop=False)
# `hop_sample` samples 200 sources and 20,000 pairs (lines 64 and 65 of
# the campaign script) from the scipy block BFS (`evaluation.py:167`),
# and it draws the targets per BFS row. The sampler keeps every reachable
# pair (`d > 0`, `evaluation.py:172`); the CALLER then drops the
# neighbours with `d >= hop_min`, and `hop_min` is 2
# (`fodiwalk/tests/harness.py:29`, `bench_fodiwalk.py:66`). The net
# minimum is 2. `distance` gives one column and `vector` gives `n_dim`
# columns (`evaluation.py:216-219`). The forest is 100 trees with the
# sklearn leaf of 1 and the MLP is (256, 128) with NO early stopping
# (`evaluation.py:193-195`). LP is 50,000 pairs (`bench_fodiwalk.py:63`)
# and 200 trees (`evaluation.py:95`).

# One entry for each recorded baseline. A cell that repeats a default is
# not written here; the comment gives the value that the code showed.
PROTOCOLS: dict[str, tuple[LPCfg, DACfg]] = {

    # `bench_other_ge.py`. LP caps the POSITIVES at 40,000 (line 395) and
    # it draws an equal count of negatives (line 397), thus `max_pairs`,
    # which counts both classes, is 80,000. The forest is 100 trees (line
    # 403) although the docstring of line 387 claims the protocol of
    # `fodined/modular.py`, which uses 200. DA is exact PLL (line 440),
    # one distance column (line 454), and it keeps `y > 1` (line 448).
    "otherge": (
        LPCfg(max_pairs=80_000, n_estimators=100),
        DACfg(hops="pll"),
    ),

    # `bench_node2vec_1M.py`. The same tasks at 1.13M nodes. The one
    # difference of LP is the DRAW: line 198 calls `rng.choice` even when
    # the edge count is under the cap, thus the generator advances and the
    # negatives are other pairs. DA restricts the pair sources to 200
    # sampled nodes (line 232) and it reads the hops from the scipy block
    # BFS (line 236), because PLL does not fit at that size.
    "n2v1m": (
        LPCfg(max_pairs=80_000, n_estimators=100, pos_draw="always"),
        DACfg(n_sources=200, hops="bfs"),
    ),

    # `fodined/modular.py`. LP is `MAX_LP_PAIRS = 50_000` (line 538) into
    # `fodined/link_prediction.py`, which is 200 trees (line 95).
    #
    # DA keeps 2,000 pairs (line 618) and NOT the 20,000 of the other
    # baselines. The recorded log confirms it: "hop-distance regression on
    # 2000 unique pairs" (`fodiwalk/dev-docs/CATALOG.md:796`).
    #
    # The feature is ONE column: line 632 builds
    # `np.linalg.norm(Z[u] - Z[v], axis=1)[:, None]`, and the two `vector`
    # forms of lines 630 and 631 are COMMENTED OUT. The docstring of
    # `fodiwalk/misc/evaluation.task_hop` said that `vector` "is the
    # protocol of `fodined/modular.py`"; that docstring was stale and it
    # is now repaired. The code produced the recorded numbers, thus
    # `feature="distance"` stands. Two agents reached this by different
    # routes -- the recorded target range of the CATALOG log, and the
    # file mtimes -- and they agree.
    #
    # WARNING -- the DA row of `fodined` REPRODUCES NO PUBLISHED NUMBER.
    # Every other cell of this row is a recorded fact, but `hops` is not:
    # `auto` is the dataclass default and NO baseline chose it.
    # `modular.py` computes no hop distance at all. It reads the target
    # from `D.data` (line 624) -- the weight that the AUGMENTATION stored
    # -- and not the true hop distance of the graph. `D` holds only the
    # pairs that the policy sampled, and its unreachable sentinel is `n`.
    # `evaluator` reads the TRUE distance from the graph.
    #
    # The two are therefore DIFFERENT MEASUREMENTS, and the difference is
    # intentional (PRD block B1). The parity gate tests the LINK
    # PREDICTION of `fodined` only. A reader must not compare a `fodined`
    # hop score of `evaluator` against a hop score of `modular.py`.
    "fodined": (
        LPCfg(neg_draw="far_pairs"),         # 50,000 pairs, 200 trees
        DACfg(n_pairs=2_000, rf_estimators=200, rf_min_leaf=1,
              mlp_hidden=(128, 64), mlp_early_stop=True),
    ),

    # TWO rows, and they differ in `feature` alone. See `_FODIWALK_DA`
    # above for the shared cells and for the reason of the split.
    "fodiwalk_dist": (_FODIWALK_LP, _FODIWALK_DA),
    "fodiwalk_vec": (_FODIWALK_LP,
                     dataclasses.replace(_FODIWALK_DA, feature="vector")),

    # No baseline. A run that names `default` claims no comparison.
    "default": (LPCfg(), DACfg()),
}


# ---------------------------------------------------------------------------
# The two functions
# ---------------------------------------------------------------------------
def get(name: str) -> tuple[LPCfg, DACfg]:
    """The two configurations of the protocol `name`.

    Raises `ValueError` on an unknown name, and the message lists the
    names. A typed protocol name that falls back to the defaults would
    give a number that claims a baseline it never used.
    """
    try:
        return PROTOCOLS[name]
    except KeyError:
        raise ValueError(
            f"unknown protocol {name!r}; "
            f"the names are {', '.join(sorted(PROTOCOLS))}") from None


def override(cfg, **kw) -> tuple[object, bool]:
    """`(cfg2, modified)`. A given keyword replaces the field of `cfg`.

    `modified` is the flag that the record carries as
    `protocol_modified`. It is `True` when the caller gave ANY keyword,
    EVEN when the value equals the value that the field already holds. The
    flag reports the intent of the caller and not the difference of the
    two objects: a caller that types a number takes responsibility for it,
    and a later edit of the protocol table must not turn that row back
    into a baseline.

    A keyword whose value is `None` is dropped, and it does not set the
    flag. That is how the public API sends "this one takes the value of
    the protocol" (PRD section 5.3).

    Raises `ValueError` on a field that the dataclass does not hold.
    """
    kw = {k: v for k, v in kw.items() if v is not None}
    if not kw:
        return cfg, False
    fields = {f.name for f in dataclasses.fields(cfg)}
    bad = sorted(set(kw) - fields)
    if bad:
        raise ValueError(
            f"{type(cfg).__name__} has no field {', '.join(bad)}; "
            f"the fields are {', '.join(sorted(fields))}")
    return dataclasses.replace(cfg, **kw), True
