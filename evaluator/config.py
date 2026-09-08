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

v2 adds six more tasks (PRD-v2 section 5.2) and the metrology blocks:
`DRCfg`, `DSCfg`, `RTCfg`, `LRCfg`, `STCfg`, `SBCfg`, one `Protocol`
object that holds all eight, the noise floors `FLOORS`, the per-class
metric table `APPLIES`, and the PROVISIONAL constants of PRD-v2 section
12. The six v1 protocols keep every v1 cell and take the new cfgs at
their dataclass defaults, so an old name runs a new task and the record
still says `protocol_modified=False`: the protocol had no opinion on the
new task, so the defaults ARE the protocol.

`get()` keeps its v1 contract and returns the `(lp, da)` pair, because
`tasks/link_prediction.py` and `tasks/dist_approx.py` are frozen and
index it. A new task reads `protocol(name)` and takes the field it wants.

Import discipline: `dataclasses` and `types` only. This module is the
bottom of the tree; every other module of the package imports it, and it
imports nothing of the package.
"""
from __future__ import annotations

import dataclasses
import types


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
# The PROVISIONAL constants (PRD-v2 section 12)
# ---------------------------------------------------------------------------
# Each of these is an OPEN DECISION. The document ships a default so the
# tree can be built and tested, and each default is ONE named constant, so
# a decision is one edit here and not a search over eight task modules
# (PRD-v2 invariant 8). The comment names the row it came from.

# metrology.md section 13 item 1: the headline distance number. The
# document proposes Somers' D_yx, with Spearman rho printed beside it.
HEADLINE_RANK = "somers_d"

# metrology.md section 13 item 2: a pair whose two nodes sit in different
# components has no hop distance. `drop` removes the pair and the record
# counts it (`sizes.pairs_dropped_cross_component`); `cap` sets
# `y = diam_est + 1`. v1 dropped such pairs silently. cora has 78
# components, so this is not a rare case.
CROSS_COMPONENT = "drop"

# metrology.md section 13 item 3: `k` for retrieval. 10 fixed on class H,
# `k = deg(u)` on class S. The class comes from the profile (PRD-v2 B14);
# `RTCfg.k` holds the 10.
K_MODE_BY_CLASS = types.MappingProxyType({"H": "fixed", "S": "degree"})

# The resolution horizon `h*`: the largest hop shell at which the
# cumulative rho stays within this much of its maximum. metrology.md
# section 13 has no row of its own for the rule; item 4 is the nearest one
# (a planted ring lattice would give a known `h*` to test it against), and
# the rule itself is PRD-v2 B9.
H_STAR_TOL = 0.02

# Graph reconstruction scores the `m` nearest pairs of Z against the edge
# set, which does not finish at tier T3. Above this node count the score is
# skipped with a warning. No metrology.md section 13 row; the number is
# PRD-v2 B11 and section 11. It is also the row count above which B5 turns
# to an approximate kNN index.
RECON_MAX_N = 200_000

# The community algorithm on A. Louvain, as NetworKit `community.PLM`.
# No metrology.md section 13 row; the source is PRD-v2 B15 and
# metrology.md section 6.6.
COMMUNITY_DEFAULT = "louvain"

# Resamples of a bootstrap, both for a score interval (B9) and for a noise
# floor (B17). No numbered metrology.md section 13 row; metrology.md
# section 8 fixes the count at 1,000 resamples of the scored pair set.
N_BOOT = 1_000


# ---------------------------------------------------------------------------
# The six v2 task configurations
# ---------------------------------------------------------------------------
# One dataclass per new task, and the fields are exactly the settings of
# that task's signature in PRD-v2 section 5.2. Frozen and flat, for the
# reason `LPCfg` is. A field carries its unit and where the default came
# from.
@dataclasses.dataclass(frozen=True)
class DRCfg:
    """`dist_rank`: does the DISTANCE ORDER of Z match the hop order?

    No model sits between the layout and the number, so this is the
    primary distance score of metrology.md section 6.1. The task samples
    pairs, reads the true hop distance, and correlates it with the
    distance in Z.
    """

    n_pairs: int = 20_000          # pairs to score; PRD-v2 B12 `metrology_h`
    n_sources: int = 0             # 0 = every node is a candidate source
    min_hop: int = 2               # drop the pairs closer than this; v1 DACfg
    max_hop: int = 0               # 0 = no cap. A cap keeps the near shells
    metric: str = "euclidean"      # the distance read in Z; v1 DACfg
    hops: str = "auto"             # auto | pll | bfs | landmark; v1 DACfg
    pair_draw: str = "reject"      # reject | bfs_rows | stratified_by_hop
    per_source: bool = False       # also score inside one BFS row, then mean
    per_shell: bool = False        # also score per hop shell, and give h_star
    bootstrap: int = 0             # resamples for the intervals; 0 = none
    cross_component: str = CROSS_COMPONENT   # drop | cap


@dataclasses.dataclass(frozen=True)
class DSCfg:
    """`dist_stress`: does the DISTANCE MAGNITUDE of Z match the hops?

    Rank correlation survives any monotone map, so it says nothing about
    where the map goes flat. Kruskal stress-1 at the best scale does
    (metrology.md section 6.2). It is the primary number for class S and
    for a 2-D or 3-D drawing.
    """

    n_pairs: int = 20_000          # pairs to score; same budget as DRCfg
    n_sources: int = 0             # 0 = every node is a candidate source
    min_hop: int = 2               # drop the pairs closer than this; v1 DACfg
    max_hop: int = 0               # 0 = no cap
    metric: str = "euclidean"      # the distance read in Z
    hops: str = "auto"             # auto | pll | bfs | landmark
    pair_draw: str = "reject"      # reject | bfs_rows | stratified_by_hop
    shepard: bool = False          # also keep the (x, y, fit) points, PRD B10
    cross_component: str = CROSS_COMPONENT   # drop | cap


@dataclasses.dataclass(frozen=True)
class RTCfg:
    """`retrieval`: are the near neighbours of a node near it in Z?

    Link prediction with no classifier (metrology.md section 6.3), and
    the primary local-structure score. The `k` nearest points of a query
    in Z are compared against the truth, which is either the adjacency
    row or the `k` nearest nodes by hop distance.
    """

    k: int = 10                    # neighbours per query; section 13 item 3
    k_mode: str = "fixed"          # fixed | degree: deg(u), floor 1, cap 100
    n_queries: int = 2_000         # queries to draw; PRD-v2 section 11
    metric: str = "euclidean"      # the distance read in Z
    truth: str = "adjacency"       # adjacency (the CSR row) | hop_knn (BFS)
    block: int = 512               # MB of the sklearn kNN working memory, B5


@dataclasses.dataclass(frozen=True)
class LRCfg:
    """`link_rank`: link prediction with no model.

    A pair is scored by `-||Z_u - Z_v||` and nothing is trained
    (metrology.md section 6.4). `neg_ratio` is why the task exists: the
    balanced AUC of `link_prediction` reads near 0.99 for every method on
    every class H graph, and average precision at 1:100 does not.
    """

    max_pairs: int = 50_000        # positives AND negatives together; v1 LPCfg
    neg_ratio: float = 1.0         # negatives per positive; up to 1000, B3
    metric: str = "euclidean"      # the distance read in Z
    neg_draw: str = "reject"       # reject | far_pairs -- the v1 LPCfg knob


@dataclasses.dataclass(frozen=True)
class STCfg:
    """`structure`: does the layout hold the communities and the hubs?

    Communities are found on A and clusters on Z, and the two labellings
    are compared (metrology.md section 6.6). Secondary for class H and off
    for class S.
    """

    n_clusters: int = 0            # 0 = as many as the communities found, B15
    community: str = COMMUNITY_DEFAULT   # louvain | label_prop | spectral
    centrality: str = "pagerank"   # degree | pagerank (damping 0.85), B15


@dataclasses.dataclass(frozen=True)
class SBCfg:
    """`stability`: do two seeds of one method give the same layout?

    Procrustes residual and kNN Jaccard over every pair of the given
    embeddings (metrology.md section 6.7). It is half of the noise-floor
    rule, and it is mandatory at level L5.
    """

    k: int = 10                    # neighbours per query, for the Jaccard
    metric: str = "euclidean"      # the distance read in Z
    n_queries: int = 2_000         # queries to draw; the RTCfg budget


@dataclasses.dataclass(frozen=True)
class Protocol:
    """The eight task configurations of one named baseline.

    A protocol is one row of the table below. It is frozen for the reason
    a cfg is: a run reports the settings that produced its score.

    The six v1 names give their new fields the dataclass defaults. That is
    a decision of PRD-v2 B1: the v1 baseline had no opinion on a task that
    did not exist, so an old name runs a new task and the record still
    reads `protocol_modified=False`. Raising instead would force a new
    name for every recorded baseline.
    """

    lp: LPCfg = dataclasses.field(default_factory=LPCfg)
    da: DACfg = dataclasses.field(default_factory=DACfg)
    dr: DRCfg = dataclasses.field(default_factory=DRCfg)
    ds: DSCfg = dataclasses.field(default_factory=DSCfg)
    rt: RTCfg = dataclasses.field(default_factory=RTCfg)
    lr: LRCfg = dataclasses.field(default_factory=LRCfg)
    st: STCfg = dataclasses.field(default_factory=STCfg)
    sb: SBCfg = dataclasses.field(default_factory=SBCfg)


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
PROTOCOLS: types.MappingProxyType = types.MappingProxyType({

    # `bench_other_ge.py`. LP caps the POSITIVES at 40,000 (line 395) and
    # it draws an equal count of negatives (line 397), thus `max_pairs`,
    # which counts both classes, is 80,000. The forest is 100 trees (line
    # 403) although the docstring of line 387 claims the protocol of
    # `fodined/modular.py`, which uses 200. DA is exact PLL (line 440),
    # one distance column (line 454), and it keeps `y > 1` (line 448).
    "otherge": Protocol(
        LPCfg(max_pairs=80_000, n_estimators=100),
        DACfg(hops="pll"),
    ),

    # `bench_node2vec_1M.py`. The same tasks at 1.13M nodes. The one
    # difference of LP is the DRAW: line 198 calls `rng.choice` even when
    # the edge count is under the cap, thus the generator advances and the
    # negatives are other pairs. DA restricts the pair sources to 200
    # sampled nodes (line 232) and it reads the hops from the scipy block
    # BFS (line 236), because PLL does not fit at that size.
    "n2v1m": Protocol(
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
    "fodined": Protocol(
        LPCfg(neg_draw="far_pairs"),         # 50,000 pairs, 200 trees
        DACfg(n_pairs=2_000, rf_estimators=200, rf_min_leaf=1,
              mlp_hidden=(128, 64), mlp_early_stop=True),
    ),

    # TWO rows, and they differ in `feature` alone. See `_FODIWALK_DA`
    # above for the shared cells and for the reason of the split.
    "fodiwalk_dist": Protocol(_FODIWALK_LP, _FODIWALK_DA),
    "fodiwalk_vec": Protocol(
        _FODIWALK_LP,
        dataclasses.replace(_FODIWALK_DA, feature="vector")),

    # No baseline. A run that names `default` claims no comparison.
    "default": Protocol(LPCfg(), DACfg()),

    # The three metrology protocols of PRD-v2 B12. They have NO recorded
    # baseline behind them: they are this document's own settings for the
    # two graph classes and for a fast pass. Their `lp` and `da` cells are
    # the dataclass defaults, so a metrology run of a v1 task claims no v1
    # baseline either.
    #
    # `evaluate(protocol="metrology")` picks `_h` or `_s` from the primary
    # class of the profile and records which one it picked. That choice is
    # made outside this module; the profile is not known here.

    # Class H: many pairs, sources capped so the BFS cost stays bounded,
    # the per-shell table on, and the bootstrap interval on. `min_hop=2`
    # and `k=10` are already the defaults. `neg_ratio=100` is the point of
    # `link_rank`: a balanced AUC reads near 0.99 on every class H graph.
    "metrology_h": Protocol(
        dr=DRCfg(n_sources=200, per_shell=True, bootstrap=N_BOOT),
        ds=DSCfg(n_sources=200),
        lr=LRCfg(neg_ratio=100.0),
    ),

    # Class S: a road network or a tree. Hop distances run into the
    # hundreds, so the pairs are drawn one shell at a time and `h = 1` is
    # kept -- on this class the near shells are the measurement. `max_hop`
    # stays 0, which is no cap; B12 writes `diam_est`, and no reachable
    # pair is farther than the diameter, so the two agree. `k = deg(u)`
    # because a fixed k = 10 asks for ten neighbours from a node that has
    # two.
    "metrology_s": Protocol(
        dr=DRCfg(min_hop=1, pair_draw="stratified_by_hop", per_shell=True,
                 cross_component="drop"),
        ds=DSCfg(min_hop=1, pair_draw="stratified_by_hop",
                 cross_component="drop"),
        rt=RTCfg(k_mode="degree"),
    ),

    # L1 and L3: a smoke run and an optimiser screen, where turnaround
    # matters. A quarter of the pairs, no bootstrap, k = 10 fixed. `k` and
    # `bootstrap` are already the defaults.
    "metrology_quick": Protocol(
        dr=DRCfg(n_pairs=5_000),
        ds=DSCfg(n_pairs=5_000),
    ),
})


# ---------------------------------------------------------------------------
# The two functions
# ---------------------------------------------------------------------------
def protocol(name: str) -> Protocol:
    """The whole `Protocol` of `name`, all eight task configurations.

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


def get(name: str) -> tuple[LPCfg, DACfg]:
    """The link-prediction and hop-distance configurations of `name`.

    This KEEPS the v1 contract and returns the pair, because the two v1
    task modules index it: `tasks/link_prediction.py:75` takes `[0]` and
    `tasks/dist_approx.py:76` takes `[1]`. Both files are frozen and must
    stay byte-identical, so the return type of this function cannot
    change. A v2 task calls `protocol(name)` and takes the field it wants.
    """
    p = protocol(name)
    return p.lp, p.da


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


# ---------------------------------------------------------------------------
# The noise floors
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Floor:
    """How large a difference must be before it counts, for one metric.

    `rel` is a fraction of the older value (0.015 is 1.5%); `abs` is a
    difference in the metric's own units. Either may be `None`. `date` is
    when the floor was set and `source` is where it came from, because a
    floor that nobody can trace back cannot be revised.

    A floor is only HALF of the rule (metrology.md section 8). The other
    half is the seed spread, and both must be passed. `source="unset"`
    means no floor was measured yet; `compare()` then returns
    `MEASUREMENT`, which is the correct answer and not a missing feature.
    """

    rel: float | None = None
    abs: float | None = None
    date: str | None = None
    source: str = "unset"


# The 1.5% floor was set for AUC and R2 and it does NOT transfer to a rank
# correlation or to stress: metrology.md section 8 says each new metric
# gets its own floor by bootstrap over the scored pairs on cora and pubmed
# at 3 seeds, BEFORE any verdict uses it. `floors.py` (PRD-v2 B17)
# measures one and prints the line to paste here; it does not edit this
# file, because a floor is a reviewed change.
_PLAN = Floor(rel=0.015, abs=None, date="2026-08-18",
              source="PLAN.md 2026-08-18T01:30")
_UNSET = Floor()

FLOORS: types.MappingProxyType = types.MappingProxyType({
    # v1, measured. The two conditions of the rule are in force for these.
    "auc": _PLAN,
    "accuracy": _PLAN,
    "f1_score": _PLAN,
    "r2": _PLAN,
    "mae": _PLAN,

    # v1, no floor of their own. They ride beside the five above.
    "precision": _UNSET,
    "recall": _UNSET,
    "rmse": _UNSET,
    "mre": _UNSET,
    "exact": _UNSET,

    # dist_rank (PRD-v2 5.2)
    "rho": _UNSET,
    "tau_b": _UNSET,
    "somers_d": _UNSET,
    "tie_frac_y": _UNSET,
    "tau_b_max": _UNSET,
    "tau_b_ratio": _UNSET,
    "n_pairs_scored": _UNSET,
    "rho_src_mean": _UNSET,
    "rho_src_std": _UNSET,

    # dist_stress
    "stress1": _UNSET,
    "scale": _UNSET,
    "distortion_mean": _UNSET,
    "distortion_max": _UNSET,
    "distortion_p99": _UNSET,
    "isotonic_r2": _UNSET,
    "h_star": _UNSET,
    "h_star_isotonic": _UNSET,

    # retrieval
    "precision_at_k": _UNSET,
    "recall_at_k": _UNSET,
    "map": _UNSET,
    "mrr": _UNSET,
    "hits_at_k": _UNSET,
    "knn_overlap": _UNSET,
    "trustworthiness": _UNSET,
    "continuity": _UNSET,
    "reconstruction": _UNSET,

    # link_rank
    "dist_auc": _UNSET,
    "dist_ap": _UNSET,
    "dist_ap_at_ratio": _UNSET,
    "neg_ratio_used": _UNSET,

    # structure
    "nmi": _UNSET,
    "ari": _UNSET,
    "conductance_mean": _UNSET,
    "conductance_max": _UNSET,
    "rho_norm_degree": _UNSET,
    "rho_norm_centrality": _UNSET,
    "n_communities": _UNSET,
    "modularity": _UNSET,

    # stability
    "procrustes_rms": _UNSET,
    "procrustes_rms_std": _UNSET,
    "knn_jaccard_mean": _UNSET,
    "knn_jaccard_std": _UNSET,
    "pairs_evaluated": _UNSET,
})


# ---------------------------------------------------------------------------
# Which metric applies to which graph class
# ---------------------------------------------------------------------------
# `APPLIES[cls][metric]` is `True`, `False` or `"secondary"`, filled from
# PRD-v2 B16 and metrology.md section 6. The key is the PRIMARY class of
# the profile, and that is `"H"` or `"S"` and nothing else (PRD-v2 5.3:
# `S` when the average degree is at most 3 or the graph is a tree, else
# `H`; `B`, `L`, `W` and `C` are secondary flags and `D` and `T` are
# reserved).
#
# `False` NEVER suppresses a number. The report prints it in brackets with
# the class note beside it. Hop R2 on a road network reads 0.00 to 0.02 at
# every epoch count and it is still printed, because every recorded table
# has it.
#
# A metric that this table does not name has no opinion recorded, and the
# report prints it plain.
_H = types.MappingProxyType({
    # B16 row 1: rank correlation on the hop distance. Primary on both.
    "rho": True, "somers_d": True, "tau_b": True,
    # B16 row 2: magnitude. Secondary on H, where the hop distance takes
    # five to eight values and the layout has little room to stretch.
    "stress1": "secondary", "distortion_mean": "secondary",
    "distortion_max": "secondary", "distortion_p99": "secondary",
    # B16 row 3: the model-based hop regression of `dist_approx`.
    "r2": "secondary", "mae": "secondary",
    # B16 row 4: `link_prediction`. It saturates near 0.99 and
    # discriminates nothing; it is reported because it is expected.
    "auc": "secondary",
    # B16 row 5: `link_rank`, the model-free form.
    "dist_auc": True, "dist_ap": True,
    # B16 row 6: retrieval, primary for local structure.
    "precision_at_k": True, "recall_at_k": True, "map": True, "mrr": True,
    "hits_at_k": True, "knn_overlap": True, "trustworthiness": True,
    "continuity": True, "reconstruction": True,
    # B16 row 7: structure without labels.
    "nmi": "secondary", "ari": "secondary",
    "conductance_mean": "secondary", "conductance_max": "secondary",
    "rho_norm_degree": "secondary", "rho_norm_centrality": "secondary",
})

_S = types.MappingProxyType({
    # B16 row 1: per shell, and up to `h_star`. Over the whole pair sample
    # a class S graph mixes shells that are hundreds of hops apart.
    "rho": True, "somers_d": True, "tau_b": True,
    # B16 row 2: primary here. A near-planar graph or a tree has a low
    # intrinsic dimension, and stress reads the map that rank ignores.
    "stress1": True, "distortion_mean": True,
    "distortion_max": True, "distortion_p99": True,
    # B16 row 3: NOT the metric for this class.
    "r2": False, "mae": False,
    # B16 row 4.
    "auc": "secondary",
    # B16 row 5.
    "dist_auc": True, "dist_ap": True,
    # B16 row 6.
    "precision_at_k": True, "recall_at_k": True, "map": True, "mrr": True,
    "hits_at_k": True, "knn_overlap": True, "trustworthiness": True,
    "continuity": True, "reconstruction": True,
    # B16 row 7: off for class S.
    "nmi": False, "ari": False,
    "conductance_mean": False, "conductance_max": False,
    "rho_norm_degree": False, "rho_norm_centrality": False,
})

APPLIES: types.MappingProxyType = types.MappingProxyType({"H": _H, "S": _S})
