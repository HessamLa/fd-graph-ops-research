#!/bin/env python3
"""The v1 parity gate: P1, P2, P5, P6a, P6b on cora, P3 and P4 on pubmed.

WHAT THIS FILE GUARDS. `evaluator` reproduces six recorded baselines
EXACTLY, and that exactness is the whole value of the package: a number is
comparable to a number that used the same protocol, and to no other. The
draws live in `evaluator/pairs.py` and `evaluator/hops.py`, and
`pairs.py` says why an edit there is dangerous:

    THE GENERATOR ORDER IS PART OF THE PARITY.

A `numpy` Generator is a stream. A moved draw gives other pairs from the
same seed, nothing raises, and the run reports a plausible WRONG NUMBER.
This file is the gate that catches such an edit.

THE REFERENCE is `tests/reference/`, the frozen verbatim copies of the
original functions. Every expected value below comes from one of those
functions, run in the test. No expected value is written down and none
comes from the package itself.

THE EMBEDDING. Every parity run needs one `Z`, and a different `Z` gives
different numbers, so this one is built in the test and written down here:

    Z = np.random.default_rng(0).standard_normal((n, 128))

128 columns, float64, generator seed 0 -- which is NOT the protocol seed
42. The scores it produces are near chance, because a random layout holds
no structure. That is correct for a parity test: the test asks whether two
code paths give the SAME number, never whether the number is good.

THE TOLERANCE IS 1e-12 AND NOT ZERO (PRD section 9, measured 2026-08-22).
The reference is not bit-stable against itself: three back-to-back runs of
`reference/other_ge.task_sp_regression` on cora gave `rf_mae` values
spanning 4.44e-16, about one ULP, because `RandomForestRegressor(n_jobs=-1)`
adds the per-tree outputs into a shared array across threads and the
summation order varies. So no test here asserts bit-equality on the output
of an estimator. Bit-equality IS asserted upstream of every estimator --
the pair arrays, the hop arrays, the feature matrices and the state of the
generator -- and those `np.array_equal` tests are what actually guards the
draw order.

Run the default gate:      .venv/bin/python -m pytest evaluator/tests/test_parity_v1.py -v
Add P3 and P4 (pubmed):    ... --runslow
"""
from __future__ import annotations

import numpy as np
import pytest

import evaluator as ev
from evaluator import hops as hops_mod
from evaluator import metrics, pairs as pairs_mod
from evaluator.tests.reference import fodined_lp, fodiwalk_eval, other_ge

TOL = 1e-12
SEED = 42                  # the seed of the whole table, PRD section 9
Z_SEED = 0                 # the embedding generator; see the docstring
N_DIM = 128

LP_KEYS = ("accuracy", "precision", "recall", "f1_score", "auc")


def ref_key(key, f1_spelling):
    """The frozen reference's own name for the package score `key`.

    One score has THREE spellings. The package says `f1_score`;
    `reference/other_ge.py:74` says `f1`; `reference/fodined_lp.py:102`
    says `f1-score`. A hyphen is not a Python identifier, which is why the
    package does not use that form. The frozen copies are never renamed,
    so every comparison maps the name here and each caller passes the
    spelling ITS reference uses. Mapping in one place and forgetting the
    other is how P1 broke on 2026-09-07.
    """
    return f1_spelling if key == "f1_score" else key

LP_SIZES = ("positives", "negatives", "edges", "pairs", "train", "test")
REG_KEYS = ("mae", "mre", "rmse", "r2", "exact")

# The recorded cells of each baseline, written here as literals and not
# read from `evaluator.config`. A test that took the pair budget from the
# package would pass after a wrong edit to that budget.
OTHERGE_PAIRS = 20_000     # `bench_other_ge.task_sp_regression(n_pairs=)`
FODINED_MAX_PAIRS = 50_000  # `fodined/modular.py:538`, MAX_LP_PAIRS
FODIWALK_SOURCES = 200     # `bench_fodiwalk.py:64`
FODIWALK_PAIRS = 20_000    # `bench_fodiwalk.py:65`
FODIWALK_MIN_HOP = 2       # `fodiwalk/tests/harness.py:29`, applied by the
                           # CALLER of `hop_sample`, never inside it


# ---------------------------------------------------------------------------
# The graphs and the embedding
# ---------------------------------------------------------------------------
def _graph(name):
    """`(A, n, info, Z)` for a registry name. `Z` is the docstring's."""
    A, n, info = ev.load_graph(name)
    Z = np.random.default_rng(Z_SEED).standard_normal((n, N_DIM))
    return A, n, info, Z


@pytest.fixture(scope="session")
def cora():
    return _graph("cora")


@pytest.fixture(scope="session")
def pubmed():
    return _graph("pubmed")


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------
def check(record_diffs, values):
    """Record `|package - reference|` for each value, fail over 1e-12.

    `values` is `(name, package value, reference value)`. Every difference
    reaches the run's summary, so a passing test still reports the number
    it measured.
    """
    got = [(name, abs(float(a) - float(b))) for name, a, b in values]
    record_diffs(got)
    over = [f"{name}={d:.3e}" for name, d in got if not d <= TOL]
    assert not over, ("over the 1e-12 tolerance: " + ", ".join(over))


def state(rng, k=4):
    """`k` draws of `rng`. Two generators left in the same state give the
    same draws, so this compares the STATE, which is what a moved draw
    changes and what a comparison of the returned pairs alone can miss."""
    return rng.integers(0, 2 ** 62, k)


# ---------------------------------------------------------------------------
# Upstream of every estimator: byte-exact, and this is the real gate
# ---------------------------------------------------------------------------
def test_pairs_lp_bitexact(cora):
    """`pairs.positives` and `pairs.negatives` against the frozen samplers.

    `negatives(draw='reject')` must equal `other_ge.sample_non_edges`, the
    sampler of `otherge` and `n2v1m`. `positives(draw='over_cap')` must
    equal `fodined_lp.sample_positives`; `max_count` is set below the edge
    count on purpose, so the `rng.choice` of the cap really runs.

    Both the PAIRS and the generator STATE are compared. Equal pairs with
    a moved generator still breaks every later draw.
    """
    A, n, _, _ = cora
    r_pkg, r_ref = np.random.default_rng(SEED), np.random.default_rng(SEED)
    cap = A.nnz // 2 // 2                    # under the edge count, so it bites
    pos_pkg = pairs_mod.positives(A, cap, r_pkg, "over_cap")
    pos_ref = fodined_lp.sample_positives(A, cap, r_ref)
    assert np.array_equal(pos_pkg, pos_ref)

    neg_pkg = pairs_mod.negatives(A, n, OTHERGE_PAIRS, r_pkg, draw="reject")
    neg_ref = other_ge.sample_non_edges(A, n, OTHERGE_PAIRS, r_ref)
    assert np.array_equal(neg_pkg, neg_ref)
    assert np.array_equal(state(r_pkg), state(r_ref))


def test_features_lp_bitexact(cora):
    """`metrics.feature(kind='hadamard')` against `fodined_lp.edge_features`.

    The feature matrix is the last object upstream of the classifier, so
    it is the last one that a test may compare byte for byte.
    """
    A, n, _, Z = cora
    rng = np.random.default_rng(SEED)
    pos = pairs_mod.positives(A, FODINED_MAX_PAIRS // 2, rng, "over_cap")
    neg = pairs_mod.negatives(A, n, pos.shape[0], rng, draw="reject")
    pr = np.vstack([pos, neg])
    X_ref, y_ref = fodined_lp.edge_features(Z, pos, neg)
    X_pkg = metrics.feature(Z, pr[:, 0], pr[:, 1], "hadamard")
    y_pkg = np.concatenate([np.ones(pos.shape[0]), np.zeros(neg.shape[0])])
    assert np.array_equal(X_pkg, X_ref)
    assert np.array_equal(y_pkg, y_ref)


def test_pairs_hops_fodiwalk_bitexact(cora):
    """`pairs.pairs_for_hops(draw='bfs_rows')` against `hop_sample`.

    This is the draw of `fodiwalk_dist` and `fodiwalk_vec`: one
    `rng.choice` for each BFS row, inside the block loop, so the draw of a
    row depends on how many rows came before it. The pairs, the HOP
    ARRAY and the generator state are all compared.

    The source draw sits in the CALLER (`tasks/dist_approx.py` step 1),
    and it is the first draw of the task; `hop_sample` makes the same draw
    on its first line.
    """
    A, n, _, Z = cora
    r_ref = np.random.default_rng(SEED)
    u_ref, v_ref, d_ref, _ = fodiwalk_eval.hop_sample(
        A, n, r_ref, FODIWALK_SOURCES, FODIWALK_PAIRS)

    r_pkg = np.random.default_rng(SEED)
    src = r_pkg.choice(n, size=min(FODIWALK_SOURCES, n), replace=False)
    pr, h = pairs_mod.pairs_for_hops(A, n, FODIWALK_PAIRS, r_pkg, src,
                                     "bfs_rows")
    assert np.array_equal(pr[:, 0], u_ref)
    assert np.array_equal(pr[:, 1], v_ref)
    assert np.array_equal(h, d_ref)
    assert np.array_equal(state(r_pkg), state(r_ref))

    # The two feature widths, after the caller's `min_hop` filter. The two
    # expressions are the ones inside `fodiwalk_eval.task_hop`.
    keep = d_ref >= FODIWALK_MIN_HOP
    u, v = pr[keep, 0], pr[keep, 1]
    assert np.array_equal(
        metrics.distance(Z, u, v, "euclidean")[:, None],
        np.linalg.norm(Z[u] - Z[v], axis=1)[:, None])
    assert np.array_equal(metrics.feature(Z, u, v, "l1"),
                          np.abs(Z[u] - Z[v]))


# ---------------------------------------------------------------------------
# P1, P3 -- link prediction, `otherge`
# ---------------------------------------------------------------------------
def _p_lp_otherge(graph, record_diffs):
    A, n, info, Z = graph
    ref = other_ge.task_link_prediction(A, n, Z, SEED)
    got = ev.link_prediction((A, n, info), Z, protocol="otherge", seed=SEED)
    check(record_diffs,
          [(k, got.scores[k], ref[ref_key(k, "f1")]) for k in LP_KEYS])


def test_p1_lp_otherge_cora(cora, record_diffs):
    """P1: LP, cora, `otherge`, seed 42, against
    `reference/other_ge.task_link_prediction`."""
    _p_lp_otherge(cora, record_diffs)


@pytest.mark.slow
def test_p3_lp_otherge_pubmed(pubmed, record_diffs):
    """P3: the same run on pubmed, where the 40,000 positive cap really
    bites and the `rng.choice` of `pos_draw='over_cap'` runs."""
    _p_lp_otherge(pubmed, record_diffs)


# ---------------------------------------------------------------------------
# P2, P4 -- distance approximation, `otherge`
# ---------------------------------------------------------------------------
def _p_da_otherge(graph, record_diffs):
    A, n, info, Z = graph
    # `method` is anything but "poincare"; both give the euclidean branch.
    ref = other_ge.task_sp_regression(A, n, Z, "euclidean", SEED,
                                      n_pairs=OTHERGE_PAIRS)
    got = ev.dist_approx((A, n, info), Z, protocol="otherge", seed=SEED)
    values = [(k, got.sizes[k], ref[k])
              for k in ("n_pairs", "hop_min", "hop_max")]
    # The reference tags the three models `base`, `rf`, `mlp`; the package
    # names the first one `baseline` (PRD block B6).
    for pkg_name, ref_tag in (("baseline", "base"), ("rf", "rf"),
                              ("mlp", "mlp")):
        values += [(f"{pkg_name}_{k}", got.scores[pkg_name][k],
                    ref[f"{ref_tag}_{k}"]) for k in REG_KEYS]
    check(record_diffs, values)


def test_p2_da_otherge_cora(cora, record_diffs):
    """P2: DA, cora, `otherge`, seed 42, against
    `reference/other_ge.task_sp_regression`. The hop truth is exact PLL on
    both sides."""
    _p_da_otherge(cora, record_diffs)


@pytest.mark.slow
def test_p4_da_otherge_pubmed(pubmed, record_diffs):
    """P4: the same run on pubmed."""
    _p_da_otherge(pubmed, record_diffs)


# ---------------------------------------------------------------------------
# P5 -- link prediction, `fodined`
# ---------------------------------------------------------------------------
def test_p5_lp_fodined_cora(cora, record_diffs):
    """P5: LP, cora, `fodined` cells, seed 42, against
    `reference/fodined_lp.link_prediction`.

    THE ONE OVERRIDE, and it is required by the reference and not by the
    package. `config.PROTOCOLS['fodined']` sets `neg_draw='far_pairs'`,
    because `fodined/link_prediction.py:sample_negatives` calls
    `graph_augmentation.sample_far_pairs`. The frozen copy replaced that
    call with the rejection sampler -- its own docstring calls the
    substitution "ONE CHANGE" -- so the copy is a reference for the
    REJECTION sampler and not for `fodined`, which `config.py` writes down
    in the note above `PROTOCOLS`. The test therefore passes
    `neg_draw='reject'` and it keeps every other `fodined` cell: 50,000
    pairs, 200 trees, hadamard, test size 0.2.

    Measured on cora with this `Z`: the unmodified `fodined` protocol
    gives accuracy 0.5194 against the frozen copy's 0.5128. That is the
    recorded `far_pairs` difference, not a defect.

    The six counts are compared with `==`: they are integers of the
    sampler, upstream of the classifier.
    """
    A, n, info, Z = cora
    ref, ref_sizes = fodined_lp.link_prediction(
        Z, A, n, FODINED_MAX_PAIRS, np.random.default_rng(SEED), SEED)
    got = ev.link_prediction((A, n, info), Z, protocol="fodined", seed=SEED,
                             neg_draw="reject")
    for k in LP_SIZES:
        assert got.sizes[k] == ref_sizes[k], k
    # `f1_score` here, `f1-score` in the frozen copy: a hyphen is not a
    # Python identifier. The frozen file must NOT be renamed.
    check(record_diffs,
          [(k, got.scores[k], ref[ref_key(k, "f1-score")])
           for k in LP_KEYS])


# ---------------------------------------------------------------------------
# P6a, P6b -- distance approximation, `fodiwalk_dist` and `fodiwalk_vec`
# ---------------------------------------------------------------------------
def _p_da_fodiwalk(graph, protocol, feature, record_diffs):
    A, n, info, Z = graph
    rng = np.random.default_rng(SEED)
    u, v, d, _ = fodiwalk_eval.hop_sample(A, n, rng, FODIWALK_SOURCES,
                                          FODIWALK_PAIRS)
    keep = d >= FODIWALK_MIN_HOP         # the CALLER's filter; see the note
    rows = fodiwalk_eval.task_hop(Z, u[keep], v[keep], d[keep], SEED,
                                  feature=feature, n_estimators=100,
                                  hidden=(256, 128), early_stopping=False)
    got = ev.dist_approx((A, n, info), Z, protocol=protocol, seed=SEED)
    assert got.sizes["n_pairs"] == int(keep.sum())

    values = []
    for (pkg_name, row) in zip(("baseline", "rf", "mlp"), rows):
        values += [(f"{pkg_name}_{k}", got.scores[pkg_name][k], row[i])
                   for i, k in enumerate(REG_KEYS, start=1)]
    check(record_diffs, values)


def test_p6a_da_fodiwalk_dist_cora(cora, record_diffs):
    """P6a: DA, cora, `fodiwalk_dist`, seed 42, against
    `reference/fodiwalk_eval.task_hop(feature="distance")`. ONE feature
    column, and the pairs come from the BFS rows."""
    _p_da_fodiwalk(cora, "fodiwalk_dist", "distance", record_diffs)


def test_p6b_da_fodiwalk_vec_cora(cora, record_diffs):
    """P6b: the same run at `fodiwalk_vec`, thus
    `task_hop(feature="vector")` and 128 feature columns. A model with 128
    features can win only because it has more of them, so the two rows are
    two protocols and never one."""
    _p_da_fodiwalk(cora, "fodiwalk_vec", "vector", record_diffs)
