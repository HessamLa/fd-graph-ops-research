#!/bin/env python3
"""evaluator.guards -- the checks that stop a meaningless run.

THE INCIDENT, from `bench_other_ge._pick_subtree`: a BFS truncation turned
NCBI into a STAR -- one hub of degree 19,999, and 19,999 leaves. Every
non-edge pair sat exactly 2 hops apart, thus EVERY method reached a
perfect score. Nothing crashed. The run printed a valid-looking number
that meant nothing.

`fodiwalk/core/plan_contract.py` answers the same class of defect for the
engine: it RAISES, never warns, never falls back. This module is that
pattern, for a measurement. Each message names three things -- the id, the
MEASURED value, the THRESHOLD -- thus a caller acts without reading here.

Nine checks, one function each, thus a task calls only what its step needs
(B9: C1, C2, C3, C5. B10: C6, C7, C8, C9) and a test calls one alone.
`strict=True` raises `DegenerateEvaluation`. `strict=False` returns a
warning string with the same id, for the caller to record -- the ONLY
concession invariant I5 allows.

Import discipline: `numpy` only. This module imports nothing of
`evaluator`.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "DegenerateEvaluation",
    "check_shape", "check_finite", "check_zero_rows", "check_metric",
    "check_class_balance", "check_target_spread", "check_target_degeneracy",
    "check_sample_shortfall", "check_reachability",
]


class DegenerateEvaluation(ValueError):
    """A sample or an embedding cannot give a meaningful score. Fatal."""


def _fail(strict: bool, msg: str):
    """Raise on `strict`, else return `msg` as the caller's warning."""
    if strict:
        raise DegenerateEvaluation(msg)
    return msg


# ---------------------------------------------------------------------------
# The embedding checks
# ---------------------------------------------------------------------------
def check_shape(Z, n, *, strict=True):
    """C1: `Z.shape[0]` must equal `n`, the node count of the graph."""
    rows = Z.shape[0]
    if rows != n:
        return _fail(strict,
            f"C1: Z has {rows} rows and the graph has n={n} nodes "
            f"(threshold: equal). A mismatched embedding scores the wrong "
            f"nodes.")
    return None


def check_finite(Z, *, strict=True):
    """C2: `Z` must hold no NaN and no inf."""
    bad = int(np.count_nonzero(~np.isfinite(Z)))
    if bad:
        return _fail(strict,
            f"C2: Z holds {bad} non-finite values (threshold: 0). A NaN "
            f"or an inf in the embedding gives a NaN or an inf score.")
    return None


def check_zero_rows(Z, *, strict=True, max_frac=0.5):
    """C3: at most `max_frac` of the rows of `Z` may be all-zero.

    A benchmark script leaves an all-zero row for a node that a word2vec
    vocabulary missed (`io.load_embedding`, the `zero_rows` field). A
    majority of zero rows means the method embedded almost nothing.
    """
    n = Z.shape[0]
    zero = int(np.count_nonzero(~Z.any(axis=1))) if n else 0
    frac = zero / n if n else 0.0
    if frac > max_frac:
        return _fail(strict,
            f"C3: {frac:.1%} of Z's rows are all-zero ({zero}/{n}) "
            f"(threshold: {max_frac:.0%}). Most nodes carry no vector, "
            f"and a score over them measures the zero row, not the "
            f"method.")
    return None


def check_metric(Z, metric, *, strict=True):
    """C4: with `metric="poincare"`, every row of `Z` must have norm < 1.

    A Euclidean distance in the Poincare ball has no meaning
    (`metrics.py`). A row on or outside the unit ball is not a point of
    the ball at all; `metrics._poincare` would clamp it into a plausible
    WRONG number instead of raising.
    """
    if metric != "poincare":
        return None
    norms = np.linalg.norm(Z, axis=1)
    worst = float(np.max(norms)) if norms.size else 0.0
    if worst >= 1.0:
        return _fail(strict,
            f"C4: metric='poincare' and max|z|={worst:.6g} (threshold: "
            f"< 1). A row on or outside the unit ball is not a point of "
            f"the Poincare ball.")
    return None


# ---------------------------------------------------------------------------
# The sample checks -- link prediction
# ---------------------------------------------------------------------------
def check_class_balance(y, *, strict=True, low=0.2, high=0.8):
    """C5: the positive fraction of `y` must fall inside `[low, high]`."""
    y = np.asarray(y)
    frac = float(np.mean(y == 1)) if y.size else 0.0
    if frac < low or frac > high:
        return _fail(strict,
            f"C5: the positive fraction is {frac:.1%} (threshold: "
            f"[{low:.0%}, {high:.0%}]). A classifier on a skewed split "
            f"scores high by predicting the majority class alone.")
    return None


# ---------------------------------------------------------------------------
# The sample checks -- hop distance
# ---------------------------------------------------------------------------
def check_target_spread(h, *, strict=True, min_distinct=2):
    """C6: the hop target `h` must hold at least `min_distinct` values."""
    h = np.asarray(h)
    distinct = int(np.unique(h).size)
    if distinct < min_distinct:
        return _fail(strict,
            f"C6: the hop target holds {distinct} distinct value(s) "
            f"(threshold: >= {min_distinct}). A regressor over one value "
            f"scores perfectly by predicting a constant.")
    return None


def check_target_degeneracy(h, *, strict=True, max_share=0.9):
    """C7: no single hop value may hold more than `max_share` of `h`.

    This is the NCBI star check. A BFS truncation, or any subtree that
    collapses to a star, gives EVERY non-edge pair the same hop distance
    (2), and a regressor over one value scores perfectly while measuring
    nothing about the graph.
    """
    h = np.asarray(h)
    if h.size == 0:
        return _fail(strict,
            f"C7: the hop target is empty (threshold: <= {max_share:.0%} "
            f"per value). No value can be measured against a threshold.")
    _, counts = np.unique(h, return_counts=True)
    share = float(counts.max()) / h.size
    if share > max_share:
        return _fail(strict,
            f"C7: one hop value holds {share:.1%} of the sample "
            f"(threshold: {max_share:.0%}). This is the star pattern "
            f"(bench_other_ge._pick_subtree): every non-edge pair at the "
            f"same distance, and every method then scores a perfect 1.0.")
    return None


def check_sample_shortfall(got: int, requested: int, *, strict=True,
                            min_frac=0.5):
    """C8: the sampler must return at least `min_frac` of `requested`."""
    frac = got / requested if requested else 1.0
    if frac < min_frac:
        return _fail(strict,
            f"C8: the sampler returned {got}/{requested} pairs "
            f"({frac:.1%}) (threshold: {min_frac:.0%}). A short sample "
            f"means the graph could not supply what the protocol asked "
            f"for.")
    return None


def check_reachability(h, *, strict=True, min_frac=0.1):
    """C9: at least `min_frac` of the sampled pairs must be reachable.

    `h` holds `np.inf` (`hops.hop_distance`) for a pair with no path. A
    sample that is mostly unreachable pairs measures disconnection, not
    the embedding.
    """
    h = np.asarray(h)
    n = h.size
    reach = int(np.count_nonzero(np.isfinite(h)))
    frac = reach / n if n else 0.0
    if frac < min_frac:
        return _fail(strict,
            f"C9: {frac:.1%} of the sampled pairs are reachable "
            f"({reach}/{n}) (threshold: {min_frac:.0%}). Most pairs carry "
            f"no path, and the finite ones are too few to score.")
    return None
