#!/bin/env python3
"""evaluator.rank -- does the distance ORDER of Z match the hop order?

The claim of this project is one sentence: the Euclidean distance in the
embedding `Z` is monotone in the graph distance. Every other score of this
package puts a trained model between the layout and the number. This
module puts nothing between them. It reads two arrays of the same length:

    x   the distance of a pair in Z, float, continuous
    y   the true hop distance of the same pair, a small integer

and it returns how well the order of `x` follows the order of `y`.

WHY A RANK SCORE AND NOT THE HOP R2. On a road network or a tree the hop
R2 reads 0.00 to 0.02 at every epoch count while link-prediction AUC on
the SAME graph reads 1.0000. The R2 is not wrong; it is the wrong
instrument. The per-shell rank correlation of this module replaces it for
those graphs (METROLOGY.md sections 4 and 6.1).

THE LIBRARY POLICY (PRD-v2 section 3.1). A coefficient that `scipy.stats`
ships is CALLED, never rewritten. `spearman`, `kendall_tau_b`, `somers_d`
and `bootstrap` are thin wrappers, and they exist only to fix the
argument order, to unpack the result object, and to return a plain
`float`. Own code is written for exactly three reasons and each own
function names its reason in its docstring. Every own function here is
reason (a): the library does not compute it.

READ THIS BEFORE YOU READ A NUMBER. A rank coefficient over a hop
distance is unreadable alone, because `y` holds maybe six distinct values
over 20,000 pairs:

  * `tie_frac(y)` is MANDATORY beside any coefficient of this module.
  * tau-b cannot reach 1.0 when `y` is tied and `x` is continuous. Report
    it against `tau_b_max(y)`, the ceiling of a PERFECT layout under the
    observed tie pattern.
  * rho and tau are never compared to each other. tau is roughly two
    thirds of rho on well-behaved data, and that gap is not a finding.

IMPORT COST. `scipy.stats` costs 0.41 s and it sits at the top of this
module. `import evaluator` does NOT reach here (this module is not in
`evaluator/__init__.py`), and it measured 0.20 s on 2026-09-02. A later
task module that imports `evaluator.rank` at its top therefore pays
0.61 s in total, under the 1.0 s budget of PRD invariant I6. No
`sklearn`, no `networkit`, no `gensim` is imported here, at any time.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from . import config, guards

__all__ = [
    "spearman", "kendall_tau_b", "somers_d",
    "tie_frac", "tau_b_max", "per_shell", "h_star", "bootstrap",
]


def _ranked(x):
    """`x` as a float64 array, after the check that it can be ranked.

    A constant `x` -- every pair at the same distance in Z -- gives `nan`
    from every coefficient below, and a `nan` in a table reads as a
    missing run and not as a collapsed layout. Guard C7
    (`check_target_degeneracy`) is the package's existing test for "one
    value holds almost the whole sample", so it is reused here on `x`. It
    RAISES `DegenerateEvaluation`; it does not warn.

    `y` is NOT checked here. A heavily tied `y` is the normal case for a
    hop distance, `tie_frac` reports it, and the task layer applies C7 to
    the hop target.
    """
    x = np.asarray(x, dtype=np.float64)
    try:
        guards.check_target_degeneracy(x, strict=True)
    except guards.DegenerateEvaluation as exc:
        # C7 was written for the hop target, so its text says "hop value".
        # The prefix names what was actually measured.
        raise guards.DegenerateEvaluation(
            f"rank: the distances in Z are degenerate, thus no rank score "
            f"can be read from them. {exc}") from exc
    return x


# ---------------------------------------------------------------------------
# The coefficients -- scipy.stats, wrapped thin
# ---------------------------------------------------------------------------
def spearman(x, y) -> float:
    """Spearman rho of `x` against `y`. `scipy.stats.spearmanr`.

    MIDRANKS, which is the library's default and what METROLOGY.md
    section 6.1 asks for. The school form `1 - 6*sum(d^2)/(n*(n^2-1))` is
    wrong under ties and is not used.

    rho = 1 means the layout orders every pair as the graph does.
    """
    x = _ranked(x)
    return float(stats.spearmanr(x, np.asarray(y, dtype=np.float64)).statistic)


def kendall_tau_b(x, y) -> float:
    """Kendall tau-b of `x` against `y`. `scipy.stats.kendalltau`.

    `(C - D) / sqrt((C + D + T_x) * (C + D + T_y))` over every pair of
    pairs. It reads as a probability difference: tau-b = 0.6 says that for
    two sampled pairs, agreement is 60 percentage points more likely than
    disagreement.

    Variant "b" removes the tied comparisons from the denominator, one
    variable at a time. Variant "a" counts a tie as a failure, and with a
    hop distance the tie count is enormous, so "a" is crushed toward zero
    for a reason that says nothing about the layout.

    THE CEILING. With `y` tied and `x` continuous, tau-b cannot reach 1.0
    even for a perfect layout. Report it against `tau_b_max(y)`.
    """
    x = _ranked(x)
    return float(stats.kendalltau(x, np.asarray(y, dtype=np.float64),
                                  variant="b").statistic)


def somers_d(x, y) -> float:
    """Somers' D of the response `y` given the predictor `x`, `D(Y|X)`.

    `D_yx = (C - D) / (C + D + T_y)`: tau conditioned on ties in the
    RESPONSE only. This is the cleaner choice here, because `y` is an
    ordinal target (a hop count) and `x` is continuous (a distance in Z).
    `config.HEADLINE_RANK` names it as the headline of the metrology.

    THE ARGUMENT ORDER IS THE TRAP OF THIS BLOCK, AND THE PRD HAS IT
    BACKWARDS. PRD-v2 B9 asks for `somersd(y, x)`. That call returns
    `D(X|Y)`, the WRONG statistic. `scipy.stats.somersd` takes the
    NORMALIZING variable first, which is the PREDICTOR, so the right call
    is `somersd(x, y)`. Two planted cases and scipy's own definition
    agree:

      * scipy defines `D(Y|X) = tau_a(X, Y) / tau_a(X, X)`, and the
        denominator counts the pairs UNTIED IN X. That is
        `(C - D) / (C + D + T_y)`, the formula METROLOGY.md section 6.1
        writes for `D_yx`.
      * four-pair example of `papers/ordered-rank-criteria.md` (C=4, D=1,
        T_y=1, T_x=0): `somersd(x, y)` gives 3/6 = 0.5, which is `D_yx`.
        `somersd(y, x)` gives 3/5 = 0.6, which conditions on ties in the
        layout and is not the question.
      * `x = [1, 1, 2, 3]`, `y = [1, 2, 3, 4]` (C=5, D=0, T_x=1, T_y=0):
        `somersd(x, y)` gives 5/5 = 1.0 and `somersd(y, x)` gives
        5/6 = 0.833. Only the first is `D_yx`.

    Verified on scipy 1.15.3 on 2026-09-02. A planted test holds the
    order, so an upstream change is caught by the gate and not by a paper.

    THE CEILING, the same caveat `tau_b_max` carries. `D_yx` divides by
    the pairs untied in `x`, and those still hold every `y` tie, so a
    PERFECT layout reaches `1 - tie_frac(y)` and not 1.0.
    """
    x = _ranked(x)
    return float(stats.somersd(x, np.asarray(y, dtype=np.float64)).statistic)


# ---------------------------------------------------------------------------
# Own code, reason (a): the library does not compute these
# ---------------------------------------------------------------------------
def tie_frac(y) -> float:
    """The fraction of pair-of-pair comparisons that `y` ties.

    `sum over the distinct values of t*(t - 1) / (n*(n - 1))`, where `t`
    is the count of one value. 0.0 means `y` has no repeat; a value near
    1.0 means almost every comparison is tied and no rank coefficient can
    say anything.

    Own code, reason (a): no `scipy.stats` function returns it. MANDATORY
    beside any coefficient of this module (METROLOGY.md section 6.1).
    """
    y = np.asarray(y)
    n = y.size
    if n < 2:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    counts = counts.astype(np.float64)
    return float(np.sum(counts * (counts - 1.0)) / (n * (n - 1.0)))


def tau_b_max(y) -> float:
    """The tau-b of a PERFECT layout under the tie pattern of `y`.

    The ceiling that `kendall_tau_b` is reported against. `x` is built so
    that it orders `y` exactly and holds NO tie of its own, thus `T_x = 0`
    and only the ties of `y` limit the answer.

    The build uses the DENSE RANK of `y` plus a step in `[0, 1)`. The step
    is distinct for every position, so `x` has no tie, and it is smaller
    than the gap of 1 between two dense ranks, so `x` never crosses a
    shell. The tie-break inside one shell is arbitrary and it does not
    matter: those comparisons are the `T_y` that leave the numerator.

    COMPUTE IT ONCE PER `y`. It does not depend on the layout, so a caller
    that scores ten embeddings of one pair sample calls this one time.

    Own code, reason (a): the library gives no attainable maximum. The
    closed form `sqrt(1 - tie_frac(y))` is the same number and is used as
    the planted check, but the library call below is what ships, per the
    library policy.
    """
    y = np.asarray(y)
    n = y.size
    if n < 2:
        return float("nan")
    _, dense = np.unique(y, return_inverse=True)
    x = dense.astype(np.float64) + np.arange(n, dtype=np.float64) / n
    return float(stats.kendalltau(x, y, variant="b").statistic)


def per_shell(x, y):
    """One row for each hop shell `h` of `sorted(unique(y))`.

    The shell at which the layout stops resolving distance IS the
    resolution horizon `h*`, and this table is how it is read.

    Each row is a dict:

      `h`          the hop value of the shell.
      `n`          how many pairs sit at exactly this hop.
      `rho_cum`    Spearman rho of `x` against `y` over the shells `<= h`.
                   This is the column `h_star` reads.
      `rho_shell`  Spearman rho over the shells `h` and the one before it.
      `x_mean`     the mean distance in Z of the pairs at exactly `h`.
      `x_std`      their standard deviation.
      `r2_shell`   R2 of a straight-line fit over the same two shells
                   `rho_shell` uses. For one predictor, R2 is the squared
                   Pearson correlation, so the direction of the fit does
                   not change it.

    TWO SHELLS AND NOT ONE, AND WHY. Inside one shell `y` is a constant,
    so a correlation and a line fit over that shell alone are both
    undefined and return `nan`. PRD-v2 B9 and METROLOGY.md section 6.1
    say "restricted to pairs at hop h", which cannot be computed. The
    smallest set on which the question "does the layout still separate
    shell `h`?" HAS an answer is the shell and the one before it, so
    `rho_shell` and `r2_shell` use that pair of shells. This reading is
    NOT in the PRD. It is flagged for the owner.

    The FIRST row has no shell before it and no second value of `y`, so
    its `rho_cum`, `rho_shell` and `r2_shell` are all `nan`. That is not a
    failure; there is nothing to correlate yet.

    Own code, reason (a): the library has no per-shell form. It calls
    `spearman` above for every row.
    """
    x = _ranked(x)
    y = np.asarray(y, dtype=np.float64)
    shells = np.unique(y)
    rows = []
    for i, h in enumerate(shells):
        at = y == h
        xs = x[at]
        cum = y <= h
        adj = at if i == 0 else (at | (y == shells[i - 1]))
        rows.append({
            "h": float(h),
            "n": int(at.sum()),
            "rho_cum": _rho_or_nan(x[cum], y[cum]),
            "rho_shell": _rho_or_nan(x[adj], y[adj]),
            "x_mean": float(xs.mean()) if xs.size else float("nan"),
            "x_std": float(xs.std()) if xs.size else float("nan"),
            "r2_shell": _r2_or_nan(x[adj], y[adj]),
        })
    return rows


def _rho_or_nan(x, y) -> float:
    """Spearman rho, or `nan` when `y` holds one value alone.

    `scipy.stats.spearmanr` returns `nan` with a `ConstantInputWarning`
    for that case. The test here keeps the warning out of a table run.
    """
    if y.size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def _r2_or_nan(x, y) -> float:
    """R2 of a straight-line fit of `x` on `y`, or `nan` when undefined.

    With one predictor, R2 equals the squared Pearson correlation, so
    `np.corrcoef` is the whole fit. Undefined when either side is a
    constant.
    """
    if y.size < 2 or np.unique(y).size < 2 or np.unique(x).size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def h_star(shells, tol=config.H_STAR_TOL) -> float:
    """The resolution horizon: the largest `h` whose `rho_cum` stays
    within `tol` of the best `rho_cum` of the table.

    `shells` is the list `per_shell` returns. Rows whose `rho_cum` is
    `nan` -- the first shell always, and any shell with one value of `y`
    below it -- take no part, and `nan` comes back when no row is left.

    THE RULE IS PROVISIONAL (PRD-v2 B9). `tol` defaults to
    `config.H_STAR_TOL`, which is 0.02 today. The number is NOT written
    here, so a change of the rule is a change of one line of
    `evaluator/config.py`.

    Own code, reason (a): the rule is this project's, not a library's.
    """
    ok = [r for r in shells if not np.isnan(r["rho_cum"])]
    if not ok:
        return float("nan")
    best = max(r["rho_cum"] for r in ok)
    within = [r["h"] for r in ok if r["rho_cum"] >= best - tol]
    return float(max(within))


# ---------------------------------------------------------------------------
# The bootstrap -- the wrapper is ours, the resampling is scipy's
# ---------------------------------------------------------------------------
def bootstrap(fn, x, y, n_boot=config.N_BOOT, rng=None):
    """A percentile bootstrap interval of `fn(x, y)`.
    Returns the `scipy.stats.BootstrapResult`.

    `paired=True` keeps a pair together: a resample takes the same index
    from `x` and from `y`, which is the only correct resampling of a
    correlation. `method="percentile"` is what METROLOGY.md section 8
    asks for, and `n_boot` defaults to `config.N_BOOT`, which is 1,000.

    `vectorized=False` is stated and not left to detection. `scipy`
    decides by looking for an `axis` argument in the signature of `fn`,
    and `scipy.stats.spearmanr` HAS one. A caller who passes the library
    function straight in would then get the vectorised path, an `axis`
    keyword `spearmanr` reads differently, and a wrong interval with no
    error. The wrappers of this module take no `axis`, so the detection
    would be right for them and wrong for the obvious mistake.

    `rng` is required. A resample is a draw, so a result without a stated
    generator is not reproducible (PRD invariant I1).

    Own code: the wrapper alone.
    """
    if rng is None:
        raise ValueError(
            "bootstrap: `rng` is required. A resample is a draw, thus an "
            "interval without a stated generator cannot be reproduced. "
            "Pass the generator of the task.")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return stats.bootstrap((x, y), fn, paired=True, n_resamples=n_boot,
                           method="percentile", vectorized=False,
                           random_state=rng)
