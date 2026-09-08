#!/bin/env python3
"""evaluator.stress -- does the distance MAP stay smooth, and where does it
saturate?

A rank correlation (`rank.py`) is invariant to any monotone map: it says
the layout ORDERS distances right and nothing about the map's SHAPE.
Kruskal stress-1 at the best scale does (metrology.md section 6.2). It is
the native quality measure of a force-directed layout, which has no fixed
scale, so a stress that depends on the scale would measure the scale and
not the layout.

Four pure functions over `(x, y)` arrays: `x` is a distance read from `Z`
(`metrics.distance`), `y` is the hop distance (`hops.py`). No task, no CLI
flag, no record field lives here; `tasks/dist_stress.py` calls this.

`sklearn` is imported inside `isotonic_fit`, not at module top (PRD
invariant I6: `import evaluator` must cost under 1.0 s and must not pull
in `sklearn`).
"""
from __future__ import annotations

import numpy as np

from . import config

__all__ = ["stress1", "distortion", "isotonic_fit", "shepard_pairs"]

# The Shepard record must fit in one `.jsonl` line. 5,000 pairs plots a
# smooth diagram and it stays small; PRD-v2 B10 names this constant, not a
# config field (`DSCfg` carries only the `shepard` on/off flag).
_SHEPARD_N_KEEP = 5_000


def stress1(x, y):
    """Kruskal stress-1 of `x` against `y`, at the scalar `s*` that best
    maps `x` onto `y` in the least-squares sense.

    `s* = sum(x*y) / sum(x*x)`, then `stress = sqrt(sum((s*x - y)^2) /
    sum(y*y))`. Own code (PRD-v2 3.1 reason 2a): no library exposes
    Kruskal stress-1 on a layout already in hand with the scale returned;
    `sklearn.manifold.MDS` computes a stress internally but only for a fit
    it drives itself.

    Returns `(stress, s_star)`.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    s_star = np.sum(x * y) / np.sum(x * x)
    stress = np.sqrt(np.sum((s_star * x - y) ** 2) / np.sum(y * y))
    return float(stress), float(s_star)


def distortion(x, y, s):
    """Per-pair distortion of the scaled layout distance `s*x` against the
    true distance `y`: `max(s*x/y, y/(s*x))`, so a pair reads 1.0 when the
    two agree and grows without bound either way it disagrees.

    These are the numbers of metric-embedding theory (Bourgain,
    Thorup-Zwick); they place the method on the space-stretch frontier.

    Returns a dict with `mean`, `max`, `p99`.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sx = s * x
    ratio = np.maximum(sx / y, y / sx)
    return {
        "mean": float(np.mean(ratio)),
        "max": float(np.max(ratio)),
        "p99": float(np.percentile(ratio, 99)),
    }


def isotonic_fit(x, y, tol=config.H_STAR_TOL):
    """Isotonic regression of `y` on `x`: the best non-decreasing map,
    the calibration curve itself. Its flat region is a SECOND estimate
    of the resolution horizon (`rank.py` gives the first, from per-shell
    ρ, `rank.h_star`): where the fit stops rising, added hop distance
    buys no more layout distance.

    `h_star_isotonic` is a HOP, the same unit as `rank.py`'s per-shell
    `h_star` -- METROLOGY 6.2 calls it "the same `h*`, read a second
    way." Sort the pairs by `x`; the flat region is every point whose
    fitted value sits within `tol` of the fit's maximum (`tol` defaults
    to `config.H_STAR_TOL`, the same constant `rank.h_star` reads, not
    inlined here either). `h_star_isotonic` is the fitted `y` at the
    START of that region -- the first point, in `x`-sorted order, whose
    fit already sits within `tol` of the top. Past that point a larger
    embedded distance buys at most `tol` more resolved hop.

    An EQUALITY test here (the region within `1e-9`, not `tol`) picks
    out only the handful of points that land on the exact maximum, so
    `h_star_isotonic` collapses to `max(fitted)` -- the largest hop in
    the data, which reads as "no saturation" even on a badly saturating
    layout. `h_star_isotonic_share`, the region's share of the pairs, is
    returned beside the estimate so a horizon resting on a sliver of the
    data is visible in the number itself, not only by inspection.

    `IsotonicRegression` depends only on the SORT ORDER of `x`, not its
    scale, so both `h_star_isotonic` and its share are invariant to
    rescaling `x` by any positive constant.

    Returns a dict with `fitted` (the fit at each `x`), `isotonic_r2`,
    `h_star_isotonic`, `h_star_isotonic_share`.
    """
    from sklearn.isotonic import IsotonicRegression

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ir = IsotonicRegression(out_of_bounds="clip")
    fitted = ir.fit_transform(x, y)

    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    order = np.argsort(x)
    fitted_sorted = fitted[order]
    flat = fitted_sorted >= fitted_sorted[-1] - tol
    h_star = float(fitted_sorted[flat][0])
    share = float(np.count_nonzero(flat)) / flat.size

    return {
        "fitted": fitted,
        "isotonic_r2": float(r2),
        "h_star_isotonic": h_star,
        "h_star_isotonic_share": share,
    }


def shepard_pairs(x, y, n_keep=_SHEPARD_N_KEEP):
    """A subsample of `(x, y, fit)` for the record's Shepard diagram: the
    isotonic curve of `isotonic_fit` runs once here, then `x`, `y` and the
    fit are subsampled together.

    At most `n_keep` pairs (default 5,000) so the record stays inside one
    `.jsonl` line; below `n_keep`, every pair comes back. The subsample is
    an even stride over `x` sorted, so the kept points still span the
    full range instead of clustering.

    Returns a dict with `x`, `y`, `fit`.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fit = isotonic_fit(x, y)["fitted"]
    n = x.shape[0]

    order = np.argsort(x)
    if n > n_keep:
        idx = order[np.linspace(0, n - 1, n_keep).round().astype(int)]
    else:
        idx = order

    return {"x": x[idx], "y": y[idx], "fit": fit[idx]}
