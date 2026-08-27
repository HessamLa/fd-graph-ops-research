#!/bin/env python3
"""evaluator.tasks.dist_approx -- does the geometry hold the HOP DISTANCE?

A regressor reads the two vectors of a node pair and must say how many
hops apart they are. The target is the graph's TRUE distance, read by
`evaluator.hops`, never a stored augmentation weight.

THE STEP ORDER IS THE PARITY CONTRACT (PRD block B10):

    1. sources = rng.choice(n, size=min(cfg.n_sources, n), replace=False)
                 when cfg.n_sources, else None
    2. pr = pairs.pairs_for_hops(A, n, cfg.n_pairs, rng, sources,
                                 cfg.pair_draw)
            -- 'bfs_rows' returns `(pairs, hops)`
    3. h  = hops.hop_distance(A, n, pr, backend, rng)
            -- SKIPPED when cfg.pair_draw == 'bfs_rows'
    4. keep = isfinite(h) & (h >= cfg.min_hop)
    5. guards C6, C7, C8, C9
    6. X = distance(...)[:, None]  when feature == 'distance'
           feature(..., 'l1')      when feature == 'vector'
    7. train_test_split(cfg.test_size, random_state=seed)
    8. baseline = the train mean; rf; mlp on a StandardScaler

STEP 1 LIVES HERE, not in `pairs.pairs_for_hops`. Both baselines draw the
sources with the same call, thus one draw in the caller keeps ONE stream
order for the `reject` and `bfs_rows` protocols alike.

STEP 4 IS AFTER THE SAMPLE, NEVER INSIDE IT. `hop_sample` keeps `d > 0`
and its CALLERS apply `d >= 2` (`fodiwalk/tests/harness.py:29`). A
`min_hop` filter inside the sampler changes the pair count, thus the
split, thus parity tests P6a and P6b.

THE FEATURE WIDTH IS THE PROTOCOL (PRD invariant I4). `distance` gives ONE
column, `vector` gives `n_dim`. Neither EVER falls back to the other: a
model with 128 features can win only because it has more of them. An
unknown name raises.

Parity to 1e-12 against `reference/other_ge.task_sp_regression`
(`otherge`) and `reference/fodiwalk_eval.task_hop` (`fodiwalk_*`).

IMPORT DISCIPLINE. `scoring` and every sklearn estimator import INSIDE
`_run`. See the docstring of `link_prediction.py`.
"""
from __future__ import annotations

import time

import numpy as np

from .. import config, guards, metrics, pairs as pairs_mod
from .. import hops as hops_mod
from ..report import TaskResult
from . import (cfg_record, load_embedding, load_graph, register,
               resolve_rng)

__all__ = ["dist_approx"]


@register("dist_approx")
def dist_approx(graph, Z, *, protocol="default", seed=42, rng=None,
                n_pairs=None, n_sources=None, min_hop=None, feature=None,
                metric=None, hops=None, models=None, test_size=None,
                strict=True) -> TaskResult:
    """Score an embedding on hop-distance approximation. Returns a
    `TaskResult` whose `scores` is `{model: {metric: value}}`.

    `graph` and `Z` accept the same forms as `link_prediction`, the
    pre-loaded triple and pair included.

    `hops` names the ground-truth backend (`auto`, `pll`, `bfs`,
    `landmark`) and `metric` names the distance of the `distance` feature.
    A keyword left at `None` takes the value of the protocol; a keyword
    given explicitly sets `protocol_modified` (PRD invariant I2).
    """
    cfg, modified = config.override(
        config.get(protocol)[1],
        n_pairs=n_pairs, n_sources=n_sources, min_hop=min_hop,
        feature=feature, metric=metric, hops=hops, models=models,
        test_size=test_size)
    A, n, _ = load_graph(graph, seed=seed)
    Z, _ = load_embedding(Z, n)
    result = _run(A, n, Z, cfg, resolve_rng(rng, seed), seed, strict)
    result.cfg = cfg_record(cfg, protocol, modified)
    result.protocol_modified = bool(modified)
    return result


def _run(A, n, Z, cfg, rng, seed, strict) -> TaskResult:
    """The eight steps of PRD B10, in the order that the module docstring
    writes down. Do not reorder them."""
    from sklearn.model_selection import train_test_split

    from .. import scoring                   # imports sklearn.metrics

    t0 = time.time()
    warns = []

    # 1. the sources. `n_sources = 0` means every node is a candidate.
    sources = None
    if cfg.n_sources:
        sources = rng.choice(n, size=min(cfg.n_sources, n), replace=False)

    # 2. the pairs. 'bfs_rows' gives the hops too: the BFS row is already
    #    in the memory, thus a second pass would cost a full sweep again.
    pr = pairs_mod.pairs_for_hops(A, n, cfg.n_pairs, rng, sources,
                                  cfg.pair_draw)

    # 3. the ground truth, unless step 2 gave it.
    if cfg.pair_draw == "bfs_rows":
        pr, h = pr
        backend = "bfs_rows"
    else:
        backend = cfg.hops
        if backend == "auto":                # resolve, so the record names it
            backend = hops_mod.choose_backend(
                n, int(np.unique(pr[:, 0]).size))
        h = hops_mod.hop_distance(A, n, pr, backend, rng)

    # 4. the filter. AFTER the sample; see the module docstring.
    reachable = int(np.count_nonzero(np.isfinite(h)))
    keep = np.isfinite(h) & (h >= cfg.min_hop)
    pr, y = pr[keep], h[keep].astype(np.float64)

    # 5. the guards. C6 and C7 read the kept target, C8 reads the count
    #    that the protocol asked for, and C9 reads the WHOLE sample --
    #    step 4 already dropped every unreachable pair, thus a C9 over the
    #    kept target would report 100% for every graph and measure nothing.
    for warn in (guards.check_target_spread(y, strict=strict),
                 guards.check_target_degeneracy(y, strict=strict),
                 guards.check_sample_shortfall(int(y.size), int(cfg.n_pairs),
                                               strict=strict),
                 guards.check_reachability(h, strict=strict)):
        if warn:
            warns.append(warn)

    # 6. the feature. ONE column or `n_dim` columns, and never a fallback.
    if cfg.feature == "distance":
        X = metrics.distance(Z, pr[:, 0], pr[:, 1], cfg.metric)[:, None]
    elif cfg.feature == "vector":
        X = metrics.feature(Z, pr[:, 0], pr[:, 1], "l1")
    else:
        raise ValueError(
            f"dist_approx: unknown feature {cfg.feature!r}; the names are "
            f"'distance' (1 column) and 'vector' (n_dim columns). The two "
            f"are different protocols and neither falls back to the other.")

    # 7. the split. NO stratify: the target is a real number.
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg.test_size, random_state=seed)

    # 8. the models, in the order of `cfg.models`.
    scores = {}
    for name in cfg.models:
        scores[name] = scoring.regress_scores(
            yte, _predict(name, cfg, seed, Xtr, ytr, Xte))

    values, counts = np.unique(y, return_counts=True)
    sizes = {"n_pairs": int(y.size),
             "hop_min": float(y.min()) if y.size else float("nan"),
             "hop_max": float(y.max()) if y.size else float("nan"),
             "reachable": reachable,
             "backend": backend,
             "hop_hist": {int(v): int(c) for v, c in zip(values, counts)},
             "sampled": int(h.size),
             "train": int(Xtr.shape[0]), "test": int(Xte.shape[0])}
    return TaskResult(task="dist_approx", cfg={}, scores=scores, sizes=sizes,
                      seconds=time.time() - t0, warnings=warns)


def _predict(name, cfg, seed, Xtr, ytr, Xte):
    """The prediction of one model of `cfg.models`.

    `baseline` is the MEAN of the TRAIN target, thus a model that beats it
    read something out of the geometry. `rf` reads the raw feature and
    `mlp` reads a standardized one: a forest splits on order alone and a
    network does not converge on an unscaled column. An unknown name
    raises and it names the three that exist.
    """
    if name == "baseline":
        return np.full(Xte.shape[0], ytr.mean())
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=cfg.rf_estimators,
                                   min_samples_leaf=cfg.rf_min_leaf,
                                   random_state=seed, n_jobs=-1)
        return rf.fit(Xtr, ytr).predict(Xte)
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        mlp = MLPRegressor(hidden_layer_sizes=tuple(cfg.mlp_hidden),
                           max_iter=300,
                           early_stopping=cfg.mlp_early_stop,
                           random_state=seed)
        mlp.fit(sc.transform(Xtr), ytr)
        return mlp.predict(sc.transform(Xte))
    raise ValueError(
        f"dist_approx: unknown model {name!r}; the names are 'baseline', "
        f"'rf', 'mlp'.")
