#!/bin/env python3
"""evaluator.tasks.link_prediction -- does the geometry hold the adjacency?

A classifier reads ONE feature of the two vectors of a node pair and it
must say whether the pair has an edge. The positives are real edges; the
negatives are drawn pairs that have none.

THE STEP ORDER IS THE PARITY CONTRACT (PRD block B9). The generator is a
stream (`evaluator.pairs`), thus a moved step changes every later number
and nothing crashes. `_run` holds this order and no other:

    0. guard C1
    1. pos = pairs.positives(A, cfg.max_pairs // 2, rng, cfg.pos_draw)
    2. neg = pairs.negatives(A, n, round(len(pos) * cfg.neg_ratio), rng,
                             draw=cfg.neg_draw)
    3. X = metrics.feature(Z, u, v, cfg.feature);  y = [1...1, 0...0]
    4. guards C2, C3, C5
    5. train_test_split(test_size, random_state=seed, stratify=y)
    6. fit cfg.model; scoring.classify_scores

STEP 0 IS THE SHAPE GUARD, before any indexing of `Z`. PRD B9 first put C1
at step 4; a short `Z` then raised a bare `IndexError` from step 3 and the
guard whose one job is to EXPLAIN the mismatch never spoke. C1 draws no
random number, thus the move costs zero parity.

`seed` and `rng` are separate and both reach the record: `seed` goes to
sklearn (`random_state`), `rng` goes to the sampling.

Parity: `tests/reference/other_ge.task_link_prediction` (protocol
`otherge`) and `fodiwalk/misc/evaluation.link_prediction` (protocol
`fodined`), both to 1e-12.

IMPORT DISCIPLINE. `scoring` and every sklearn estimator import INSIDE
`_run`. `evaluator.scoring` imports `sklearn.metrics` at its top, thus a
top-level import here would pull sklearn into `import evaluator` and break
the 1.0 s budget of PRD B11.1.
"""
from __future__ import annotations

import time

import numpy as np

from .. import config, guards, metrics, pairs as pairs_mod
from ..report import TaskResult
from . import (cfg_record, load_embedding, load_graph, register,
               resolve_rng)

__all__ = ["link_prediction"]


@register("link_prediction")
def link_prediction(graph, Z, *, protocol="default", seed=42, rng=None,
                    max_pairs=None, test_size=None, neg_ratio=None,
                    feature=None, model=None, n_estimators=None,
                    pos_draw=None, neg_draw=None,
                    strict=True) -> TaskResult:
    """Score an embedding on link prediction. Returns a `TaskResult`.

    `graph` is a `scipy.sparse` matrix, an `(m, 2)` edge array, a path, a
    registry name, or the `(A, n, info)` triple of `io.load_graph`. `Z` is
    an `(n, d)` array, a path, or the `(Z, info)` pair of
    `io.load_embedding`.

    A keyword left at `None` takes the value of the protocol. A keyword
    given EXPLICITLY overrides the protocol and it sets
    `protocol_modified` on the result, thus a changed number cannot claim
    the name of a recorded baseline (PRD invariant I2).

    `rng` is the sampling generator. `None` gives
    `np.random.default_rng(seed)`, which is the first line of every frozen
    reference.
    """
    cfg, modified = config.override(
        config.get(protocol)[0],
        max_pairs=max_pairs, test_size=test_size, neg_ratio=neg_ratio,
        feature=feature, model=model, n_estimators=n_estimators,
        pos_draw=pos_draw, neg_draw=neg_draw)
    A, n, _ = load_graph(graph, seed=seed)
    Z, _ = load_embedding(Z, n)
    result = _run(A, n, Z, cfg, resolve_rng(rng, seed), seed, strict)
    result.cfg = cfg_record(cfg, protocol, modified)
    result.protocol_modified = bool(modified)
    return result


def _run(A, n, Z, cfg, rng, seed, strict) -> TaskResult:
    """The six steps of PRD B9, in the order that the module docstring
    writes down. Do not reorder them."""
    from sklearn.model_selection import train_test_split

    from .. import scoring                   # imports sklearn.metrics

    t0 = time.time()
    warns = []

    # 0. C1 FIRST, before any indexing of `Z`. A `Z` with fewer rows than
    # `n` makes step 3 raise a bare `IndexError`, thus a guard that runs
    # after it never speaks. The check draws nothing, thus the move costs
    # no parity (PRD B9, step 0).
    warn = guards.check_shape(Z, n, strict=strict)
    if warn:
        warns.append(warn)

    # 1. the positive pairs -- the real edges, capped
    pos = pairs_mod.positives(A, cfg.max_pairs // 2, rng, cfg.pos_draw)

    # 2. the negative pairs -- drawn, and rejected when an edge exists
    neg = pairs_mod.negatives(A, n, round(pos.shape[0] * cfg.neg_ratio),
                              rng, draw=cfg.neg_draw)

    # 3. the feature and the label
    pr = np.vstack([pos, neg])
    X = metrics.feature(Z, pr[:, 0], pr[:, 1], cfg.feature)
    y = np.concatenate([np.ones(pos.shape[0]), np.zeros(neg.shape[0])])

    # 4. the guards. C2 and C3 read the embedding; C5 reads the split.
    # C1 already ran at step 0.
    for warn in (guards.check_finite(Z, strict=strict),
                 guards.check_zero_rows(Z, strict=strict),
                 guards.check_class_balance(y, strict=strict)):
        if warn:
            warns.append(warn)

    # 5. the split. `stratify` holds the class balance in both parts.
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg.test_size, random_state=seed, stratify=y)

    # 6. the model and the scores
    clf = _model(cfg.model, cfg.n_estimators, seed).fit(Xtr, ytr)
    scores = scoring.classify_scores(yte, clf.predict(Xte),
                                     clf.predict_proba(Xte)[:, 1])

    sizes = {"positives": int(pos.shape[0]), "negatives": int(neg.shape[0]),
             "edges": int(A.nnz // 2), "pairs": int(pr.shape[0]),
             "train": int(Xtr.shape[0]), "test": int(Xte.shape[0])}
    return TaskResult(task="link_prediction", cfg={}, scores=scores,
                      sizes=sizes, seconds=time.time() - t0,
                      warnings=warns)


def _model(name, n_estimators, seed):
    """The classifier of `cfg.model`. An unknown name RAISES and it names
    the two that exist; a silent fallback to the forest would give a
    number under a model that the record does not describe."""
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=n_estimators,
                                      random_state=seed, n_jobs=-1)
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=seed)
    raise ValueError(
        f"link_prediction: unknown model {name!r}; the names are "
        f"'rf', 'logreg'.")
