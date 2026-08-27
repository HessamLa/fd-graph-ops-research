#!/bin/env python3
"""evaluator.scoring -- the two frozen score sets.

The keys are FROZEN: `report.py` writes them into the record, and a
rename breaks a recorded baseline. Every value is a plain `float`, because
a numpy scalar does not serialize to JSON.

THE RENAME. `fodined/link_prediction.py:89` and
`fodiwalk/misc/evaluation.py:113` spell the harmonic mean `f1-score`. This
package spells it `f1`, because a hyphen is not a Python identifier and it
blocks the attribute and keyword forms. Map one to the other when you
compare an old record to a new one.

`zero_division=0` makes a degenerate split give 0.0, not a NaN and a
warning. `roc_auc_score` still raises on a one-class test part; `guards.py`
stops that run before it reaches here.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)

__all__ = ["classify_scores", "regress_scores"]


def classify_scores(y, pred, prob):
    """The five classification scores. `prob` scores the positive class."""
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "auc": float(roc_auc_score(y, prob))}


def regress_scores(y, pred):
    """The five regression scores of a hop-distance target.

    `mre` is the mean relative error, a scale-free companion of the MAE.
    `exact` is `mean(rint(pred) == y)`: a hop distance is an integer, thus
    the share of rounded hits reads more plainly than the MAE.
    """
    y = np.asarray(y)
    pred = np.asarray(pred)
    return {"mae": float(mean_absolute_error(y, pred)),
            "mre": float(mean_absolute_percentage_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred)),
            "exact": float(np.mean(np.rint(pred) == y))}
