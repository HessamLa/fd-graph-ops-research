#!/bin/env python3
"""reference/fodined_lp.py -- the FROZEN reference of the `fodined` link
prediction protocol, for parity test P5.

Verbatim copies of five functions of `fodiwalk/misc/evaluation.py`, taken
on 2026-08-22. That file states in its own docstring that these five came
verbatim from `fodined/link_prediction.py`; thus one copy serves both, and
P5 tests the `fodined` protocol against it.

WHY A COPY: the source belongs to ANOTHER active session, which edited the
file on 2026-08-22. A reference that a second party can edit is not a
reference. See `fodiwalk_eval.py` for the same reasoning.

ONE CHANGE, and it is the reason this file is not a plain import:
`sample_negatives` delegated to `fodiwalk.augment_graph.far_pairs`. That
import is replaced by the rejection sampler INLINE, copied from the same
package at the same moment, thus this module needs no `fodiwalk` at all
and it cannot drift when that package changes.

NOTE the key spelling: `classify` here returns `f1-score`. `evaluator`
returns `f1`. The parity test must map the one to the other; the rename is
recorded in PRD block B6.1.

DO NOT TIDY THIS FILE. See `other_ge.py`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

from .other_ge import sample_non_edges as _reject


def sample_far_pairs(n, count, ball, rng, **kw):
    """The `fodiwalk.augment_graph.far_pairs` call, served by the frozen
    rejection sampler of `other_ge.py`. With `ball = A` the two agree: a
    pair that `A` does not hold is a pair with no edge."""
    return _reject(ball, n, count, rng)


def sample_positives(A, max_count: int, rng):
    """At most `max_count` of the ORIGINAL undirected edges, once each.

    The upper triangle gives each edge one time. A graph with more edges than
    `max_count` gets a sample without replacement, thus no edge arrives two
    times.
    """
    pos = np.column_stack(sp.triu(A, k=1).nonzero())
    if pos.shape[0] <= max_count:
        return pos
    return pos[rng.choice(pos.shape[0], max_count, replace=False)]


def sample_negatives(A, n: int, count: int, rng):
    """`count` random pairs that have NO edge.

    `sample_far_pairs` rejects every pair that the matrix it receives holds.
    With `A` that means "connected", thus the result is exactly the set of
    unconnected pairs, and no pair arrives two times.

    It works on sorted composite keys and it needs no Python set. The older
    loop used a set, which cost 1.8 GB at a million nodes. See
    `experiments/large-graph-node2vec/REPORT.md`.
    """
    return sample_far_pairs(n, count, A, rng)


def edge_features(Z, pos, neg):
    """`(X, y)`: the Hadamard product of the two endpoints, and the label.

    The elementwise product is the standard edge feature of an embedding, and
    it is symmetric in `(u, v)`, as an undirected edge feature must be.
    """
    pairs = np.vstack([pos, neg])
    X = Z[pairs[:, 0]] * Z[pairs[:, 1]]
    y = np.concatenate([np.ones(pos.shape[0]), np.zeros(neg.shape[0])])
    return X, y


def classify(X, y, seed: int, test_size: float = 0.2, n_estimators: int = 200):
    """Fit a random forest on a part of the pairs, score it on the rest.

    Returns `(scores, sizes)`. `stratify` holds the balance of the classes in
    both parts, thus the accuracy of the test part stays comparable.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return ({"accuracy": accuracy_score(y_test, y_pred),
             "precision": precision_score(y_test, y_pred),
             "recall": recall_score(y_test, y_pred),
             "f1-score": f1_score(y_test, y_pred),
             "auc": roc_auc_score(y_test, y_prob)},
            {"train": X_train.shape[0], "test": X_test.shape[0]})


def link_prediction(Z, A, n: int, max_pairs: int, rng, seed: int,
                    test_size: float = 0.2, n_estimators: int = 200):
    """The whole test. Returns `(scores, info)`.

    `max_pairs` counts the positives AND the negatives together, thus each
    class gets one half of it.

    `info` holds the numbers that a caller prints: `positives`, `negatives`,
    `edges` (how many the graph has), `pairs`, `train`, and `test`.
    """
    pos = sample_positives(A, max_pairs // 2, rng)
    neg = sample_negatives(A, n, pos.shape[0], rng)
    X, y = edge_features(Z, pos, neg)
    scores, sizes = classify(X, y, seed, test_size, n_estimators)
    return scores, {"positives": pos.shape[0], "negatives": neg.shape[0],
                    "edges": A.nnz // 2, "pairs": X.shape[0], **sizes}
