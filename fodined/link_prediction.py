#!/bin/env python3
"""link_prediction.py -- the link prediction test of fodined, as functions.

The question: does the embedding know which node pairs have an edge? A
classifier reads only the two vectors of a pair, thus a good score means that
the geometry holds the adjacency of the graph.

The test is a SAMPLE of the pairs, and not a census of them. A graph of a
million nodes has millions of edges, thus all of them give a feature matrix
of gigabytes and a forest that trains for hours. A balanced sample measures
the same property at a constant cost.

Every function here is pure: it reads only its arguments, it writes nothing
outside its result, and it prints nothing. The caller keeps the configuration
and the messages. `modular.py` is that caller.

The random generator arrives as an argument, and this module never makes one.
That is what keeps a run reproducible.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

from fodined.graph_augmentation import sample_far_pairs


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
