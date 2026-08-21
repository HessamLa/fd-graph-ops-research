#!/bin/env python3
"""weights.py -- axis A of the experiment: what weight a pair gets.

The force law of fodined reads the weight `h` of a stored pair as a
distance: the attraction decreases with `h`, and the repulsion increases
with it. Thus a SMALL `h` must mean "these two nodes are near".

Every rule here gives `h` as an INTEGER in the range `[1, window]`. The same
range for every rule is what makes the comparison honest: the force
constants `K1..K4` then see the same numbers, and only the ORDER of the
pairs changes.

The integer is not a decoration, it is a requirement of the force law.
`degrees_from_D` counts the entries that are exactly 1. A float weight
almost never is exactly 1, thus the degree becomes 0 and the attraction
goes with it.

The first version of `mean_gap` and `pmi` gave floats. Both collapsed to an
AUC of 0.55 and 0.70, with `||dZ||` at 0.000. That is the reason for the
`np.rint` in each rule. See FINDINGS.md.

The rules:

  flat      1.0 for every pair. The control of hypothesis H3.
  min_gap   the smallest step gap that a walk gave. It is an upper bound of
            the hop distance, and hypothesis H2 measures how tight it is.
  mean_gap  the mean of the gaps. It uses every visit, and not only the
            best one, thus it is smoother and it is not an integer.
  pmi       from the count of the visits, through the pointwise mutual
            information. Levy and Goldberg showed that word2vec with
            negative sampling factors a shifted PMI matrix [5], and Qiu et
            al. showed the same for DeepWalk and node2vec [6]. Thus this
            rule gives the force law the same quantity that node2vec reads.

Every function is pure and it prints nothing.
"""
# Provenance: moved from `experiments/fdwalk/weights.py` on 2026-08-19.
# Verbatim: not one line of a body changed.
from __future__ import annotations

import numpy as np


def weight_flat(stats, window: int, n: int):
    """1.0 for every pair."""
    return np.ones(stats["key"].size, dtype=np.float64)


def weight_min_gap(stats, window: int, n: int):
    """The smallest step gap. An integer in `[1, window]`."""
    return stats["mn"].astype(np.float64)


def weight_mean_gap(stats, window: int, n: int):
    """The mean step gap, to the nearest integer in `[1, window]`."""
    mean = stats["sm"].astype(np.float64) / np.maximum(stats["cnt"], 1)
    return np.clip(np.rint(mean), 1, window)


def weight_pmi(stats, window: int, n: int):
    """From the PMI of the pair, mapped into `[1, window]`.

    `pmi = log(c_uv * T / (c_u * c_v))`, where `c_u` is how many times the
    node `u` appears in a kept pair and `T` is the total. A large PMI means
    "these two meet more than chance", thus it must become a SMALL `h`.

    The map clips at the 1st and the 99th percentile before it scales. One
    pair of two rare nodes can otherwise take the whole range, and every
    other pair then collects at one end of it.
    """
    key, cnt = stats["key"], stats["cnt"].astype(np.float64)
    u, v = key // n, key % n
    total = cnt.sum()
    node_cnt = np.bincount(np.concatenate([u, v]),
                           weights=np.concatenate([cnt, cnt]), minlength=n)
    pmi = np.log(np.maximum(cnt, 1e-12) * total
                 / np.maximum(node_cnt[u] * node_cnt[v], 1e-12))

    lo, hi = np.percentile(pmi, [1.0, 99.0])
    if hi <= lo:                                 # every pair is the same
        return np.ones(key.size, dtype=np.float64)
    x = np.clip((pmi - lo) / (hi - lo), 0.0, 1.0)    # 1 = the nearest
    return np.clip(np.rint(1.0 + (window - 1) * (1.0 - x)), 1, window)


RULES = {"flat": weight_flat, "min_gap": weight_min_gap,
         "mean_gap": weight_mean_gap, "pmi": weight_pmi}
