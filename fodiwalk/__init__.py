"""fodiwalk -- walk-augmented force-directed graph embedding.

    from fodiwalk import Fodiwalk
    from fodiwalk.make_graph import load

    A, n = load("cora")
    fw = Fodiwalk(n_dim=64, seed=42, pairs="nbr_walk", weight="min_gap",
                  force="fdlinear", optim="plain", lr=1.0)
    fw.embed(A, epochs=200)
    Z = fw.get_embeddings()

The tree, and the one-way dependency between the stages::

    core/            csr, force_directed, sell_c_sigma, forces, plan_contract
    make_graph/      datasets                     (stage 1)
    augment_graph/   walks, pairs, weights, buckets, far_pairs, landmarks
    misc/            optim, drop, evaluation      (stage 3 support)
    fodiwalk.py      class Fodiwalk

`core` imports nothing of the package except `core`. Every other stage may
import `core`. A cycle is a defect.

This package is a REFACTOR of `experiments/fdwalk/` with a parity
requirement: the physics, the augmentation rules and the numbers do not
change. `fodiwalk/tests/test_parity.py` is the gate, and
`fodiwalk/dev-docs/PRD.md` section 8 holds the recorded numbers it reproduces.
"""
from __future__ import annotations

from .fodiwalk import Fodiwalk, Config

__all__ = ["Fodiwalk", "Config"]
