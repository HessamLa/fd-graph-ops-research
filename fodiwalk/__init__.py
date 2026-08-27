"""fodiwalk -- walk-augmented force-directed graph embedding.

    from fodiwalk import Fodiwalk
    from fodiwalk.make_graph import load

    A, n = load("cora")
    fw = Fodiwalk(n_dim=64, seed=42, pairs="nbr_walk", weight="min_gap",
                  force="fdlinear", optim="plain", lr=1.0)
    fw.embed(A, epochs=200)
    Z = fw.get_embeddings()

The tree, and the one-way dependency between the stages::

    config.py        class Config -- every knob, FLAT
    core/            forces, plan_contract -- the PHYSICS of this
                     package. The engine `ForceDirected`, the
                     SELL-C-sigma kernel and the CSR helpers are NOT
                     here: they live in the root package
                     `forcedirected`, which `fodined` reads too.
                     `core/__init__.py` re-exports their names.
                     CATALOG section 23.
    base.py          class Fodiwalk_base -- the pipeline contract: the
                     stage hooks a concrete model implements
    make_graph/      datasets                            (stage 1)
    augment_graph/   policies, walks, pairs, weights, buckets, far_pairs,
                     landmarks, merge, planes, degrees    (stage 2, incl.
                     the law's planes and the degree divisor)
    embed/           planner                          (stage 3, CONSUMES)
    misc/            optim, drop, evaluation             (stage 3 support)
    model.py         class Fodiwalk(Fodiwalk_base) -- the three stages, WIRED

`core` imports nothing of the package except `core`. Every other stage may
import `core`. A cycle is a defect.

This package is a REFACTOR of `experiments/fdwalk/` with a parity
requirement: the physics, the augmentation rules and the numbers do not
change. `fodiwalk/tests/test_parity.py` is the gate, and
`fodiwalk/dev-docs/PRD.md` section 8 holds the recorded numbers it reproduces.
"""
from __future__ import annotations

from .config import Config
from .model import Fodiwalk

__all__ = ["Fodiwalk", "Config"]
