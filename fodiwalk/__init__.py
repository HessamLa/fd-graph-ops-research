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
    base.py          class Fodiwalk_base -- the pipeline contract: the
                     stage hooks a concrete model implements
    make_graph/      datasets                            (stage 1)
    augment_graph/   policies, walks, pairs, weights, buckets, far_pairs,
                     landmarks, merge                    (stage 2, PRODUCES
                     the matrix `D` and its `freq`. It knows no force law.)
    embed/           forces, plan_contract, planes, degrees, planner
                     (stage 3, CONSUMES. Everything that knows a LAW: the
                     laws, the plane registry, the asserter, the plane and
                     degree builders, and the batch plan.)
    misc/            drop, evaluation                    (stage 3 support)
    model.py         class Fodiwalk(Fodiwalk_base) -- the three stages, WIRED

The engine `ForceDirected`, the SELL-C-sigma kernel and the CSR helpers are
in the ROOT package `forcedirected`, which `fodined` reads too. CATALOG
section 23.

STAGE 2 AND STAGE 3 DO NOT IMPORT EACH OTHER. Stage 2 produces and stage 3
consumes, across a fixed data contract -- `augment_graph.result.Augmentation`
holds `D`, `freq`, `stats` and `info`, with fixed types and shapes -- and
neither calls a function of the other. `model.py` carries the data across;
it is the composition root and belongs to neither stage.

The tree reached this shape on 2026-08-28. `fodiwalk/core/` held `forces.py`
and `plan_contract.py` and is DELETED; `planes.py` and `degrees.py` went
back to `embed/` from `augment_graph/`, because building a plane meant
asking a law what it reads, and that question was an import from stage 2
into stage 3.

This package is a REFACTOR of `experiments/fdwalk/` with a parity
requirement: the physics, the augmentation rules and the numbers do not
change. `fodiwalk/tests/test_parity.py` is the gate, and
`fodiwalk/dev-docs/PRD.md` section 8 holds the recorded numbers it reproduces.
"""
from __future__ import annotations

from .config import Config
from .model import Fodiwalk

__all__ = ["Fodiwalk", "Config"]
