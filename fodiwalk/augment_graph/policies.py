#!/bin/env python3
"""policies.py -- the stage-2 registry, and the one entry point.

`build(A, n, spec, rng)` takes the un-augmented graph and gives an
`Augmentation`. The policy comes from a REGISTRY and never from an `if`
chain, exactly as `core.forces.FORCE_PLANES`, `weights.RULES` and
`forcedirected.RULES` already do. A chain on a name is what let a missing
`freq` fall back to the planes of another law, in silence.

EVERY POLICY HERE IS WALK-BASED, and that is the scope of the package
(2026-08-20). `h` is a WALK GAP -- an upper bound of the hop distance,
which a walk of `t` steps proves -- and never a measured distance. The
exact-distance policies `ball` and `sampled` were removed; see
`dev-docs/CATALOG.md` section 18.

Import discipline: this package imports numpy and scipy only. It imports no
`core`, no `embed` and no model.
"""
from __future__ import annotations

from . import policy_nbr_walk, policy_walk
from .result import Augmentation, AugmentSpec

# name -> the builder of stage 2. `walk` and `walk_edges` share a builder
# and differ by one axis inside it.
POLICIES = {"walk": policy_walk.build,
            "walk_edges": policy_walk.build,
            "nbr_walk": policy_nbr_walk.build}

# name -> the walk that the policy runs, and its pair statistics.
WALK_STATS = {"walk": policy_walk.pair_stats,
              "walk_edges": policy_walk.pair_stats,
              "nbr_walk": policy_nbr_walk.row_stats}

UNKNOWN_PAIRS = (
    "Unknown pairs policy {!r}. Known: walk, walk_edges, nbr_walk. Every "
    "policy of this package is walk-based; the exact-distance policies were "
    "removed on 2026-08-20.")


def build(A, n: int, spec: AugmentSpec, rng) -> Augmentation:
    """Stage 2. The un-augmented graph into `D`, by the policy of `spec`.

    ONE generator flows through the whole stage: the walks, the bucket
    sample, the far-pair draw and the landmarks read it in that order. A
    moved call, an extra draw or a second generator changes the pairs and
    looks like a defect of the physics.
    """
    if spec.pairs not in POLICIES:
        raise ValueError(UNKNOWN_PAIRS.format(spec.pairs))
    return POLICIES[spec.pairs](A, n, spec, rng)


def graph_walk(A, n: int, spec: AugmentSpec, rng):
    """The random walks alone, and the statistics of the pairs they give.

    `nbr_walk` gives `walks.walk_rows`: directed, no window, no cap. `walk`
    and `walk_edges` give `walks.walk_pair_stats`: windowed, undirected,
    with a cap and a prune. The two must never be unified -- `walk_rows` has
    no prune BY DESIGN, because its bound IS `walks * walk_len` for a row.
    """
    if spec.pairs not in WALK_STATS:
        raise ValueError(UNKNOWN_PAIRS.format(spec.pairs))
    return WALK_STATS[spec.pairs](A, n, spec, rng)
