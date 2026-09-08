#!/bin/env python3
"""fodiwalk.config -- `Config`, every knob of the augmentation and the physics.

`Config` is FLAT and it is the PUBLIC surface: `experiments/fodiwalk/*.py`
builds one, `tests/harness.py` reads `fw.cfg.chunks`, and
`tests/check_api.py` holds the 39 field names and their default values
literally. A field that moves, that is renamed or that changes its default
breaks a recorded command line, thus the flat form stays.

The narrowing happens at the SEAM and not here. Each stage reads its own
frozen spec, thus a stage cannot reach the knobs of another stage:

    augment_graph.result.AugmentSpec.from_config(cfg)    stage 2, the pairs
    embed.planes.ForceSpec.from_config(cfg)              stage 3, the physics
    embed.planner.PlanSpec.from_config(cfg)              stage 3, the layout

`ForceSpec` was `augment_graph.planes.ForceSpec` between 2026-08-21 and
2026-08-28. It went back to stage 3 with the law it describes: stage 2
knows no force law at all.

Each spec stays in the package that reads it, and this module does NOT
re-export them. A re-export would pull `embed`, thus jax, into the import
of the configuration, and it would give a second name for one class.
`from_config` reads the attributes by NAME, thus a stage package needs no
import of this module at all and the dependency runs one way.

Import discipline: `dataclasses` only. This module is the bottom of the tree.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Config:
    """Every knob of the augmentation and the physics. One name for one flag.

    The names are the flags of `experiments/fdwalk/bench_fdwalk.py` with
    the dashes turned into underscores, thus a driver script maps one to
    one and a recorded command line stays readable.
    """

    # -- the pairs -----------------------------------------------------
    pairs: str = "walk"            # walk | walk_edges | nbr_walk
    edge_rule: str = "both"        # both | low_deg
    policy: str = "cap"            # cap | buckets
    walks: int = 10                # walks from each node
    walk_len: int = 20
    window: int = 5
    cap: int = 16                  # pairs kept for each node
    row_cap: int = 0               # nbr_walk: the m best of EACH ROW
    p: float = 1.0                 # node2vec return parameter
    q: float = 1.0                 # node2vec in-out parameter
    prune_max: int = 4_000_000
    prune_factor: int = 4

    # -- the weight ----------------------------------------------------
    weight: str = "min_gap"        # flat | min_gap | mean_gap | pmi
    freq_mode: str = "pair"        # pair | node

    # -- the long-range term -------------------------------------------
    far: int = 0                   # 0 = n*log10(n) for the cap policy
    far_weight: float = 100.0
    far_bias: float = 0.0          # deg^alpha draw; 0.75 = node2vec
    far_with_buckets: bool = False
    bucket_total: int = 0          # 0 = n*log10(n)
    landmarks: int = 0             # 0 = one constant for every far pair
    far_max: int = 32
    far_scale: float = 1.0

    # -- the physics ---------------------------------------------------
    force: str = "fdlinear"        # a key of embed.forces.FORCE_PLANES
    fuse_planes: bool = False      # fdlinear -> fdlinear_fused
    no_deg_norm: bool = False
    deg_source: str = "auto"       # auto | D | A
    k1: float = 0.999
    k2: float = 1.0                # hop decay of the attraction
    k4: float = 0.01
    kr: float = 1.0
    fdlinear_sign: float = -1.0
    random_drop_rate: float = 0.5
    drop_strategy: str = "random_rows"

    # -- the layout ----------------------------------------------------
    b_cells: int = 16_384
    k_max: int = 256
    ladder_base: float = 1.5
    chunks: int = 1
    chunk_host: bool = False

    # -- the asserter --------------------------------------------------
    check_planes: bool = True      # I1, I2, I4, I5 at the seam
    check_padding: bool = True     # I3, one pass over the built tiles
