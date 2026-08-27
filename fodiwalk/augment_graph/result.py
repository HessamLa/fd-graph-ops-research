#!/bin/env python3
"""result.py -- the seam of stage 2, and the narrow config it reads.

`Augmentation` is EVERYTHING that stage 2 gives stage 3. Nothing else
crosses: stage 3 reads `D`, `freq`, `stats`, `info`, and now `planes`,
`degrees` and `params` -- the three fields a policy's `build` leaves at
their default of `None` and the model fills in once the law is known
(`Fodiwalk.augment_graph`, `model.py`). They widened the seam on 2026-08-21:
a plane, a degree and a force param are prepared DATA of the recipe (one
pair policy plus one force law), not kernel state, thus they leave stage 2
and stage 3 never builds them (`dev-docs/fodiwalk-module.md`).

`AugmentSpec` is the other half of the seam. It holds the pair-policy
fields of the flat `Config` and no other knob, thus a policy cannot reach
the layout. `from_config` takes ANY object with these attribute names, thus
this package stays free of the model.

Import discipline: numpy, scipy and `core` (for the plane/degree contract
that `planes.py` and `degrees.py` read). No `embed`, no model.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import scipy.sparse as sp


@dataclasses.dataclass
class Augmentation:
    """What stage 2 builds. `D.data` IS the `h` plane.

    `planes`, `degrees` and `params` default to `None`: a policy's `build`
    does not know the force law, thus it cannot fill them. `Fodiwalk`
    fills them once, right after `build` returns and the law is known
    (`augment_graph.planes.build_planes`, `.degrees.resolve_degrees`,
    `.planes.force_params`) -- still stage-2 CODE, called before stage 3
    ever sees the result. A `None` reaching `embed.build_plans` is a
    defect: the caller skipped that fill.
    """

    D: sp.csr_matrix
    freq: np.ndarray | None     # (nnz,), aligned to `D.indices`, or None
    stats: dict                 # the walk statistics, plus `freq` as a CSR
    info: dict                  # the counts and the timings a log prints
    planes: tuple | None = None    # `(nnz,)` arrays, filled after `build`
    degrees: np.ndarray | None = None   # `(n,)`, filled after `build`
    params: dict | None = None          # the law's scalars, filled after `build`


@dataclasses.dataclass(frozen=True)
class AugmentSpec:
    """The stage-2 knobs of `Config`, spelled the same. Nothing else."""

    # -- the pairs -----------------------------------------------------
    pairs: str = "walk"            # walk | walk_edges | nbr_walk
    edge_rule: str = "both"        # both | low_deg
    policy: str = "cap"            # cap | buckets
    walks: int = 10
    walk_len: int = 20
    window: int = 5
    cap: int = 16
    row_cap: int = 0
    p: float = 1.0
    q: float = 1.0
    prune_max: int = 4_000_000
    prune_factor: int = 4

    # -- the weight ----------------------------------------------------
    weight: str = "min_gap"        # flat | min_gap | mean_gap | pmi
    freq_mode: str = "pair"        # pair | node

    # -- the long-range term -------------------------------------------
    far: int = 0
    far_weight: float = 100.0
    far_bias: float = 0.0
    far_with_buckets: bool = False
    bucket_total: int = 0
    landmarks: int = 0
    far_max: int = 32
    far_scale: float = 1.0

    @classmethod
    def from_config(cls, cfg):
        """The stage-2 fields of any config object with these names."""
        return cls(**{f.name: getattr(cfg, f.name)
                      for f in dataclasses.fields(cls)})


def take(stats, idx):
    """The same statistics dict, on the selected pairs only."""
    return {k: (v[idx] if isinstance(v, np.ndarray) else v)
            for k, v in stats.items()}
