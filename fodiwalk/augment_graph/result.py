#!/bin/env python3
"""result.py -- the seam of stage 2, and the narrow config it reads.

`Augmentation` is EVERYTHING that stage 2 gives stage 3, and it is DATA:
`D`, `freq`, `stats`, `info`, with fixed types and shapes. Nothing else
crosses, and no stage calls a function of the other
(`dev-docs/fodiwalk-module.md`).

The dataclass also carried `planes`, `degrees` and `params` between
2026-08-21 and 2026-08-28. All three are stage-3 things, defined by the
force law; no code ever filled them or read them, and they are REMOVED with
the builders that went back to `embed/`.

`AugmentSpec` is the other half of the seam. It holds the pair-policy
fields of the flat `Config` and no other knob, thus a policy cannot reach
the layout. `from_config` takes ANY object with these attribute names, thus
this package stays free of the model.

Import discipline: numpy and scipy. No `embed`, no model.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import scipy.sparse as sp

from .rows import RowCSR, RowStats


@dataclasses.dataclass
class Augmentation:
    """What stage 2 builds, and the WHOLE of what stage 3 receives.

    `D.data` IS the `h` value of each stored pair. Stage 3 reads these four
    fields and builds its own planes, degrees and force params from them;
    stage 2 neither knows nor names a force law.

    `D` is a `scipy.sparse.csr_matrix` on the `walk`/`walk_edges` policies.
    `nbr_walk` builds a plain `rows.RowCSR` instead (2026-08-28,
    `agentic-log/10.mem-agent/`) -- no consumer of `D` calls a scipy MATRIX
    method on it, only `.indptr`/`.indices`/`.data`/`.shape`/`.nnz`, which
    `RowCSR` gives for less memory and no validation pass. `far > 0` still
    ends in a real `sp.csr_matrix`, through `RowCSR.tocoo()`
    (`merge.add_far_pairs`). `stats` is a plain `dict` on `walk`/
    `walk_edges` and a `rows.RowStats` on `nbr_walk`, for the same reason
    (`RowStats`'s own docstring).
    """

    D: sp.csr_matrix | RowCSR
    freq: np.ndarray | None     # (nnz,), aligned to `D.indices`, or None
    stats: dict | RowStats      # the walk statistics, plus `freq`
    info: dict                  # the counts and the timings a log prints


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
