"""fodiwalk.augment_graph -- stage 2: the graph into the matrix `D`.

A stored entry `D[u, v] = h >= 1` is a distance. The walks make the pairs,
`weights.py` turns the walk statistics into `h`, `pairs.py` caps them and
builds the CSR, `buckets.py` and `far_pairs.py` choose which pairs survive
and which long-range pairs are added, and `landmarks.py` gives the far
pairs a real distance in place of one constant.

Every policy here is WALK-BASED. `h` is a walk GAP -- an upper bound of the
hop distance -- and never a measured distance. The exact-distance policies
`ball` and `sampled` were removed on 2026-08-20; see `dev-docs/CATALOG.md`
section 18.

`planes.py` and `degrees.py` moved here from `embed/` on 2026-08-21: a
plane and a degree are properties of the RECIPE (one pair policy plus one
force law) and not of the kernel that consumes them, thus they are
DATA-PREPARATION and belong to stage 2. `embed/` keeps only the plan build
and the jitted kernel wiring -- consumption, and nothing else
(`dev-docs/fodiwalk-module.md`, `dev-docs/CATALOG.md` the RECIPE entry).

THREE INVARIANTS, and each one has already caused a silent defect:

  I4  a stored weight is an INTEGER in `[1, window]`, and 1 means
      adjacency. A continuous weight gives every pair its own shell,
      `degrees_from_D` gives 0 for every row, and the whole force
      vanishes: AUC 0.55 with `||dZ|| = 0.000`.
  I5  attraction exists at `h = 1` only, thus a policy that can leave a
      row with no `h = 1` entry MUST pass an explicit `degrees` array, or
      that row freezes in silence.
  --  `walk_rows` has NO prune: its bound IS `n_walks * walk_len` for each
      row, thus the walk budget of the caller is the memory bound.
      `walk_pair_stats` DOES prune, and its prune is an approximation that
      `prunes > 0` reports.

Import discipline: this package imports numpy, scipy and `core`. `core` is
read for the plane contract only (`planes_of`, `fuse`, `degrees_from_D`,
`PLANE_CHECKS`) -- the recipe's data preparation needs to know what a law's
planes are named and what a degree promises. It never imports `embed` or
the model.
"""
from __future__ import annotations

from .pairs import (split_key, to_csr, to_csr_directed, cap_per_node,
                    row_cap)
from .walks import (uniform_walks, node2vec_walks, make_walker,
                    walk_pair_stats, walk_rows, with_all_neighbours,
                    with_neighbours_low_deg)
from .weights import RULES as WEIGHT_RULES
from .buckets import bucket_sample, budget, row_buckets
from .far_pairs import degree_table, sample_far_pairs
from . import landmarks
from .result import Augmentation, AugmentSpec, take
from .merge import add_far_pairs, drop_pairs_of
from .policies import POLICIES, build, graph_walk
from .planes import ForceSpec, PLANE_BUILDERS, build_planes, force_params
from .degrees import resolve_degrees

__all__ = [
    "split_key", "to_csr", "to_csr_directed", "cap_per_node", "row_cap",
    "uniform_walks", "node2vec_walks", "make_walker", "walk_pair_stats",
    "walk_rows", "with_all_neighbours", "with_neighbours_low_deg",
    "WEIGHT_RULES", "bucket_sample", "budget", "row_buckets",
    "degree_table", "sample_far_pairs", "landmarks",
    "Augmentation", "AugmentSpec", "take", "add_far_pairs", "drop_pairs_of",
    "POLICIES", "build", "graph_walk",
    "ForceSpec", "PLANE_BUILDERS", "build_planes", "force_params",
    "resolve_degrees",
]
