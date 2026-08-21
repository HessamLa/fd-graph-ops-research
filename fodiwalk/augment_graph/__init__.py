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

Import discipline: this package imports numpy and scipy only. It never
imports `core`.
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

__all__ = [
    "split_key", "to_csr", "to_csr_directed", "cap_per_node", "row_cap",
    "uniform_walks", "node2vec_walks", "make_walker", "walk_pair_stats",
    "walk_rows", "with_all_neighbours", "with_neighbours_low_deg",
    "WEIGHT_RULES", "bucket_sample", "budget", "row_buckets",
    "degree_table", "sample_far_pairs", "landmarks",
]
