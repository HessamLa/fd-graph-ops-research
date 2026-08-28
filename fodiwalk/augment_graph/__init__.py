"""fodiwalk.augment_graph -- stage 2: the graph into the matrix `D`.

A stored entry `D[u, v] = h >= 1` is a distance. The walks make the pairs,
`weights.py` turns the walk statistics into `h`, `pairs.py` caps them and
builds the CSR, `buckets.py` and `far_pairs.py` choose which pairs survive
and which long-range pairs are added, and `landmarks.py` gives the far
pairs a real distance in place of one constant. `nbr_walk` carries its
pairs as `rows.RowStats` instead, and merges `A`'s edges into them in
`row_merge.py` -- both split out of `pairs.py`/`walks.py` on 2026-08-28,
under `tests/test_structure.py`'s line caps; see their own docstrings.

Every policy here is WALK-BASED. `h` is a walk GAP -- an upper bound of the
hop distance -- and never a measured distance. The exact-distance policies
`ball` and `sampled` were removed on 2026-08-20; see `dev-docs/CATALOG.md`
section 18.

THIS PACKAGE KNOWS NO FORCE LAW. `planes.py` and `degrees.py` were here
between 2026-08-21 and 2026-08-28 and went back to `embed/`: a plane and a
degree are defined by the LAW that reads them, and to build one this
package had to ask the law what it wanted. That question was an import from
stage 2 into stage 3. Stage 2 PRODUCES and stage 3 CONSUMES, across a fixed
data contract, and neither calls the other
(`dev-docs/fodiwalk-module.md`).

THREE INVARIANTS, and each one has already caused a silent defect:

  I4  a stored weight is an INTEGER in `[1, window]`, and 1 means
      adjacency. A continuous weight gives every pair its own shell, the
      stage-3 degree count then gives 0 for every row, and the whole force
      vanishes: AUC 0.55 with `||dZ|| = 0.000`.
  I5  attraction exists at `h = 1` only, thus a policy that can leave a
      row with no `h = 1` entry MUST report a real degree, or that row
      freezes in silence.
  --  `walk_rows` has NO prune: its bound IS `n_walks * walk_len` for each
      row, thus the walk budget of the caller is the memory bound.
      `walk_pair_stats` DOES prune, and its prune is an approximation that
      `prunes > 0` reports.

THE SEAM is `result.Augmentation`, and it is DATA: `D`, `freq`, `stats`,
`info`, with fixed types and shapes. Nothing else crosses.

Import discipline: numpy, scipy and its own modules. NOTHING else -- no
`embed`, no `forcedirected`, no `make_graph`, no `misc`, no model.
`tests/test_structure.py` asserts it in both directions.
"""
from __future__ import annotations

from .pairs import (split_key, to_csr, to_csr_directed, cap_per_node,
                    row_cap)
from .rows import RowStats, RowCSR, to_csr_directed_rows, assert_row_sorted
from .walks import (uniform_walks, node2vec_walks, make_walker,
                    walk_pair_stats, walk_rows)
from .row_merge import with_all_neighbours, with_neighbours_low_deg
from .weights import RULES as WEIGHT_RULES
from .buckets import bucket_sample, budget, row_buckets
from .far_pairs import degree_table, sample_far_pairs
from . import landmarks
from .result import Augmentation, AugmentSpec, take
from .merge import add_far_pairs, drop_pairs_of
from .policies import POLICIES, build, graph_walk

__all__ = [
    "split_key", "to_csr", "to_csr_directed", "cap_per_node", "row_cap",
    "RowStats", "RowCSR", "to_csr_directed_rows", "assert_row_sorted",
    "uniform_walks", "node2vec_walks", "make_walker", "walk_pair_stats",
    "walk_rows", "with_all_neighbours", "with_neighbours_low_deg",
    "WEIGHT_RULES", "bucket_sample", "budget", "row_buckets",
    "degree_table", "sample_far_pairs", "landmarks",
    "Augmentation", "AugmentSpec", "take", "add_far_pairs", "drop_pairs_of",
    "POLICIES", "build", "graph_walk",
]
