"""augment_graph.degrees -- the divisor of the row sum. THREE sources, an
order.

`core.sell_c_sigma.make_plan` turns this array into `inv_deg_ext`, and the
kernel multiplies every row of `dZ` by it. A degree of 0 becomes 0.0 and it
zeroes the WHOLE force of the row -- the repulsion too. The row then never
moves, for the whole run, and nothing raises (trap 4 of
`dev-docs/REFACTOR.md`, invariant I5). `plan_contract.check_degrees`
asserts against exactly that, and this module is what must not produce it.

THIS MODULE IS STAGE 2. The degree a row gets depends on WHICH pairs the
augmentation stored (`h == 1` entries, or an explicit array keyed to the
`edge_rule`), thus resolving it is a property of the recipe's data, not of
the kernel that consumes the result.

Provenance: `Fodiwalk._build_degrees` of `fodiwalk/fodiwalk.py`
(2026-08-20); `fodiwalk/embed/degrees.py` (2026-08-20, the first split);
moved into `augment_graph` (2026-08-21) when the stage boundary moved. No
line of the body changed across either move.

Import discipline: numpy, scipy and `core` only.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..core.forces import degrees_from_D


def resolve_degrees(D, G, spec, explicit=None) -> np.ndarray:
    """The divisor of the row sum. Three sources, and the order matters.

    `D`         the augmented matrix.
    `G`         the un-augmented adjacency `A`, or None.
    `spec`      a `planes.ForceSpec`: it reads `no_deg_norm`, `deg_source`
                and `edge_rule`.
    `explicit`  an array the caller handed over (`set_D(degrees=...)`), or
                None.

    `no_deg_norm` gives 1, thus the engine does not divide -- the law that
    the fdlinear specification writes.

    An explicit array (`deg_source = "A"`, or `set_D(degrees=...)`) is the
    true degree of the graph. `edge_rule = "low_deg"` needs it: a hub can
    then hold no entry at `h = 1`, `degrees_from_D` would give 0, and
    `inv_deg_ext` turns that into 0.0, which freezes the row.

    Otherwise the count of `h = 1` entries of `D`, which is the package
    default.
    """
    n = D.shape[0]
    if spec.no_deg_norm:
        return np.ones(n, dtype=np.int64)
    if explicit is not None:
        return explicit
    use_A = (spec.deg_source == "A"
             or (spec.deg_source == "auto"
                 and spec.edge_rule == "low_deg"))
    if use_A and G is not None and sp.issparse(G):
        return np.maximum(np.diff(G.indptr), 1).astype(np.int64)
    return degrees_from_D(D)
