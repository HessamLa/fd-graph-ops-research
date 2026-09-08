"""forcedirected -- the shared engine: the epoch loop, the kernel, the rules.

    from forcedirected import ForceDirected, Callback_Base
    from forcedirected import build_batch, make_step, step
    from forcedirected import row_of, n_rows, RULES, state_arrays

`ForceDirected` relaxes an embedding `Z` against a GIVEN weighted matrix
`D`. It owns the epoch loop, the batching, the callback events and the `Z`
update, and NO force law: a subclass supplies `forces`. A pipeline that
builds `D` from raw data first is one layer up, e.g.
`fodiwalk.Fodiwalk_base`.

THE ONE IMPORT RULE: this package imports numpy, scipy, jax and its own
modules, and NOTHING else of this repository. `fodined` and `fodiwalk` both
read the engine and neither may be made to depend on the other, thus every
dependency points AT this package and a cycle is impossible.

`tests/m1_old_vs_new.py` and `tests/reconstruct_pre_unification.py` are
closed-window records of the SELL-C-sigma kernel this package no longer
holds (see below): no module imports either one and pytest does not
collect them. The rule above is about what the PACKAGE imports, and it
still holds.

The tree, and why each file is here::

    force_directed.py  `ForceDirected` and `Callback_Base` -- the loop
    flat_batch.py      the flat batched kernel `step` uses (no layout)
    csr.py             `row_of` / `n_rows`, the bottom of the tree
    optim.py           `RULES` -- what `updateZ` dispatches through
    tests/             the kernel gates and the reconstruction record
    PARITY.md          the evidence the OLD (tiled) kernel was one algorithm

REPLACED 2026-09-02 (owner's decision): `sell_c_sigma.py` -- the SELL-C-sigma
layout (sort rows by width, pack them onto a padded rung of a geometric
ladder, split a hub row across virtual rows) -- is DELETED. `fodiwalk`
already bounds a batch to the rows that fit on the card, which is plain
batch processing; the layout earned nothing once that bound was already in
place. `flat_batch.py` replaces it: one flat array of `(row, partner)`
pairs per batch, reduced with `jax.ops.segment_sum`, with no padding to a
batch's widest row. See `flat_batch.py`'s own docstring for the full
reasoning and what changed in the math. `forcedirected_old/sell_c_sigma.py`
is the frozen copy of the deleted module -- the "before" side of the
measured comparison; nothing in it is edited.

Names retired with `sell_c_sigma.py`: `build_ladder`, `PlanCache`, `to_csr`
and `_to_csr` served the tiled layout only, and nothing outside that module
and its own tests used any of them (checked before deletion). `make_plan`
and `step` keep no old meaning under those names; `build_batch` and `step`
of `flat_batch.py` are their replacements, with a different signature (see
that module).

Created 2026-08-26, from three moves that all landed here:

  * `fodiwalk/core/force_directed.py` -- the engine, and the class is
    `ForceDirected` again (it was `ForceDirectedEmbedding` from
    2026-08-21). That path was a forwarder and is DELETED (2026-08-27);
    `fodiwalk` imports this package directly.
  * `sellcsigma/` -- the root package made 2026-08-25 for the kernel,
    ABSORBED here byte for byte. A separate package for one kernel that
    only `ForceDirected` calls was overkill. `PARITY.md` came with it.
  * `fodiwalk/core/csr.py` and `fodiwalk/misc/optim.py` -- the two modules
    the engine reads. Both old paths are DELETED (2026-08-27).
"""
from __future__ import annotations

from .csr import row_of, n_rows
from .force_directed import ForceDirected, Callback_Base
from .optim import RULES, STATE_ARRAYS, state_arrays
from .flat_batch import build_batch, make_step, step

__all__ = [
    "ForceDirected", "Callback_Base",
    "build_batch", "make_step", "step",
    "row_of", "n_rows",
    "RULES", "STATE_ARRAYS", "state_arrays",
]
