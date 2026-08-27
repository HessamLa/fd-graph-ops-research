"""forcedirected -- the shared engine: the epoch loop, the kernel, the rules.

    from forcedirected import ForceDirected, Callback_Base
    from forcedirected import build_ladder, make_plan, step, PlanCache, to_csr
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

`tests/m1_old_vs_new.py` is the ONE file here that names `fodined` and
`fodiwalk`. It is a closed-window record, not a gate: no module imports it,
pytest does not collect it, and the comparison it ran cannot be repeated.
The rule above is about what the PACKAGE imports, and it holds.

The tree, and why each file is here::

    force_directed.py  `ForceDirected` and `Callback_Base` -- the loop
    sell_c_sigma.py    the SELL-C-sigma layout and kernel `step` uses
    csr.py             `row_of` / `n_rows`, the bottom of the tree
    optim.py           `RULES` -- what `updateZ` dispatches through
    tests/             the kernel gates and the reconstruction record
    PARITY.md          the evidence that the kernel is the same algorithm

`_step` and `_to_csr` stay as aliases, because live code and tests name
them. `to_csr` is the public name of `_to_csr`: gate D6
(`fodiwalk/tests/test_structure.py`) forbids a plain module to import a
private name of another module, thus a shared package must offer one.

Created 2026-08-26, from three moves that all landed here:

  * `fodiwalk/core/force_directed.py` -- the engine, and the class is
    `ForceDirected` again (it was `ForceDirectedEmbedding` from
    2026-08-21). The old path is a forwarder and holds NO alias.
  * `sellcsigma/` -- the root package made 2026-08-25 for the kernel,
    ABSORBED here byte for byte. A separate package for one kernel that
    only `ForceDirected` calls was overkill. `PARITY.md` came with it.
  * `fodiwalk/core/csr.py` and `fodiwalk/misc/optim.py` -- the two modules
    the engine reads. Both old paths are forwarders.
"""
from __future__ import annotations

from .csr import row_of, n_rows
from .force_directed import ForceDirected, Callback_Base
from .optim import RULES, STATE_ARRAYS, state_arrays
from .sell_c_sigma import (
    build_ladder,
    make_plan,
    step,
    _step,
    PlanCache,
    to_csr,
    _to_csr,
)

__all__ = [
    "ForceDirected", "Callback_Base",
    "build_ladder", "make_plan", "step", "_step", "PlanCache",
    "to_csr", "_to_csr",
    "row_of", "n_rows",
    "RULES", "STATE_ARRAYS", "state_arrays",
]
