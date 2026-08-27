"""misc.optim -- A FORWARDER. Do not edit the update rules here.

The eight rules, `RULES`, `STATE_ARRAYS` and `state_arrays` live in ONE
place now:

    forcedirected/optim.py

They moved 2026-08-26 with the engine that dispatches through them:
`ForceDirected.set_rule` is at the repository root, and `from ..misc import
optim` would point outside its package. `updateZ` cannot run without the
rules, thus the rules follow the engine.

The names below are re-exported BY NAME and never rebuilt. `fw.rule is
optim.RULES["plain"]` holds -- `fodiwalk/tests/test_smoke.py` asserts it
with `is` -- because this module and `forcedirected.optim` name the SAME
function objects. A forwarder that copied the dict would break that
assertion, and the assertion is right.

To change a rule, edit `forcedirected/optim.py`.
"""
from __future__ import annotations

from forcedirected.optim import RULES, STATE_ARRAYS, state_arrays

__all__ = ["RULES", "STATE_ARRAYS", "state_arrays"]
