"""fodiwalk.misc -- the update rules, the drop, and the measurements.

    from fodiwalk.misc import RULES, state_arrays
    from fodiwalk.misc import drop_steady_rate
    from fodiwalk.misc import link_prediction, hop_sample, task_hop

`evaluation.py` is measurement and not engine, thus it is here and not in
`core`. A later move to `fodiwalk/eval/` is a rename, and not a redesign.

The update rules live in `forcedirected/optim.py`, beside the engine that
dispatches through them. `misc/optim.py` was a FORWARDER to that file from
2026-08-26 and was DELETED 2026-08-27; the three names below still resolve
here, but `from fodiwalk.misc import optim` does not. Use
`from forcedirected import optim`.
"""
from __future__ import annotations

from forcedirected import RULES, STATE_ARRAYS, state_arrays
from .drop import drop_steady_rate, FallbackKeys
from .evaluation import link_prediction, hop_sample, task_hop

__all__ = ["RULES", "STATE_ARRAYS", "state_arrays", "drop_steady_rate",
           "FallbackKeys",
           "link_prediction", "hop_sample", "task_hop"]
