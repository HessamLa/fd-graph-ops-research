"""fodined -- force-directed graph embedding with JAX and a SELL-C-sigma engine.

This package holds the parts of `fdge_jax_sell_c_sigma` that `modular.py`
uses, and nothing more. It has two sub-packages:

    core/       the `Fodined` base class, and the CSR helpers.
    embedding/  the force law, the SELL-C-sigma layout machinery, and the
                random drop regularizer.

`modular.py` is the script. It loads a graph from an edgelist. Then it
augments the graph, it embeds the nodes, and it does link prediction. Refer
to `README.md` in this directory.

This package keeps the float32 default of JAX, because the engine is for
throughput. If you need float64, call
`jax.config.update("jax_enable_x64", True)` before you import this package.
"""
from __future__ import annotations
