"""reference/shell_force.py -- the FROZEN shell-averaged force law.

WHY THIS FILE EXISTS. `forcedirected/tests/test_parity.py` measured
`make_plan` and `step` against `fodined.embedding.shell_force`, an
independent implementation of the same physics. The user removed `fodined`
in commit `ea9ae67` ("Moved fdwalk to archive. Removed fodined."), so four
tests stopped at `ModuleNotFoundError` -- not because a number changed, but
because the oracle they compare against was gone.

The functions below are copied VERBATIM from
`archive/root-2026-08-16/fdge_jax_sell_c_sigma/embedding/shell_force.py`,
the last surviving copy. Only the import of `row_of` is rewritten, from
`fdge_jax_sell_c_sigma.core.csr` to `forcedirected`, which holds the same
function today. No line of a body changed.

DO NOT EDIT. This is a reference copy and its whole value is that it does
not move. The live laws are in `fodiwalk/embed/forces.py`; the
shell-averaged laws were removed from that package on 2026-08-19 because
the project settled on the linear ones.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import jax.numpy as jnp

from forcedirected import row_of


def shell_counts(D: sp.csr_matrix):
    """Per-node shell sizes, over only the hop values `D` actually stores.

    Returns `(hop_values, counts)` where `counts[u, j]` is how many stored
    neighbours node `u` has at `hop_values[j]`. The values are COMPACTED to
    the distinct hops present, which is a fixed O(n^2) memory defect and not
    a micro-optimisation: a table indexed by the raw hop value is
    `(n, D.data.max() + 1)`, and a disconnected graph stamps an
    `unreachable = n` sentinel, so `max()` IS `n` and the table becomes
    about 20 GB at n = 50,000.
    """
    n = D.shape[0]
    hop_values = np.unique(D.data)
    m = hop_values.size
    if m == 0:
        return hop_values, np.zeros((n, 0), dtype=np.int64)
    col = np.searchsorted(hop_values, D.data)
    flat = np.bincount(row_of(D.indptr) * m + col, minlength=n * m)
    return hop_values, flat.reshape(n, m).astype(np.int64)


def shell_coeff_data(D: sp.csr_matrix) -> np.ndarray:
    """Per-stored-entry `shell_coeff = 1 / |S_h(u)|`, in `D`'s own CSR order.

    Aligned 1:1 with `D.indices` / `D.data`, thus computed BEFORE any
    hub-split, width sort, ladder quantisation or padding -- the alignment
    `make_plan` requires of a coefficient plane.
    """
    hop_values, counts = shell_counts(D)
    u_of = row_of(D.indptr)
    col = np.searchsorted(hop_values, D.data)
    return (1.0 / counts[u_of, col]).astype(np.float32)


def degrees_from_D(D: sp.csr_matrix, degrees=None) -> np.ndarray:
    """Force-law degree: the count of `D.data == 1` entries per source row.

    NOT a CSR row's stored width. `fodiwalk.embed.forces.degrees_from_D` is
    the same function; this copy is kept so the oracle stands alone.
    """
    if degrees is None:
        u_of = row_of(D.indptr)
        return np.bincount(u_of[D.data == 1],
                           minlength=D.shape[0]).astype(np.float64)
    return np.ascontiguousarray(degrees, dtype=np.float64)


def shell_force(x, planes, params):
    """`Fa + Fr` for every cell of an `(R, k)` tile. Pure, elementwise.

    `planes` is `(shell_coeff, h)`; `params` holds `k1, k2, k3, k4,
    h_shift` and, since 2026-09-09, `node_degree`. Returns the force
    MAGNITUDE along `u -> v`; the engine applies the direction and the
    padding guards.

    THE DEGREE DIVISION IS HERE, and no longer in the engine. `step` used
    to divide the row sum by `deg(u)` for every law; on 2026-09-09 that
    averaging coefficient moved into the laws, because the right
    denominator is the size of the set a law sums over. This oracle
    reproduces it inline rather than calling
    `fodiwalk.embed.forces.averaged`: `forcedirected` imports NOTHING of
    this repository (`test_b1_engine_package_imports_nothing_of_this_repository`)
    and this file is inside it. A degree of 0 gives exactly 0, which is what
    the old `inv_deg_ext` array did.

    A pad cell arrives with both planes zeroed, so each term vanishes on its
    own. Keep that property: the engine's `x == 0` guard is the second layer
    of a double safety, not the only one.
    """
    shell_coeff, h = planes
    Fa = (params["k1"] * shell_coeff * x
          * jnp.exp(-params["k2"] * (h - params["h_shift"])))
    Fr = -params["k3"] * h * jnp.exp(-params["k4"] * x)
    d = params["node_degree"]
    return jnp.where(d > 0, (Fa + Fr) / jnp.where(d > 0, d, 1.0), 0.0)
