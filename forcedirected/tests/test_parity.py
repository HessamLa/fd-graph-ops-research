"""test_parity.py -- the gates that keep the kernel the same algorithm.

These outlive milestone 2. They name NO old module: after the unification
`fodined/embedding/sell_c_sigma.py` and `fodiwalk/core/sell_c_sigma.py`
forward here, thus a comparison against them would be a comparison against
this file and would pass for the wrong reason. The one-time old-against-new
evidence is `../PARITY.md`, made by `m1_old_vs_new.py`.

Run: .venv/bin/pytest forcedirected/tests/test_parity.py
"""
from __future__ import annotations

import collections

import numpy as np
import scipy.sparse as sp
import pytest

from forcedirected import build_ladder, make_plan, step, to_csr, _to_csr, _step

B_CELLS, K_MAX, LADDER_BASE = 16_384, 256, 1.5

# Pinned on 2026-08-25 from the implementation `PARITY.md` proves identical
# to both originals. A change here is a change of the layout, and it must be
# argued and not absorbed.
GOLDEN_LADDERS = {
    (256, 1.5): [1, 2, 3, 5, 8, 12, 18, 27, 41, 62, 93, 140, 210, 256],
    (256, 2.0): [1, 2, 4, 8, 16, 32, 64, 128, 256],
    (16, 1.3): [1, 2, 3, 4, 6, 8, 11, 15, 16],
}


def cora():
    from fodiwalk.make_graph.datasets import read_edges, to_csr as edges_to_csr
    return edges_to_csr(read_edges("cora"))


def plan_inputs(A):
    from fodined.embedding.shell_force import shell_coeff_data, degrees_from_D
    return (shell_coeff_data(A), A.data), degrees_from_D(A)


def max_owner_multiplicity(plan, n: int) -> int:
    """The largest count of one real owner id inside ONE batch. 3 or more
    makes that batch's scatter nondeterministic on the GPU (see
    `experiments/fdwalk/FINDINGS.md` lines 1100-1182)."""
    worst = 0
    for rung in plan:
        for batch in np.asarray(rung[0]):
            c = collections.Counter(int(v) for v in batch if int(v) < n)
            if c:
                worst = max(worst, max(c.values()))
    return worst


# ---------------------------------------------------------------------------
def test_the_public_and_private_names_are_one_object():
    """`shell_force.py` imports `_to_csr` and `fodiwalk/core/__init__.py`
    re-exports `_step`. Both must stay, and both must BE the public one."""
    assert to_csr is _to_csr
    assert step is _step


@pytest.mark.parametrize("key,expected", sorted(GOLDEN_LADDERS.items()))
def test_build_ladder_matches_the_pinned_output(key, expected):
    k_max, base = key
    assert list(build_ladder(k_max, base)) == expected


def test_build_ladder_never_stalls_and_ends_at_k_max():
    """The `max(ks[-1] + 1, ...)` guard forces progress where
    `ceil(k * base) == k`. Without it a base under 2 loops at k = 1."""
    for k_max in (1, 2, 3, 7, 16, 64, 256, 1024):
        for base in (1.1, 1.3, 1.5, 2.0, 3.0):
            ladder = build_ladder(k_max, base)
            assert ladder[0] == 1
            assert ladder[-1] == k_max
            assert np.all(np.diff(ladder) > 0), (k_max, base)


@pytest.mark.parametrize("k_max", [K_MAX, 16])
def test_make_plan_is_deterministic_run_to_run(k_max):
    """Host side, numpy and scipy only. Two builds of one graph must agree
    on every array, every dtype and every stat."""
    A, n = cora()
    planes, deg = plan_inputs(A)
    args = dict(degrees=deg, b_cells=B_CELLS, k_max=k_max,
                ladder_base=LADDER_BASE)
    pa, ia, sa = make_plan(A, planes, **args)
    pb, ib, sb = make_plan(A, planes, **args)

    assert sa == sb
    assert ia.dtype == ib.dtype and np.array_equal(ia, ib)
    assert len(pa) == len(pb)
    for ra, rb in zip(pa, pb):
        assert len(ra) == len(rb)
        for x, y in zip(ra, rb):
            assert x.dtype == y.dtype
            assert np.array_equal(x, y)


def test_make_plan_rejects_a_plane_of_the_wrong_length():
    """The plane contract: every plane is `(D.nnz,)` and aligned to
    `D.indices`. A wrong length is the one part the builder can catch."""
    A, n = cora()
    _, deg = plan_inputs(A)
    with pytest.raises(ValueError):
        make_plan(A, (np.ones(A.nnz - 1, dtype=np.float32),), degrees=deg)


def test_step_is_exact_on_cora_where_the_scatter_is_deterministic():
    """Cora's widest row is 168, thus `k_max = 256` splits nothing, thus no
    two virtual rows share an owner and the scatter has ONE addend per
    address. The multiplicity is MEASURED here and not assumed: if a future
    `D` or `k_max` pushes it to 3, this test says so instead of failing for
    a reason that is the GPU's and not the code's."""
    import jax
    import jax.numpy as jnp
    import functools
    from fodined.embedding.shell_force import shell_force

    A, n = cora()
    planes, deg = plan_inputs(A)
    plan_np, inv_np, stats = make_plan(
        A, planes, degrees=deg, b_cells=B_CELLS, k_max=K_MAX,
        ladder_base=LADDER_BASE)
    mult = max_owner_multiplicity(plan_np, n)
    assert mult < 3, (
        f"cora reaches {mult} addends on one address inside a batch; "
        f"exactness is not available and this test needs a new graph")

    plan = jax.tree_util.tree_map(jax.device_put, plan_np)
    inv = jax.device_put(inv_np)
    Z = jax.random.normal(jax.random.PRNGKey(42), (n, 128), dtype=jnp.float32)
    kernel = jax.jit(functools.partial(step, n=n, force_fn=shell_force))
    params = dict(k1=0.999, k2=1.0, k3=10.0, k4=0.01, h_shift=1.0)

    a = np.asarray(kernel(Z, plan, inv, params).block_until_ready())
    b = np.asarray(kernel(Z, plan, inv, params).block_until_ready())
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()
