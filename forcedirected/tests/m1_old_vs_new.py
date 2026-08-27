"""m1_old_vs_new.py -- the parity evidence of milestone 1, 2026-08-25.

THIS SCRIPT HAS ONE WINDOW. It compares the two ORIGINAL copies against
the shared kernel, thus it is meaningful only while the originals still hold
the algorithm. After milestone 2 they are forwarders and the comparison
becomes a module against itself. The result is written to `PARITY.md` and
this file stays only as the record of how the numbers were made.

What it does NOT do: assert bit equality of a long run. A hub row that the
plan splits gives several virtual rows ONE owner id, thus
`dZ.at[rows].add` accumulates in a nondeterministic order and float32
addition is not associative. See `experiments/fdwalk/FINDINGS.md` lines
1100-1182. The threshold is 3 addends on one address INSIDE one batch. The
script MEASURES that multiplicity and asserts exactness only where it is
below 3.

Run: .venv/bin/python -m forcedirected.tests.m1_old_vs_new
     (the window is CLOSED since 2026-08-25; kept as the record only)

This is the one file of `forcedirected/` that imports `fodined` and
`fodiwalk`, and it is deliberate: the comparison it ran needed both. It is
a record and not a gate -- no module of this package imports it, and pytest
does not collect it -- thus the package's import rule is not touched.
"""
from __future__ import annotations

import collections
import importlib
import sys

import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp

from fodined.embedding.shell_force import (shell_coeff_data, degrees_from_D,
                                           shell_force)
from fodiwalk.make_graph.datasets import read_edges, to_csr

OLD_FODINED = importlib.import_module("fodined.embedding.sell_c_sigma")
# `fodiwalk.core.sell_c_sigma` was the forwarder that carried this name;
# it was DELETED 2026-08-27. The window this script measures closed
# 2026-08-25, thus this leg now compares the shared module against
# itself, exactly as the docstring above says it would.
OLD_FODIWALK = importlib.import_module("forcedirected.sell_c_sigma")
NEW = importlib.import_module("forcedirected.sell_c_sigma")

PARAMS = dict(k1=0.999, k2=1.0, k3=10.0, k4=0.01, h_shift=1.0)
B_CELLS, K_MAX, LADDER_BASE = 16_384, 256, 1.5

fail = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'ok ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        fail.append(name)


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------
def star(n_leaf: int, n_pad: int = 5):
    """A star of `n_leaf` leaves. One row of width `n_leaf`, thus `k_max`
    splits it and every virtual row carries the SAME owner. This is the
    graph that makes the scatter nondeterministic on purpose."""
    n = n_leaf + 1 + n_pad
    rows = np.concatenate([np.zeros(n_leaf, np.int64), np.arange(1, n_leaf + 1)])
    cols = np.concatenate([np.arange(1, n_leaf + 1), np.zeros(n_leaf, np.int64)])
    A = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    A.sort_indices()
    return A, n


def path(n: int):
    """A path. Every row is width 2, thus no split and no duplicate owner."""
    rows = np.concatenate([np.arange(n - 1), np.arange(1, n)])
    cols = np.concatenate([np.arange(1, n), np.arange(n - 1)])
    A = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    A.sort_indices()
    return A, n


# `k_max` per case. Cora's widest row is 168, thus the default 256 never
# splits it. The `k_max = 16` case is the SAME graph forced to split, which
# is how a real graph reaches 3 addends on one address.
def graphs():
    yield "path-64", path(64), K_MAX
    yield "star-1000 (k_max splits it)", star(1000), K_MAX
    yield "cora", to_csr(read_edges("cora")), K_MAX
    yield "cora k_max=16 (forced split)", to_csr(read_edges("cora")), 16


def plan_inputs(A):
    return (shell_coeff_data(A), A.data), degrees_from_D(A)


# ---------------------------------------------------------------------------
# 1. build_ladder, over a grid
# ---------------------------------------------------------------------------
def test_build_ladder():
    same = True
    n_case = 0
    for k_max in (1, 2, 3, 7, 16, 64, 256, 1024, 4096):
        for base in (1.1, 1.3, 1.5, 2.0, 3.0):
            a = OLD_FODINED.build_ladder(k_max, base)
            b = OLD_FODIWALK.build_ladder(k_max, base)
            c = NEW.build_ladder(k_max, base)
            n_case += 1
            same &= np.array_equal(a, c) and np.array_equal(b, c)
    check("build_ladder exact over the grid", same, f"{n_case} cases")


# ---------------------------------------------------------------------------
# 2. make_plan, exact on every array
# ---------------------------------------------------------------------------
def plans_equal(p, q) -> bool:
    if len(p) != len(q):
        return False
    for rp, rq in zip(p, q):
        if len(rp) != len(rq):
            return False
        for a, b in zip(rp, rq):
            if a.dtype != b.dtype or not np.array_equal(a, b):
                return False
    return True


def test_make_plan():
    for label, (A, n), k_max in graphs():
        planes, deg = plan_inputs(A)
        out = [m.make_plan(A, planes, degrees=deg, b_cells=B_CELLS,
                           k_max=k_max, ladder_base=LADDER_BASE)
               for m in (OLD_FODINED, OLD_FODIWALK, NEW)]
        (pa, ia, sa), (pb, ib, sb), (pc, ic, sc) = out
        ok = (plans_equal(pa, pc) and plans_equal(pb, pc)
              and np.array_equal(ia, ic) and np.array_equal(ib, ic)
              and ia.dtype == ic.dtype and sa == sc and sb == sc)
        check(f"make_plan exact  {label}", ok, f"n={n} {sc}")


# ---------------------------------------------------------------------------
# 3. the in-batch owner multiplicity, measured and not assumed
# ---------------------------------------------------------------------------
def max_owner_multiplicity(plan, n: int) -> int:
    """The largest count of ONE real owner id inside ONE batch. 3 or more
    makes the scatter of that batch nondeterministic (FINDINGS 1100-1182)."""
    worst = 0
    for rung in plan:
        rows = np.asarray(rung[0])
        for batch in rows:
            c = collections.Counter(int(v) for v in batch if int(v) < n)
            if c:
                worst = max(worst, max(c.values()))
    return worst


# ---------------------------------------------------------------------------
# 4. step, on the SAME plan and the SAME Z
# ---------------------------------------------------------------------------
def test_step():
    for label, (A, n), k_max in graphs():
        planes, deg = plan_inputs(A)
        plan_np, inv_np, stats = NEW.make_plan(
            A, planes, degrees=deg, b_cells=B_CELLS, k_max=k_max,
            ladder_base=LADDER_BASE)
        mult = max_owner_multiplicity(plan_np, n)
        plan = jax.tree_util.tree_map(jax.device_put, plan_np)
        inv = jax.device_put(inv_np)
        Z = jax.random.normal(jax.random.PRNGKey(42), (n, 128), dtype=jnp.float32)

        outs = []
        for m in (OLD_FODINED, OLD_FODIWALK, NEW):
            fn = jax.jit(lambda Z, p, i, q, _m=m: _m._step(
                Z, p, i, q, n=n, force_fn=shell_force))
            outs.append(np.asarray(fn(Z, plan, inv, PARAMS).block_until_ready()))
        a, b, c = outs
        exact = np.array_equal(a, c) and np.array_equal(b, c)
        denom = np.linalg.norm(c)
        rel = max(np.linalg.norm(a - c), np.linalg.norm(b - c)) / (denom or 1.0)
        note = (f"n_split={stats['n_split']} max in-batch owner "
                f"multiplicity={mult} rel={rel:.3e}")
        if mult < 3:
            check(f"step EXACT  {label}", exact, note)
        else:
            # 3 addends on one address: the scatter order is the GPU's, thus
            # a difference here is the hardware and not the refactor.
            check(f"step within tolerance  {label}", rel <= 1e-6,
                  note + "  [>=3 addends: exactness not claimed]")


if __name__ == "__main__":
    print(f"jax {jax.__version__} on {jax.devices()}")
    print(f"numpy {np.__version__}  scipy {sp.__version__ if hasattr(sp,'__version__') else ''}")
    print()
    test_build_ladder()
    test_make_plan()
    test_step()
    print()
    print("RESULT:", "ALL PASS" if not fail else f"{len(fail)} FAILED: {fail}")
    sys.exit(1 if fail else 0)
