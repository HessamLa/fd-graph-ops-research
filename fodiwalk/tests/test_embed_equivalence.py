"""test_embed_equivalence -- `embed/` gives what the god class gave. Byte for byte.

The milestone M1 gate of `dev-docs/REFACTOR.md`: the stage-3 assembly moved
out of `Fodiwalk` into `fodiwalk/embed/`, and a code move that changes a
number is a defect. Each case builds the model the OLD way, keeps what the
old methods gave, and compares it with `build_planes`, `resolve_degrees`,
`build_plans` and `force_params` on the same `D`.

The cases are keys of `golden.AUGMENT_CASES`, one for each branch that
stage 3 has: chunked, fused, `no_deg_norm`, `deg_source = A` through
`edge_rule = low_deg`, `freq_mode = node`, and a `buckets` policy.

This file RETIRES ITSELF: `fodiwalk.fodiwalk` holds the old methods, and
milestone M2 deletes it. The `importorskip` below then skips the file.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fodiwalk.fodiwalk")     # M2 deletes the old class

from fodiwalk.embed.forces import force_fn
from fodiwalk.embed.plan_contract import PlaneContractError
from fodiwalk.embed import (ForceSpec, PlanSpec, build_planes, build_plans,
                            force_params, resolve_degrees)
from fodiwalk.fodiwalk import Fodiwalk
from fodiwalk.tests.golden import AUGMENT_CASES

# Every case of the golden gate. Cora is small, thus the 16 cost seconds
# and no branch of stage 3 is left out: chunked, fused, `no_deg_norm`,
# `deg_source = A` through `edge_rule = low_deg`, `freq_mode = node`, and
# the `buckets` policy.
CASES = list(AUGMENT_CASES)


def _host(x):
    """A device array or a host array, as NumPy."""
    return np.asarray(x)


@pytest.fixture(scope="session")
def old(cora):
    """The OLD result of every case, built one time. `{name: (fw, D)}`."""
    A, n = cora
    out = {}
    for name in CASES:
        fw = Fodiwalk(n_dim=64, seed=42, **AUGMENT_CASES[name])
        D = fw.augment_graph(A)
        out[name] = fw
    return out


def _new(fw, A):
    """The NEW result, from `embed/` alone, on the same `D` and `freq`."""
    fspec = ForceSpec.from_config(fw.cfg)
    pspec = PlanSpec.from_config(fw.cfg)
    planes = build_planes(fspec.law, fw.D, fw.freq,
                          fw.cfg.pairs, fw.cfg.policy)
    degrees = resolve_degrees(fw.D, A, fspec, explicit=fw.deg_from_A)
    plan_set = build_plans(fw.D, planes, degrees, pspec,
                           force_fn(fspec.law))
    return fspec, planes, degrees, plan_set


@pytest.mark.parametrize("name", CASES)
def test_planes_and_degrees_are_identical(name, old, cora):
    A, n = cora
    fw = old[name]
    fspec, planes, degrees, _ = _new(fw, A)

    assert fspec.law == fw.law
    ref_planes = fw._build_planes(fw.D)
    assert len(planes) == len(ref_planes)
    for p, q in zip(planes, ref_planes):
        assert np.array_equal(np.asarray(p), np.asarray(q))
        assert np.asarray(p).dtype == np.asarray(q).dtype
    ref_deg = np.asarray(fw._build_degrees(fw.D, A))
    assert np.array_equal(np.asarray(degrees), ref_deg)
    assert np.asarray(degrees).dtype == ref_deg.dtype


@pytest.mark.parametrize("name", CASES)
def test_params_are_identical(name, old, cora):
    fw = old[name]
    got = force_params(ForceSpec.from_config(fw.cfg))
    assert set(got) == set(fw.params)
    for k in fw.params:
        assert got[k] == fw.params[k], k


@pytest.mark.parametrize("name", CASES)
def test_plan_stats_and_layout_are_identical(name, old, cora):
    A, n = cora
    fw = old[name]
    _, _, _, ps = _new(fw, A)

    assert ps.chunk_rows == fw.chunk_rows
    assert ps.resident == fw.resident
    assert len(ps.plans) == len(fw.plans) == len(ps.steps)
    assert set(ps.stats) == set(fw.plan_stats)
    for k, v in fw.plan_stats.items():
        assert ps.stats[k] == v, k
    assert np.array_equal(_host(ps.deg_ext), _host(fw.deg_ext))


@pytest.mark.parametrize("name", CASES)
def test_the_plan_pytree_is_identical(name, old, cora):
    """Every leaf of every rung of every chunk, on the host."""
    A, n = cora
    fw = old[name]
    _, _, _, ps = _new(fw, A)

    for c, (new_plan, ref_plan) in enumerate(zip(ps.plans, fw.plans)):
        assert len(new_plan) == len(ref_plan)
        for r, (new_rung, ref_rung) in enumerate(zip(new_plan, ref_plan)):
            assert len(new_rung) == len(ref_rung)
            for j, (a, b) in enumerate(zip(new_rung, ref_rung)):
                a, b = _host(a), _host(b)
                assert a.dtype == b.dtype, (c, r, j)
                assert np.array_equal(a, b), f"chunk {c}, rung {r}, leaf {j}"


def test_a_missing_freq_raises_and_names_the_plane_and_the_law(old, cora):
    """No fallback to the planes of another law. The 2026-08-18 defect."""
    fw = old["nbr_walk"]
    for law, plane in (("fdlinear", "freq"), ("fdlinear_fused", "w")):
        with pytest.raises(PlaneContractError) as e:
            build_planes(law, fw.D, None, "nbr_walk", "cap")
        msg = str(e.value)
        assert law in msg and plane in msg


def test_the_registry_mirrors_the_asserter():
    """One plane name, one builder, one check. `PLANE_BUILDERS` <-> `PLANE_CHECKS`."""
    from fodiwalk.embed.forces import FORCE_PLANES
    from fodiwalk.embed.plan_contract import PLANE_CHECKS
    from fodiwalk.embed.planes import PLANE_BUILDERS

    assert set(PLANE_BUILDERS) == set(PLANE_CHECKS)
    for names in FORCE_PLANES.values():
        for name in names:
            assert name in PLANE_BUILDERS
