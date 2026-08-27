"""test_parity.py -- PRD section 8. THE GATE OF THE WHOLE PROJECT.

The package must reproduce these recorded runs. A difference is a defect
and not a finding: the refactor moved the physics, it did not change it.

Every scenario names the log it comes from, thus a reader can check the
number without trusting this file.

    P1  results/fused/cora_3plane.log and cora_fused.log, 2026-08-17
    P2  results/lrladder/const_plain_lr0.999.log, 2026-08-18
    P3  results/grid_cora/C_buckets_far_s{42,56,88}.log, 2026-08-18
    P4  results/grid_cora/D_p2.0_q1.0_s{42,56,88}.log, 2026-08-18
    P5  the row-cap rule, on com_youtube
    P6  results/optim/smoke_*.log, 2026-08-18 -- RETIRED 2026-08-19, see
        the note above `FDLINEAR_ROSTER` at the end of this file

A CORRECTION TO THE PRD, and it is a label and not a number. PRD section 8
writes P2 as `walk/min_gap/v1/plain`. The recorded run behind
`auc = 0.9986, r2_dist = 0.523` is `nbr_walk/min_gap/fdlinear/plain` with
`k4 = 1.0`, from `run_lrladder.sh`; `walk/v1` has no such row. The
configuration below is the one that produced the number.

One test in this file is NOT parity: `test_the_eight_optimizers_on_fdlinear`
is a baseline this package measured, and it replaced P6 when the law P6 ran
was removed. Its docstring says so. Everything else here reproduces a run
that the campaign recorded.

These tests are marked `parity`: they need a GPU and several minutes.

    .venv/bin/python -m pytest fodiwalk/tests/test_parity.py -m parity -q
"""
from __future__ import annotations

import statistics

import numpy as np
import pytest

from fodiwalk.augment_graph import pairs as PR
from fodiwalk.augment_graph import walks as W
from fodiwalk.tests.harness import run

pytestmark = pytest.mark.parity


# The flags that every 2026-08-18 grid shares (`run_grids_2026-08-18.sh`).
GRID = dict(weight="min_gap", far_bias=0.75, far_weight=100.0)
FDL = dict(force="fdlinear", k4=1.0, kr=1.0)


def test_p1_fdlinear_on_cora_200_epochs(cora):
    """P1. EXACT: the path is `uniform_walks`, thus it is bit-exact.

    `dz = 0.5016  auc = 0.9962  r2_dist = 0.253`, and the fused plane must
    give the same numbers as the three-plane reference.

    The tight `abs=5e-5` tolerances hold because of the PLAN, and not
    because the graph is Cora. A row wider than `k_max` splits into virtual
    rows that share an owner id, and `dZ.at[rows].add(...)` then
    accumulates in a free order on a GPU.

    `n_split == 0` is asserted below as a CONSERVATIVE PROXY. The real
    invariant is `max in-batch owner multiplicity < 3`: three addends
    landing on one address cost exactness, and splitting alone does not.
    `n_split == 0` forces multiplicity 1, thus it IMPLIES the real
    condition; it is not equivalent to it. `forcedirected/PARITY.md`
    section 5 is the proof -- an augmented Cora `D` at `n_split = 3` with
    multiplicity 2 was bit-identical at 0.000e+00. The proxy is used
    because `n_split` is already in `plan_stats` and the multiplicity is
    not.

    THUS, WHEN THIS ASSERT FAILS: re-measure the multiplicity. Do NOT
    relax the tolerances, and do NOT delete the assert. A legitimate change
    can reach `n_split = 2` with multiplicity still 2 and stay exact; the
    repair is then to assert the multiplicity directly, as
    `forcedirected/tests/test_parity.py` does with `max_owner_multiplicity`.
    Without the guard, such a change surfaces as an `acc`/`f1` mismatch and
    reads as a defect of the physics, which is the wrong diagnosis this
    scenario must not invite.
    """
    A, n = cora
    for fuse_planes in (False, True):
        out = run(A, n, dim=64, epochs=200, lr=1.0, seed=42, optim="plain",
                  pairs="nbr_walk", weight="min_gap", force="fdlinear",
                  k4=0.01, kr=1.0, fuse_planes=fuse_planes)
        assert out["dnnz"] == 209_542
        # The precondition of every tolerance below. See the docstring.
        assert out["plan"]["n_split"] == 0
        assert out["plan"]["pad_frac"] == pytest.approx(0.20059361671282838)
        assert out["dz"] == pytest.approx(0.5016, abs=5e-5)
        assert out["acc"] == pytest.approx(0.9754, abs=5e-5)
        assert out["f1"] == pytest.approx(0.9752, abs=5e-5)
        assert out["auc"] == pytest.approx(0.9962, abs=5e-5)
        assert out["r2_dist"] == pytest.approx(0.253, abs=5e-4)
        assert out["mae_dist"] == pytest.approx(1.291, abs=5e-4)


def test_p2_the_learning_rate_ladder_at_0999(cora):
    """P2. EXACT. `auc = 0.9986  r2_dist = 0.523`, and `dz = 0.1274`."""
    A, n = cora
    out = run(A, n, dim=64, epochs=2000, lr=0.999, seed=42, optim="plain",
              pairs="nbr_walk", **GRID, **FDL)
    assert out["dnnz"] == 209_542
    assert out["dz"] == pytest.approx(0.1274, abs=5e-5)
    assert out["auc"] == pytest.approx(0.9986, abs=5e-5)
    assert out["acc"] == pytest.approx(0.9858, abs=5e-5)
    assert out["r2_dist"] == pytest.approx(0.523, abs=5e-4)


def test_p3_buckets_with_far_pairs_three_seeds(cora):
    """P3. Inside the recorded seed spread: `r2_dist = 0.636 +-0.024`."""
    A, n = cora
    got = [run(A, n, dim=64, epochs=2000, lr=0.1, seed=s, optim="plain",
               pairs="walk", policy="buckets", far_with_buckets=True,
               **GRID, **FDL)["r2_dist"] for s in (42, 56, 88)]
    mean = statistics.mean(got)
    print(f"\nP3 r2_dist per seed {got}, mean {mean:.4f} "
          f"+-{statistics.pstdev(got):.4f} (recorded 0.6357 +-0.024)")
    assert mean == pytest.approx(0.6357, abs=0.024), got
    assert statistics.pstdev(got) < 0.06, got


def test_p4_the_second_order_walk_at_p2_three_seeds(cora):
    """P4. Inside the recorded seed spread: `r2_dist = 0.452 +-0.016`."""
    A, n = cora
    got = [run(A, n, dim=64, epochs=2000, lr=0.1, seed=s, optim="plain",
               pairs="nbr_walk", p=2.0, q=1.0, **GRID, **FDL)["r2_dist"]
           for s in (42, 56, 88)]
    mean = statistics.mean(got)
    print(f"\nP4 r2_dist per seed {got}, mean {mean:.4f} "
          f"+-{statistics.pstdev(got):.4f} (recorded 0.452 +-0.016)")
    assert mean == pytest.approx(0.452, abs=0.016), got
    assert statistics.pstdev(got) < 0.05, got


def test_p5_row_cap_on_cora(cora):
    """P5, on the graph that every other scenario uses.

    The rule: row `u` keeps its own best `m`, thus its width is exactly
    `min(m, what the walks of that row found)`.

    On com_youtube every row finds more than 16 partners, thus the width is
    16 everywhere and the total is `n * 16` -- which is how the PRD writes
    the criterion, and `test_p5_row_cap_on_com_youtube` asserts it there.
    Cora is not like that: it holds small components, thus 208 of its 2708
    rows (7.7%) find fewer than 16 partners and keep what they found. The
    exact form of the rule holds on both graphs.
    """
    A, n = cora
    st = W.walk_rows(A, n, 10, 20, np.random.default_rng(42))
    before = np.bincount(st["key"] // n, minlength=n)
    capped = PR.row_cap(st, n, 16)
    width = np.bincount(capped["key"] // n, minlength=n)
    assert width.max() == 16
    assert np.array_equal(width, np.minimum(before, 16))
    assert capped["key"].size == int(np.minimum(before, 16).sum())
    assert int((before < 16).sum()) == 208        # and the rest are full


@pytest.mark.big
def test_p5_row_cap_on_com_youtube():
    """P5 at 1,134,890 nodes. A few GB and about two minutes.

    PRD section 8 writes the criterion as "max row width 16, `n*16`
    entries". `n * 16` is the count only when EVERY row finds more than 16
    partners, which is the qualifier PRD B6.4 itself carries and section 8
    drops. com_youtube nearly meets it and does not quite: 18,154,004
    entries of the 18,158,240 that `n * 16` would ask for, thus a few
    hundred rows find fewer than 16 and keep what they found. The exact
    rule -- `width == min(16, what the row found)` -- holds everywhere, and
    it is what this asserts.
    """
    from fodiwalk.make_graph import load
    A, n = load("com_youtube")
    assert n == 1_134_890
    st = W.walk_rows(A, n, 10, 20, np.random.default_rng(42))
    before = np.bincount(st["key"] // n, minlength=n)
    capped = PR.row_cap(st, n, 16)
    width = np.bincount(capped["key"] // n, minlength=n)
    assert width.max() == 16
    assert np.array_equal(width, np.minimum(before, 16))
    assert capped["key"].size == int(np.minimum(before, 16).sum())
    assert capped["key"].size == 18_154_004        # 99.98% of n * 16


# ---------------------------------------------------------------------------
# P6 IS RETIRED, and this is what replaced it. Read the docstring below.
# ---------------------------------------------------------------------------
# The recorded P6 run of 2026-08-18 used `v1`, the shell-averaged law. The
# package reproduced it on 2026-08-19 -- all eight rules to three decimals,
# `dev-docs/BUILD.md` section 3 -- and `v1` was removed on the same day
# (`dev-docs/CATALOG.md` section 13). A package with no `v1` cannot run that
# scenario again, thus the parity claim retires with the law and the record
# in BUILD.md stands in its place.
#
# The roster is still an axis of the project, thus it still needs a
# regression test. This is that test, on the law the project now runs.
FDLINEAR_ROSTER = {"plain": 0.609, "sqn": 0.607, "velocity": 0.606,
                   "sgd": 0.522, "adam": 0.213, "momentum": 0.085,
                   "nesterov": -0.000, "fa2": None}


def test_the_eight_optimizers_on_fdlinear(cora):
    """The optimizer roster under `fdlinear`. A BASELINE, and NOT parity.

    Cora, `walk/min_gap/fdlinear`, 60 epochs, dim 32, lr 1.0, seed 42 --
    the shape of the 2026-08-18 smoke run, with the law it used replaced.
    The numbers were measured on 2026-08-19 by this package. They are NOT
    the numbers of the campaign and they must never be quoted as such: the
    campaign ran `v1` and this runs `fdlinear`, thus the two are different
    experiments and only their SHAPE is comparable.

    What the shape says, and it survived the change of the law: `plain`
    leads, `sqn` and `velocity` follow within 0.003 of it, `sgd` is one
    step behind, and `fa2` diverges. What changed: the whole leading group
    rose from 0.373 to 0.609, and `adam` moved above `momentum` and
    `nesterov`.

    `momentum` and `nesterov` are inside a near-divergent regime at this
    rate -- `||dZ||` ends at 1.8e4 and 1.3e6 against 1.8 for `plain` --
    thus their hop R2 is near zero because the layout blew up, and not
    because a fit came out badly. `fa2` goes the whole way and gives a
    non-finite `Z`, which the run RECORDS (I7) and does not raise.
    """
    A, n = cora
    got = {}
    for name in FDLINEAR_ROSTER:
        out = run(A, n, dim=32, epochs=60, lr=1.0, seed=42, optim=name,
                  pairs="walk", weight="min_gap")
        got[name] = None if out["diverged"] else out["r2_dist"]

    print("\nroster hop R2:", {k: (None if v is None else round(v, 3))
                               for k, v in got.items()})
    assert got["fa2"] is None, "fa2 gives a non-finite Z at lr 1.0"
    for name, want in FDLINEAR_ROSTER.items():
        if want is None:
            continue
        assert got[name] == pytest.approx(want, abs=0.06), (name, got)

    # the leading group, and it must stay ahead of the rest
    lead = [got[k] for k in ("plain", "sqn", "velocity")]
    assert min(lead) > got["sgd"] > got["adam"], got
    assert max(lead) - min(lead) < 0.02, lead


@pytest.mark.big
def test_b2_the_com_youtube_plan_of_2026_08_17():
    """PRD B2.3: `pad_frac == 0.15407662320379392` on the com_youtube plan.

    The recorded run is `results/fused/yt1M_fused.log` and
    `results/mixmatch/mm_yt1M_fdlinear_nbr_walk_both_s42.log`, which agree
    on every plan field. Its walk budget is in no surviving script; it is
    `2 walks x 10 steps`, which is the only pair that gives the recorded
    `19,668,991` raw pairs.

    `pad_frac` is the LAST CHUNK's, and the chunk count is part of it: the
    run used 8 chunks on the host. `cells` is the sum over the chunks.

    This asserts the whole augmentation of a 1.13M-node graph at once --
    the walks, the neighbours, the far pairs at `deg^0.75`, and the plan.
    It needs about 4 GB and a few minutes.
    """
    from fodiwalk import Fodiwalk
    from fodiwalk.make_graph import load

    A, n = load("com_youtube")
    fw = Fodiwalk(n_dim=64, seed=42, lr=0.999, optim="plain",
                  pairs="nbr_walk", weight="min_gap", walks=2, walk_len=10,
                  far=2_270_000, far_bias=0.75, far_weight=100.0,
                  chunks=8, chunk_host=True, force="fdlinear",
                  fuse_planes=True, k4=1.0, kr=1.0)
    D = fw.augment_graph(A)

    assert fw.info["raw_pairs"] == 19_668_991
    assert fw.info["unique_pairs"] == 18_624_811
    assert fw.info["h1_entries"] == 5_975_248 == fw.info["edges_of_A"]
    assert fw.info["far_pairs"] == 2_269_516
    assert D.nnz == 23_163_843

    ps = fw.plan_stats
    assert ps["cells"] == 23_163_843
    assert ps["n_virtual"] == 141_943
    assert ps["n_split"] == 50
    assert ps["rungs"] == 14
    assert ps["chunks"] == 8
    assert ps["rows_per_chunk"] == 141_862
    assert ps["pad_frac"] == 0.15407662320379392
