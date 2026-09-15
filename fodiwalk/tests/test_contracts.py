"""test_contracts.py -- the success criteria of PRD sections 6 and 7.

One test for one criterion, and the name of the test says which. These
tests use a small graph and they do not need a recorded run; the recorded
runs are in `test_parity.py`.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest
import scipy.sparse as sp

from forcedirected import csr, sell_c_sigma
from fodiwalk.embed import forces, plan_contract
from fodiwalk.embed.plan_contract import PlaneContractError
from fodiwalk.augment_graph import pairs as PR
from fodiwalk.augment_graph import walks as W
from fodiwalk.augment_graph import weights as WT
from fodiwalk.augment_graph.far_pairs import sample_far_pairs
from forcedirected import optim

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent


def _random_csr(rng, n, density, empty_rows=0):
    A = sp.random(n, n, density=density, format="csr", random_state=int(
        rng.integers(1 << 30)))
    A.data[:] = rng.integers(1, 6, A.nnz).astype(np.float64)
    if empty_rows:
        keep = np.ones(n, dtype=bool)
        keep[rng.choice(n, empty_rows, replace=False)] = False
        A = sp.csr_matrix(A.multiply(keep[:, None]))
    A.sort_indices()
    return A


# ===========================================================================
# B1 -- the CSR helpers, `forcedirected/csr.py`
# ===========================================================================
def test_b1_row_of_matches_repeat_on_20_random_csrs():
    """B1.1. Also on matrices that hold empty rows."""
    rng = np.random.default_rng(0)
    for i in range(20):
        n = int(rng.integers(5, 200))
        A = _random_csr(rng, n, density=0.05, empty_rows=int(n // 4))
        want = np.repeat(np.arange(n), np.diff(A.indptr))
        assert np.array_equal(csr.row_of(A.indptr), want)
        assert csr.n_rows(A) == n
    # a CSR whose every row is empty is the degenerate end of the same rule
    E = sp.csr_matrix((7, 7))
    assert csr.row_of(E.indptr).size == 0


def test_b1_engine_package_imports_nothing_of_this_repository():
    """B1.2. `csr.py` is the bottom of the tree, and `forcedirected/` is
    the bottom of the repository.

    `row_of` and `n_rows` moved to `forcedirected/csr.py` 2026-08-26 with
    the engine that reads them, and `fodiwalk/core/csr.py` stayed as a
    forwarder until 2026-08-27. The contract widened with the move: EVERY
    module of `forcedirected` imports numpy, scipy, jax and its own
    modules, and no package of this repository. That is what let `fodined`
    and `fodiwalk` share the engine without either depending on the other.

    `tests/` is excluded: `m1_old_vs_new.py` is a closed record that names
    both callers on purpose, and the rule is about the package."""
    repo = PKG.parent
    pkgs = {d.name for d in repo.iterdir()
            if d.is_dir() and ((d / "__init__.py").exists()
                               or any(d.glob("*.py")))} - {"forcedirected"}
    for path in sorted((repo / "forcedirected").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0 or not (node.module or "").startswith(
                    ".."), f"{path.name} imports above its own package"
                if node.level == 0 and node.module:
                    names = [node.module]
            for name in names:
                assert name.split(".")[0] not in pkgs, (
                    f"forcedirected/{path.name} imports {name!r}, a package "
                    f"of this repository")


def test_embed_imports_only_embed():
    """The risk of section 9, at its new address. This gate was
    `test_core_imports_only_core` until 2026-08-28, when `forces.py` and
    `plan_contract.py` moved from `fodiwalk/core/` to `fodiwalk/embed/` and
    `fodiwalk/core/` was deleted. `embed` now holds the physics, thus it is
    the package at the BOTTOM of the stages and it must import no other
    stage at module level.

    `embed` reaches the ROOT package `forcedirected` for the engine, the
    kernel and the CSR helpers; that is no stage of `fodiwalk` and it reads
    nothing of this repository, thus the one-way rule is unchanged, and it
    is what this gate keeps. Stage 2 reads `embed` (it asks the law which
    planes to build), thus an import the other way would close a cycle;
    `test_structure.py` holds that gate at function level too."""
    for path in sorted((PKG / "embed").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:                       # module level only
            mods = []
            if isinstance(node, ast.ImportFrom):
                mods = [("." * node.level) + (node.module or "")]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                bad = ("..augment_graph", "..misc", "..make_graph",
                       "..fodiwalk", "fodiwalk.augment_graph",
                       "fodiwalk.misc", "fodiwalk.make_graph")
                assert not m.startswith(bad), f"{path.name} imports {m}"


# ===========================================================================
# B2 -- the kernel, `forcedirected/sell_c_sigma.py`, and the plane contract
# ===========================================================================
def test_b2_make_plan_raises_on_a_wrong_plane_shape(tiny):
    """B2.1."""
    A, n = tiny
    deg = np.diff(A.indptr).astype(np.float64)
    with pytest.raises(ValueError, match="expected"):
        sell_c_sigma.make_plan(A, (np.ones(A.nnz - 1),), degrees=deg)


def test_b2_hub_split_keeps_every_row(tiny):
    """B2.2. A row wider than `k_max` splits, and the row set survives."""
    A, n = tiny
    deg = np.diff(A.indptr).astype(np.float64)
    plan, inv, stats = sell_c_sigma.make_plan(
        A, (A.data,), degrees=deg, k_max=8, b_cells=64)
    assert stats["n_split"] > 0
    seen = {}
    for rows, nbrs, tile in plan:
        rows = np.asarray(rows).ravel()
        nbrs = np.asarray(nbrs).reshape(rows.size, -1)
        tile = np.asarray(tile).reshape(rows.size, -1)
        for r, nb, tl in zip(rows, nbrs, tile):
            if r == n:                                   # a pad ROW
                continue
            seen.setdefault(int(r), []).extend(
                int(x) for x, t in zip(nb, tl) if t != 0)
    for u in range(n):
        want = set(A.indices[A.indptr[u]:A.indptr[u + 1]].tolist())
        assert set(seen.get(u, [])) == want


def test_b2_a_pad_cell_contributes_zero_for_a_law_with_a_constant(tiny):
    """B2.4 and I3. `fdlinear` has the constant `kr`, thus a pad cell that
    did not carry 0 in every plane would push."""
    import jax.numpy as jnp
    A, n = tiny
    freq = np.ones(A.nnz)
    deg = np.diff(A.indptr).astype(np.float64)
    plan, inv, stats = sell_c_sigma.make_plan(
        A, (A.data, freq), degrees=deg, k_max=8, b_cells=64)
    assert stats["pad_frac"] > 0                    # the test needs padding
    plan_contract.check_plan(plan, 2)
    # `node_degree` of 1: this test asks whether the LAW vanishes on a
    # pad cell, and a divisor of 1 cannot hide a non-zero answer.
    params = dict(k1=0.999, k4=0.01, kr=1.0, sign=-1.0, node_degree=1.0)
    for rung in plan:
        h, fq = np.asarray(rung[2]), np.asarray(rung[3])
        pad = (h == 0) & (fq == 0)
        # x is exactly 0 on a pad cell too; take x = 1 to show that the LAW
        # alone vanishes, and not only the kernel's `x == 0` guard.
        F = np.asarray(forces.fdlinear(jnp.ones_like(jnp.asarray(h)),
                                       (jnp.asarray(h), jnp.asarray(fq)),
                                       params))
        assert np.all(F[pad] == 0.0)


# ===========================================================================
# B3 -- embed/forces.py and the registry
# ===========================================================================
def test_b3_registry_covers_every_law():
    """B3.1. One law, one plane tuple, and no branch anywhere else."""
    assert set(forces.FORCE_PLANES) == set(forces.FORCE_FN)
    assert forces.planes_of("fdlinear") == ("h", "freq")
    assert forces.planes_of("fdlinear_fused") == ("w",)
    with pytest.raises(ValueError):
        forces.planes_of("no_such_law")


def test_b3_fused_reproduces_fdlinear_cellwise(tiny):
    """B3.2, at the level of the law. The whole-run form is P1."""
    import jax.numpy as jnp
    rng = np.random.default_rng(3)
    h = rng.integers(1, 6, 500).astype(np.float32)
    fq = rng.integers(1, 9, 500).astype(np.float32)
    x = rng.random(500).astype(np.float32) * 4
    w = forces.fuse(h, fq)
    # one `node_degree` for both, so this compares the physics only
    p = dict(k1=0.999, k4=0.01, kr=1.0, sign=-1.0, node_degree=1.0)
    a = np.asarray(forces.fdlinear(jnp.asarray(x), (jnp.asarray(h),
                                                    jnp.asarray(fq)), p))
    b = np.asarray(forces.fdlinear_fused(jnp.asarray(x), (jnp.asarray(w),), p))
    assert np.allclose(a, b, atol=1e-6)


def test_b3_every_law_divides_by_its_own_node_degree():
    """B3.5, since 2026-09-09. `1 / deg(u)` is the LAW's averaging
    coefficient. `forcedirected.sell_c_sigma.step` supplies
    `params["node_degree"]` and divides NOTHING; a law that forgets
    the inline divide keeps the whole row sum and the row diverges with no
    error. This is the gate that replaced the kernel's own division.

    Every law is LINEAR in `1 / node_degree`, thus a degree of 2 gives
    exactly half of a degree of 1. Both are powers of two, so float32
    makes the comparison BIT-exact and no tolerance is needed.
    """
    import jax.numpy as jnp
    rng = np.random.default_rng(11)
    m = 400
    h = rng.integers(1, 5, m).astype(np.float32)
    fq = rng.integers(1, 9, m).astype(np.float32)
    plane = {"h": h, "freq": fq, "w": forces.fuse(h, fq),
             "deg_le": rng.choice([-1.0, 1.0], m).astype(np.float32)}
    x = (rng.random(m).astype(np.float32) * 4.0) + 0.1
    base = dict(k1=0.999, k2=1.0, k4=0.01, kr=1.0, sign=-1.0)

    for law, names in forces.FORCE_PLANES.items():
        fn = forces.FORCE_FN[law]
        pl = tuple(jnp.asarray(plane[nm]) for nm in names)
        def run(d):
            return np.asarray(fn(jnp.asarray(x), pl,
                                 dict(base, node_degree=jnp.float32(d))))
        one, two = run(1.0), run(2.0)
        assert np.array_equal(two, one * np.float32(0.5)), (
            f"{law} does not divide by params['node_degree']. The kernel "
            f"stopped dividing on 2026-09-09, thus every law must end with "
            f"an inline divide by params['node_degree'], or it loses it in silence.")
        assert np.all(run(0.0) == 0.0), (
            f"{law}: a degree of 0 must give EXACTLY 0 and never a "
            f"division by zero. That is what the old `inv_deg_ext` did, "
            f"and invariant I5 exists to catch the frozen row it makes.")


def test_b3_degrees_from_D_counts_the_h1_entries(tiny):
    """B3.4."""
    A, n = tiny
    A = A.copy()
    A.data[:] = np.random.default_rng(2).choice([1, 2, 3], A.nnz)
    want = np.bincount(csr.row_of(A.indptr)[A.data == 1], minlength=n)
    assert np.array_equal(forces.degrees_from_D(A), want.astype(np.float64))


# ===========================================================================
# plan_contract -- the invariants of PRD section 7
# ===========================================================================
def test_i2_a_plane_that_is_not_D_data_in_the_h_position_raises(tiny):
    """THE 2026-08-18 DEFECT. `fdlinear` reads `(h, freq)`. Given the
    planes of another law it read a coefficient plane AS `h`, the run
    diverged to NaN, and nothing raised. The guard tests the VALUES: a
    plane in the `h` position must be `D.data` itself."""
    A, n = tiny
    D = A.copy()
    D.data[:] = np.random.default_rng(4).integers(1, 5, D.nnz)
    freq = np.random.default_rng(8).integers(1, 9, D.nnz).astype(np.float64)
    with pytest.raises(PlaneContractError, match="`h` plane"):
        plan_contract.check("fdlinear", (freq, D.data), D)


def test_i2_a_wrong_plane_count_raises(tiny):
    A, n = tiny
    D = A.copy()
    D.data[:] = 1.0
    with pytest.raises(PlaneContractError, match="reads 2 planes"):
        plan_contract.check("fdlinear", (D.data,), D)
    with pytest.raises(PlaneContractError, match="reads 1 planes"):
        plan_contract.check(
            "fdlinear_fused", (forces.fuse(D.data, np.ones(D.nnz)), D.data),
            D)


def test_i1_a_wrong_plane_shape_raises(tiny):
    A, n = tiny
    D = A.copy()
    D.data[:] = 1.0
    with pytest.raises(PlaneContractError, match="shape"):
        plan_contract.check("fdlinear", (D.data[:-1], D.data[:-1]), D)


def test_i4_a_continuous_weight_raises(tiny):
    """A stored weight is an INTEGER >= 1. A float weight gave every pair
    its own shell, `degrees_from_D` gave 0 for every row, and the whole
    force vanished: AUC 0.55 with ||dZ|| = 0.000."""
    A, n = tiny
    D = A.copy()
    D.data[:] = np.random.default_rng(5).random(D.nnz) * 3 + 1
    with pytest.raises(PlaneContractError, match="INTEGER"):
        plan_contract.check("fdlinear", (D.data, np.ones(D.nnz)), D)


def test_i5_a_row_with_stored_pairs_and_degree_zero_raises(tiny):
    """The row would freeze for the whole run, in silence."""
    A, n = tiny
    D = A.copy()
    D.data[:] = 2.0                      # no entry at h = 1 anywhere
    with pytest.raises(PlaneContractError, match="freeze"):
        plan_contract.check_degrees(forces.degrees_from_D(D), D)
    plan_contract.check_degrees(np.ones(n), D)          # explicit: legal


def test_the_contract_accepts_what_the_registry_asks_for(tiny):
    A, n = tiny
    D = A.copy()
    D.data[:] = np.random.default_rng(6).integers(1, 5, D.nnz)
    freq = np.random.default_rng(7).integers(1, 9, D.nnz).astype(np.float64)
    plan_contract.check("fdlinear", (D.data, freq), D)
    plan_contract.check("fdlinear_fused", (forces.fuse(D.data, freq),), D)


# ===========================================================================
# B6 -- augment_graph
# ===========================================================================
def test_b6_make_walker_returns_uniform_walks_itself(tiny):
    """B6.1. Not the rejection sampler with trivial weights: every result
    before 2026-08-18 used the uniform walk, thus only this keeps a default
    run bit-exact."""
    A, n = tiny
    walker = W.make_walker(A, n, 1.0, 1.0)
    assert walker.func is W.uniform_walks

    a = W.walk_rows(A, n, 4, 10, np.random.default_rng(42), walker=walker)
    b = W.walk_rows(A, n, 4, 10, np.random.default_rng(42))
    assert np.array_equal(a["key"], b["key"])          # BIT-identical
    assert np.array_equal(a["mn"], b["mn"])
    assert np.array_equal(a["cnt"], b["cnt"])


def test_b6_node2vec_at_p_q_1_matches_uniform_distributionally(tiny):
    """B6.2. NOT bit-exact: the rejection loop draws one more number for
    each accept test, thus the two consume the stream differently."""
    A, n = tiny
    L = 12
    u = W.uniform_walks(A, np.tile(np.arange(n), 40), L,
                        np.random.default_rng(1))
    v = W.node2vec_walks(A, np.tile(np.arange(n), 40), L,
                         np.random.default_rng(1), p=1.0, q=1.0)
    assert not np.array_equal(u, v)                    # not the same array
    du = np.array([np.unique(r).size for r in u]).mean()
    dv = np.array([np.unique(r).size for r in v]).mean()
    assert abs(du - dv) < 0.15 * du                    # the same walk


def test_b6_node2vec_p_and_q_move_the_walk_as_predicted(tiny):
    """B6.3. All four of p<1, p>1, q<1, q>1."""
    A, n = tiny
    starts = np.tile(np.arange(n), 60)

    def distinct(p, q):
        w = W.node2vec_walks(A, starts, 12, np.random.default_rng(9),
                             p=p, q=q)
        return float(np.mean([np.unique(r).size for r in w]))

    base = distinct(1.0, 1.0)
    # p < 1 makes the return to the previous node cheap, thus the walk
    # doubles back and it sees FEWER distinct nodes. p > 1 is the opposite.
    assert distinct(0.25, 1.0) < base
    assert distinct(4.0, 1.0) > base
    # q < 1 sends the walk away (DFS-like), thus MORE distinct nodes.
    # q > 1 holds it near the previous node (BFS-like), thus fewer.
    assert distinct(1.0, 0.25) > base
    assert distinct(1.0, 4.0) < base


def test_b6_row_cap_gives_every_row_the_same_width(tiny):
    """B6.4. Exactly `m` wide, and exactly `n * m` entries."""
    A, n = tiny
    st = W.walk_rows(A, n, 10, 20, np.random.default_rng(3))
    m = 8
    capped = PR.row_cap(st, n, m)
    rows = capped["key"] // n
    width = np.bincount(rows, minlength=n)
    assert width.max() == m
    assert capped["key"].size == n * m               # every row is full
    assert width.min() == m


def test_b6_every_weight_rule_gives_an_integer_in_the_window(tiny):
    """B6.5. A continuous weight makes the whole force vanish (I4)."""
    A, n = tiny
    window = 5
    st = W.walk_pair_stats(A, n, 10, 20, window, np.random.default_rng(4),
                           cap=16)
    for name, rule in WT.RULES.items():
        h = rule(st, window, n)
        assert np.array_equal(h, np.rint(h)), name
        assert h.min() >= 1 and h.max() <= window, name


def test_b6_sample_far_pairs_rejects_a_pair_stored_in_either_direction(tiny):
    """B6.6. On a DIRECTED `D` the old test rejected one direction only,
    thus the CSR build SUMMED the far weight onto a stored walk gap and the
    histogram showed entries at 101..119."""
    A, n = tiny
    st = W.walk_rows(A, n, 6, 12, np.random.default_rng(5))
    near = PR.to_csr_directed(st["key"], st["mn"].astype(np.float64), n)
    stored = set(zip(*near.nonzero()))
    both = {(u, v) for u, v in stored} | {(v, u) for u, v in stored}

    far = sample_far_pairs(n, 200, near, np.random.default_rng(6),
                           directed=True)
    for u, v in far:
        assert (int(u), int(v)) not in both
    # and the default keeps the recorded behaviour: it tests the stored
    # direction only, which is exact for a SYMMETRIC D.
    sym = PR.to_csr(st["key"], st["mn"].astype(np.float64), n)
    far2 = sample_far_pairs(n, 200, sym, np.random.default_rng(6))
    sym_stored = set(zip(*sym.nonzero()))
    for u, v in far2:
        assert (int(u), int(v)) not in sym_stored
        assert (int(v), int(u)) not in sym_stored


# ===========================================================================
# B7 -- misc
# ===========================================================================
def test_b7_state_arrays():
    """B7.2."""
    assert optim.state_arrays("sqn", 3) == 8
    assert optim.state_arrays("sqn", 5) == 12
    assert optim.state_arrays("plain") == 0
    assert set(optim.RULES) == set(optim.STATE_ARRAYS)
    assert len(optim.RULES) == 8


def test_b7_link_prediction_caps_and_never_samples_a_stored_edge(cora):
    """B7.3."""
    from fodiwalk.misc.evaluation import (link_prediction, sample_positives,
                                          sample_negatives)
    A, n = cora
    rng = np.random.default_rng(0)
    max_pairs = 2_000
    pos = sample_positives(A, max_pairs // 2, rng)
    neg = sample_negatives(A, n, pos.shape[0], rng)
    assert pos.shape[0] == max_pairs // 2
    assert len({tuple(p) for p in pos}) == pos.shape[0]
    assert np.asarray(A[neg[:, 0], neg[:, 1]]).ravel().sum() == 0
    assert np.asarray(A[neg[:, 1], neg[:, 0]]).ravel().sum() == 0

    Z = np.random.default_rng(1).normal(size=(n, 8))
    scores, info = link_prediction(Z, A, n, max_pairs, rng, seed=0)
    assert info["pairs"] <= max_pairs
    assert set(scores) == {"accuracy", "precision", "recall", "f1_score",
                           "auc"}


# ===========================================================================
# B5 -- make_graph/datasets.py
# ===========================================================================
def test_b5_cora_is_symmetric_with_a_zero_diagonal(cora):
    """B5.1 and B5.2, on the graph every fast test uses."""
    A, n = cora
    assert n == 2708
    assert A.shape == (n, n)
    assert A.has_sorted_indices
    assert (A != A.T).nnz == 0
    assert A.diagonal().sum() == 0
    assert set(np.unique(A.data)) == {1.0}


@pytest.mark.slow
def test_b5_pubmed_has_19717_nodes():
    """B5.1."""
    from fodiwalk.make_graph import load
    A, n = load("pubmed")
    assert n == 19717
    assert (A != A.T).nnz == 0
    assert A.diagonal().sum() == 0


@pytest.mark.big
def test_b5_com_youtube_has_1134890_nodes_and_a_hub_of_28754():
    """B5.1, at the size the campaign runs on."""
    from fodiwalk.make_graph import load
    A, n = load("com_youtube")
    assert n == 1_134_890
    assert int(np.diff(A.indptr).max()) == 28_754
    assert A.has_sorted_indices


def test_b5_induced_keeps_the_edges_of_the_kept_nodes(cora):
    A, n = cora
    keep = np.arange(0, n, 3)
    from fodiwalk.make_graph.datasets import induced
    B = induced(A, keep, n)
    assert B.shape == (keep.size, keep.size)
    assert B.nnz == A[keep][:, keep].nnz


# ===========================================================================
# The far pairs of a DIRECTED D -- the repair of 2026-08-17
# ===========================================================================
def test_the_nbr_walk_policy_stores_no_far_pair_it_already_holds(tiny):
    """`sample_far_pairs` rejects on the keys AS STORED, thus on a directed
    `near` a pair held as (v, u) passed and the CSR build SUMMED the two:
    the weight became 100 + the walk gap, and the histogram showed entries
    at 101..119. `Fodiwalk` keeps the filter of `archive/fdwalk/bench_fdwalk.py`, thus
    every stored weight is either a walk gap or exactly `far_weight`."""
    from fodiwalk import Fodiwalk
    A, n = tiny
    fw = Fodiwalk(n_dim=4, seed=7, pairs="nbr_walk", force="fdlinear",
                  walks=3, walk_len=8, far=150, far_weight=100.0)
    D = fw.augment_graph(A)
    weights = np.unique(D.data)
    assert weights.max() == 100.0
    assert not ((weights > 1.5) & (weights < 100.0) &
                (weights != np.rint(weights))).any()
    assert weights[(weights > 20) & (weights < 100)].size == 0


# ===========================================================================
# The augmentation against a recorded run -- cheap, and it guards the whole
# `walk` path that the slow parity gate exercises only end to end.
# ===========================================================================
@pytest.mark.parametrize("pairs,dnnz", [("walk", 75_068),
                                        ("walk_edges", 75_072)])
def test_the_augmentation_reproduces_the_recorded_dnnz(cora, pairs, dnnz):
    """`results/fdlinear/fdl_cora_{walk,walk_edges}_lr1.0_s42.log`, 2026-08-17
    (`run_fdlinear.sh`: 5 walks x 20 steps, window 5, cap 16, seed 42).

    It costs seconds because it stops after `augment_graph`, and it covers
    what a parity run covers only through the embedding: the walk pair
    statistics, `cap_per_node`, the symmetric CSR build, and the far pairs.
    """
    from fodiwalk import Fodiwalk
    A, n = cora
    fw = Fodiwalk(n_dim=64, seed=42, pairs=pairs, weight="min_gap",
                  force="fdlinear", k4=1.0, kr=1.0, walks=5, walk_len=20,
                  window=5)
    D = fw.augment_graph(A)
    assert D.nnz == dnnz
    assert fw.info["capped_pairs"] == 28_239
    assert fw.info["far_pairs"] == 9_295
