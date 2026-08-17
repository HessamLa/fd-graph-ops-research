"""Correctness tests for embedding/ (force law + SELL-C-sigma layout engine).

Independent of the bucketed kernel's own reasoning -- checked against a
naive, unvectorized Python/NumPy oracle, same methodology
``fdge_jax/embedding/test_shell_force.py`` uses (and, one level further
back, ``fdge2/embedding/test_shell_force.py``): don't trust your own port,
compare to a dead-simple reference. ``naive_oracle`` below is the same
formula, copied unchanged.

PLUS a direct cross-engine parity check against the sibling
``fdge_jax.embedding.shell_force.ShellForce`` (the flat-edge-list +
segment_sum engine) -- this is the test that actually proves the bucketed
layout reproduces the exact same physics as the existing engine, just with
a different (faster, at scale) execution shape. Parity tests are exactly
the place a cross-package import of ``fdge_jax`` is legitimate (see
``sell_c_sigma.py``'s module docstring for why the *engine* module itself
must never do this).

PLUS three tests that exist specifically to protect the three-file split
(``shell_force.py`` = physics, ``sell_c_sigma.py`` = layout, ``drop.py`` =
regularizer): a force-law swap through the untouched engine, a no-recompile
proof for the traced ``params``, and an O(n^2) regression guard on the shell
count table.

NOTE on imports: ``embedding/`` itself must never import
``graph_augmenting/`` or the sibling ``fdge_jax`` package (see
``sell_c_sigma.py``'s module docstring / ``docs/DESIGN.md``) -- but this
test file is not part of that package, so the import boundary doesn't
apply to it. It uses this package's own
``graph_augmenting.hopfill``/``graph_augmenting.sparse_hops`` to build
fixtures, and (only in the parity test) the sibling
``fdge_jax.embedding.shell_force.ShellForce`` for cross-checking.

Run:
    source /home/h/gnn/fd-graph-embedding/fdmap/.venv/bin/activate
    python -m fdge_jax_sell_c_sigma.embedding.test_sell_c_sigma   # from repo root (fdmap/)
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import jax
import jax.numpy as jnp

from fdge_jax_sell_c_sigma.core import ForceDirected
from fdge_jax_sell_c_sigma.graph_augmenting.hopfill import augment_graph
from fdge_jax_sell_c_sigma.graph_augmenting import sparse_hops
from fdge_jax_sell_c_sigma.embedding.sell_c_sigma import (
    PlanCache, make_plan, build_ladder, _to_csr,
)
from fdge_jax_sell_c_sigma.embedding.shell_force import (
    ShellForce, shell_force, shell_counts, shell_coeff_data, degrees_from_D,
)


# ---------------------------------------------------------------------------
# Naive dense oracle (same formula as shell_force's naive_oracle -- the
# physics is unchanged across engines, only the execution shape differs)
# ---------------------------------------------------------------------------
def naive_oracle(Z, hops, degrees, k1, k2, k3, k4):
    """Dense, unvectorized reference of the shell force law (drop disabled)."""
    n, d = Z.shape
    out = np.zeros((n, d))
    for u in range(n):
        vals, counts = np.unique(hops[u], return_counts=True)
        size = dict(zip(vals.tolist(), counts.tolist()))
        acc = np.zeros(d)
        for v in range(n):
            h = hops[u, v]
            if h <= 0:
                continue
            diff = Z[v] - Z[u]
            x = np.linalg.norm(diff)
            if x == 0:
                continue
            Fa = k1 * (1.0 / size[h]) * x * np.exp(-k2 * (h - 1))
            Fr = -k3 * h * np.exp(-k4 * x)
            acc += (Fa + Fr) / x * diff
        out[u] = acc / degrees[u] if degrees[u] > 0 else 0.0
    return out


def _two_component_graph():
    # karate club + a disjoint path -> disconnected pairs -> h = n sentinel
    return nx.disjoint_union(nx.karate_club_graph(), nx.path_graph(6))


# ---------------------------------------------------------------------------
# 1. kernel correctness vs. naive oracle (small k_max/b_cells so the tiny
#    test graph actually exercises hub-splitting and multiple rungs)
# ---------------------------------------------------------------------------
def _check_matches_oracle(is_sparse):
    key = jax.random.PRNGKey(0)
    rng = np.random.default_rng(0)
    G = _two_component_graph()
    n = G.number_of_nodes()
    d = 3
    Z = np.ascontiguousarray(rng.standard_normal((n, d))).astype(np.float32)

    D = augment_graph(G, is_sparse=is_sparse)          # real D, csr_matrix or dense
    k1, k2, k3, k4 = 0.999, 1.0, float(n), 0.01

    # Small b_cells/k_max relative to n so hub-splitting and several rungs
    # actually get exercised by this tiny two-component fixture, not just
    # one giant rung.
    sf = ShellForce(k1=k1, k2=k2, k3=k3, k4=k4, random_drop_rate=0.0,
                    b_cells=64, k_max=8, ladder_base=1.5)
    got = np.asarray(sf(Z, D, 0, n, degrees=None, key=key))  # degrees derived from D

    Dd = D.toarray() if hasattr(D, "toarray") else np.asarray(D)
    hops = Dd.astype(np.int64)                          # includes sentinel n
    degrees = (Dd == 1).sum(axis=1)
    want = naive_oracle(Z.astype(np.float64), hops, degrees, k1, k2, k3, k4)

    # RELATIVE error, not absolute: this fixture uses the legacy k3=n
    # convention on a two-component graph where n itself is the
    # disconnected-pair sentinel hop value, so entries blow up to O(1e4)
    # magnitude (unlike shell_force's own oracle test, which can afford a
    # tight ABSOLUTE tolerance only because it runs in float64 throughout
    # -- fdge_jax enables jax_enable_x64 at import time. This package is
    # float32 by design (docs/DESIGN.md), so the right bar is "matches to
    # float32 ULP scale," i.e. a small RELATIVE error, not a small
    # absolute one that happens to be tiny compared to O(1e4) values.
    rel = np.abs(got - want) / (np.abs(want) + 1e-6)
    err = float(rel.max())
    assert err < 1e-4, (
        f"kernel mismatch (is_sparse={is_sparse}), max relative err={err}")
    return err


def test_kernel_correctness_dense():
    err = _check_matches_oracle(is_sparse=False)
    print(f"[ok] dense D: bucketed kernel matches naive oracle (max rel err {err:.2e}, "
          f"two components, sentinel h=n exercised)")


def test_kernel_correctness_sparse():
    err = _check_matches_oracle(is_sparse=True)
    print(f"[ok] csr_matrix D: bucketed kernel matches naive oracle (max rel err {err:.2e}, "
          f"CSR-native path)")


# ---------------------------------------------------------------------------
# 2. per-D cache avoids recomputation
# ---------------------------------------------------------------------------
def test_cache_avoids_recomputation():
    key = jax.random.PRNGKey(1)
    rng = np.random.default_rng(1)
    G = _two_component_graph()
    n = G.number_of_nodes()
    Z = np.ascontiguousarray(rng.standard_normal((n, 3))).astype(np.float32)
    D = augment_graph(G, is_sparse=True)

    sf = ShellForce(random_drop_rate=0.0, b_cells=64, k_max=8)
    # many calls on the SAME D (several epochs x several batches)
    for _ in range(20):
        key, k1_, k2_ = jax.random.split(key, 3)
        sf(Z, D, 0, n // 2, key=k1_)
        sf(Z, D, n // 2, n, key=k2_)
    assert sf.n_derivations == 1, (
        f"cache recomputed {sf.n_derivations}x on the same D (expected 1)")

    # a genuinely different D object must trigger exactly one more derivation
    D2 = augment_graph(G, is_sparse=True)
    key, sub = jax.random.split(key)
    sf(Z, D2, 0, n, key=sub)
    assert sf.n_derivations == 2, (
        f"new D did not re-derive (n_derivations={sf.n_derivations})")
    print(f"[ok] single-slot identity cache: {40} calls on one D -> 1 derivation; "
          f"new D -> +1")


# ---------------------------------------------------------------------------
# 3. degrees derived from D == degrees from nx.to_scipy_sparse_array
# ---------------------------------------------------------------------------
def test_degrees_derivation():
    G = _two_component_graph()
    nodes = list(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, weight=None, format="csr")
    degrees_csr = np.diff(A.indptr)
    D = augment_graph(G, is_sparse=False)
    degrees_from_D = (np.asarray(D) == 1).sum(axis=1)
    assert np.array_equal(degrees_csr, degrees_from_D), "degree derivation mismatch"
    print("[ok] degrees derived from D (hop-1 counts) == nx.to_scipy_sparse_array degrees")


# ---------------------------------------------------------------------------
# 4. plan-builder stats: real cells == D.nnz, padding fraction is sane
# ---------------------------------------------------------------------------
def test_plan_stats_sane():
    G = nx.barabasi_albert_graph(300, 3, seed=1)
    D = _to_csr(sparse_hops.augment_graph(G, is_sparse=True, radius=2))
    planes = (shell_coeff_data(D), D.data)
    plan, inv_deg_ext, stats = make_plan(
        D, planes, degrees=degrees_from_D(D), b_cells=512, k_max=32)

    assert stats["cells"] == D.nnz, (
        f"plan lost or duplicated real cells: stats['cells']={stats['cells']} "
        f"!= D.nnz={D.nnz}")
    assert 0.0 <= stats["pad_frac"] < 1.0
    assert stats["rungs"] == len(plan)
    assert inv_deg_ext.shape == (D.shape[0] + 1,)
    assert inv_deg_ext[-1] == 0.0, "trailing pad slot of inv_deg_ext must be 0"
    # one (nb, R, k) tile per plane, plus rows + nbrs -- the plane count is
    # the whole interface between the layout and the force law, so pin it.
    for rung in plan:
        assert len(rung) == 2 + len(planes), (
            f"rung must be (rows, nbrs, *plane_tiles), got {len(rung)} arrays")
    print(f"[ok] plan stats sane: cells={stats['cells']} (== D.nnz), "
          f"pad_frac={stats['pad_frac']:.3f}, rungs={stats['rungs']}, "
          f"{len(planes)} coefficient planes")


def test_build_ladder_monotone_and_capped():
    ladder = build_ladder(k_max=256, base=1.5)
    assert ladder[0] == 1 and ladder[-1] == 256
    assert np.all(np.diff(ladder) > 0), "ladder must be strictly increasing"
    print(f"[ok] build_ladder(256, 1.5): {len(ladder)} rungs, "
          f"1 .. 256")


# ---------------------------------------------------------------------------
# 5. cross-engine parity: bucketed engine vs. the sibling fdge_jax's
#    flat-edge-list segment_sum ShellForce, same D, same Z, same k1..k4
# ---------------------------------------------------------------------------
def test_cross_engine_parity_vs_shell_force():
    # Deliberate cross-package import -- legitimate only in a parity test,
    # see this file's and sell_c_sigma.py's module docstrings.
    from fdge_jax.embedding.shell_force import ShellForce as FlatShellForce

    G = nx.barabasi_albert_graph(200, 3, seed=7)
    D = sparse_hops.augment_graph(G, is_sparse=True, radius=2)
    n = D.shape[0]

    rng = np.random.default_rng(3)
    Z = np.ascontiguousarray(rng.standard_normal((n, 8))).astype(np.float32)

    k1, k2, k3, k4 = 0.999, 1.0, 10.0, 0.01

    bucketed = ShellForce(k1=k1, k2=k2, k3=k3, k4=k4, random_drop_rate=0.0)
    flat = FlatShellForce(k1=k1, k2=k2, k3=k3, k4=k4, random_drop_rate=0.0)

    key = jax.random.PRNGKey(42)
    got_bucketed = np.asarray(bucketed(Z, D, 0, n, key=key))
    # fdge_jax (sibling package, not touched by this package's move to plain
    # scipy.sparse.csr_matrix) still expects its own HopMatrix/dense duck
    # type -- hand it the dense equivalent instead of D itself. Same values,
    # so it's still a same-D parity check, just via a container both engines
    # independently accept.
    got_flat = np.asarray(flat(Z, D.toarray(), 0, n, key=key))

    err = np.abs(got_bucketed - got_flat).max()
    assert err < 1e-4, (
        f"bucketed engine diverges from flat segment_sum engine: max abs err={err:.3e}")
    print(f"[ok] cross-engine parity vs. fdge_jax.embedding.shell_force.ShellForce: "
          f"max abs diff {err:.2e} (n={n}, nnz={D.nnz})")


# ---------------------------------------------------------------------------
# 6. momentum updateGradient override against the REAL forces()
# ---------------------------------------------------------------------------
class _RealSellModel(ForceDirected):
    """Minimal real model wiring the real (local) augment + real bucketed forces.

    Stands in for the real ``fdge_jax_sell_c_sigma/models.py`` composition
    root -- exercised here so the momentum seam runs against the real
    kernel, not a dummy.
    """

    def __init__(self, *a, random_drop_rate=0.0, **kw):
        super().__init__(*a, **kw)
        self._sell = ShellForce(random_drop_rate=random_drop_rate)
        self._degrees = None

    def augment_graph(self, G, is_sparse: bool = True, **kwargs):
        D = augment_graph(G, is_sparse=is_sparse)
        A = nx.to_scipy_sparse_array(G, nodelist=list(G.nodes()), weight=None, format="csr")
        self._degrees = np.diff(A.indptr)
        return D

    def forces(self, Z, D, row_start, row_end, **kwargs):
        return self._sell(Z, D, row_start, row_end,
                           degrees=self._degrees, **kwargs)


class _MomentumSellModel(_RealSellModel):
    """updateGradient override implementing velocity (per core's documented pattern)."""

    def updateGradient(self, Z, D, **kwargs):
        import jax.numpy as jnp
        if self.V is None:
            self.V = jnp.zeros_like(Z)
        rs, re = kwargs["row_start"], kwargs["row_end"]
        self.V = self.V.at[rs:re].set(
            self.beta * self.V[rs:re] + self.forces(Z, D, **kwargs))
        return self.V[rs:re]


def test_passthrough_forces_runs():
    G = nx.karate_club_graph()
    m = _RealSellModel(n_dim=2, verbosity=0, seed=7)
    Z = m.embed(G, epochs=8, lr=0.5)
    assert np.isfinite(np.asarray(Z)).all()
    # cache held across all epochs/batches -> exactly one derivation
    assert m._sell.n_derivations == 1, m._sell.n_derivations
    print(f"[ok] real passthrough forces: 8 epochs finite, "
          f"1 derivation across the whole embed()")


def test_momentum_override_runs():
    G = nx.karate_club_graph()
    m = _MomentumSellModel(n_dim=2, beta=0.7, verbosity=0, seed=3)
    Z = m.embed(G, epochs=8, lr=0.3)          # batch_count=1 (whole-array velocity)
    assert np.isfinite(np.asarray(Z)).all()
    print("[ok] momentum updateGradient override (beta=0.7) runs finite vs real forces()")


# ---------------------------------------------------------------------------
# 7. key=None fallback path (non-reproducible, but must run + be finite)
# ---------------------------------------------------------------------------
def test_fallback_key_when_unspecified():
    rng = np.random.default_rng(3)
    G = _two_component_graph()
    n = G.number_of_nodes()
    Z = np.ascontiguousarray(rng.standard_normal((n, 2))).astype(np.float32)
    D = augment_graph(G, is_sparse=True)

    sf = ShellForce(random_drop_rate=0.5)
    out1 = np.asarray(sf(Z, D, 0, n))   # key=None -> fallback
    out2 = np.asarray(sf(Z, D, 0, n))   # a second fallback call -> different drop mask
    assert np.isfinite(out1).all() and np.isfinite(out2).all()
    assert not np.array_equal(out1, out2), (
        "fallback key did not advance between calls (expected different drop masks)")
    print("[ok] key=None fallback path runs and advances across calls")


# ---------------------------------------------------------------------------
# 8. THE SEAM: swap the force law without touching sell_c_sigma.py
# ---------------------------------------------------------------------------
def linear_attraction(x, planes, params):
    """A deliberately different force law, defined entirely in this test file.

    Pure shell-averaged linear attraction: no repulsion term, no exp, and it
    ignores the ``h`` plane completely. If this runs through the unmodified
    engine, then the engine really is force-law agnostic -- which is the
    entire point of splitting ``shell_force.py`` out of ``sell_c_sigma.py``.
    """
    shell_coeff, _h = planes
    return params["k1"] * shell_coeff * x


def test_force_law_is_swappable():
    """A researcher can change the physics by writing one function.

    Proves three things at once:
      * ``PlanCache`` takes an arbitrary ``force_fn`` -- no edit to
        ``sell_c_sigma.py``, no subclass of anything in it;
      * a force law may ignore planes and params it doesn't need;
      * the alternative law produces genuinely DIFFERENT output (so the
        swap took effect) that is still finite (so the padding contract and
        the layout survived the swap).
    """
    rng = np.random.default_rng(5)
    G = _two_component_graph()
    n = G.number_of_nodes()
    Z = jnp.asarray(rng.standard_normal((n, 3)), dtype=jnp.float32)
    D = _to_csr(augment_graph(G, is_sparse=True))

    planes = (shell_coeff_data(D), D.data)
    degrees = degrees_from_D(D)
    params = dict(k1=0.999, k2=1.0, k3=10.0, k4=0.01, h_shift=1.0)

    def run(force_fn):
        # Same D, same planes, same params, same layout tunables -- the ONLY
        # thing that differs between the two runs is force_fn.
        cache = PlanCache(force_fn, b_cells=64, k_max=8)
        der = cache.derive(D, lambda: (D, planes, degrees))
        return np.asarray(der["step"](Z, der["plan"], der["inv_deg_ext"], params))

    stock = run(shell_force)
    swapped = run(linear_attraction)

    assert np.isfinite(swapped).all(), "swapped force law produced non-finite dZ"
    assert swapped.shape == stock.shape == (n, 3)
    assert not np.allclose(stock, swapped), (
        "swapped force law gave the same answer as the stock one -- the seam "
        "is not actually wired to force_fn")
    # Pure attraction toward stored neighbours: strictly no repulsion, so
    # this is not merely "different numbers", it is the different physics.
    assert np.abs(swapped).max() < np.abs(stock).max()
    print(f"[ok] force law swapped via force_fn alone (sell_c_sigma.py untouched): "
          f"stock |dZ|max={np.abs(stock).max():.3e} vs "
          f"linear-attraction |dZ|max={np.abs(swapped).max():.3e}")


# ---------------------------------------------------------------------------
# 9. traced params: a hyperparameter sweep must NOT recompile
# ---------------------------------------------------------------------------
def test_params_are_traced_not_static():
    """``_cache_size()`` must stay at 1 across many distinct ``k1`` values.

    ``n`` and ``force_fn`` are closure constants (static); ``params`` is a
    plain dict pytree of traced scalars. If someone were to make k1..k4
    static -- e.g. by threading them through ``functools.partial`` -- this
    counter would climb by one per distinct value, and a hyperparameter
    sweep against one graph would pay a full XLA compile per point.
    """
    rng = np.random.default_rng(6)
    G = _two_component_graph()
    n = G.number_of_nodes()
    Z = jnp.asarray(rng.standard_normal((n, 3)), dtype=jnp.float32)
    D = _to_csr(augment_graph(G, is_sparse=True))

    cache = PlanCache(shell_force, b_cells=64, k_max=8)
    der = cache.derive(D, lambda: (D, (shell_coeff_data(D), D.data), degrees_from_D(D)))
    step = der["step"]

    sizes = []
    for k1 in (0.999, 0.5, 1.5, 2.25, 42.0):
        params = dict(k1=k1, k2=1.0, k3=10.0, k4=0.01, h_shift=1.0)
        jax.block_until_ready(step(Z, der["plan"], der["inv_deg_ext"], params))
        sizes.append(step._cache_size())

    assert sizes == [1] * len(sizes), (
        f"traced params recompiled: _cache_size() went {sizes} across 5 distinct k1")
    # And the plan itself was never rebuilt either.
    assert cache.n_derivations == 1
    print(f"[ok] 5 distinct k1 values -> _cache_size() {sizes}, "
          f"n_derivations={cache.n_derivations} (no recompile, no re-plan)")


# ---------------------------------------------------------------------------
# 10. shell_counts must not allocate O(n^2) on a DISCONNECTED graph
# ---------------------------------------------------------------------------
def test_shell_counts_not_quadratic_on_disconnected_graph():
    """Regression guard for a real O(n^2) memory bug.

    Dense ``hopfill`` stamps an ``unreachable = n`` sentinel for every
    disconnected pair, so the old ``(n, D.data.max() + 1)`` shell-count
    table became ``(n, n + 1)`` int64 == ``8 n^2`` bytes on any graph with
    more than one component -- 32 MB at n=2000, ~20 GB at n=50k. Connected
    graphs bound ``max()`` by the diameter, which is why no existing test
    saw it.

    Two disjoint cliques store exactly two distinct hop values (1 within a
    component, the sentinel across), so the compacted table must be (n, 2)
    at BOTH sizes below -- constant width, linear total bytes. The old code
    would give (n, n+1) here.
    """
    widths, sizes = [], (200, 600)
    for n in sizes:
        G = nx.disjoint_union(nx.complete_graph(n // 2), nx.complete_graph(n // 2))
        D = _to_csr(augment_graph(G, is_sparse=True))
        hop_values, counts = shell_counts(D)

        assert counts.shape == (n, hop_values.size)
        assert set(np.unique(D.data).tolist()) == set(hop_values.tolist())
        # Linear, not quadratic: <= 8 bytes * n * (a small constant).
        assert counts.nbytes <= 8 * n * 4, (
            f"shell_counts allocated {counts.nbytes} bytes at n={n} "
            f"(shape {counts.shape}) -- the O(n^2) table is back")
        widths.append(counts.shape[1])

    assert widths[0] == widths[1] == 2, (
        f"table width must not grow with n: {dict(zip(sizes, widths))}")
    print(f"[ok] shell_counts stays (n, 2) on a disconnected graph at "
          f"n={sizes[0]} and n={sizes[1]} (old code: (n, n+1) = 8n^2 bytes)")


if __name__ == "__main__":
    test_kernel_correctness_dense()
    test_kernel_correctness_sparse()
    test_cache_avoids_recomputation()
    test_degrees_derivation()
    test_plan_stats_sane()
    test_build_ladder_monotone_and_capped()
    test_cross_engine_parity_vs_shell_force()
    test_passthrough_forces_runs()
    test_momentum_override_runs()
    test_fallback_key_when_unspecified()
    test_force_law_is_swappable()
    test_params_are_traced_not_static()
    test_shell_counts_not_quadratic_on_disconnected_graph()
    print("\nAll embedding tests passed.")
