"""Throughput benchmark: the SELL-C-sigma bucketed engine on LARGE,
power-law-degree graphs -- the whole reason this package exists.

Generates synthetic Barabasi-Albert graphs directly with
``networkx.barabasi_albert_graph(n, m=3, ...)`` (``m=3`` matches
``dev_docs/fdmap_engine_design_notes.md``'s own reference BA parameter),
skipping ``graph_building``'s kNN/MST machinery entirely -- that stage is
orthogonal to embedding-engine throughput and would make a 500k-node
timing dominated by graph construction, not the kernel this benchmark
exists to measure. Augments with ``graph_augmenting.sparse_hops``
(bounded radius; dense ``hopfill`` is ``O(n^2)`` and infeasible at this
scale -- see ``docs/DESIGN.md``'s "Which augmentation policy to use at
scale"). Runs a real ``SparseFDModel.embed(...)`` for 5 epochs at
``n_dim=128``, reporting real (non-padded) cells/s, steady-state ms/epoch,
padding fraction, and total 5-epoch wall time -- plus a correctness
spot-check at the smallest size against the sibling
``fdge_jax.embedding.shell_force.ShellForce`` engine.

Warmup strategy (why this isn't just "call embed() twice"): JIT compile
time must be excluded from the steady-state number, but
``ForceDirected.embed()`` doesn't expose a lower-level single-step call --
it always re-runs ``augment_graph`` itself as its first step. Since
this package's ``embedding.shell_force.ShellForce`` plan +
compiled-kernel cache is keyed on ``D``
object IDENTITY (not equality), a second, independent call to
``augment_graph`` inside a timed ``embed()`` call would produce a
DIFFERENT ``D`` object and miss the cache entirely, silently re-including
the ENGINE's compile time in what's supposed to be a steady-state
measurement. The fix used here: build ``D`` once ourselves, warm up the
model's own ``_sell`` engine directly against that exact ``D`` object (one
level below ``SparseFDModel``, exactly as its own docstring anticipates),
THEN monkeypatch ``model.augment_graph`` to hand back that SAME cached
``D`` object before starting the timed ``embed()`` call.

That removes the engine's own compile cost from the timed run -- but
measurement (below) still shows epoch 1 inside ``embed()`` taking ~150x
longer than epochs 2-5, even with the engine pre-warmed. The remaining
cost is XLA's first-time dispatch/compile of ``ForceDirected.embed()``'s
OWN bare (non-jitted) array ops at this shape -- ``self.dZ.at[...].set(...)``,
``self.Z + lr * self.dZ``, ``jnp.linalg.norm(dZ, axis=-1).mean()`` -- each
of which is a separate XLA program the first time it runs against a given
shape, regardless of whether the force-law kernel itself was pre-warmed.
There is no way to pre-warm these from outside ``embed()`` (it owns them
internally, no lower-level per-step call is exposed), so per-epoch timing
via a callback is used instead, exactly as anticipated: epoch 1 is
reported separately, and steady-state ms/epoch is the mean of epochs
2..``EPOCHS`` -- the fair number, since it excludes both compile sources
(engine AND loop-plumbing) rather than just one.

Run:
    source /home/h/gnn/fd-graph-embedding/fdmap/.venv/bin/activate
    python -m fdge_jax_sell_c_sigma.validation.bench_embedding_perf   # from repo root (fdmap/)
"""
from __future__ import annotations

import time

import numpy as np
import networkx as nx
import jax
import jax.numpy as jnp

from fdge_jax_sell_c_sigma.core import Callback_Base
from fdge_jax_sell_c_sigma.graph_augmenting import sparse_hops
from fdge_jax_sell_c_sigma.embedding.shell_force import ShellForce
from fdge_jax_sell_c_sigma.models import SparseFDModel

K1, K2, K3, K4 = 0.999, 1.0, 10.0, 0.01
RANDOM_DROP_RATE = 0.5
M = 3                 # BA attachment parameter (dev_docs' own reference value)
N_DIM = 128
EPOCHS = 5
SEED = 0
SIZES = (1_000, 100_000, 200_000, 300_000, 400_000, 500_000)
RADIUS = 1             # radius=1 -> D is essentially the plain adjacency, hop
                       # distance 1 everywhere, shell_coeff = 1/deg(u) -- the
                       # case most directly comparable to dev_docs' own
                       # Mcells/s figures (same m=3 BA graphs, same core
                       # gather/elementwise/reduce shape).


def _make_ba_graph(n: int, m: int = M, seed: int = SEED) -> nx.Graph:
    return nx.barabasi_albert_graph(n, m, seed=seed)


class _EpochTimer(Callback_Base):
    """Records wall-clock seconds per epoch, GPU-synced at epoch end.

    ``on_epoch_end`` fires AFTER ``updateZ`` in ``ForceDirected.embed``'s
    loop (see ``core/force_directed.py``), so blocking on ``model.Z`` here
    waits for that epoch's full pipeline (forces -> dZ -> Z update) to
    actually finish on-device before stopping the clock -- a wall-clock
    timer that didn't block would just measure how fast Python can enqueue
    async dispatches, not real epoch latency.
    """

    def __init__(self):
        self.epoch_seconds = []
        self._t0 = None

    def on_epoch_begin(self, model, **kwargs):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, model, **kwargs):
        jax.block_until_ready(model.Z)
        self.epoch_seconds.append(time.perf_counter() - self._t0)


def bench_one(n: int, m: int = M, radius: int = RADIUS, epochs: int = EPOCHS,
              n_dim: int = N_DIM, seed: int = SEED,
              b_cells: int = 16_384, k_max: int = 256, ladder_base: float = 1.5):
    """Time one (graph size, augmentation) configuration end-to-end.

    Returns a stats dict, or raises if this size is genuinely infeasible
    on this machine (caller decides whether to catch and skip -- see
    ``main``).
    """
    t0 = time.perf_counter()
    G = _make_ba_graph(n, m=m, seed=seed)
    graph_gen_s = time.perf_counter() - t0

    model = SparseFDModel(n_dim=n_dim, verbosity=0, seed=seed,
                           k1=K1, k2=K2, k3=K3, k4=K4,
                           random_drop_rate=RANDOM_DROP_RATE,
                           b_cells=b_cells, k_max=k_max, ladder_base=ladder_base)

    # ---- stage 2 (timed once, separately -- this is genuinely O(n) to
    # O(n * avg_degree^radius) preprocessing, not the hot loop) -----------
    t0 = time.perf_counter()
    D = model.augment_graph(G, radius=radius)     # also sets model.D
    augment_s = time.perf_counter() - t0
    n_nodes = D.shape[0]

    # ---- warm up the JIT against this EXACT D object (see module
    # docstring's "Warmup strategy") -----------------------------------
    key = jax.random.PRNGKey(seed)
    key, zkey, warmkey = jax.random.split(key, 3)
    Z0 = jax.random.normal(zkey, (n_nodes, n_dim), dtype=jnp.float32)

    t0 = time.perf_counter()
    jax.block_until_ready(model._sell(Z0, D, 0, n_nodes, key=warmkey))
    compile_s = time.perf_counter() - t0

    stats = model._sell.stats   # cells, n_virtual, n_split, rungs, pad_frac

    # Freeze augment_graph to hand back the SAME D object (identity, not
    # just equality) so the timed embed() call below doesn't (a) redo the
    # sparse_hops BFS or (b) miss the engine's identity-keyed cache
    # and silently recompile mid-measurement.
    model.augment_graph = lambda G, **kw: D

    timer = _EpochTimer()
    model.attach_callback(timer)

    t0 = time.perf_counter()
    Z = model.embed(G, epochs=epochs, Z=Z0, batch_count=1, verbosity=0, radius=radius)
    jax.block_until_ready(Z)
    total_s = time.perf_counter() - t0

    epoch_s = timer.epoch_seconds
    epoch1_s = epoch_s[0]
    # Steady-state: mean of epochs 2..EPOCHS -- excludes epoch 1's one-time
    # dispatch/compile cost for embed()'s own bare ops (see module
    # docstring's "Warmup strategy" for why this can't be pre-warmed away).
    steady_epochs = epoch_s[1:] if len(epoch_s) > 1 else epoch_s
    ms_per_epoch = (sum(steady_epochs) / len(steady_epochs)) * 1e3
    mcells = (stats["cells"] / (ms_per_epoch * 1e-3) / 1e6) if stats["cells"] else 0.0

    return dict(n=n, n_nodes=n_nodes, nnz=D.nnz, cells=stats["cells"],
                pad_frac=stats["pad_frac"], rungs=stats["rungs"],
                n_split=stats["n_split"], graph_gen_s=graph_gen_s,
                augment_s=augment_s, compile_s=compile_s,
                epoch1_s=epoch1_s, total_s=total_s,
                ms_per_epoch=ms_per_epoch, mcells=mcells)


def correctness_spot_check(n: int = 1_000, m: int = M, radius: int = RADIUS,
                            seed: int = SEED, n_dim: int = N_DIM) -> float:
    """1-epoch bucketed-vs-flat cross-engine check at the smallest size.

    Deliberate cross-package import of ``fdge_jax`` -- legitimate here
    (a validation script, not the engine module itself) for exactly the
    reason ``embedding/test_sell_c_sigma.py``'s parity test does it: this
    is the check that the large-n numbers below are numbers for the SAME
    physics as the existing, already-validated engine, not a different
    (and differently wrong) one that merely happens to run fast.
    """
    from fdge_jax.embedding.shell_force import ShellForce as FlatShellForce

    G = _make_ba_graph(n, m=m, seed=seed)
    D = sparse_hops.augment_graph(G, radius=radius)

    n_rows = D.shape[0]
    rng = np.random.default_rng(seed)
    Z = np.ascontiguousarray(rng.standard_normal((n_rows, n_dim))).astype(np.float32)
    key = jax.random.PRNGKey(seed)

    bucketed = ShellForce(k1=K1, k2=K2, k3=K3, k4=K4, random_drop_rate=0.0)
    flat = FlatShellForce(k1=K1, k2=K2, k3=K3, k4=K4, random_drop_rate=0.0)

    got_bucketed = np.asarray(bucketed(Z, D, 0, n_rows, key=key))
    # fdge_jax (sibling package) still expects its own HopMatrix/dense duck
    # type, so hand it the dense equivalent -- same values, a container both
    # engines accept. See test_sell_c_sigma.py's parity test, same reason.
    got_flat = np.asarray(flat(Z, D.toarray(), 0, n_rows, key=key))
    diff = float(np.abs(got_bucketed - got_flat).max())
    print(f"[correctness spot-check] n={n} radius={radius} nnz={D.nnz}: "
          f"bucketed vs. fdge_jax.embedding.shell_force.ShellForce, "
          f"1 epoch, random_drop_rate=0 -> max abs diff = {diff:.3e}")
    return diff


def main():
    print(f"jax {jax.__version__} | backend: {jax.default_backend()} "
          f"| devices: {jax.devices()}")
    print(f"config: n_dim={N_DIM}  epochs={EPOCHS}  BA m={M}  radius={RADIUS}  "
          f"k1..k4=({K1},{K2},{K3},{K4})  random_drop_rate={RANDOM_DROP_RATE}\n")

    results = []
    for n in SIZES:
        print(f"--- n={n} ---")
        t0 = time.perf_counter()
        try:
            r = bench_one(n)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: report
            # and move on rather than losing the rest of the sweep to one
            # infeasible size (OOM, excessive runtime, etc.) -- the task
            # explicitly wants an honest report either way.
            elapsed = time.perf_counter() - t0
            print(f"n={n:>7}  SKIPPED after {elapsed:6.1f}s: "
                  f"{type(exc).__name__}: {str(exc)[:200]}")
            print(f"  (if this is RESOURCE_EXHAUSTED / OOM: expected on a "
                  f"small-VRAM GPU at n_dim={N_DIM} -- "
                  f"dev_docs/fdmap_engine_design_notes.md Sec 6.2 documents "
                  f"this exact engine family's fp32 ceiling on a 2 GB card "
                  f"as sitting between n=400k and n=700k at d=128; not an "
                  f"algorithm bug, just this machine's VRAM.)")
            continue

        print(f"n={r['n']:>7}  nodes={r['n_nodes']:>7}  nnz={r['nnz']:>10}  "
              f"cells={r['cells']:>10}  rungs={r['rungs']:>2}  "
              f"split_hubs={r['n_split']:>3}  pad={r['pad_frac']*100:5.1f}%  "
              f"graph_gen={r['graph_gen_s']:6.2f}s  augment={r['augment_s']:7.2f}s  "
              f"engine_compile={r['compile_s']:6.2f}s  epoch1={r['epoch1_s']*1e3:8.2f}ms  "
              f"{EPOCHS}-epoch total={r['total_s']:7.3f}s  "
              f"steady ms/epoch={r['ms_per_epoch']:8.3f}  {r['mcells']:8.1f} Mcells/s")
        results.append(r)

    print()
    correctness_spot_check(n=SIZES[0])

    print("\nsummary (steady ms/epoch = mean of epochs 2..N, excluding the "
          "engine's own JIT compile AND embed()'s first-epoch loop-plumbing "
          "dispatch cost -- see module docstring):")
    header = (f"{'n':>8} {'nodes':>8} {'cells':>10} {'pad%':>6} "
              f"{'augment(s)':>11} {'compile(s)':>11} {'epoch1(ms)':>11} "
              f"{'5ep total(s)':>13} {'steady ms/ep':>13} {'Mcells/s':>10}")
    print(header)
    for r in results:
        print(f"{r['n']:>8} {r['n_nodes']:>8} {r['cells']:>10} "
              f"{r['pad_frac']*100:>5.1f}% {r['augment_s']:>11.2f} "
              f"{r['compile_s']:>11.2f} {r['epoch1_s']*1e3:>11.2f} "
              f"{r['total_s']:>13.3f} {r['ms_per_epoch']:>13.3f} {r['mcells']:>10.1f}")

    if not results:
        print("\nNo size completed -- see per-size SKIPPED lines above for why.")


if __name__ == "__main__":
    main()
