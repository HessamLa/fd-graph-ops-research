"""fodiwalk.model -- `class Fodiwalk`, the user-facing model. THE WIRING ONLY.

`Fodiwalk(Fodiwalk_base)` implements the whole contract: `make_graph`
(stage 1, a STUB), `graph_walk`/`augment_graph` (stage 2 whole: `D`, the
planes, the degrees, the force params -- `augment_graph`'s own docstring
says why all four are stage-2 data), `forces` (stage 3, the law) and
`embed` (`augment_graph` once, then the engine's loop). No `fit`
(2026-08-21): `make_graph` is a stub, and `fit` promised a stage that was
never real.

THE ONE RULE THAT THIS CLASS EXISTS TO KEEP. The plane list comes from
`core.forces.FORCE_PLANES`, through `augment_graph.build_planes`, and never
an `if` chain: a missing plane RAISES and never falls back to another
law's planes, the fallback that once read a coefficient plane as `h` and
went to NaN with no error. `core.plan_contract` asserts it before `make_plan`.

ONE GENERATOR. `self.rng` of the base class flows walks -> far pairs ->
link prediction -> hop sample, in that order. A second generator, a moved
call or one more draw changes every recorded number.

`embed()` CALLS `augment_graph()` ITSELF, one time. A second call embeds a
`D` that is not the `D` the log reports, and `D.nnz` still agrees, because
the pair counts are stable. Time the stage with a `train_begin` callback.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp

from forcedirected import ForceDirected

from .base import Fodiwalk_base
from .config import Config
from .core import plan_contract
from .core.forces import force_fn
from .misc.drop import drop_steady_rate
from .augment_graph import (policies, AugmentSpec, ForceSpec, build_planes,
                            force_params, resolve_degrees)
from .embed import PlanSpec, build_plans


class Fodiwalk(Fodiwalk_base):
    """The walk-augmented force-directed model. The THREE stages, wired."""

    VER_MAJ = "01"
    VER_MIN = "00"
    DESCRIPTION = "Walk-augmented force-directed embedding (JAX, SELL-C-sigma)"

    def __init__(self, n_dim: int | None = None, lr: float = 1.0,
                 seed: int | None = None, verbosity: int = 0,
                 optim: str = "plain", lr_decay: str = "const",
                 eta: float = 0.3, sgd_frac: float = 0.5,
                 sqn_memory: int = 3, **cfg):
        super().__init__(n_dim=n_dim, lr=lr, verbosity=verbosity, seed=seed)
        unknown = set(cfg) - {f.name for f in dataclasses.fields(Config)}
        if unknown:
            raise TypeError(
                f"Unknown Fodiwalk option(s) {sorted(unknown)}. The names "
                f"are the flags of bench_fdwalk.py with underscores; see "
                f"`fodiwalk.Config`.")
        self.cfg = Config(**cfg)
        self.seed = seed        # `self.rng` comes from the base class

        # The three stage specs, narrowed from `self.cfg` ONE time. They
        # are frozen, thus a stage cannot edit the configuration of another
        # one and `self.cfg` stays the single source.
        self.aug_spec = AugmentSpec.from_config(self.cfg)
        self.force_spec = ForceSpec.from_config(self.cfg)
        self.plan_spec = PlanSpec.from_config(self.cfg)
        # the law, resolved once: `fuse_planes` picks the fused form, thus
        # the plane list follows from ONE name.
        self.law = self.force_spec.law
        self.force_fn = force_fn(self.law)

        rule_kw = {"velocity": {"eta": eta},
                   "sgd": {"frac": sgd_frac, "seed": seed or 0},
                   "sqn": {"memory": sqn_memory}}.get(optim, {})
        self.set_rule(optim, lr, **rule_kw)
        if lr_decay == "linear":
            # lr_t = lr * (1 - epoch/epochs): at the last epoch this is
            # lr/epochs and not 0, thus the final step still moves.
            self.lr_schedule = lambda lr, ep, eps: lr * (1.0 - ep / max(1, eps))
        elif lr_decay != "const":
            raise ValueError(f"Unknown lr_decay {lr_decay!r}")

        # filled by augment_graph / set_D
        self.stats = None          # the walk statistics
        self.info = {}             # the numbers a log prints
        self.freq = None           # the `freq` plane data, or None
        self.deg_from_A = None     # an explicit degree array, or None
        self._D_given = None       # a matrix handed over by set_D
        self.plans, self.steps = [], []
        self.plan_stats = None

    # ---------------------------------------------------------- stage 1
    def make_graph(self, data, **kwargs):
        """Stage 1. A STUB: it returns the graph it is given.

        The graphs arrive as edge lists on disk and `make_graph.load` reads
        them. The stage stays in the API because a later project builds the
        graph from raw data (kNN, MST) and changes this method only.
        """
        return data

    # ---------------------------------------------------------- the walk
    def graph_walk(self, A, n: int | None = None, rng=None, **kwargs):
        """The walks alone, and the statistics of the pairs they give.

        A thin forward to `policies.graph_walk`. The default generator is
        `self.rng`, thus a call ADVANCES the one generator of the run.
        """
        n = A.shape[0] if n is None else n
        rng = self.rng if rng is None else rng
        return policies.graph_walk(A, n, self.aug_spec, rng)

    # ---------------------------------------------------------- stage 2
    def set_D(self, D, stats=None, freq=None, degrees=None):
        """Hand over a `D` that is already augmented, and skip the walks.

        `augment_graph` then builds the planes and the plan from it alone --
        the path of a CACHED augmentation: com_youtube's needs ten minutes.
        `freq` aligns to `D.indices`; a law that reads it and gets none RAISES.
        """
        self._D_given = D
        self.stats = stats
        self.freq = None if freq is None else np.asarray(freq)
        self.deg_from_A = None if degrees is None else np.asarray(degrees)
        return self

    def augment_graph(self, G, **kwargs):
        """Stage 2, whole: `D`, the planes, the degrees, the force params
        -- all stage-2 DATA (`augment_graph` docstring). The plan below
        only CONSUMES them. `G` is un-augmented, unless `set_D` skipped
        the walks.
        """
        if self._D_given is not None:
            D = self._D_given
        else:
            aug = policies.build(G, G.shape[0], self.aug_spec, self.rng)
            self.freq, self.stats, self.info = aug.freq, aug.stats, aug.info
            D = aug.D
        self.D = D

        planes = self._build_planes(D)
        degrees = self._build_degrees(D, G)
        self.params = force_params(self.force_spec)

        if self.plan_spec.check_planes:
            plan_contract.check(self.law, planes, D)
            plan_contract.check_degrees(degrees, D)

        ps = build_plans(D, planes, degrees, self.plan_spec, self.force_fn)
        self.plans, self.steps = ps.plans, ps.steps
        self.inv_deg_ext, self.chunk_rows = ps.inv_deg_ext, ps.chunk_rows
        self.resident, self.plan_stats = ps.resident, ps.stats
        return D

    def _build_planes(self, D):
        """The planes of `self.law`. `golden.py` and `check_api.py` call it."""
        return build_planes(self.law, D, self.freq, self.cfg.pairs,
                            self.cfg.policy)

    def _build_degrees(self, D, G):
        """The divisor of the row sum. `golden.py` calls it directly."""
        return resolve_degrees(D, G, self.force_spec,
                               explicit=self.deg_from_A)

    # ---------------------------------------------------------- stage 3
    def forces(self, Z, D, row_start: int, row_end: int, key=None, **kwargs):
        """The dZ rows of `[row_start, row_end)`.

        A chunk is a ROW RANGE, thus the first row picks the plan. The
        kernel runs over that chunk and the result is sliced: the plan
        groups rows by WIDTH and not by node id.
        """
        if key is None:
            self.key, key = jax.random.split(self.key)
        i = min(row_start // self.chunk_rows, len(self.plans) - 1)
        plan = self.plans[i]
        if not self.resident:
            plan = jax.tree_util.tree_map(jax.device_put, plan)
        full = self.steps[i](jnp.asarray(Z, dtype=jnp.float32),
                             plan, self.inv_deg_ext, self.params)
        out = drop_steady_rate(full[row_start:row_end], key,
                               self.cfg.random_drop_rate,
                               strategy=self.cfg.drop_strategy)
        del full, plan
        return out

    def embed(self, G, epochs: int = 1000, lr: float | None = None, Z=None,
              batch_count: int = 1, epsilon: float | None = None, **kwargs):
        """`augment_graph(G)` ONCE (trap 12), then the engine's loop on `D`."""
        self.G = G
        D = self.augment_graph(G, **kwargs)
        return ForceDirected.embed(
            self, D, epochs=epochs, lr=lr, Z=Z, batch_count=batch_count,
            epsilon=epsilon, **kwargs)
