#!/bin/env python3
"""fodiwalk.fodiwalk -- `class Fodiwalk`, the user-facing model.

`Fodiwalk` is `core.ForceDirected` with the three stages of the fdwalk
campaign bound to it:

    make_graph     stage 1. A STUB in this project: it returns the graph it
                   is given. The graphs arrive as edge lists on disk, and
                   `make_graph.datasets.load` is the reader.
    graph_walk     the random walks, and the statistics of the pairs they
                   give. `(key, mn, cnt, ...)`.
    augment_graph  stage 2. The statistics into the matrix `D`, then the
                   PLANES and the batch plan of the force law.
    embed          stage 3. Inherited from `core.ForceDirected` unchanged.
    fit            not implemented in this project. It raises.

THE ONE RULE THAT THIS CLASS EXISTS TO KEEP. The plane list comes from
`core.forces.FORCE_PLANES`, and never from an `if` chain:

    for name in planes_of(self.cfg.force):
        planes.append(self._plane(name, D))

and `_plane` RAISES when the augmentation did not build what the law reads.
The 896-line script that this class replaces chose its planes with a chain
of `if`s and a fallback, thus a missing `freq` gave `fdlinear` the planes of
another law, the law read a coefficient plane as `h`, and the run went to
NaN under an `fdlinear` label with no error. `core/plan_contract.py`
asserts the result of every build, at the seam, before `make_plan` sees it.

EVERY POLICY HERE IS WALK-BASED, and that is the scope of the package
(2026-08-20). `h` is a WALK GAP -- an upper bound of the hop distance,
which a walk of `t` steps proves -- and never a measured distance.
`pairs = walk | walk_edges | nbr_walk`.

The exact-distance policies `ball` and `sampled` were removed. They needed
a shortest-path search or `k` sparse products, and a walk does not lose to
them: on Cora `nbr_walk` exceeds the measured 2-hop ball of
`fodined/modular.py` on every score. See `dev-docs/CATALOG.md` section 18.

Usage::

    from fodiwalk import Fodiwalk
    from fodiwalk.make_graph import load

    A, n = load("cora")
    fw = Fodiwalk(n_dim=64, seed=42, optim="plain", lr=1.0,
                  pairs="nbr_walk", weight="min_gap", force="fdlinear")
    fw.embed(A, epochs=200)
    Z = fw.get_embeddings()
"""
from __future__ import annotations

import dataclasses
import functools
import time

import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp

from .core.force_directed import ForceDirected
from .core.forces import planes_of, force_fn, degrees_from_D, fuse
from .core.sell_c_sigma import make_plan, _step
from .core import plan_contract
from .misc.drop import drop_steady_rate
from .augment_graph import pairs as PR
from .augment_graph import walks as W
from .augment_graph import weights as WT
from .augment_graph import buckets as BK
from .augment_graph import landmarks as LM
from .augment_graph.far_pairs import degree_table, sample_far_pairs


@dataclasses.dataclass
class Config:
    """Every knob of the augmentation and the physics. One name for one flag.

    The names are the flags of `experiments/fdwalk/bench_fdwalk.py` with
    the dashes turned into underscores, thus a driver script maps one to
    one and a recorded command line stays readable.
    """

    # -- the pairs -----------------------------------------------------
    pairs: str = "walk"            # walk | walk_edges | nbr_walk
    edge_rule: str = "both"        # both | low_deg
    policy: str = "cap"            # cap | buckets
    walks: int = 10                # walks from each node
    walk_len: int = 20
    window: int = 5
    cap: int = 16                  # pairs kept for each node
    row_cap: int = 0               # nbr_walk: the m best of EACH ROW
    p: float = 1.0                 # node2vec return parameter
    q: float = 1.0                 # node2vec in-out parameter
    prune_max: int = 4_000_000
    prune_factor: int = 4

    # -- the weight ----------------------------------------------------
    weight: str = "min_gap"        # flat | min_gap | mean_gap | pmi
    freq_mode: str = "pair"        # pair | node

    # -- the long-range term -------------------------------------------
    far: int = 0                   # 0 = n*log10(n) for the cap policy
    far_weight: float = 100.0
    far_bias: float = 0.0          # deg^alpha draw; 0.75 = node2vec
    far_with_buckets: bool = False
    bucket_total: int = 0          # 0 = n*log10(n)
    landmarks: int = 0             # 0 = one constant for every far pair
    far_max: int = 32
    far_scale: float = 1.0

    # -- the physics ---------------------------------------------------
    force: str = "fdlinear"        # a key of core.forces.FORCE_PLANES
    fuse_planes: bool = False      # fdlinear -> fdlinear_fused
    no_deg_norm: bool = False
    deg_source: str = "auto"       # auto | D | A
    k1: float = 0.999
    k4: float = 0.01
    kr: float = 1.0
    fdlinear_sign: float = -1.0
    random_drop_rate: float = 0.5
    drop_strategy: str = "random_rows"

    # -- the layout ----------------------------------------------------
    b_cells: int = 16_384
    k_max: int = 256
    ladder_base: float = 1.5
    chunks: int = 1
    chunk_host: bool = False

    # -- the asserter --------------------------------------------------
    check_planes: bool = True      # I1, I2, I4, I5 at the seam
    check_padding: bool = True     # I3, one pass over the built tiles


class Fodiwalk(ForceDirected):
    """The walk-augmented force-directed model."""

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

        # the law, resolved once. `fuse_planes` picks the fused form of
        # `fdlinear`, thus the plane list follows from ONE name.
        self.law = ("fdlinear_fused"
                    if self.cfg.fuse_planes and self.cfg.force == "fdlinear"
                    else self.cfg.force)
        self.force_fn = force_fn(self.law)

        rule_kw = {"velocity": {"eta": eta},
                   "sgd": {"frac": sgd_frac, "seed": seed or 0},
                   "sqn": {"memory": sqn_memory}}.get(optim, {})
        self.set_rule(optim, lr, **rule_kw)
        if lr_decay == "linear":
            # lr_t = lr * (1 - epoch/epochs). At the last epoch this is
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

        The graphs of this project arrive as edge lists on disk and
        `make_graph.datasets.load` reads them. The stage stays in the API
        because a later project builds a graph from raw data (kNN, MST),
        and that project changes this method and nothing else.
        """
        return data

    # ---------------------------------------------------------- the walk
    def graph_walk(self, A, n: int | None = None, rng=None, **kwargs):
        """The random walks, and the statistics of the pairs they give.

        `pairs = "nbr_walk"` gives exactly `walks.walk_rows`: row `u` holds
        every node that a walk FROM `u` reached, at `h` = the first step
        that reached it. Directed, no window, and no cap -- the budget IS
        `walks * walk_len` for each row.

        `pairs = "walk"` and `"walk_edges"` give `walks.walk_pair_stats`:
        every pair inside the window of a walk, undirected, with a cap and
        a prune.

        The walk itself is `make_walker(A, n, p, q)`, thus `p = q = 1`
        gives `uniform_walks` itself and a default run reproduces every
        earlier number bit-exactly.
        """
        c = self.cfg
        n = A.shape[0] if n is None else n
        rng = self.rng if rng is None else rng
        walker = W.make_walker(A, n, c.p, c.q)
        if c.pairs == "nbr_walk":
            return W.walk_rows(A, n, c.walks, c.walk_len, rng, walker=walker)
        if c.pairs in ("walk", "walk_edges"):
            return W.walk_pair_stats(A, n, c.walks, c.walk_len, c.window,
                                     rng, cap=c.cap, prune_max=c.prune_max,
                                     prune_factor=c.prune_factor,
                                     walker=walker)
        raise ValueError(
            f"Unknown pairs policy {c.pairs!r}. Known: walk, walk_edges, "
            f"nbr_walk. Every policy of this package is walk-based; the "
            f"exact-distance policies were removed on 2026-08-20.")

    # ---------------------------------------------------------- stage 2
    def set_D(self, D, stats=None, freq=None, degrees=None):
        """Hand over a `D` that is already augmented, and skip the walks.

        `augment_graph` then builds the planes and the plan from it and
        nothing else. This is the path of a cached augmentation: the
        augmentation of com_youtube needs ten minutes, thus a retry of the
        embedding alone must not pay it again.

        `freq` is the `(nnz,)` data of the frequency plane, aligned to
        `D.indices`. A law that reads `freq` and does not get one RAISES --
        it never falls back to other physics.
        """
        self._D_given = D
        self.stats = stats
        self.freq = None if freq is None else np.asarray(freq)
        self.deg_from_A = None if degrees is None else np.asarray(degrees)
        return self

    def augment_graph(self, G, **kwargs):
        """Stage 2. `G` into `D`, then the planes and the batch plan.

        `G` is the un-augmented adjacency `A`, unless `set_D` handed a
        matrix over. The three steps:

        1. the walks and the policy build `D` and the `freq` data
           (`_build_D`);
        2. the plane list comes FROM `core.forces.FORCE_PLANES` and
           `plan_contract.check` asserts it;
        3. `make_plan` builds one plan, or `chunks` plans of a row range
           each.

        Why a ROW RANGE is the correct cut for a chunk, and not any set of
        pairs: the kernel writes `dZ.at[rows].add(...)`, thus only
        DISJOINT rows make the parts additive. `embed` already slices `dZ`
        by the same row range for its batches, thus the chunk and the
        batch are the same object and the engine needs no change.

        The global quantities stay global, and they must: `degrees_from_D`
        reads the WHOLE `D`, and a chunk only slices the result. A degree counted on one chunk is not the degree of the
        node.
        """
        if self._D_given is not None:
            D = self._D_given
        else:
            D = self._build_D(G)
        self.D = D
        n = D.shape[0]

        planes = self._build_planes(D)
        degrees = self._build_degrees(D, G)
        if self.cfg.check_planes:
            plan_contract.check(self.law, planes, D)
            plan_contract.check_degrees(degrees, D)

        self.params = dict(k1=self.cfg.k1, k4=self.cfg.k4,
                           kr=self.cfg.kr, sign=self.cfg.fdlinear_sign)
        self._build_plan(D, planes, degrees, n)
        return D

    # -- the plane list, FROM the registry ---------------------------------
    def _build_planes(self, D):
        """The planes of `self.law`, in the order the registry gives.

        No branch on the law name. A plane that the augmentation did not
        build stops the run here, with the name of the plane and the name
        of the law -- never with different physics.
        """
        out = []
        for name in planes_of(self.law):
            out.append(self._plane(name, D))
        return tuple(out)

    def _plane(self, name: str, D):
        if name == "h":
            return D.data
        if name == "freq":
            if self.freq is None:
                raise plan_contract.PlaneContractError(
                    f"{self.law} reads the plane `freq` and the "
                    f"augmentation for pairs={self.cfg.pairs!r} "
                    f"policy={self.cfg.policy!r} did not build one. This "
                    f"is a defect of the augmentation, not a configuration "
                    f"error, and it is never a reason to run the planes of "
                    f"another law.")
            return self.freq
        if name == "w":
            if self.freq is None:
                raise plan_contract.PlaneContractError(
                    f"{self.law} reads the fused plane `w = h/freq`, and "
                    f"the augmentation built no `freq`.")
            return fuse(D.data, self.freq)
        raise plan_contract.PlaneContractError(
            f"{self.law} declares a plane `{name}` that this class cannot "
            f"build. Add it here and to core.forces.FORCE_PLANES together.")

    def _build_degrees(self, D, G):
        """The divisor of the row sum. Three sources, and the order matters.

        `no_deg_norm` gives 1, thus the engine does not divide -- the law
        that the fdlinear specification writes.

        An explicit array (`deg_source = "A"`, or `set_D(degrees=...)`) is
        the true degree of the graph. `edge_rule = "low_deg"` needs it: a
        hub can then hold no entry at `h = 1`, `degrees_from_D` would give
        0, and `inv_deg_ext` turns that into 0.0, which freezes the row.

        Otherwise the count of `h = 1` entries of `D`, which is the
        package default.
        """
        n = D.shape[0]
        if self.cfg.no_deg_norm:
            return np.ones(n, dtype=np.int64)
        if self.deg_from_A is not None:
            return self.deg_from_A
        use_A = (self.cfg.deg_source == "A"
                 or (self.cfg.deg_source == "auto"
                     and self.cfg.edge_rule == "low_deg"))
        if use_A and G is not None and sp.issparse(G):
            return np.maximum(np.diff(G.indptr), 1).astype(np.int64)
        return degrees_from_D(D)

    # -- the plan ----------------------------------------------------------
    def _build_plan(self, D, planes, degrees, n):
        """One plan, or `chunks` plans of a row range each.

        The whole plan of a graph of a million nodes does not fit beside
        `Z` and `dZ` on a 2 GB card. A chunk holds the rows `[a, b)` only,
        thus the device holds ONE chunk at a time. The rows outside `[a, b)`
        become empty, and `make_plan` gives an isolated row no virtual row
        at all, thus an empty row costs nothing in the plan of another
        chunk.
        """
        c = self.cfg
        resident = not c.chunk_host
        self.chunk_rows = max(1, (n + max(1, c.chunks) - 1) // max(1, c.chunks))
        self.plans, self.steps, cells = [], [], 0
        stats = None
        for a in range(0, n, self.chunk_rows):
            b = min(a + self.chunk_rows, n)
            lo, hi = int(D.indptr[a]), int(D.indptr[b])
            indptr = np.zeros(n + 1, dtype=D.indptr.dtype)
            indptr[a + 1:b + 1] = D.indptr[a + 1:b + 1] - lo
            indptr[b + 1:] = indptr[b]
            Dc = sp.csr_matrix((D.data[lo:hi], D.indices[lo:hi], indptr),
                               shape=D.shape)
            plan, inv_deg_ext, stats = make_plan(
                Dc, tuple(p[lo:hi] for p in planes), degrees=degrees,
                b_cells=c.b_cells, k_max=c.k_max, ladder_base=c.ladder_base)
            cells += stats["cells"]
            # `resident` keeps the plan on the device, which is the fast
            # form and the one that needs the memory. Otherwise the plan
            # stays in the host memory and it moves for each use.
            self.plans.append(jax.tree_util.tree_map(jax.device_put, plan)
                              if resident else plan)
            self.steps.append(jax.jit(functools.partial(
                _step, n=n, force_fn=self.force_fn)))
            if self.cfg.check_padding:
                # I3, on the HOST tiles and before the device copy: a pad
                # cell carries 0 in every plane. One pass for each chunk,
                # one time for each D.
                plan_contract.check_plan(plan, len(planes))
        self.resident = resident
        self.plan_stats = dict(stats, cells=cells, chunks=len(self.plans),
                               rows_per_chunk=self.chunk_rows)
        self.inv_deg_ext = jax.device_put(inv_deg_ext)

    # ---------------------------------------------------------- stage 3
    def forces(self, Z, D, row_start: int, row_end: int, key=None, **kwargs):
        """The dZ rows of `[row_start, row_end)`.

        The kernel runs over the plan CHUNK that owns the row range, then
        the result is sliced. With one chunk it runs over the whole graph
        for every batch, which is the accepted trade: the plan groups rows
        by WIDTH and not by node id, thus a row range has no cheaper
        correct correspondence to a subset of the rungs.
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

    def fit(self, data, epochs: int = 1000, **kwargs):
        """Not implemented in the fodiwalk project. It raises.

        `fit` is `make_graph(data)` and then `embed(G)`, and `make_graph`
        is a stub here: the graphs arrive as edge lists on disk. A `fit`
        that calls a stub would promise a stage that does not exist.
        """
        raise NotImplementedError(
            "Fodiwalk.fit(.) is not implemented in the fodiwalk project: "
            "make_graph is a stub, thus fit would call it and promise a "
            "stage that does not exist. Read a graph with "
            "fodiwalk.make_graph.load(name) and call embed(A, epochs=...).")

    # =====================================================================
    # The augmentation. One method for each policy branch, and the branches
    # are the ones of `bench_fdwalk.build_D`, line for line.
    # =====================================================================
    def _build_D(self, A):
        c = self.cfg
        n = A.shape[0]
        rng = self.rng
        self.info = info = {}
        t0 = time.time()
        info["walk_order"] = 1 if (c.p == 1.0 and c.q == 1.0) else 2

        if c.pairs == "nbr_walk":
            return self._build_D_nbr_walk(A, n, rng, info, t0)
        return self._build_D_walk(A, n, rng, info, t0)

    # -- the directed policy of 2026-08-17 --------------------------------
    def _build_D_nbr_walk(self, A, n, rng, info, t0):
        """Row `u` = every neighbour of `u`, plus every node a walk reached.

        The ORDER is the contract of 2026-08-17: `row_cap` bounds the WALK
        partners only, and it runs BEFORE the neighbours are added, thus a
        neighbour is never dropped. A cap after would break that contract
        in silence.
        """
        c = self.cfg
        st = self.graph_walk(A, n, rng)
        if c.row_cap:
            info["walk_pairs_before_rowcap"] = int(st["key"].size)
            st = PR.row_cap(st, n, c.row_cap)
            info["walk_pairs_after_rowcap"] = int(st["key"].size)
        st = (W.with_neighbours_low_deg(st, A, n) if c.edge_rule == "low_deg"
              else W.with_all_neighbours(st, A, n))
        if c.policy == "buckets":
            st, info["buckets"] = BK.row_buckets(
                st, A, n, c.bucket_total or BK.budget(n), rng)
        info["prunes"] = 0
        info["raw_pairs"] = int(st["raw"])
        info["unique_pairs"] = int(st["key"].size)
        info["capped_pairs"] = int(st["key"].size)
        info["t_pairs"] = time.time() - t0

        h = st["mn"].astype(np.float64)
        near = PR.to_csr_directed(st["key"], h, n)
        info["near_nnz"] = int(near.nnz)
        info["h1_entries"] = int((near.data == 1).sum())
        info["edges_of_A"] = int(A.nnz)
        # `freq`, on the SAME sparsity, thus the two `.data` arrays match
        # entry by entry after the CSR build.
        if c.freq_mode == "node":
            visits = np.bincount(st["key"] % n, weights=st["cnt"],
                                 minlength=n)
            fq = visits[st["key"] % n]
        else:
            fq = st["cnt"].astype(np.float64)
        freq = PR.to_csr_directed(st["key"], fq.astype(np.float64), n)
        info["far_pairs"] = 0
        info["far_asked"] = 0

        if c.far > 0:
            cum = degree_table(A, c.far_bias)
            far = sample_far_pairs(n, c.far, near, rng, cum=cum)
            # `sample_far_pairs` rejects on the keys of `near` AS STORED,
            # and `near` is DIRECTED here, thus a pair stored as (v, u)
            # does not reject (u, v). The CSR build then SUMS the two and
            # the weight becomes 100 + the walk gap; the histogram showed
            # entries at 101..119 on 2026-08-17. This drops any far pair
            # that `near` holds in EITHER direction.
            #
            # Why the filter and not `sample_far_pairs(..., directed=True)`,
            # which rejects both ways inside the loop: a changed rejection
            # changes how many pairs one batch accepts, thus it changes the
            # draw of the next batch and the far set of every recorded run.
            # The two give the same PROPERTY and not the same pairs, and
            # parity decides. See CATALOG.md 8.2.
            if far.shape[0]:
                nk = near.tocoo()
                have = np.sort(nk.row.astype(np.int64) * n
                               + nk.col.astype(np.int64))

                def _absent(k):
                    pos = np.searchsorted(have, k)
                    pos[pos >= have.size] = 0
                    return have[pos] != k

                k1 = far[:, 0].astype(np.int64) * n + far[:, 1]
                k2 = far[:, 1].astype(np.int64) * n + far[:, 0]
                far = far[_absent(k1) & _absent(k2)]
                del nk, have
            if far.shape[0]:
                fc = freq.tocoo()
                cc = near.tocoo()
                fw = np.full(2 * far.shape[0], c.far_weight)
                rr = np.concatenate([cc.row, far[:, 0], far[:, 1]])
                cl = np.concatenate([cc.col, far[:, 1], far[:, 0]])
                near = sp.csr_matrix((np.concatenate([cc.data, fw]),
                                      (rr, cl)), shape=(n, n))
                freq = sp.csr_matrix(
                    (np.concatenate([fc.data, np.ones(2 * far.shape[0])]),
                     (rr, cl)), shape=(n, n))
                info["far_pairs"] = int(far.shape[0])
                info["far_asked"] = int(c.far)
                info["near_nnz"] = int(near.nnz)

        info["t_aug"] = time.time() - t0
        self.stats = dict(st, freq=freq)
        self.freq = freq.data
        return near

    # -- the undirected policies ------------------------------------------
    def _build_D_walk(self, A, n, rng, info, t0):
        c = self.cfg
        stats = self.graph_walk(A, n, rng)
        info["prunes"] = int(stats.get("prunes", 0))
        info["raw_pairs"] = int(stats["raw"])
        info["unique_pairs"] = int(stats["key"].size)
        info["t_pairs"] = time.time() - t0

        keep = PR.cap_per_node(stats["key"], stats["cnt"], n, c.cap)
        stats = _take(stats, keep)
        info["capped_pairs"] = int(stats["key"].size)

        h = WT.RULES[c.weight](stats, c.window, n)

        if c.policy == "buckets":
            return self._build_D_buckets(A, n, rng, info, t0, stats, h)

        if c.pairs == "walk_edges":
            # The first-order edges always go in, at the distance 1. A walk
            # can miss the edge of a low-degree node, and that edge is the
            # pair that we trust most. EVERY array of `stats` grows, and
            # not only the key.
            e = sp.triu(A, k=1).tocoo()
            ekey = e.row.astype(np.int64) * n + e.col.astype(np.int64)
            new = ~np.isin(ekey, stats["key"])
            add = int(new.sum())
            stats["key"] = np.concatenate([stats["key"], ekey[new]])
            stats["mn"] = np.concatenate([stats["mn"],
                                          np.ones(add, dtype=np.int32)])
            stats["sm"] = np.concatenate([stats["sm"],
                                          np.ones(add, dtype=np.int64)])
            stats["cnt"] = np.concatenate([stats["cnt"],
                                           np.ones(add, dtype=np.int64)])
            h = np.concatenate([h, np.ones(add)])
            info["edges_added"] = add

        near = PR.to_csr(stats["key"], h, n)
        info["near_nnz"] = int(near.nnz)
        # `freq` on the SAME sparsity as `near`, thus the two `.data`
        # arrays line up entry by entry after the CSR build.
        fq_near = PR.to_csr(stats["key"], stats["cnt"].astype(np.float64), n)

        n_far = c.far or int(n * np.log10(max(n, 10)))
        cum = degree_table(A, c.far_bias)
        far = sample_far_pairs(n, n_far, near, rng, cum=cum)
        if cum is not None:
            deg = np.diff(A.indptr)
            info["far_mean_deg"] = (float(deg[far.ravel()].mean())
                                    if far.size else 0.0)
        info["far_pairs"] = int(far.shape[0])
        info["far_asked"] = n_far
        freq_csr = fq_near
        D = near
        if far.shape[0]:
            fc = fq_near.tocoo()
            freq_csr = sp.csr_matrix(
                (np.concatenate([fc.data, np.ones(2 * far.shape[0])]),
                 (np.concatenate([fc.row, far[:, 0], far[:, 1]]),
                  np.concatenate([fc.col, far[:, 1], far[:, 0]]))),
                shape=(n, n))
            if c.landmarks:
                # A real distance for the far pairs, and not one constant.
                t1 = time.time()
                lm = LM.pick(A, n, c.landmarks, rng)
                table = LM.distances(A, n, lm)
                info["t_landmark"] = time.time() - t1
                info["landmark_mb"] = table.nbytes / 1e6
                fw = LM.pair_distance(table, far[:, 0], far[:, 1], c.far_max,
                                      c.far_weight / c.far_scale)
                fw = fw * c.far_scale
                info["far_unreached"] = int((fw == c.far_weight).sum())
                del table
            else:
                fw = np.full(far.shape[0], c.far_weight)
            cc = near.tocoo()
            D = sp.csr_matrix(
                (np.concatenate([cc.data, fw, fw]),
                 (np.concatenate([cc.row, far[:, 0], far[:, 1]]),
                  np.concatenate([cc.col, far[:, 1], far[:, 0]]))),
                shape=(n, n))
        info["t_aug"] = time.time() - t0
        self.stats = dict(stats, freq=freq_csr)
        self.freq = freq_csr.data
        return D

    def _build_D_buckets(self, A, n, rng, info, t0, stats, h):
        """Every original edge at `h = 1`, plus a stratified `h >= 2` budget.

        The pairs at `h = 1` are not candidates: the policy keeps all of
        them, thus only the walk pairs at `h >= 2` enter the sampler.

        `freq` is built HERE, on the same sparsity as `near`. Until
        2026-08-18 this branch returned none, thus an `fdlinear` run took
        the planes of another law from a fallback and read a coefficient
        plane as `h`. The physics was wrong, nothing raised, and the run
        went to NaN at 2000 epochs.
        """
        c = self.cfg
        far2 = h >= 2
        idx, info["buckets"] = BK.bucket_sample(
            h[far2], c.bucket_total or BK.budget(n), rng)
        sel = np.flatnonzero(far2)[idx]
        stats = _take(stats, sel)
        h = h[sel]
        e = sp.triu(A, k=1).tocoo()
        ekey = e.row.astype(np.int64) * n + e.col.astype(np.int64)
        stats["key"] = np.concatenate([stats["key"], ekey])
        stats["mn"] = np.concatenate([stats["mn"],
                                      np.ones(ekey.size, dtype=np.int32)])
        stats["sm"] = np.concatenate([stats["sm"],
                                      np.ones(ekey.size, dtype=np.int32)])
        stats["cnt"] = np.concatenate([stats["cnt"],
                                       np.ones(ekey.size, dtype=np.int32)])
        h = np.concatenate([h, np.ones(ekey.size)])
        info["edges_added"] = int(ekey.size)
        near = PR.to_csr(stats["key"], h, n)
        info["near_nnz"] = int(near.nnz)
        freq_b = PR.to_csr(stats["key"], stats["cnt"].astype(np.float64), n)
        info["far_pairs"] = 0
        info["far_asked"] = 0

        if c.far_with_buckets:
            # `buckets` has NO far pairs by construction, and every
            # measurement says the long-range term carries the hop R2.
            # This adds them ON TOP of the bucket budget, thus it separates
            # two causes that the policy confounds.
            nf = c.far or int(n * np.log10(max(n, 10)))
            cum = degree_table(A, c.far_bias)
            far = sample_far_pairs(n, nf, near, rng, cum=cum)
            info["far_pairs"] = int(far.shape[0])
            info["far_asked"] = nf
            if far.shape[0]:
                fw = np.full(far.shape[0], c.far_weight)
                cc = near.tocoo()
                near = sp.csr_matrix(
                    (np.concatenate([cc.data, fw, fw]),
                     (np.concatenate([cc.row, far[:, 0], far[:, 1]]),
                      np.concatenate([cc.col, far[:, 1], far[:, 0]]))),
                    shape=(n, n))
                info["near_nnz"] = int(near.nnz)
                fc = freq_b.tocoo()
                freq_b = sp.csr_matrix(
                    (np.concatenate([fc.data, np.ones(2 * far.shape[0])]),
                     (np.concatenate([fc.row, far[:, 0], far[:, 1]]),
                      np.concatenate([fc.col, far[:, 1], far[:, 0]]))),
                    shape=(n, n))
        info["t_aug"] = time.time() - t0
        self.stats = dict(stats, freq=freq_b)
        self.freq = freq_b.data
        return near


def _take(stats, idx):
    """The same statistics dict, on the selected pairs only."""
    return {k: (v[idx] if isinstance(v, np.ndarray) else v)
            for k, v in stats.items()}
