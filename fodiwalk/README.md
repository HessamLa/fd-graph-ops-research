# fodiwalk

Walk-augmented force-directed graph embedding, on JAX.

A graph goes in. Random walks turn it into a weighted matrix `D`. A force
law relaxes an embedding `Z` against `D`. The layout, not a loss, carries
the structure.

Run everything through the project interpreter:

```bash
.venv/bin/python -m pytest fodiwalk/tests -q -m "not parity and not big"
```

---

## Use

```python
from fodiwalk import Fodiwalk
from fodiwalk.make_graph import load

A, n = load("cora")                       # (symmetric CSR, node count)
fw = Fodiwalk(n_dim=64, seed=42, lr=1.0, optim="plain",
              pairs="nbr_walk", weight="min_gap", force="fdlinear")
fw.embed(A, epochs=200)                   # augments, then relaxes
Z = fw.get_embeddings()                   # (n, 64) numpy
```

`embed` takes the UN-augmented graph. It calls `augment_graph` first, thus
one call covers both stages.

Measure it:

```python
import numpy as np
from fodiwalk.augment_graph import pairs as PR
from fodiwalk.misc.evaluation import link_prediction, hop_sample, task_hop

scores, info = link_prediction(Z, A, n, 50_000, fw.rng, seed=42)
gap = PR.to_csr(fw.stats["key"], fw.stats["mn"].astype(np.float64), n)
u, v, d, h2 = hop_sample(A, n, fw.rng, 200, 20_000, gap)
rows = task_hop(Z, u, v, d, seed=42, feature="distance")
```

Pass `fw.rng` and not a fresh generator: the order of the draws is part of
a reproducible run.

Read what happened:

```python
fw.info         # pair counts, timings, far pairs
fw.plan_stats   # cells, n_split, rungs, pad_frac, chunks
fw.diverged     # True when Z went non-finite. It is RECORDED, never raised
```

### Knobs

Every knob is a `Config` field, and the names are the flags of
`experiments/fdwalk/bench_fdwalk.py` with underscores. An unknown name
raises `TypeError`.

```python
fw = Fodiwalk(n_dim=64, seed=42,
              pairs="walk", policy="buckets",   # what enters D
              walks=10, walk_len=20, window=5, cap=16,
              p=2.0, q=1.0,                     # node2vec walk
              far=0, far_bias=0.75,             # long-range pairs
              force="fdlinear", k1=0.999, k4=1.0, kr=1.0,
              optim="adam", lr=0.1, lr_decay="linear",
              chunks=8, chunk_host=True)        # a 1M-node graph on 2 GB
```

`Config` is a dataclass. `print(fodiwalk.Config())` lists the defaults.

### Skip the walks

The augmentation of a 1M-node graph costs minutes. Hand a built `D` over
and only the plan is rebuilt:

```python
fw.set_D(D, stats=stats, freq=freq_csr.data)
fw.embed(A, epochs=500)
```

---

## Inherit

### A new stage on `Fodiwalk`

`make_graph` is a stub: it returns what it is given. Override it to make
stage 1 real. There is no `fit` (removed 2026-08-21, see `dev-docs/CATALOG.md`
section 21): call `make_graph` then `embed` yourself.

```python
class KnnFodiwalk(Fodiwalk):
    def make_graph(self, data, k=10, **kw):
        return knn_csr(data, k)                # -> a symmetric CSR

knn = KnnFodiwalk(n_dim=64)
knn.embed(knn.make_graph(raw_data, k=10), epochs=200)
```

### A new augmentation policy

A policy is NOT a subclass. It is a module function behind the registry
`augment_graph.policies.POLICIES`, thus it takes a graph and gives an
`Augmentation`, and it holds no state of the model:

```python
def build(A, n, spec, rng) -> Augmentation
```

Write it in a new `augment_graph/policy_*.py`, then add ONE entry:

```python
import numpy as np, scipy.sparse as sp
from fodiwalk import Fodiwalk
from fodiwalk.augment_graph import policies
from fodiwalk.augment_graph.pairs import to_csr
from fodiwalk.augment_graph.result import Augmentation

def build_edges_only(A, n, spec, rng):
    """The edges of `A` and nothing else. `spec` is an `AugmentSpec`."""
    e = sp.triu(A, k=1).tocoo()
    key = e.row.astype(np.int64) * n + e.col.astype(np.int64)
    h = np.ones(key.size)                  # INTEGER h >= 1, and 1 = adjacent
    D, freq = to_csr(key, h, n), to_csr(key, h, n)
    return Augmentation(D=D, freq=freq.data,
                        stats={"key": key, "mn": h, "cnt": h, "freq": freq},
                        info={"near_nnz": int(D.nnz)})

policies.POLICIES["edges_only"] = build_edges_only
fw = Fodiwalk(n_dim=16, seed=42, pairs="edges_only")
fw.embed(A, epochs=3)
```

Add the walk of the policy to `policies.WALK_STATS` under the same name
when `Fodiwalk.graph_walk` must work for it too. An unknown `pairs` name
raises with the list of the known ones; it never falls back.

Three rules the builder must keep, and each one has already cost a run:

1. `freq` must be built on the SAME sparsity as `D`, thus the two `.data`
   arrays line up entry by entry. Give it `None` only for a law that reads
   no `freq` plane -- a missing plane RAISES and never falls back to the
   planes of another law.
2. `h` is an INTEGER `>= 1` and `h = 1` means adjacency.
3. Use the `rng` you are given, one time, in one order. A second generator
   changes every recorded number.

### A different engine

`ForceDirected` owns the epoch loop, the batching, the callbacks and the
`Z` update, and NOTHING about how `D` is built (it was
`ForceDirectedEmbedding` between 2026-08-21, when
`make_graph`/`augment_graph`/`fit` left it, and 2026-08-26 --
`dev-docs/CATALOG.md` sections 21 and 23). The class itself lives in the
ROOT package `forcedirected`, which `fodined` reads too; `fodiwalk.core`
re-exports the name, thus either import below works. Subclass it directly
when you already have `D` and need only a new force law:

```python
from fodiwalk.core import ForceDirected

class MyModel(ForceDirected):
    def forces(self, Z, D, row_start, row_end, **kw): ...   # -> (rows, d)

model = MyModel(n_dim=32, seed=0)
model.embed(D, epochs=200)      # D already built. No augment_graph at all.
```

`forces` gets a ROW RANGE, never a set of pairs: the kernel writes
`dZ.at[rows].add(...)`, thus only disjoint rows make the parts additive.

A model that needs the FULL pipeline -- a raw graph in, not a ready `D` --
subclasses `fodiwalk.Fodiwalk_base` instead, and implements `make_graph`,
`augment_graph` and `embed` (the orchestration: build `D`, then call
`ForceDirected.embed` on it). `Fodiwalk` itself is exactly that
kind of subclass, walk-based.

---

## The tree

| path | what it holds |
| --- | --- |
| `config.py` | `Config` -- every knob, FLAT and public |
| `base.py` | `class Fodiwalk_base` -- the pipeline CONTRACT, every stage abstract |
| `model.py` | `class Fodiwalk(Fodiwalk_base)` -- the three stages, WIRED. Nothing else |
| `core/` | the engine (`ForceDirected`), the layout, the physics, the asserter. `force_directed.py`, `sell_c_sigma.py` and `csr.py` FORWARD to the root package `forcedirected` |
| `make_graph/` | stage 1, the datasets |
| `augment_graph/` | stage 2, the walks, the pair policies, the planes and the degrees |
| `embed/` | stage 3, CONSUMPTION ONLY: the chunked plan and the jitted kernel wiring |
| `misc/` | the update rules, the drop, the measurements |
| `tests/` | contracts, smoke, golden, api, parity |
| `dev-docs/` | PRD, ORCHESTRATION, CATALOG, BUILD, REFACTOR |

**The dependency runs ONE WAY, and a cycle is a defect.** Four rules:

```
core            imports core only. It never learns about `Config`.
make_graph      imports numpy and scipy.
augment_graph   imports numpy, scipy, core (the plane/degree contract). NO embed, NO model.
embed           imports core only. NO augment_graph, NO model.
model.py        imports config, make_graph, augment_graph, embed, core, misc.
```

`core` may not read a configuration, and that is why the plane, degree and
force-param builders live in `augment_graph/` and the plan assembly in
`embed/`, and neither is in `core/`. A plane and a degree are DATA the
recipe (one pair policy plus one law) prepares, not kernel state
(`dev-docs/fodiwalk-module.md`; `dev-docs/CATALOG.md` section 20 -- they
moved here from `embed/` on 2026-08-21). No module imports a `_private`
name of another module. `tests/test_structure.py` and
`tests/test_contracts.py` assert every rule above with the `ast` module,
thus the next change either keeps the shape or fails.

---

## `core/`

### `force_directed.py` -- `ForceDirected`, `Callback_Base`

A FORWARDER since 2026-08-26: the class lives in
`forcedirected/force_directed.py`, at the repository root, because
`fodined` uses the same engine and must not depend on `fodiwalk`.

The epoch loop, the row batching, the callback events, the `Z` update. A
PURE engine: `embed(D, ...)` relaxes `Z` against a `D` it is GIVEN. It owns
NO augmentation and NO force law -- and (it was `ForceDirectedEmbedding`
from 2026-08-21 to 2026-08-26) no `make_graph`, no `fit`, either. The
pipeline that builds
`D` from raw data is `fodiwalk.Fodiwalk_base`, one layer up
(`dev-docs/CATALOG.md` section 21).

Events, in order: `train_begin`, then per epoch `epoch_begin`, per batch
`batch_begin` / `batch_end`, `epoch_end`, then `train_end`.

```python
from fodiwalk.core import Callback_Base

class Watch(Callback_Base):
    def on_epoch_end(self, model, **kw):
        print(kw["epoch"], model.Th(model.dZ))

fw.attach_callback(Watch())
```

Replace the integrator by overriding `updateZ`, or dispatch to a rule:

```python
fw.set_rule("velocity", lr=0.1, eta=0.3)
fw.lr_schedule = lambda lr, ep, eps: lr * (1 - ep / eps)   # linear decay
```

### `forces.py` -- the laws, and `FORCE_PLANES`

The laws, the plane builders, and the registry that says which law reads
which planes, in what order.

| law | planes | attraction |
| --- | --- | --- |
| `fdlinear` | `(h, freq)` | `h <= 1` only |
| `fdlinear_fused` | `(w,)` | `w < 0` |

Add a law. When it reuses the plane names that exist, NOTHING else needs an
edit -- that is what the registry is for:

```python
import jax.numpy as jnp
from fodiwalk.core import forces

def fdsquare(x, planes, params):
    h, freq = planes
    live, near = h > 0, h <= 1
    Fa = jnp.where(near & live, params["k1"] * x * x, 0.0)
    coeff = jnp.where(near, params["kr"], h / jnp.maximum(freq, 1.0))
    return Fa + jnp.where(live, -coeff * jnp.exp(-params["k4"] * x), 0.0)

forces.FORCE_PLANES["fdsquare"] = ("h", "freq")
forces.FORCE_FN["fdsquare"] = fdsquare

fw = Fodiwalk(n_dim=64, force="fdsquare", pairs="nbr_walk")
```

A law returns the force MAGNITUDE along `u -> v`. The kernel applies the
direction, the degree division and the padding guards.

A law that needs a NEW plane needs two more edits, and the error message
names both: an entry in `augment_graph.planes.PLANE_BUILDERS` that builds
the values, and a validator in `plan_contract.PLANE_CHECKS` that states
what the name promises. The two tables must hold the same keys, and
`planes.py` asserts that at import. The plane builder lives in
`augment_graph/`, not `embed/`: it is data preparation for the recipe (one
augmentation policy plus one law), and `embed/` only ever consumes it
(`dev-docs/CATALOG.md` section 20).

### `plan_contract.py` -- the asserter

The one new module of the package. It asserts the plane contract at the
seam, and it RAISES: a missing or wrong plane is never a reason to run
other physics.

```python
plan_contract.check("fdlinear", (freq, h), D)   # RAISES: h is not D.data
```

A plane name is a promise about the values, thus a swap is caught even
though the COUNT is right. `check_degrees` catches a row that would freeze;
`check_plan` catches a pad cell that would push. Turn them off for a very
large run with `check_planes=False, check_padding=False`.

### `sell_c_sigma.py` -- the layout. No physics.

Hub split, width sort, ladder quantize, balanced pack, pad. It turns `D`
into fixed-shape `(nb, R, k)` batches and reduces each with a dense
`sum(axis=1)`. `make_plan` never interprets a plane; `step` never knows
the law. Do not edit it to change the physics.

`step` is the per-epoch kernel and it is PUBLIC. It was `_step`, and the
model class imported it across a module boundary, which the underscore
forbids. `_step` stays as an alias, thus every older caller works.

### `csr.py`

`row_of(indptr)`, `n_rows(D)`. The bottom of the tree; it imports numpy
only.

---

## `make_graph/`

`datasets.py`: `load(name, max_nodes=0, seed=42) -> (A, n)`. `A` is a
symmetric CSR with a zero diagonal and sorted indices. `cora`, `pubmed`,
`com_youtube`, `as_skitter`, `roadnet_ca`, `wordnet`, `ncbi_taxonomy`.

A graph larger than `max_nodes` is truncated, and the way depends on the
kind of graph. Both wrong choices give a perfect score on an empty problem:

| kind | truncation | what the other choice does |
| --- | --- | --- |
| a graph (`roadnet_ca`, `com_youtube`, ...) | a BFS ball, which stays connected | a subtree walk read roadNet-CA as "parent -> child" and returned 500,000 nodes with 504 edges |
| a tree (`TREE`: `wordnet`, `ncbi_taxonomy`) | one subtree, which keeps the DEPTH | a BFS ball from an NCBI hub is a STAR of depth 1, thus every non-edge pair is 2 hops apart |

Point it elsewhere with `FDMAP_DATA=/path .venv/bin/python ...`.

---

## `augment_graph/`

| module | what it does |
| --- | --- |
| `policies.py` | `POLICIES` and `WALK_STATS` -- the registry, and `build(A, n, spec, rng)`, the ONE entry point of the stage |
| `result.py` | `Augmentation` -- what the stage gives stage 3. `AugmentSpec` -- the knobs it may read |
| `policy_walk.py` | `walk` and `walk_edges`: undirected, windowed, capped |
| `policy_buckets.py` | `policy = buckets` on the undirected pairs |
| `policy_nbr_walk.py` | `nbr_walk`: directed rows, and its `cap` and `buckets` axes |
| `merge.py` | `add_far_pairs` -- the ONE far/`freq` CSR merge. `drop_pairs_of` -- the far filter of `nbr_walk` |
| `planes.py` | `PLANE_BUILDERS`, `build_planes`, `ForceSpec`, `force_params` -- the law's DATA (moved from `embed/` 2026-08-21) |
| `degrees.py` | `resolve_degrees` -- the divisor of the row sum (moved from `embed/` 2026-08-21) |
| `walks.py` | `uniform_walks`, `node2vec_walks`, `make_walker`, `walk_rows`, `walk_pair_stats`, the neighbour merges |
| `pairs.py` | the pair key, `cap_per_node`, `row_cap`, the CSR builds |
| `weights.py` | `flat`, `min_gap`, `mean_gap`, `pmi` -- the walk statistics into `h` |
| `buckets.py` | the stratified budget, undirected and row-wise |
| `far_pairs.py` | the long-range pairs, and the `deg^alpha` draw |
| `landmarks.py` | a real distance for a far pair, from exact BFS |

A policy is a module function `(A, n, spec, rng) -> Augmentation` and never
a method: it reads an `AugmentSpec` and it writes nothing on the model.
Drive the stage alone with it:

```python
import numpy as np
from fodiwalk import Config
from fodiwalk.augment_graph import build
from fodiwalk.augment_graph.result import AugmentSpec

spec = AugmentSpec.from_config(Config(pairs="nbr_walk"))
aug = build(A, n, spec, np.random.default_rng(42))
aug.D, aug.freq, aug.stats, aug.info      # a policy's own build() fills these
aug.planes, aug.degrees, aug.params       # None here; Fodiwalk fills them once
                                          # the law is known -- see below
```

**THE ORDER OF THE COO TRIPLES IS PART OF THE RESULT.** `add_far_pairs`
writes the near entries, then the forward pairs, then the backward pairs.
`sp.csr_matrix` SUMS a duplicate coordinate and the sum order decides the
last bit. Do not sort and do not group.

### The planes come from a REGISTRY

`build_planes` walks `core.forces.planes_of(law)` and takes the builder of
each name from `PLANE_BUILDERS`. There is NO branch on the law name. A
plane that the augmentation did not build RAISES, with the name of the
plane and the name of the law -- it never falls back to the planes of
another law. That fallback made `fdlinear` read a coefficient plane as `h`,
and the run went to NaN under an `fdlinear` label with no error.

```python
from fodiwalk.augment_graph import ForceSpec, build_planes, force_params

fspec = ForceSpec.from_config(cfg)
planes = build_planes(fspec.law, D, freq, cfg.pairs, cfg.policy)
params = force_params(fspec)          # dict(k1=, k4=, kr=, sign=)
```

### The degree has three sources, and the order matters

`no_deg_norm` gives 1. An explicit array (`deg_source="A"`, or
`set_D(degrees=...)`) is the true degree of the graph. Otherwise
`degrees_from_D` counts the `h == 1` entries of a row.

A row with no `h = 1` entry gets degree 0, `inv_deg_ext` becomes 0.0, and
the kernel zeroes the WHOLE force of that row -- the repulsion too. The row
never moves again and nothing raises. `edge_rule="low_deg"` can do that to
a hub, thus `deg_source="auto"` reads `edge_rule` and takes the degree from
`A` instead.

```python
from fodiwalk.augment_graph import resolve_degrees

degrees = resolve_degrees(D, A, fspec, explicit=None)
```

Both are STAGE-2 DATA (`dev-docs/fodiwalk-module.md`): a different
augmentation, or a different law, prepares a different plane set or a
different degree array for the same `D`. `embed/`, below, only ever
consumes what this stage already built.

**Every policy is WALK-BASED.** `h` is a walk GAP -- an upper bound of the
hop distance, which a walk of `t` steps proves -- and never a measured
distance.

| `pairs` | a row holds | direction |
| --- | --- | --- |
| `walk` | every pair inside the window of a walk, capped at `cap` for each node | undirected |
| `walk_edges` | the same, plus every original edge forced to `h = 1` | undirected |
| `nbr_walk` | EVERY neighbour at `h = 1`, plus every node a walk from the row reached | directed |

The exact-distance policies `ball` and `sampled` were removed on
2026-08-20. A walk does not lose to them: on Cora `nbr_walk` exceeds the
measured 2-hop ball of `fodined/modular.py` on every score -- accuracy
0.9801 against 0.9777, AUC 0.9982 against 0.9953 -- and it stores no exact
distance at all. `dev-docs/CATALOG.md` sections 17 and 18.

Three invariants, each of which has already caused a silent defect:

1. **A stored weight is an INTEGER in `[1, window]`, and 1 means
   adjacency.** A float weight makes `degrees_from_D` return 0 for every
   row and the whole force vanishes: AUC 0.55, `||dZ|| = 0.000`, no error.
2. **A row with no `h = 1` entry MUST get an explicit degree**, or it
   freezes for the whole run.
3. **`walk_rows` has NO prune by design**: its bound IS
   `walks * walk_len` per row. `walk_pair_stats` DOES prune, and its prune
   is an approximation that `prunes > 0` reports.

Add a weight rule:

```python
from fodiwalk.augment_graph import weights

def weight_log_gap(stats, window, n):
    """The minimum gap on a log scale. An INTEGER in [1, window]."""
    x = np.log1p(stats["mn"] - 1.0)                 # exactly 0 at gap 1
    return np.clip(np.rint(1 + (window - 1) * x / max(x.max(), 1e-9)),
                   1, window)

weights.RULES["log_gap"] = weight_log_gap     # -> Fodiwalk(weight="log_gap")
```

Two details, and both decide whether the rule works at all.

The `np.rint` is not decoration: see invariant 1.

The rule MUST give exactly 1 to the nearest pairs, which is what the
`- 1.0` does. A rule that never emits 1 leaves every row at degree 0, thus
`inv_deg_ext` is 0.0 and NOTHING moves. `mean_gap` and `pmi` have this
defect today -- `mean_gap` leaves 2,574 of Cora's 2,708 rows with no
`h = 1` entry -- and it is why they score AUC 0.55 and 0.70 with
`||dZ||` near 0. `plan_contract.check_degrees` stops such a run before the
first epoch, thus a new rule cannot repeat it in silence. The repair, when
a rule genuinely has no `h = 1`, is an explicit degree:
`Fodiwalk(weight=..., deg_source="A")`.

---

## `embed/`

Stage 3, CONSUMPTION ONLY (`dev-docs/fodiwalk-module.md`): "This stage
shall not do any graph analysis or data preparation. It must only consume
the data." It takes the `D`, the planes, the degrees and the force params
that `augment_graph/` already prepared, and builds the chunked plan and the
jitted steps the kernel of `core` runs. It builds NO plane and resolves NO
degree -- `dev-docs/CATALOG.md` section 20 says why they left in
2026-08-21.

| module | what it does |
| --- | --- |
| `planner.py` | `PlanSpec`, `PlanSet`, `build_plans` -- the chunked plan |

`Fodiwalk.augment_graph` assembles the whole recipe, and `build_plans` is
its last, CONSUMING step:

```python
from fodiwalk.core import plan_contract
from fodiwalk.core.forces import force_fn
from fodiwalk.augment_graph import (ForceSpec, build_planes, force_params,
                                    resolve_degrees)
from fodiwalk.embed import PlanSpec, build_plans

fspec, pspec = ForceSpec.from_config(cfg), PlanSpec.from_config(cfg)
planes = build_planes(fspec.law, D, freq, cfg.pairs, cfg.policy)   # augment_graph
degrees = resolve_degrees(D, A, fspec)                             # augment_graph
plan_contract.check(fspec.law, planes, D)          # I1, I2, I4
plan_contract.check_degrees(degrees, D)            # I5
params = force_params(fspec)                                       # augment_graph
plan_set = build_plans(D, planes, degrees, pspec, force_fn(fspec.law))  # embed
```

### A chunk is a ROW RANGE

`build_plans` gives `PlanSpec.chunks` plans, one for the rows `[a, b)` of
each. The whole plan of a million-node graph does not fit beside `Z` and
`dZ` on a 2 GB card, thus the device holds ONE chunk at a time
(`chunk_host=True` keeps them in the host memory).

A chunk is never a set of pairs: the kernel writes `dZ.at[rows].add(...)`,
thus only DISJOINT rows make the parts additive. The global quantities stay
global -- `degrees` is counted over the whole `D`, and a chunk only slices
the planes.

```python
plan_set.plans, plan_set.steps          # one plan and one jitted step each
plan_set.inv_deg_ext, plan_set.chunk_rows
plan_set.resident, plan_set.stats       # -> fw.plan_stats
```

---

## `misc/`

### the update rules -- eight of them, in `forcedirected/optim.py`

`plain`, `momentum`, `nesterov`, `adam`, `fa2`, `velocity`, `sgd`, `sqn`.
A rule is pure: `step(Z, dZ, lr, state, epoch, **kw) -> (Z_new, state)`,
with its state in a dict the model keeps.

The rules sit beside the engine that dispatches through them, thus they are
in `forcedirected` and not in `misc`. `fodiwalk.misc` re-exports `RULES`,
`STATE_ARRAYS` and `state_arrays`.

```python
from forcedirected import optim

def step_signsgd(Z, dZ, lr, state, epoch):
    return Z + lr * jnp.sign(dZ), state

optim.RULES["signsgd"] = step_signsgd
optim.STATE_ARRAYS["signsgd"] = 0             # what fits on the card
```

`dZ` is a FORCE and not the gradient of a loss, thus a large `dZ` means
"this node must move far". A rule that divides by the size of `dZ` throws
that away.

### `drop.py`

`drop_steady_rate(dZ, key, rate, strategy)`. `random_rows` zeroes a whole
row; `random_cells` zeroes single cells. They are different regularizers.

### `evaluation.py`

`link_prediction` (does the geometry hold the ADJACENCY?), `hop_sample`
and `task_hop` (does it hold the DISTANCE?). Measurement, not engine:
nothing in `core` imports it.

`link_prediction` is the Hadamard product `Z[u] * Z[v]` into a random
forest, verbatim from `fodined/link_prediction.py`.

`hop_sample` draws its pairs by BFS from `n_sources` sources, thus most of
them are pairs `D` never held and the measurement is OUT OF SAMPLE. It also
returns H2 -- the walk gap against the true distance, on the pairs `D`
stores.

**Do not read H2 as a quality score.** It says how TIGHT the bound is, and
that does not order the policies by result: `walk` is exact on 98.1% of its
stored pairs and `nbr_walk` on 7.3%, and `nbr_walk` wins every score. The
layout needs the ORDER of `h` and enough partners, not the value of `h`.
`dev-docs/CATALOG.md` section 17.4.

`task_hop` takes two feature sets and they must not be mixed: `vector` is
`|Z[u] - Z[v]|`, thus `n_dim` features, and `distance` is the Euclidean
distance alone, thus ONE feature -- the protocol of
`experiments/other-ge/`, which the node2vec and Poincare numbers use. Its
`n_estimators`, `hidden` and `early_stopping` arguments carry the same
rule.

---

## `tests/`

```bash
.venv/bin/python -m pytest fodiwalk/tests -q -m "not parity and not big"
.venv/bin/python -m pytest fodiwalk/tests -q -s -m parity   # a GPU, ~18 min
.venv/bin/python -m pytest fodiwalk/tests -q -m big         # com_youtube
```

`test_contracts.py` holds one test for one contract. `test_structure.py`
holds the SHAPE of the tree -- the one-way dependency, no private
cross-module import, no dispatch chain, the module sizes, and every policy
builder a module function. `test_smoke.py` holds the engine and the class,
on a 60-node graph. `test_parity.py` reproduces the recorded runs of the
campaign, and `harness.py` runs one scenario end to end.

Two gates cost seconds, thus they run between two edits:

```bash
.venv/bin/python -m fodiwalk.tests.golden --check      # the BEHAVIOUR
.venv/bin/python -m fodiwalk.tests.check_api           # the SURFACE
```

`golden.py` hashes what the augmentation and the plan build on Cora, for
one configuration of every policy branch. The augment half is pure NumPy
and it is compared BYTE EXACT, the digest of the generator state included.
The embed half is JAX, thus it runs on the CPU backend and compares numbers
at `rtol = 1e-4`: a split hub row makes the GPU scatter-add order free.

`check_api.py` holds the public surface LITERALLY -- the 39 `Config` fields
with their defaults, the methods, the attributes, and the key sets of
`info`, `plan_stats` and `stats`. A lost key fails and a new key passes.
A check that read `dataclasses.fields(Config)` would compare `Config` to
itself and pass whatever a refactor did.

**The package must reproduce the numbers of `experiments/fdwalk/`.** A
difference is a defect and not a finding. Run the parity gate before you
trust a change to the physics, the augmentation or the plan.

---

## `dev-docs/`

`PRD.md` what each block must satisfy. `ORCHESTRATION.md` who built it and
in what order. `CATALOG.md` the entities, incrementing -- nothing is ever
removed from it. `BUILD.md` what was built, what it reproduces, and what it
does not do. `REFACTOR.md` the defects of the 698-line god class, the
target tree, the traps and the gates of the split of 2026-08-20;
`CATALOG.md` section 19 is its record.
