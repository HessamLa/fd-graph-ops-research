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
stage 1 real.

```python
class KnnFodiwalk(Fodiwalk):
    def make_graph(self, data, k=10, **kw):
        return knn_csr(data, k)                # -> a symmetric CSR

    def fit(self, data, epochs=1000, **kw):    # the base class raises
        return self.embed(self.make_graph(data, **kw), epochs=epochs)
```

### A new augmentation policy

`augment_graph` builds `D`, then the planes, then the plan. Override
`_build_D` to change only what enters `D`, and keep the plane and plan
machinery:

```python
class MyPolicyFodiwalk(Fodiwalk):
    def _build_D(self, A):
        D, freq = my_pair_policy(A, self.rng)
        self.stats = {"key": ..., "mn": ..., "cnt": ...}
        self.freq = freq                       # the `freq` plane, (nnz,)
        self.info = {"near_nnz": int(D.nnz)}
        return D
```

`self.freq` must be set when the law reads a `freq` plane. A missing plane
RAISES; it never falls back to another law.

### A different engine

`ForceDirected` owns the epoch loop, the batching, the callbacks and the
`Z` update, and nothing else. Subclass it directly for a force law that is
not walk-based:

```python
from fodiwalk.core import ForceDirected

class MyModel(ForceDirected):
    def augment_graph(self, G, **kw): ...      # -> D
    def forces(self, Z, D, row_start, row_end, **kw): ...   # -> (rows, d)
```

`forces` gets a ROW RANGE, never a set of pairs: the kernel writes
`dZ.at[rows].add(...)`, thus only disjoint rows make the parts additive.

---

## The tree

| path | what it holds |
| --- | --- |
| `fodiwalk.py` | `Fodiwalk`, `Config` |
| `core/` | the engine, the layout, the physics, the asserter |
| `make_graph/` | stage 1, the datasets |
| `augment_graph/` | stage 2, the walks and the policies |
| `misc/` | the update rules, the drop, the measurements |
| `tests/` | contracts, smoke, parity |
| `dev-docs/` | PRD, ORCHESTRATION, CATALOG, BUILD |

`core` imports nothing of the package except `core`. The dependency runs
one way, and a cycle is a defect.

---

## `core/`

### `force_directed.py` -- `ForceDirected`, `Callback_Base`

The epoch loop, the row batching, the callback events, the `Z` update. It
owns NO augmentation and NO force law.

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
names both: a branch in `Fodiwalk._plane` that builds it, and a validator
in `plan_contract.PLANE_CHECKS` that states what the name promises.

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
`sum(axis=1)`. `make_plan` never interprets a plane; `_step` never knows
the law. Do not edit it to change the physics.

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
| `walks.py` | `uniform_walks`, `node2vec_walks`, `make_walker`, `walk_rows`, `walk_pair_stats`, the neighbour merges |
| `pairs.py` | the pair key, `cap_per_node`, `row_cap`, the CSR builds |
| `weights.py` | `flat`, `min_gap`, `mean_gap`, `pmi` -- the walk statistics into `h` |
| `buckets.py` | the stratified budget, undirected and row-wise |
| `far_pairs.py` | the long-range pairs, and the `deg^alpha` draw |
| `landmarks.py` | a real distance for a far pair, from exact BFS |

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

## `misc/`

### `optim.py` -- eight update rules

`plain`, `momentum`, `nesterov`, `adam`, `fa2`, `velocity`, `sgd`, `sqn`.
A rule is pure: `step(Z, dZ, lr, state, epoch, **kw) -> (Z_new, state)`,
with its state in a dict the model keeps.

```python
from fodiwalk.misc import optim

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

`test_contracts.py` holds one test for one contract. `test_smoke.py` holds
the engine and the class, on a 60-node graph. `test_parity.py` reproduces
the recorded runs of the campaign, and `harness.py` runs one scenario end
to end.

**The package must reproduce the numbers of `experiments/fdwalk/`.** A
difference is a defect and not a finding. Run the parity gate before you
trust a change to the physics, the augmentation or the plan.

---

## `dev-docs/`

`PRD.md` what each block must satisfy. `ORCHESTRATION.md` who built it and
in what order. `CATALOG.md` the entities, incrementing -- nothing is ever
removed from it. `BUILD.md` what was built, what it reproduces, and what it
does not do.
