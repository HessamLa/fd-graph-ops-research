# fodiwalk -- catalog of entities

Created 2026-08-19. Companion to [PRD.md](PRD.md), which says what each
block must satisfy, and to [ORCHESTRATION.md](ORCHESTRATION.md), which says
who builds it.

This file catalogs the NAMED entities of the package: the force laws, the
plane names, the weight rules, the pair policies, the update rules, and the
invariants. It continues [the catalog of the experiment](../../experiments/fdwalk/CATALOG.md),
which holds the same entities in their original location.

It is an INCREMENTING document: nothing is removed from it. An update
states its reason, gives the new version, and names the step at which the
update is adopted.

Each entry holds: what the entity is, how it works, a code example where
one applies, and the provenance -- a link to the code.

---

## 0. The map of the package

```
fodiwalk/
  fodiwalk.py            class Fodiwalk, class Config
  core/
    csr.py               row_of, n_rows
    force_directed.py    Callback_Base, class ForceDirected
    sell_c_sigma.py      build_ladder, make_plan, _step, PlanCache
    forces.py            the laws, the plane builders, FORCE_PLANES
    plan_contract.py     check, check_degrees, check_plan          [NEW]
  make_graph/datasets.py read_edges, to_csr, induced, load
  augment_graph/
    walks.py             uniform_walks, node2vec_walks, make_walker,
                         walk_pair_stats, walk_rows, with_*_neighbours
    pairs.py             split_key, to_csr, to_csr_directed,
                         cap_per_node, row_cap
    weights.py           flat, min_gap, mean_gap, pmi
    buckets.py           bucket_sample, budget, row_buckets
    far_pairs.py         degree_table, sample_far_pairs
    landmarks.py         pick, distances, pair_distance
  misc/
    optim.py             the eight update rules, RULES, state_arrays
    drop.py              drop_steady_rate, FallbackKeys
    evaluation.py        link_prediction, hop_sample, task_hop
  tests/                 test_contracts, test_smoke, test_parity, harness
```

`core` imports nothing of the package except `core`. Every other stage may
import `core`. The one exception is documented at its site:
`ForceDirected.set_rule` reaches `misc/optim.py` through a FUNCTION-local
import, thus the module-level graph still runs one way.

---

## 1. Force laws

The law and the plan are ONE contract: `make_plan` takes a SEQUENCE of
planes, and `_step` gives that sequence to the law, which unpacks it
POSITIONALLY. `FORCE_PLANES` is the one statement of which planes a law
reads, and in what order.

| law | planes, in order | attraction | status |
| --- | --- | --- | --- |
| `v1` (`shell_force`) | `(shell_coeff, h)` | `h == 1` only | REMOVED 2026-08-19, see 13 |
| `v2` (`shell_force_v2`) | `(shell_coeff, h)` | `h == 1` only | REMOVED 2026-08-19, see 13 |
| `fdlinear` | `(h, freq)` | `h <= 1` only | live |
| `fdlinear_3plane` | `(shell_coeff, h, freq)` | `h <= 1` only | REMOVED 2026-08-19, see 13 |
| `fdlinear_fused` | `(w,)` | `w < 0` | live |

**Provenance.** [fodiwalk/core/forces.py](../core/forces.py)

### 1.1 `shell_force` (`v1`) -- the law of the package

```
guard = where(h > 1, 0, 1)
Fa    = guard * k1 * shell_coeff * x * exp(-k2 * (h - h_shift))
Fr    = -k3 * h * exp(-k4 * x)
```

`shell_coeff = 1 / |S_h(u)|`, thus at `h = 1` it is `1 / deg(u)`. `k2` is a
dead parameter: `guard` kills every `h > 1`, thus the exponent is one
constant.

### 1.2 `shell_force_v2` (`v2`) -- the off-branch law

`h == 1` keeps `Fa + Fr`. `h > 1` gets `Fr = -k3 * h` with NO decay, thus a
far pair pushes with a constant magnitude at any distance.

### 1.3 `fdlinear` -- the linear law of 2026-08-17

```
h == 1:  Fa = k1 * x           Fr = -kr * exp(sign * k4 * x)
h >= 2:  Fa = 0                Fr = -(h / freq) * exp(sign * k4 * x)
```

`sign = -1` is the default and it decays; `sign = +1` is the literal
reading of the specification, and it diverges. `freq` divides the
repulsion, thus a node that a row reached MANY times is pushed away LESS.

### 1.4 `fdlinear_fused` -- the same law from ONE plane

```
w = -1.0      for h == 1     (a SENTINEL)
w = h / freq  for h >= 2     (always > 0)
w = 0.0       for a pad cell
```

**The sentinel is load-bearing.** A plain `w = h/freq` cannot be decoded:
`h=1, freq=3` and `h=2, freq=6` both give 0.333. The `h == 1` branch reads
NEITHER `h` NOR `freq` -- its coefficient is the constant `kr` -- thus it
needs one flag and no value, and the sign carries the flag at no cost. The
three regions are disjoint: negative, positive, exactly zero.

**Cost of fusing.** `h / freq` becomes a BUILD-time quantity: `k1`, `k4`
and `kr` still sweep for free, but a law of the form `h / freq**beta` could
no longer sweep `beta` without rebuilding `D`.

**Verified identical** on Cora (`nbr_walk/min_gap`, dim 64, 200 epochs,
lr 1.0, seed 42): `||dZ||` 0.5016, accuracy 0.9754, F1 0.9752, AUC 0.9962,
hop R2 0.253. Reproduced by the package on 2026-08-19, both variants
([test_p1](../tests/test_parity.py)).

---

## 2. Planes, and what each name promises

A "plane" is a `(nnz,)` array carrying one number for each stored pair, in
`D`'s OWN pre-split, pre-sort CSR order. `make_plan` slices, reorders and
pads it exactly as it does `D.indices`.

| name | what it is | the promise `plan_contract` asserts |
| --- | --- | --- |
| `h` | the stored weight of the pair | it IS `D.data`, and every value is an integer `>= 1` |
| `shell_coeff` | `1 / \|S_h(u)\|` | REMOVED 2026-08-19 with the laws that read it, see 13 |
| `freq` | how often the row reached the partner | every value `>= 1` |
| `w` | the fused plane | exactly `-1`, or `> 0`; never 0 for a stored pair |

**Why a name and not only a count.** A wrong plane in the `h` position
passes a COUNT test -- two planes for a two-plane law -- and it is still
the defect of 2026-08-18. The value test catches it: only `D.data` itself
is `D.data`. `check("fdlinear", (freq, h), D)` therefore RAISES.

**Provenance.** [fodiwalk/core/plan_contract.py](../core/plan_contract.py)

---

## 3. `plan_contract` -- the asserter [NEW in this package]

**What it is.** The only new module of the refactor. It asserts the
invariants of PRD section 7 at the seam between the augmentation and the
plan. It RAISES `PlaneContractError`; it never warns, and it never falls
back to other physics.

```python
planes = fw._build_planes(D)              # FROM the registry, not an `if`
plan_contract.check(law, planes, D)       # I1, I2, I4
plan_contract.check_degrees(degrees, D)   # I5
plan, inv_deg_ext, stats = make_plan(D, planes, degrees=degrees)
plan_contract.check_plan(plan, len(planes))   # I3
```

| # | invariant | what breaks in silence |
| --- | --- | --- |
| I1 | planes are `(D.nnz,)` and aligned to `D.indices` | wrong physics, no error |
| I2 | the plane COUNT and ORDER match the registry | a law reads one plane as another |
| I3 | a pad cell has every plane at 0 | pad cells contribute force |
| I4 | stored weights are integers, and 1 means adjacency | the whole force vanishes |
| I5 | a row with no `h = 1` entry gets an explicit degree | the row freezes, forever |
| I6 | a chunk is a row range | `dZ` is summed twice or not at all |
| I7 | a non-finite `Z` is RECORDED, never evaluated | a crash inside sklearn, no measurement |

I6 is a property of `ForceDirected.embed` and `Fodiwalk._build_plan`: a
chunk is `[a, b)` and `embed` slices `dZ` by the same range. I7 is a
property of `ForceDirected.embed`, which records `self.diverged`.

**Adopted at:** the package build of 2026-08-19, block A2 of
[ORCHESTRATION.md](ORCHESTRATION.md).

---

## 4. Pair policies -- what enters `D`

| `pairs` | what a row holds | direction |
| --- | --- | --- |
| `walk` | every pair inside the window of a walk, capped at `cap` for each node | undirected |
| `walk_edges` | the same, plus every original edge forced to `h = 1` | undirected |
| `nbr_walk` | EVERY neighbour at `h = 1`, plus every node a walk from the row reached | directed |

| `policy` | the budget |
| --- | --- |
| `cap` | `cap` pairs for each node, plus `n * log10(n)` far pairs |
| `buckets` | every edge at `h = 1`, plus `n * log10(n)` pairs at 50% `h=2`, 25% `h=3`, 25% `h>=4`, and NO far pairs |

**The budget contract.** `walk_rows` has NO prune: its bound IS
`n_walks * walk_len` for each row, thus the walk budget of the caller is
the memory bound. `walk_pair_stats` DOES prune, and its prune is an
approximation that `prunes > 0` reports. The two are not unified.

**`--pairs ball` is NOT in this package.** It needs `k_hop_ball`, which the
PRD's module list for `augment_graph/` does not carry. The control stays in
`experiments/fdwalk/bench_fdwalk.py`.

> **SUPERSEDED 2026-08-19. See section 16.** `ball` and `sampled` are now
> policies of the package, in `augment_graph/ball.py`. The paragraph above
> records the scope of the refactor as the PRD wrote it, and it is kept
> for that reason.

**Provenance.** [fodiwalk/augment_graph/walks.py](../augment_graph/walks.py),
[buckets.py](../augment_graph/buckets.py),
[fodiwalk.py](../fodiwalk.py) (`_build_D_nbr_walk`, `_build_D_walk`,
`_build_D_buckets`)

---

## 5. Caps

### 5.1 `cap_per_node(key, cnt, n, m)` -- the undirected cap

A pair survives when EITHER endpoint keeps it, thus a row can hold MORE
than `m` and a hub collects many. This is the symmetrisation of a
neighbour graph in LargeVis and UMAP: a low-degree node keeps its
partners, although a hub does not choose it.

### 5.2 `row_cap(stats, n, m)` -- the directed cap

Row `u` keeps its own best `m`, thus EVERY row has the same width. The
payoff is in the PLAN and not only in `D`: no hub split (`n_split -> 0`),
little padding, balanced batches, and an exact bound of `n * m` cells.

It runs on the output of `walk_rows`, which is already directed and
row-disjoint. `_pairs_of` cannot be used for it: that function collapses
the direction at the source with the key `min*n + max`.

**Provenance.** [fodiwalk/augment_graph/pairs.py](../augment_graph/pairs.py)

---

## 6. Weight rules -- the walk statistics into `h`

| rule | `h` |
| --- | --- |
| `flat` | 1.0 for every pair |
| `min_gap` | the smallest step gap a walk gave |
| `mean_gap` | the mean gap, rounded |
| `pmi` | the pointwise mutual information, mapped into `[1, window]` |

**`mean_gap` and `pmi` are BROKEN today, and section 14 holds the
evidence.** They give integers, thus they satisfy the rule below, and they
still leave most rows with no entry at `h = 1`, thus those rows freeze.

**Every rule gives an INTEGER in `[1, window]`, and that is a requirement
of the force law and not a decoration.** `shell_counts` makes one column
for each DISTINCT value of `D.data`, and `degrees_from_D` counts the
entries that are exactly 1. A float weight gives almost every pair its own
value, thus the table has millions of columns, the degree becomes 0, and
the whole force vanishes: AUC fell to 0.55 with `||dZ|| = 0.000`.

**Provenance.** [fodiwalk/augment_graph/weights.py](../augment_graph/weights.py)

---

## 7. Walks

### 7.1 `uniform_walks` -- the first-order walk

The walk of DeepWalk, and the walk of node2vec at `p = q = 1`. All the
walks take one step at the same time.

### 7.2 `node2vec_walks` -- the second-order walk, by rejection sampling

The transition from `v`, having come from `t`, to a candidate `x` has the
unnormalised weight `1/p` when `x == t`, `1` when `x` is adjacent to `t`,
and `1/q` otherwise.

**Why rejection sampling and not alias tables.** An alias table for each
EDGE costs `sum(deg^2)`. com_youtube holds a node of degree 28,754, thus
that node alone contributes 827 million entries.

**The gate is DISTRIBUTIONAL and not bit-exact.** At `p = q = 1` every
weight is 1 and the accept test never fails, thus the walk has the same
DISTRIBUTION as `uniform_walks`. It is NOT the same array: the rejection
loop draws one more number for each accept test.

### 7.3 `make_walker(A, n, p, q)` -- the choice

At `p = q = 1` it gives `uniform_walks` ITSELF, and not the rejection
sampler with trivial weights. Every result recorded before 2026-08-18 used
the uniform walk, thus only this rule keeps a default run bit-exact.

**UPDATE 2026-08-19, and it changes no number.** The binding is
`functools.partial(uniform_walks, A)` and no longer a lambda of the same
call, thus the identity of the walk stays visible to a test:

```python
assert make_walker(A, n, 1.0, 1.0).func is uniform_walks
```

`partial(uniform_walks, A)(starts, L, rng)` IS
`uniform_walks(A, starts, L, rng)`. **Reason for the update:** PRD B6.1
asks for an assertion of identity, and a lambda cannot carry one.
**Adopted at:** the package build of 2026-08-19, block A5.

**Provenance.** [fodiwalk/augment_graph/walks.py](../augment_graph/walks.py)

---

## 8. Long-range pairs

### 8.1 `degree_table(A, alpha)` -- the biased draw

The cumulative weights `deg^alpha`. `alpha = 0` gives a uniform draw and
returns `None`, thus the caller keeps the fast path. `alpha = 0.75` is the
negative-sample distribution of word2vec and node2vec: on com_youtube a
node of degree 28,754 is then 636 times more likely than a node of the
average degree.

### 8.2 `sample_far_pairs(n, count, ball, rng, cum=None, directed=False)`

`count` pairs that `ball` does not hold. The test needs no search: a sorted
array of the composite keys and one `searchsorted` for a whole batch.

**UPDATE 2026-08-19 -- the parameter `directed`.** The rejection reads the
keys of `ball` as they are STORED. A symmetric `ball` holds both
directions, thus the old behaviour is already correct there. A DIRECTED
`ball` -- the `D` of `walk_rows` -- holds `(u, v)` and not `(v, u)`, thus a
pair the rows already hold as `(v, u)` passed the test, the CSR build
SUMMED the two entries, and the weight became `far_weight + the walk gap`.
The histogram showed entries at 101..119 on 2026-08-17, and
`bench_fdwalk.py` repaired it with a filter AFTER the sampler.

`directed=True` rejects both directions inside the sampler. **The default
stays `False`, and `Fodiwalk` does not use the flag.** A changed rejection
changes how many pairs one batch accepts, thus it changes the draw of the
next batch, thus it changes the far set of every recorded run.
`Fodiwalk._build_D_nbr_walk` therefore keeps the FILTER of
`bench_fdwalk.py`: sample with the default, then drop any pair that `near`
holds in either direction. The two paths give the same PROPERTY and not the
same pairs, and parity decides which one the augmentation takes.

The flag exists for a caller that wants the rejection inside the sampler --
a new run, where no recorded number is at stake.

**Adopted at:** the package build of 2026-08-19, block A4, to satisfy PRD
B6 criterion 6 without moving a recorded number.

### 8.3 `landmarks` -- a real distance for a far pair

`d(u, v) <= min over L of (d(u, L) + d(L, v))`, from `count` exact BFS. The
estimate is an UPPER bound, thus it has the same character as the walk gap,
which is also an upper bound. Without it every pair beyond the window takes
ONE constant, and the force law cannot separate a pair at 6 hops from a
pair at 19.

**Provenance.** [far_pairs.py](../augment_graph/far_pairs.py),
[landmarks.py](../augment_graph/landmarks.py)

---

## 9. Update rules (axis C)

`ForceDirected.updateZ` is the seam. `plain` IS the one line the base class
had; every other rule replaces it. A rule is
`step(Z, dZ, lr, state, epoch, **kw) -> (Z_new, state)`, pure in `Z` and
`dZ`, with its state in a dict the model keeps.

| rule | state arrays of the shape of `Z` |
| --- | --- |
| `plain` | 0 |
| `sgd` | 0 |
| `momentum`, `nesterov`, `fa2`, `velocity` | 1 |
| `adam` | 2 |
| `sqn` | `2 * memory + 2`, thus 8 at the default |

`dZ` here is a FORCE and not the gradient of a loss, thus a large `dZ`
means "this node must move far", and a rule that divides by the size of
`dZ` throws that information away. `fa2` keeps the size and `adam` does
not.

```python
fw = Fodiwalk(n_dim=64, optim="velocity", eta=0.3, lr=1.0)
fw.set_rule("adam")            # or change it later
```

**Provenance.** [fodiwalk/misc/optim.py](../misc/optim.py),
[core/force_directed.py](../core/force_directed.py)

---

## 10. `class Fodiwalk` -- the composition

```python
from fodiwalk import Fodiwalk
from fodiwalk.make_graph import load

A, n = load("cora")
fw = Fodiwalk(n_dim=64, seed=42, lr=1.0, optim="plain",
              pairs="nbr_walk", weight="min_gap", force="fdlinear")
fw.embed(A, epochs=200)
Z = fw.get_embeddings()
```

| method | stage |
| --- | --- |
| `make_graph(data)` | 1. A STUB: it returns what it is given |
| `graph_walk(A, n, rng)` | the walks, and the pair statistics |
| `augment_graph(G)` | 2. `D`, then the planes and the batch plan |
| `embed(G, epochs, ...)` | 3. Inherited from `ForceDirected`, unchanged |
| `fit(data)` | raises `NotImplementedError` |
| `set_D(D, freq=...)` | hand over an augmented `D` and skip the walks |

**`Config`** carries every knob, and the names are the flags of
`bench_fdwalk.py` with the dashes turned into underscores. An unknown name
raises `TypeError`. `force` defaults to `fdlinear` since 2026-08-19; it
defaulted to `v1` before, and `v1` no longer exists (see 13).

**The rule this class exists to keep:** the plane list comes from
`FORCE_PLANES`, and `_plane` RAISES when the augmentation did not build
what the law reads.

**Provenance.** [fodiwalk/fodiwalk.py](../fodiwalk.py)

---

## 11. Measurements

| entity | what it answers |
| --- | --- |
| `link_prediction(Z, A, n, max_pairs, rng, seed)` | does the geometry hold the ADJACENCY? A random forest on the Hadamard product of the two vectors of a pair |
| `hop_sample(A, n, rng, n_sources, n_pairs, gap_csr)` | the exact hop distance of a sample of pairs, in blocks of BFS sources, plus the H2 comparison of the walk gap against the truth |
| `task_hop(Z, u, v, d, seed, feature)` | does the geometry hold the DISTANCE? `distance` gives ONE feature and `vector` gives `n_dim`, thus the two rows must not be mixed |
| ~~`hop_from_D(D, rng, sentinel, max_pairs, min_hop)`~~ | the pairs `D` STORES, the protocol of `fodined/modular.py`. Added 2026-08-19 (section 16), REMOVED 2026-08-20 (section 18): it needs `D.data` to be a measured distance, and every policy of this package stores a walk gap |

**Provenance.** [fodiwalk/misc/evaluation.py](../misc/evaluation.py)

---

## 12. The parity gate

[test_parity.py](../tests/test_parity.py) reproduces the six recorded runs
of PRD section 8. [harness.py](../tests/harness.py) is the runner, and it
keeps the ORDER of the generator, which is itself part of the parity: the
script makes ONE `default_rng(seed)` and passes it to the augmentation,
then to the link prediction, then to the hop sample.

**A CORRECTION TO THE PRD, and it is a label and not a number.** PRD
section 8 writes P2 as `walk/min_gap/v1/plain`. The recorded run behind
`auc = 0.9986, r2_dist = 0.523` is `nbr_walk/min_gap/fdlinear/plain` at
`k4 = 1.0`, from `run_lrladder.sh`
(`results/lrladder/const_plain_lr0.999.log`). No `walk/v1` row carries
those numbers. The test uses the configuration that produced them.

**Status, 2026-08-19: all six scenarios reproduce, and so do the three
`big` ones.** P1, P2 and P5 are exact. P3 gives `0.6357 +-0.0234` against
`0.6357 +-0.024` and P4 gives `0.4518 +-0.0166` against `0.452 +-0.016`.
P6 reproduces every one of the eight rules to three decimals. At 1.13M
nodes the whole augmentation and its plan reproduce too, `pad_frac` to
every digit. See [BUILD.md](BUILD.md) section 3.

---

## 13. UPDATE 2026-08-19 -- the shell-averaged laws are removed

**Reason.** The project settled on the linear laws. `fdlinear` and
`fdlinear_fused` carry every result the campaign now builds on, and the
three laws that read `shell_coeff` were not used by any of them. A law that
nothing runs is a law that nothing tests, and it costs a plane builder, a
`(n, m)` shell table, and three registry entries that every reader of
`forces.py` has to hold in the head.

**Adopted at:** the package build of 2026-08-19, after the parity gate of
section 12 passed. The gate is the reason the order matters: the removal
comes AFTER the proof that the package reproduces the campaign, thus the
proof stands on the code as it was, and this entry records exactly what
left it.

**What was removed from `fodiwalk`:**

| entity | what it was |
| --- | --- |
| `shell_coeff` | the plane `1 / \|S_h(u)\|`, the size of the hop shell of the source node |
| `shell_counts` | the `(n, distinct hop values)` table the plane is built from |
| `shell_coeff_data` | the plane builder, in `D`'s own CSR order |
| `shell_force` (`v1`) | the law of the origin package |
| `shell_force_v2` (`v2`) | the off-branch law, with a constant repulsion at `h > 1` |
| `fdlinear_3plane` | the first form of `fdlinear`, which bound `shell_coeff` and never read it |
| `k2`, `k3`, `h_shift` | the parameters that ONLY those laws read |
| `plan_contract._check_shell_coeff` | the value test of the removed plane |

**What did NOT change.** No surviving function body. `degrees_from_D`,
`fdlinear`, `fdlinear_fused` and `fuse` are character for character what
they were, which an AST comparison against the frozen sources asserts. The
`FORCE_PLANES` registry, `plan_contract` and the `_plane` builder of
`Fodiwalk` keep their shape -- the registry simply holds two rows now, and
that is the property the design was for: a law leaves by deleting one row,
and no `if` chain anywhere needs an edit.

**Nothing is lost.** `experiments/fdwalk/` is frozen and still holds every
removed law, the plane builder, and every result they produced.
`fodined/embedding/shell_force.py` holds the origin form.

**The cost, and it is real: PRD parity scenario P6 is retired.** The
recorded run of 2026-08-18 -- the eight optimizers on Cora at 60 epochs,
dim 32, lr 1.0 -- used `v1`, thus its numbers cannot be reproduced by a
package that has no `v1`. They were reproduced, to three decimals for all
eight rules, on 2026-08-19 BEFORE the removal; [BUILD.md](BUILD.md)
section 3 holds that table and it stands as the record. What replaces the
test is a baseline of the same eight rules under `fdlinear`, measured on
2026-08-19 and pinned as `FDLINEAR_ROSTER`. It is NOT parity with the
campaign, and the test says so in its own docstring.
[BUILD.md](BUILD.md) section 7 holds both columns side by side: the shape
of the roster survived the change of the law, and the leading group rose
from 0.373 to 0.609.

**Provenance.** [forces.py](../core/forces.py),
[plan_contract.py](../core/plan_contract.py), [fodiwalk.py](../fodiwalk.py)

---

## 14. FINDING 2026-08-19 -- `mean_gap` and `pmi` freeze most rows

Found by `plan_contract.check_degrees` when a new weight rule was written
for [README.md](../README.md). It is a defect of the weight rules of
`experiments/fdwalk/weights.py`, which this package moved verbatim, thus it
is present in both.

**The claim on record was wrong, or at least incomplete.** `weights.py`
says the first version of `mean_gap` and `pmi` gave FLOATS, that both
collapsed to AUC 0.55 and 0.70 with `||dZ||` at 0.000, and that the
`np.rint` in each rule is the repair. The `np.rint` repaired one cause of
two. The collapse survives it.

**The surviving cause is I5.** Attraction exists at `h = 1` only.
`degrees_from_D` counts the entries that are exactly 1, thus a row with
none gets a degree of 0, `inv_deg_ext` turns that into 0.0, and EVERY force
of the row -- the repulsion too -- is multiplied by zero. The row never
moves.

Measured on Cora, `pairs=walk`, 10 walks x 20 steps, window 5, cap 16,
seed 42, with the CURRENT integer rules:

| rule | `h` range | rows with no `h = 1` |
| --- | --- | --- |
| `flat` | 1..1 | 0 of 2708 |
| `min_gap` | 1..5 | 0 of 2708 |
| `mean_gap` | 1..5 | **2574 of 2708** |
| `pmi` | 1..5 | **1181 of 2708** |

`min_gap` is safe because `mn` is a MINIMUM: any pair a walk crossed in one
step gets exactly 1. A mean or a rescaled statistic almost never lands on
1 exactly.

**The recorded runs agree.** `results/g1_cora_walk_edges_mean_gap_plain_s42.log`
has integer weights -- its histogram is `{1: 148, 2: 9854, 3: 25480,
4: 19868, 5: 1400, 100: 18590}` -- thus it is a post-`np.rint` run, and only
148 of about 75,000 entries are at `h = 1`. It ends with
`||dZ|| = 0.000926`, `auc = 0.5603`, `r2_dist = 0.019`. That is a frozen
layout and not a weak one.

**The repair, and it needs no new code.** Pass an explicit degree:
`Fodiwalk(weight="mean_gap", deg_source="A")` gives the force law the true
degree of `A` in place of the `h = 1` count of `D`. The option exists
already, for the `edge_rule="low_deg"` case, which is the same defect from
another direction.

**Status.** Not repaired here, and it must not be repaired quietly: a
`mean_gap` or `pmi` run now RAISES at the seam with the row count in the
message, thus the choice -- an explicit degree, or a rule that emits 1 --
is the caller's and it is made in the open. `experiments/fdwalk/` is frozen
and still runs the defect silently.

**Provenance.** [weights.py](../augment_graph/weights.py),
[plan_contract.check_degrees](../core/plan_contract.py)

---

## 15. FINDING 2026-08-19 -- a split hub row makes the GPU run non-reproducible

Found while `experiments/fodiwalk/bench_fodiwalk.py` was checked against a
recorded run of the campaign. It bounds what the word "parity" can mean for
a given configuration, thus it belongs next to section 12.

**The measurement.** Cora, `walk/min_gap/fdlinear`, dim 128, 2000 epochs,
lr 1.0, seed 42. The same code, the same seed, three runs:

| backend | run 1 | run 2 | run 3 |
| --- | --- | --- | --- |
| GPU | `dz = 0.117347` | `0.117157` | `0.118518` |
| CPU (400 epochs) | `dz = 0.129451305` | `0.129451305` | `0.129451305` |

The CPU repeats are identical to nine decimals, and `sum(abs(Z))` is
identical too. The GPU repeats spread about 1.2%.

**The cause.** `_step` ends with `dZ.at[rows].add(F, mode="drop")`. That is
a SCATTER-add. A hub row wider than `k_max` is split into several virtual
rows that all carry the SAME owner id, thus several writes land on one row
of `dZ` and the order of the floating-point accumulation is not fixed on a
GPU. The configuration above has `n_split = 3`. Over 2000 epochs the
difference compounds, because a force-directed layout is a chaotic
dynamical system.

**Why the parity gate is still exact.** P1 and P2 reproduce `||dZ||` to
six decimals, and they can: their plan has `n_split = 0`. The `nbr_walk`
policy holds every row under `k_max`, thus no row is split, thus no two
writes contend and the scatter is deterministic. The exactness of P1 and P2
is therefore a property of those configurations and NOT a promise about
every configuration.

**What is stable, and what is not.**

| quantity | behaviour with `n_split > 0` |
| --- | --- |
| `D`, the plan, `pad_frac` | EXACT. The augmentation is pure NumPy. |
| hop R2, AUC, accuracy | stable to the third decimal |
| `\|\|dZ\|\|` | about 1% between runs |

`||dZ||` is the last epoch's mean row norm, thus it reads the state of one
step and it carries the whole compounded difference. The scores read `Z`,
which is an accumulation of 2000 steps, thus the differences average out.
Against the recorded run of `results/fdlinear/fdl_cora_walk_lr1.0_s42.log`
the package gives `r2_dist = 0.689` against `0.689`, `mae_dist = 0.829`
against `0.830`, `h2_exact = 0.965` against `0.965`, and
`dz = 0.118` against `0.1144`.

**How to read a `||dZ||` comparison.** On a plan with `n_split = 0` it is
exact and a difference is a defect. On a plan with `n_split > 0` a
difference below about 2% is the backend, and only a larger one is
evidence. `fw.plan_stats["n_split"]` says which case a run is in, and a
report should quote it beside `||dZ||`.

**Not a defect to repair.** The determinism is available -- one scatter per
row, or a CPU run -- and both cost more than the number is worth. The
alternative is to state the property, which this entry does.

**Provenance.** [sell_c_sigma._step](../core/sell_c_sigma.py),
[make_plan](../core/sell_c_sigma.py) (the hub split, `n_split`)

---

## 16. UPDATE 2026-08-19 -- the exact-distance policies, from `fodined/modular.py`

> **SUPERSEDED 2026-08-20. The code of this section is REMOVED; see
> section 18.** `augment_graph/ball.py`, the `ball` and `sampled` policies,
> `hop_from_D` and `tests/test_ball.py` are gone from the package. Section
> 16 is kept because the CATALOG is an incrementing document: it records
> what was built, what it proved (the augmentation and the plan were
> cell-identical to `fodined/modular.py`), and the two defects it found in
> the reference, which are still true of `fodined/`. Section 17 holds the
> walk answer and section 18 holds the removal.

**Reason for the update.** The request: "we want the results in fodiwalk to
be similar to `fodined/modular.py`; make sure the same functionality is
implemented in fodiwalk". `modular.py` is the reference pipeline of this
repository. An audit of the two found TWO things missing -- the whole
exact-distance augmentation stage, and the two tree datasets with the
truncation a tree needs -- and the rest already shared.

**Adopted at.** The fodiwalk package after the removal of the
shell-averaged laws (section 13). The parity gate of section 12 was re-run
and it is unchanged.

### 16.1 What was already the same, and needed no work

| stage | the shared code |
| --- | --- |
| link prediction | `misc/evaluation.py` holds `sample_positives`, `sample_negatives`, `edge_features` (the Hadamard product), `classify` and `link_prediction`, verbatim from `fodined/link_prediction.py` |
| the far pairs | `augment_graph/far_pairs.py` holds `degree_table`, `_draw` and `sample_far_pairs`, verbatim from `fodined/graph_augmentation.py` |
| the engine | `core/sell_c_sigma.py` -- `make_plan` and `_step` are the kernel `modular.py` imports |
| the regularizer | `misc/drop.py`, at `random_drop_rate = 0.5` and `random_rows` |
| the loader | `make_graph/datasets.py`, with the BFS-ball truncation. The two TREE datasets were the exception; see 16.4 |
| the hop features | `task_hop(feature="vector")` IS `modular.py`'s `|Z[u] - Z[v]|` |
| the constants | `n_dim = 128`, `epochs = 2000`, `k1 = 0.999`, `k4 = 0.01`, `far_weight = 100`, `far = n log10 n`, `b_cells = 16384`, `k_max = 256`, `ladder_base = 1.5` |

### 16.2 What was missing: `augment_graph/ball.py`

The whole `AUGMENT` stage of `modular.py`. A verbatim move of
`fodined/graph_augmentation.py`, less the three functions `far_pairs.py`
already holds.

| entity | what it does |
| --- | --- |
| `k_hop_ball(A, k, block)` | every pair within `k` hops, with its TRUE distance as the value, from `k` sparse products. It never computes a distance it does not keep, thus the cost follows the SIZE OF THE RESULT |
| `augment_k_hop(A, n, k, far_count, far_weight, rng, block, cum)` | the ball, plus `far_count` random pairs beyond it at `far_weight`. `pairs = "ball"` |
| `augment(A, n, count, unreachable_w, edge_set, rng, chunk)` | the older policy: draw pairs, then one BFS for each distinct source. `pairs = "sampled"` |
| `hop_distances`, `replace_unreachable`, `augmented_csr`, `edge_set_of`, `sample_unconnected_pairs`, `source_block_size` | the steps of `augment`, each usable alone |

`cum` is the one added argument. Its default `None` draws the far pairs
uniformly, which is what `modular.py` does.

**Two families of `h` now, and the difference decides what a number
means.** A walk policy gives a pair its walk GAP, an upper bound. An exact
policy gives it the measured distance. H2 -- the comparison of the gap
against the truth -- therefore does not exist for `ball` or `sampled`, and
the bench prints `h2_exact=na`.

### 16.3 The `freq` plane of an exact policy is 1

`fdlinear` reads `(h, freq)`, and `freq` counts how many times the walks
saw a pair. An exact policy takes no walk. It sets the plane to 1 for every
pair, thus

    h >= 2:   coeff = h / freq = h

and the repulsion of a pair grows with its TRUE distance. This is a
physical statement, not a filler: a walk policy divides by the count
instead, which damps a pair the walks met often. The same `D` under the two
planes is two different laws, and the plane is what makes the difference
explicit and checkable (`plan_contract` asserts `freq >= 1`).

`degrees_from_D` needs no help here. The ball stores EVERY 1-hop neighbour
at `h = 1`, thus the count of `h = 1` entries of a row IS the degree of
that node in `A`, and I5 holds by construction. A test asserts it.

### 16.4 The two tree datasets, and the truncation a tree needs

`modular.py` reads `wordnet` (the hypernym edges of the WordNet 3.0 nouns)
and `ncbi_taxonomy` (`nodes.dmp`, one parent for each node), and
`make_graph/datasets.py` did not. Both readers moved in, with `remap`,
`subtree` and the `TREE` list.

**The truncation is not one rule.** Both wrong choices give a perfect score
on an EMPTY problem, and they fail in mirror image:

| the graph | the wrong choice | what it returns |
| --- | --- | --- |
| a tree | a BFS ball | a STAR of depth 1 from an NCBI hub, thus every non-edge pair is 2 hops apart |
| not a tree | a subtree walk | roadNet-CA read as "parent -> child": 500,000 nodes with 504 edges between them |

`load` therefore branches on `name in TREE`. Measured after the move, at
`max_nodes = 20000`: wordnet gives 20,000 nodes at an average degree of
1.98 and hop distances up to 12, and ncbi_taxonomy gives 17,672 nodes at
2.00 and up to 13. A star would give 2.

#### FINDING and REPAIR -- `subtree` returns a node two times on a DAG

`modular.py` collects the descendants of the root with NO `seen` set:

```python
for _u in frontier:
    for _k in _child_of[_lo[_u]:_hi[_u]]:
        out.append(int(_k))          # no test for a repeat
```

WordNet is a DAG and not a tree -- a synset may name two hypernyms -- thus
a node below two kept parents is collected TWO times. `induced` then writes
`new_id[keep] = arange(keep.size)`, the last write wins, and every other
copy becomes a row with NO edge.

Measured on WordNet at `cap = 20000`: **20,000 entries for 19,581 distinct
nodes**, thus 419 isolated rows and **420 connected components** in
something called a subtree. An isolated row has no `h = 1` entry, and the
evaluation reads its distance to every other node as unreachable.

NCBI is a true tree -- `nodes.dmp` gives one parent for each node -- thus it
never met the defect. That is why it stayed hidden.

**Repaired in `fodiwalk`, and this is a DIVERGENCE from the reference.**
`fodiwalk.make_graph.datasets.subtree` carries a `seen` set, thus it returns
`cap` DISTINCT nodes. After the repair WordNet gives 20,000 nodes, ONE
component and a minimum degree of 1. `fodined/` is not changed -- it is
outside the scope of this work -- thus a WordNet number of `modular.py` and
one of `fodiwalk` are not comparable, and this is the reason.

`test_datasets.py::test_a_tree_dataset_is_one_component` is the gate, and
`test_subtree_of_a_dag_returns_no_duplicate` is the four-node case of the
same defect.

**Provenance.** [make_graph/datasets.py](../make_graph/datasets.py)
(`_edges_wordnet`, `_edges_ncbi`, `remap`, `subtree`, `TREE`),
[tests/test_datasets.py](../tests/test_datasets.py)

### 16.5 `hop_from_D` -- the in-sample hop protocol of `modular.py`

`modular.py` scores the hop regression on the pairs `D` STORES, and not on
a fresh BFS sample: mask `(u < v) & (D.data != sentinel) & (D.data > 1)`,
capped at 2000 pairs. `misc/evaluation.hop_from_D` is that mask, and
`bench_fodiwalk.py --hop-protocol D` selects it.

**It measures a different thing from `hop_sample`, and the difference is
not small.** These are the pairs the embedding was BUILT from. `hop_sample`
draws random pairs, most of which `D` never held. The `hop_sample` number
is the harder one, and it stays the default.

### 16.6 FINDING -- the reference protocol is EMPTY at `K_HOP = 2`

`modular.py` runs at `K_HOP = 2`. `D.data` is then `{1, 2, far_weight}`;
the mask drops the sentinel and drops the ones, thus EVERY target is 2.
A constant target gives `R2` no meaning. The recorded run says so:

```
[modular] hop-distance regression on 2000 unique pairs (hops 2..2)
           model      MAE      MRE     RMSE       R2    exact
   mean baseline    0.000    0.000    0.000    1.000   100.0%
   random forest    0.000    0.000    0.000    1.000   100.0%
             MLP    1.647    0.823    1.654    0.000     0.0%
```

`R2 = 1.000` for the baseline is not a result. `modular.py`'s own comment
names the cause ("With K_HOP = 3 the target has TWO values") and the
consequence at `k = 2` is stronger than the comment says.

Not repaired -- `fodined/` is outside the scope of this work. `hop_from_D`
carries the warning in its docstring, and `bench_fodiwalk.py` prints it
when the target is constant.

### 16.7 The measured result

Cora, `n_dim = 128`, 2000 epochs, `lr = 1.0`, seed 42, `k = 2`.

| quantity | `modular.py` (`shell_force`) | `fodiwalk` (`pairs=ball`, `fdlinear`) |
| --- | --- | --- |
| ball | 96,888 `{1: 10556, 2: 86332}` | 96,888 `{1: 10556, 2: 86332}` |
| far pairs | 9,295 | 9,295 |
| `D.nnz` | 115,478 | 115,478 |
| plan | cells 115478, n_virtual 2711, n_split 3, rungs 13, pad_frac 0.16212823694158449 | IDENTICAL, to the last digit |
| accuracy | 0.9777 | 0.9740 |
| F1 | 0.9777 | 0.9739 |
| AUC | 0.9953 | 0.9932 |
| embed | 8.9 s | 8.4 s |

The augmentation and the layout are bit-identical. The scores differ by
0.4 point of accuracy and 0.2 point of AUC, and the FORCE LAW is the only
remaining difference -- `shell_force` against `fdlinear`, by the decision
of section 13.

### 16.8 The proof

[experiments/fodiwalk/verify_modular.py](../../experiments/fodiwalk/verify_modular.py)
drives `fodiwalk` and `fodined` side by side on the same graph, seed and
generator, and it compares the arrays cell by cell. Eleven checks, and it
exits non-zero on a failure:

```
[1] k_hop_ball(A, 2) is cell-identical                  PASS   nnz 96888
[2] augment_k_hop D is cell-identical                   PASS   nnz 115478, far 9295
[3] augment (sampled pairs) D is cell-identical         PASS   nnz 14556
[4] sample_far_pairs and degree_table are identical     PASS   9295 pairs
[5] link_prediction gives the identical scores          PASS
[6] edge_features is the same Hadamard product          PASS
[7a] Fodiwalk(pairs='ball').D == modular.py's D         PASS   nnz 115478
[7b] the plan matches the plan modular.py printed       PASS
[8] degrees_from_D(D) == the true degree of A           PASS
[9] the freq plane is 1 for every pair, thus h/freq = h PASS
[10] every shared constant has the modular.py value     PASS
```

[test_ball.py](../tests/test_ball.py) holds the fifteen property tests:
the ball against a full BFS, the block size, the symmetry, I4, I5, the
plane contract, and the `hop_from_D` mask.

**Provenance.** [augment_graph/ball.py](../augment_graph/ball.py),
[fodiwalk.py](../fodiwalk.py) (`_build_D_ball`, `_build_D_sampled`),
[misc/evaluation.py](../misc/evaluation.py) (`hop_from_D`),
[tests/test_ball.py](../tests/test_ball.py)

---

## 17. UPDATE 2026-08-20 -- the walk policies against `modular.py`

**Reason for the update.** On request: "you may avoid k-ball augmentation,
only focus on walk-based augmentations, verify again". Section 16 answered
"is the same functionality implemented" by matching `modular.py`'s own
policy cell for cell. This section answers the harder question that the
narrowed scope asks: **does a WALK policy reach the result of a measured
2-hop ball?**

**Adopted at.** The fodiwalk package after section 16. No package code
changed for this section -- the walk policies were already there. The
parity gate of section 12 is untouched and it was not re-run, because
nothing it covers moved.

### 17.1 What a walk policy shares with `modular.py`, and what it does not

| | |
| --- | --- |
| shared | the LONG-RANGE term. `n log10(n)` pairs the near policy did not find, drawn by `sample_far_pairs` at `far_weight = 100`. `modular.py`'s own code, and the `walk` policies call it |
| shared | the engine, the regularizer, the loader, the link prediction, and every constant |
| NOT shared | the NEAR term. `modular.py` MEASURES the distance of every pair within 2 hops. A walk takes the walk GAP, an upper bound, over the pairs the walks reached |

The two `D` matrices are therefore not the same matrix, and they are not
meant to be. `verify_modular.py` asserts that everything AROUND the near
term is the same code, and that the walk `D` keeps every contract the law
needs (I4, I5, the plane contract).

### 17.2 The near term, and the one number that separates the policies

`modular.py` stores EVERY edge of `A` at `h = 1`, because it measures. A
walk MAY miss the edge of a low-degree node. Cora, 5,278 undirected edges:

| policy | `D.nnz` | `h = 1` cells | edges of `A` at `h = 1` |
| --- | --- | --- | --- |
| `walk` (the baseline) | 75,334 | 10,550 | 5,275 = 99.9% |
| `walk` at `cap = 32` | 127,092 | 10,556 | 5,278 = 100% |
| `walk_edges` | 75,340 | 10,556 | 5,278 = 100% |
| `nbr_walk` | 209,542 | 10,556 | 5,278 = 100% |

`walk_edges` is the cheap repair: it forces the missing edges in at
`h = 1`, and it costs SIX cells on Cora (75,334 -> 75,340).

### 17.3 The measured result -- a walk policy BEATS the ball

Cora, `n_dim = 128`, 2000 epochs, `lr = 1.0`, seed 42, `fdlinear`, `plain`.

| policy | `D.nnz` | accuracy | F1 | AUC | hop R2 dist | hop R2 vec | `||dZ||` | H2 exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fodined/modular.py` (2-hop ball, `shell_force`) | 115,478 | 0.9777 | 0.9777 | 0.9953 | -- | -- | 0.1462 | -- |
| `walk` (the baseline) | 75,334 | 0.9702 | 0.9702 | 0.9956 | 0.648 | 0.766 | 0.0153 | 98.1% |
| `walk`, `cap = 32` | 127,092 | 0.9702 | 0.9703 | 0.9950 | 0.659 | 0.766 | 0.0154 | 94.2% |
| `walk_edges` | 75,340 | 0.9706 | 0.9706 | 0.9956 | 0.650 | 0.753 | 0.0153 | 98.1% |
| **`nbr_walk`** | 209,542 | **0.9801** | **0.9800** | **0.9982** | **0.676** | **0.824** | 0.1727 | 7.3% |

**`nbr_walk` exceeds `modular.py` on every comparable number**: accuracy
0.9801 against 0.9777, and AUC 0.9982 against 0.9953. It is a walk policy,
thus the narrowed scope costs nothing on Cora -- it gains.

The reference row has no hop R2. Its own protocol is empty at `K_HOP = 2`;
see section 16.6.

### 17.4 FINDING -- a TIGHT walk gap is not what makes the embedding good

The H2 column reads against intuition and it is the most useful line of the
table. H2 is the walk gap against the true hop distance, on the pairs `D`
stores. `walk` is exact on 98.1% of them. `nbr_walk` is exact on **7.3%**.

`nbr_walk` stores `h` = the FIRST STEP that reached a node, directed, with
no window and no cap. A walk of ten steps that wanders reports `h = 9` for
a node two hops away, thus the gap is a loose upper bound almost every
time. And that policy wins.

The reading: the layout does not need `h` to be the distance. It needs the
ORDER of `h` to rank the partners of a row, and it needs enough partners --
`nbr_walk` stores 77.4 for each node against 27.8 for `walk`. A policy that
buys accuracy in `h` by keeping fewer pairs (`walk` at `cap = 16`, H2
98.1%) loses to one that keeps many loose ones.

Do not read H2 as a quality score. It measures what it says -- how tight
the bound is -- and that quantity does not order the policies by result.
`cap = 32` is the control: it raises the pair count on the same policy, H2
falls from 98.1% to 94.2%, and the hop R2 RISES from 0.648 to 0.659.

### 17.5 The state of `ball` and `sampled`

They stay in the package. They are tested (`tests/test_ball.py`, 15 tests)
and they are cell-identical to `fodined/graph_augmentation.py`; check 10 of
`verify_modular.py` re-asserts that in one line. They are no longer the
path a comparison against `modular.py` takes, and no walk result depends on
them.

The two tree datasets of section 16.4 stay too, with the DAG repair. They
are loader work, and they are policy-independent.

### 17.6 The proof

[experiments/fodiwalk/verify_modular.py](../../experiments/fodiwalk/verify_modular.py),
rewritten for the walk family. Ten checks, non-zero exit on a failure:

```
--- the shared code, which every policy uses --------------------
[1] sample_far_pairs is the identical sampler            PASS   9295 pairs
[2] degree_table is identical, and 0.0 stays uniform     PASS
[3] link_prediction gives the identical scores           PASS
[4] edge_features is the same Hadamard product           PASS
[5] every shared constant has the modular.py value       PASS
--- the WALK policies, which are the focus ----------------------
[6] every walk D holds INTEGER weights >= 1 (I4)         PASS
[7] no row reaches the law with a degree of 0 (I5)       PASS
[8] the planes ('h', 'freq') keep their contract         PASS
[9] the walk D carries modular.py's long-range term      PASS   9295 pairs at 100.0
[10] ball/sampled stay cell-identical (out of focus)     PASS   nnz 115478
```

The runs are in
[results/walk_policies_cora.tsv](../../experiments/fodiwalk/results/walk_policies_cora.tsv).

**Provenance.** [augment_graph/walks.py](../augment_graph/walks.py),
[fodiwalk.py](../fodiwalk.py) (`_build_D_walk`, `_build_D_nbr_walk`),
[experiments/fodiwalk/verify_modular.py](../../experiments/fodiwalk/verify_modular.py)

---

## 18. UPDATE 2026-08-20 -- the non-walk methods are REMOVED

**Reason for the update.** On request: "remove all traces of non-walk-based
methods and functions from fodiwalk". Section 17 measured that a walk does
not lose to a measured ball -- `nbr_walk` exceeds `fodined/modular.py` on
every score on Cora -- thus the exact-distance code earned no place in the
package. One family of `h`, and one meaning for every number.

**Adopted at.** The fodiwalk package after section 17. The parity gate of
section 12 covers the walk policies only, and nothing it touches moved.

### 18.1 What is gone

| removed | what it was |
| --- | --- |
| `augment_graph/ball.py` | `k_hop_ball`, `augment_k_hop`, `augment`, `edge_set_of`, `sample_unconnected_pairs`, `hop_distances`, `replace_unreachable`, `augmented_csr`, `source_block_size` |
| `Config.pairs = "ball" | "sampled"` | the two dispatch branches, and `Fodiwalk._build_D_ball` / `_build_D_sampled` |
| `Config.k_hop`, `Config.ball_block`, `Config.unreachable_w` | their knobs. An unknown key already raises `TypeError`, thus an old script fails loudly |
| `misc.evaluation.hop_from_D` | the in-sample hop protocol. It needs `D.data` to be a measured distance |
| `tests/test_ball.py` | 15 tests of the removed code |
| `bench_fodiwalk.py --k-hop`, `--hop-protocol`, `--hop-max-pairs` | the flags that reached them |

`Fodiwalk(pairs="ball")` now raises with the list of the three walk
policies and a pointer to this section. It does not fall back.

### 18.2 What STAYS, and why it is not a non-walk method

| kept | why |
| --- | --- |
| `far_pairs.sample_far_pairs`, `degree_table` | the LONG-RANGE term of every walk policy. `modular.py` uses the same function, which is a shared dependency and not a policy |
| `make_graph`'s `wordnet`, `ncbi_taxonomy`, `subtree`, `TREE` | loader work, policy-independent. A tree still needs a subtree and not a BFS ball (section 16.4), and the DAG repair still holds |
| `task_hop`'s `n_estimators`, `hidden`, `early_stopping` | measurement knobs with the campaign's defaults. They name no policy |
| `experiments/fodiwalk/verify_modular.py` | it now compares only the code the two files SHARE, plus the contracts a walk `D` must keep. Ten checks became nine |

### 18.3 One rename, and the bug it exposed

`sample_far_pairs`'s third parameter was named `ball`, from the k-hop ball
of the origin. There is no ball in this package, thus it is now `near` --
the matrix the sampler rejects against, which is the near `D` of a walk.

**The rename found a live defect.** The `directed=True` branch ended with

```python
near = near | (keys[pos] == kr)      # `near` was the boolean mask here
```

which re-bound the name that also held the sparse matrix. Under the new
name the line reads `hit = hit | ...` and the shadowing is gone. The branch
is unused in the package -- `Fodiwalk` keeps the post-filter of
`bench_fdwalk.py` for parity, see section 8.2 -- and `test_contracts.py`
covers it, thus the defect was latent and not active.

### 18.4 The state after the removal

`52 passed` on the non-parity suite (67 before, less the 15 tests of the
removed file). `verify_baseline.py` and `verify_modular.py` both exit 0.
The import surface holds no `ball`, `augment_k_hop`, `k_hop_ball`,
`augment` or `hop_from_D`.

### 18.5 The run after the removal -- Cora and PubMed

`n_dim = 128`, 2000 epochs, `lr = 1.0`, seed 42, `fdlinear`, `plain`,
`min_gap`. One GPU (CudaDevice, 2 GB), 15 GB of host memory.

| graph | policy | `n` | `D.nnz` | augment | embed | TOTAL | peak RSS | accuracy | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cora | `walk` | 2,708 | 75,334 | 1.0 s | 8.3 s | 126.0 s | 1,021 MB | 0.9702 | 0.9956 |
| cora | `nbr_walk` | 2,708 | 209,542 | 0.8 s | 14.8 s | 140.6 s | 1,045 MB | **0.9801** | **0.9982** |
| pubmed | `walk` | 19,717 | 654,190 | 8.6 s | 44.5 s | 250.1 s | 1,269 MB | 0.9691 | 0.9948 |
| pubmed | `nbr_walk` | 19,717 | 2,238,752 | 1.6 s | 95.0 s | 220.8 s | 1,312 MB | **0.9825** | **0.9986** |

Read the memory against a floor of about 620 MB, which is JAX and Python
before any graph is read.

**`nbr_walk` wins on both graphs**, and it widens on the larger one:
+1.0 accuracy point on Cora and +1.3 on PubMed. It costs 2.8x the entries
on Cora and 3.4x on PubMed, thus the embed time roughly doubles.

**Its AUGMENTATION is CHEAPER, and by a lot on PubMed** -- 1.6 s against
8.6 s, while it builds 3.4x more entries. `walk_rows` takes no window, no
cap, no prune and no far-pair rejection: the budget IS `walks * walk_len`
for each row. The cost of `walk` is the `n log10(n)` far pairs and the cap.

**A note on the TOTAL.** The model costs `augment + embed`. The rest is
measurement: the hop regression alone is 80 to 122 s, and it varies between
runs of the SAME configuration -- Cora `walk` took 86.2 s on 2026-08-19 and
111.5 s here, with `accuracy`, `AUC` and `||dZ||` identical to the digit.
Compare model cost with model cost, and do not read the TOTAL as a speed.

**Provenance.** [augment_graph/__init__.py](../augment_graph/__init__.py),
[augment_graph/far_pairs.py](../augment_graph/far_pairs.py),
[fodiwalk.py](../fodiwalk.py), [misc/evaluation.py](../misc/evaluation.py),
[results/walk_only.tsv](../../experiments/fodiwalk/results/walk_only.tsv)
