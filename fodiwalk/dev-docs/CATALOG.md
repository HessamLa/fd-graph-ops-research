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

**Provenance.** [fodiwalk/embed/forces.py](../embed/forces.py) (moved from
`core/forces.py` 2026-08-28, section 26).

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

**Provenance.** [fodiwalk/embed/plan_contract.py](../embed/plan_contract.py)
(moved from `core/plan_contract.py` 2026-08-28, section 26).

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
[policy_nbr_walk.py](../augment_graph/policy_nbr_walk.py),
[policy_walk.py](../augment_graph/policy_walk.py),
[policy_buckets.py](../augment_graph/policy_buckets.py)
(was `_build_D_nbr_walk`, `_build_D_walk`, `_build_D_buckets`)

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

**Provenance.** [fodiwalk/model.py](../model.py)

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
[plan_contract.py](../core/plan_contract.py),
[augment_graph/planes.py](../augment_graph/planes.py) (repointed
2026-08-21: this code was `embed/planes.py` from 2026-08-20 to 2026-08-21,
see CATALOG section 20)

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
[policy_walk.py](../augment_graph/policy_walk.py),
[policy_nbr_walk.py](../augment_graph/policy_nbr_walk.py)
(was `_build_D_walk`, `_build_D_nbr_walk`),
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
[model.py](../model.py), [misc/evaluation.py](../misc/evaluation.py),
[results/walk_only.tsv](../../experiments/fodiwalk/results/walk_only.tsv)

---

## 19. UPDATE 2026-08-20 -- the god class is SPLIT into stage packages

**Reason for the update.** On request: `fodiwalk/fodiwalk.py` had become a
slop. It was 698 lines and it held FOUR jobs -- the configuration, the whole
stage-2 augmentation, the stage-3 assembly, and the model class that should
hold only the wiring. The module specification
([fodiwalk-module.md](fodiwalk-module.md)) asks for three categories that
the TREE can show, and one file that holds three of them shows none.

The ten defects, with the evidence that
[REFACTOR.md](REFACTOR.md) section 2 records:

| # | defect | evidence |
| --- | --- | --- |
| D1 | god class: config, walk dispatch, three `_build_D_*`, the plane builder, the degree source, the plan builder, the kernel call | 698 lines, 4 jobs |
| D2 | policy dispatch by an `if` chain on a string, in two methods, while the package already used registries for the laws, the weights and the update rules | `_build_D`, `graph_walk`, `_plane` |
| D3 | the far-pair CSR merge written THREE times, each copy different | `np.concatenate` 26 times in one file |
| D4 | hidden temporal coupling: `_build_planes` read `self.freq`, which an earlier method had to set | `self.freq =` at lines 179, 244, 539, 625, 691 |
| D5 | parameter tunnels: `_build_D_nbr_walk(self, A, n, rng, info, t0)` passed a log dict and a wall clock into the algorithm | the signature |
| D6 | a cross-module import of a PRIVATE name: `from .core.sell_c_sigma import make_plan, _step` | line 64 |
| D7 | stage leak: stage 2 built `self.params`, which is physics; `jax.jit` and `device_put` lived in the model class | `augment_graph`, `_build_plan` |
| D8 | `stats` was an untyped bag, and `_take` was a private helper of the wrong module | `_take` at the file end |
| D9 | cryptic locals against the naming rule: `fw`, `fq`, `cc`, `fc`, `nk`, `cl`, `rr`, `k1`, `k2`. `fw` means "far weight" there and "the model" in the README | `_build_D_walk` |
| D10 | an empty `models/` directory | `fodiwalk/models/` |

**The tree that replaced it.** `config.py` (88) and `model.py` (200) at the
top, the stage-2 policies in `augment_graph/`, the stage-3 assembly in
`embed/`. `fodiwalk/fodiwalk.py` and `fodiwalk/models/` are REMOVED.

**Nothing about the behaviour changed. Not one number.** The entries below
are the entities the split ADDS. They are names for seams that already
existed inside one file, and the code inside them moved line for line.

---

### 19.1 `Augmentation` -- the seam of stage 2 to stage 3

EVERYTHING that stage 2 gives stage 3, and nothing else crosses. Before the
split, a policy method wrote `self.freq`, `self.stats` and `self.info` on
the model and `_build_planes` read `self.freq` later (D4). The order was
the contract and no signature showed it. Now a policy RETURNS the four
values and the model binds them at one place.

```python
@dataclasses.dataclass
class Augmentation:
    """What stage 2 builds. `D.data` IS the `h` plane."""
    D: sp.csr_matrix
    freq: np.ndarray | None     # (nnz,), aligned to `D.indices`, or None
    stats: dict                 # the walk statistics, plus `freq` as a CSR
    info: dict                  # the counts and the timings a log prints
```

`freq` is the `(nnz,)` DATA and `stats["freq"]` is the same values as a
CSR. The two must keep the sparsity of `D`, thus the `.data` arrays line up
entry by entry. A policy builds `freq` with the same `to_csr` call and the
same keys as `D` for exactly that reason.

**Provenance.** [augment_graph/result.py](../augment_graph/result.py)

---

### 19.2 `AugmentSpec`, `ForceSpec`, `PlanSpec` -- the narrow configs

`Config` stays FLAT, public and unchanged: 39 fields, spelled the same,
with the same default values. It is what a driver script builds, what
`experiments/fodiwalk/*.py` passes, and what `tests/check_api.py` holds
literally. A field that moves breaks a recorded command line.

The narrowing happens at the SEAM. Each stage reads a FROZEN spec of its
own knobs, thus a stage function cannot reach a knob of another stage and
cannot edit the configuration of the run.

```python
AugmentSpec.from_config(cfg)    # 22 fields, augment_graph/result.py
ForceSpec.from_config(cfg)      # 11 fields, embed/planes.py
PlanSpec.from_config(cfg)       #  7 fields, embed/planner.py
```

The three cover every field of `Config` exactly one time, except
`edge_rule`, which `AugmentSpec` and `ForceSpec` both hold: the `auto`
degree source reads it, because a `low_deg` edge rule can leave a hub with
no `h = 1` entry and the degree must then come from `A` (see 19.5 of
`degrees.resolve_degrees`, and trap 4).

`from_config` reads the attributes BY NAME and takes any object that has
them:

```python
    @classmethod
    def from_config(cls, cfg):
        """The stage-2 fields of any config object with these names."""
        return cls(**{f.name: getattr(cfg, f.name)
                      for f in dataclasses.fields(cls)})
```

**WHY the specs live in the stage packages and not beside `Config`.** A
spec in `config.py` would make `config.py` import `embed`, thus `core` and
jax, for the import of a configuration. It would also give one class two
names. `from_config` needs no import in the other direction, thus
`augment_graph` and `embed` stay free of the model layer and the dependency
runs one way. `Fodiwalk.__init__` builds the three one time and holds them
as `self.aug_spec`, `self.force_spec` and `self.plan_spec`.

**Provenance.** [config.py](../config.py),
[augment_graph/result.py](../augment_graph/result.py),
[augment_graph/planes.py](../augment_graph/planes.py) (repointed
2026-08-21, see section 20),
[embed/planner.py](../embed/planner.py)

---

### 19.3 `POLICIES` -- the pair-policy registry

The stage-2 dispatch, as a table. It replaces an `if` chain on
`cfg.pairs` that ran in two methods (D2). A chain on a name is what let a
missing `freq` fall back to the planes of another law in silence (section
13, and `plan_contract`).

```python
POLICIES = {"walk": policy_walk.build,
            "walk_edges": policy_walk.build,
            "nbr_walk": policy_nbr_walk.build}

WALK_STATS = {"walk": policy_walk.pair_stats,
              "walk_edges": policy_walk.pair_stats,
              "nbr_walk": policy_nbr_walk.row_stats}
```

Two tables and not one, because the two questions are different.
`POLICIES` gives the whole augmentation. `WALK_STATS` gives the WALKS
ALONE, which `Fodiwalk.graph_walk` exposes: `walk` runs
`walk_pair_stats` (windowed, undirected, capped, pruned) and `nbr_walk`
runs `walk_rows` (directed, no window, no cap, NO PRUNE by design).

Every builder is a MODULE FUNCTION with the same signature and no `self`:

```python
def build(A, n: int, spec: AugmentSpec, rng) -> Augmentation
```

`info` and the wall clock are made INSIDE the builder and not passed in
(D5). The generator is the ONE generator of the run: the walks, the bucket
sample, the far draw and the landmarks read it in that order.

**To add a policy**, write that function in a `policy_*.py` module of
`augment_graph/`, then add one entry to `POLICIES` -- and one to
`WALK_STATS` when `graph_walk` must also work for it:

```python
from fodiwalk.augment_graph import policies
policies.POLICIES["my_policy"] = build_my_policy
fw = Fodiwalk(n_dim=64, pairs="my_policy")
```

An unknown name raises with the list of the known ones. Nothing else in
the package needs an edit -- `README.md` holds the example that this
document verified on Cora.

The three builders keep their own second axis, and each axis is also a
table: `EDGE_RULES` (`walk` against `walk_edges`) and `POLICY_RULES`
(`cap` against `buckets`) in `policy_walk.py`; `NEIGHBOUR_RULES`,
`SELECT` and `FREQ_MODES` in `policy_nbr_walk.py`. Each table keeps the
default of the old `if` chain: an unknown name falls to `cap`, to `both`
or to `pair`, exactly as the old branch did.

**Provenance.** [augment_graph/policies.py](../augment_graph/policies.py),
[policy_walk.py](../augment_graph/policy_walk.py),
[policy_buckets.py](../augment_graph/policy_buckets.py),
[policy_nbr_walk.py](../augment_graph/policy_nbr_walk.py)

---

### 19.4 `add_far_pairs` -- the ONE far-pair merge

Three policies add long-range pairs to the matrix they built, and each one
carried its own copy of the block (D3). The copies had DRIFTED: the
`nbr_walk` copy shared one row array between `D` and `freq` and made
`np.full(2m, w)`; the `walk` copy built `freq` first, `D` second, and wrote
`np.full(m, w)` two times; the `buckets` copy built `D` first and `freq`
second. A repair of one left the other two wrong.

```python
def add_far_pairs(D, freq, far, weights):
    """The far pairs onto `D` and `freq`, in BOTH directions."""
    m = far.shape[0]
    near = D.tocoo()
    D_out = sp.csr_matrix(
        (np.concatenate([near.data, weights, weights]),
         (np.concatenate([near.row, far[:, 0], far[:, 1]]),
          np.concatenate([near.col, far[:, 1], far[:, 0]]))),
        shape=D.shape)
    ...
```

**THE ORDER OF THE COO TRIPLES IS PART OF THE RESULT.**
`sp.csr_matrix((data, (row, col)))` SUMS a duplicate coordinate, and the
sum order decides the last bit of a float. The near entries go first, then
the pairs in the forward direction, then the pairs in the backward
direction. Do not sort and do not group.

The far weight is the only real difference between the callers, thus it is
an ARGUMENT: one constant for each pair (`cap`, `buckets`, `nbr_walk`), or
a landmark distance for each pair (`landmarks`). `freq` always gets 1.0,
because a far pair was DRAWN and never observed. `np.full(2m, w)` and two
`np.full(m, w)` hold the same bytes, thus the one merge is byte-exact for
all three callers -- the golden gate proves it on the 16 cases.

`merge.py` also holds **`drop_pairs_of(far, near, n)`**, the far filter of
`nbr_walk`. It drops a far pair that `near` holds in EITHER direction,
AFTER `sample_far_pairs` has drawn. `directed=True` inside the sampler
gives the same PROPERTY and OTHER pairs, thus it breaks parity (section
8.2). The filter must stay after the draw.

**Provenance.** [augment_graph/merge.py](../augment_graph/merge.py)

---

### 19.5 `PLANE_BUILDERS` -- the plane registry, and its pair

A plane is a `(nnz,)` array with one value for each stored pair of `D`, in
`D`'s own pre-split CSR order. The law unpacks the tiles POSITIONALLY, thus
a wrong plane is wrong PHYSICS and not an error. `Fodiwalk._plane` built
them in a branch; the branch is now a table.

```python
PLANE_BUILDERS = {"h": _build_h, "freq": _build_freq, "w": _build_w}

# The two tables are one contract: a name that one holds and the other
# does not is a plane that is built and not asserted, or asserted and not
# built.
assert set(PLANE_BUILDERS) == set(PLANE_CHECKS)
```

**THE PAIRING IS NOW VISIBLE, and the module asserts it at import.**

| table | file | what it holds |
| --- | --- | --- |
| `FORCE_PLANES` | `core/forces.py` | which planes a law reads, and in what ORDER |
| `PLANE_BUILDERS` | `embed/planes.py` | how the VALUES of a plane name are made |
| `PLANE_CHECKS` | `core/plan_contract.py` | what the name PROMISES about the values |

**To add a plane, a person edits those three tables and nothing else.**
Section 1.4 and section 13 record what happens when the three drift: a
missing `freq` made `fdlinear` read a coefficient plane as `h`, the run
went to NaN under an `fdlinear` label, and nothing raised.

`build_planes(law, D, freq, pairs, policy)` walks `planes_of(law)` and
takes the builder of each name. There is NO branch on the law. `pairs` and
`policy` are carried only to NAME the augmentation in the message of a
missing plane.

`planes.py` also holds `ForceSpec` (19.2) and **`force_params(spec)`**, the
traced scalars `dict(k1=, k4=, kr=, sign=)` that the law reads. That is
PHYSICS, thus it belongs to stage 3; `Fodiwalk.augment_graph` built it
inside stage 2 (D7).

`degrees.py` holds **`resolve_degrees(D, G, spec, explicit=None)`**, the
divisor of the row sum, with three sources in this order: `no_deg_norm`
gives 1; an explicit array (`deg_source="A"` or `set_D(degrees=...)`) is
the true degree of the graph; otherwise `degrees_from_D` counts the
`h == 1` entries of a row. A row with none gets degree 0, `inv_deg_ext`
becomes 0.0, and the kernel zeroes the WHOLE force of the row. The node
freezes and nothing raises (section 14).

**Provenance.** [augment_graph/planes.py](../augment_graph/planes.py),
[augment_graph/degrees.py](../augment_graph/degrees.py) (both repointed
2026-08-21 from `embed/`, see section 20),
[core/plan_contract.py](../core/plan_contract.py),
[core/forces.py](../core/forces.py)

---

### 19.6 `PlanSet` and `build_plans` -- the chunked row-range plan

`build_plans(D, planes, degrees, spec, force_fn) -> PlanSet` is the
stage-3 layout. It was `Fodiwalk._build_plan`, with the `jax.jit`,
`jax.device_put` and `functools.partial` plumbing moved out of the model
class (D7).

```python
@dataclasses.dataclass
class PlanSet:
    plans: list         # one plan for each chunk, on the device when resident
    steps: list         # the jitted kernel of each chunk
    inv_deg_ext: object # (n + 1,) on the device. GLOBAL, thus one array
    chunk_rows: int     # the row count of a chunk. `forces` picks with it
    resident: bool      # the plans stay on the device
    stats: dict         # the model exposes it as `plan_stats`
```

**A CHUNK IS A ROW RANGE and never a pair set.** `core.sell_c_sigma.step`
writes `dZ.at[rows].add(...)`, thus only DISJOINT rows make the parts
additive. `ForceDirected.embed` slices `dZ` by the same range for its
batches, thus the chunk and the batch are the same object. The rows outside
`[a, b)` become empty, and `make_plan` gives an isolated row no virtual row
at all, thus an empty row costs nothing in the plan of another chunk.

**THE GLOBAL QUANTITIES STAY GLOBAL.** `degrees` is counted over the WHOLE
`D` and a chunk only SLICES the planes with the CSR span of its rows. A
degree counted on one chunk is not the degree of the node.

`Fodiwalk.forces` then holds three lines: the chunk index from
`row_start // chunk_rows`, the jitted step of that chunk, and the drop.

**Provenance.** [embed/planner.py](../embed/planner.py),
[core/sell_c_sigma.py](../core/sell_c_sigma.py)

---

### 19.7 `core.sell_c_sigma.step` -- the kernel, renamed and published

The per-epoch kernel was `_step`, and the model imported it across a module
boundary: `from .core.sell_c_sigma import make_plan, _step` (D6). A leading
underscore is a promise that no other module reads the name. A
cross-module import of it is a defect of the NAME and not of the code: the
kernel was always public in fact, thus the promise was false and a reader
could not know what `core` guarantees.

```python
def step(Z, plan, inv_deg_ext, params, n, force_fn):
    """One fused pass over the whole padded plan -> full ``(n, d)`` dZ."""
    ...

# `step` is the public name (defect D6). `_step` stays as an alias:
# `PlanCache` below, the docstrings of three modules and the tests name it.
_step = step
```

The rename is a rename and nothing else. `_step` stays an ALIAS and
`core/__init__.py` exports both names, thus every existing caller and every
test that names `_step` keeps working. `embed/planner.py` imports `step`.

**Provenance.** [core/sell_c_sigma.py](../core/sell_c_sigma.py),
[core/__init__.py](../core/__init__.py)

---

### 19.8 `tests/golden.py` and `tests/check_api.py` -- the two new gates

A code move needs a gate that costs seconds. The parity gate of section 12
costs a GPU and about 18 minutes, thus it cannot run between two edits.

**`golden.py` -- BEHAVIOUR.** It hashes what the augmentation and the plan
build, on Cora, for one configuration of every branch of the pair policies:
16 augment cases and 4 short embed runs.

```bash
.venv/bin/python -m fodiwalk.tests.golden --write   # record
.venv/bin/python -m fodiwalk.tests.golden --check   # compare
```

Two halves, because the determinism differs. The AUGMENT half is pure
NumPy, thus it is compared BYTE EXACT: the sha1 of `D.indptr`, `D.indices`,
`D.data`, of every plane, of the degrees, of `freq`, of three `stats`
arrays, plus the `info` counts, the plan statistics, and a digest of four
draws of `fw.rng` AFTER the stage -- which catches a second generator, a
moved call or one extra draw (trap 11). The EMBED half is JAX, and a split
hub row makes the GPU scatter-add order free (section 15), thus it runs on
the CPU backend and compares NUMBERS at `rtol = 1e-4`.

**What golden.py CANNOT catch.** It reads ONE graph (Cora, 2708 nodes) and
one seed (42). It holds no `chunk_host` case, no landmark case beyond
`walk_landmarks`, and no case above `k_max`, thus `n_split > 0` -- the hub
split of section 15 -- is never exercised. A digest of `Z` is also too
sharp a tool: the same run differs in the last bit between two processes,
thus the embed half compares five scalars and not a hash. It measures a
code MOVE. It cannot say that a new feature is right.

**`check_api.py` -- THE SURFACE.** Every list in it is LITERAL, and that is
the point of the file. A check that reads `dataclasses.fields(Config)`
compares `Config` to itself and passes whatever the refactor does.

```bash
.venv/bin/python -m fodiwalk.tests.check_api     # 0 = pass
```

It holds the 39 field names of `Config` with their default values, the 9
keyword names of `Fodiwalk.__init__`, 13 methods, 17 attributes after a
run, and the key sets of `info`, `plan_stats` and `stats`. The keys are a
SUBSET rule: a LOST key fails, a NEW key passes. A lost key breaks a log or
a `RESULT` line; a stage split may legitimately add one. It also asserts
the behaviours a name cannot show: an unknown keyword RAISES `TypeError`,
`fit` RAISES `NotImplementedError`, and `_build_planes` / `_build_degrees`
stay callable on the model, because `golden.py` drives them.

**What check_api.py CANNOT catch.** It proves that a NAME exists and that
it answers; it does not prove that the number behind the name is right. It
runs on a 60-node synthetic graph for 5 epochs. `golden.py` is the gate for
the values, and this file is the gate for the shape of the surface.

**The measured state, 2026-08-20:**

```
$ .venv/bin/python -m fodiwalk.tests.golden --check
GOLDEN OK (both)

$ .venv/bin/python -m pytest fodiwalk/tests -q -m "not parity and not big"
53 passed, 2 skipped, 9 deselected in 80.09s (0:01:20)

$ .venv/bin/python -m fodiwalk.tests.check_api
[check_api] API OK: 39 Config fields, 13 methods, 17 attributes, 24 keys
```

**The count of the suite MOVES, and only a failure is a defect.** It was 52
before the split. It reached 136 while `fodiwalk.py` and the two
equivalence test files existed together. It is 53 above, on the tree with
the god class deleted, and 66 after `tests/test_structure.py` added the
thirteen tests that make the new shape a gate:

```
$ .venv/bin/python -m pytest fodiwalk/tests -q -m "not parity and not big"
66 passed, 2 skipped, 9 deselected in 72.74s (0:01:12)
```

The 2 skips are `test_policy_equivalence.py` and
`test_embed_equivalence.py`. They compared the new modules against the
methods of `fodiwalk.py` and they SKIP by an `importorskip` guard now that
the file is gone. They retire with the thing they compared against, and
`golden.py` is what carries the property forward.

**Provenance.** [tests/golden.py](../tests/golden.py),
[tests/check_api.py](../tests/check_api.py),
[tests/test_golden.py](../tests/test_golden.py)

---

### 19.9 What did NOT change

| kept | proof |
| --- | --- |
| the physics: the laws, the planes, the parameters, the degree rule | the plane digests and `Z` of the golden gate |
| the walks: `uniform_walks`, `node2vec_walks`, `make_walker`, `walk_rows`, `walk_pair_stats` | not one line of `walks.py` moved. `make_walker` still returns `uniform_walks` ITSELF at `p = q = 1` |
| the weight rules, the caps, the buckets, the far draw, the landmarks | `weights.py`, `pairs.py`, `buckets.py`, `far_pairs.py`, `landmarks.py` are untouched |
| the update rules, the drop, the measurements | `misc/` is untouched |
| the ORDER of the one generator | the `rng_state` digest of all 16 golden cases is byte exact |
| the numbers | `GOLDEN OK (both)`: the augment half byte exact, the embed half inside `rtol = 1e-4` |
| the public surface | `API OK: 39 Config fields, 13 methods, 17 attributes, 24 keys` |

`Config` did not lose a field, gain a field or change a default.
`Fodiwalk` did not lose a method or an attribute. `_build_planes` and
`_build_degrees` stay on the model as thin forwards to `embed/`.

---

### 19.10 Where the CODE and REFACTOR.md disagree

The code is the authority. Three differences, and each one is deliberate:

1. **The stage specs are NOT in `config.py`.** REFACTOR.md section 3 puts
   "Config (FLAT, unchanged fields) + the stage specs" in `config.py`. Each
   spec instead lives in the package that READS it (19.2). A spec in
   `config.py` would pull `embed`, thus `core` and jax, into the import of
   a configuration, and it would break the one-way dependency the same
   document asks for in section 3.
2. **`ForceSpec` holds `edge_rule`, which section 3.2 does not list.** The
   `auto` degree source reads it. Without it, `resolve_degrees` cannot see
   that a `low_deg` rule may leave a hub with no `h = 1` entry.
3. **`merge.py` holds a second function, `drop_pairs_of`.** Section 3 names
   only `add_far_pairs`. The far filter of `nbr_walk` is the other half of
   the same seam and it belongs beside the merge (19.4).

Two gate COMMANDS of section 5 do not measure what they say, and both
were true before the split as well:

- **G4** ("no module > 300 lines except `core/sell_c_sigma.py`") also fails
  on `tests/test_contracts.py` (468), `core/force_directed.py` (432) and
  `augment_graph/walks.py` (422). All three are PRE-EXISTING files that the
  split did not touch. `model.py` is 200 and `config.py` is 88, thus the
  rule the split owns holds. `tests/test_structure.py` now encodes the rule
  with the three files PINNED at their measured size and the test files
  excluded, thus a file that needs more needs a split and not a raised cap.
- **G6** (`grep -c 'sp.csr_matrix((np.concatenate' fodiwalk/augment_graph/*.py`)
  prints 0 for every file, including `merge.py`. The pattern needs the two
  calls on ONE line and `merge.py` wraps them. It printed 1 on the old
  `fodiwalk.py`, which held THREE copies. The command under-counts and it
  cannot show what it was written to show; `grep -c 'np.concatenate'` gives
  6 in `merge.py` and the duplication is gone by reading.

One small duplication remains and it changes no number: `Fodiwalk.forces`
reads `self.cfg.random_drop_rate` and `self.cfg.drop_strategy` directly,
although `ForceSpec` also holds the two fields.

---

### 19.11 REPAIR -- the provenance links that pointed at the deleted file

Five sections cited `[fodiwalk.py](../fodiwalk.py)`, which no longer
exists. A dead link is a factual error and not content, thus the LINKS are
repointed at the modules that now hold that code, and **not one other word
of those sections changed.**

| section | was | now |
| --- | --- | --- |
| 4. Pair policies | `fodiwalk.py` (`_build_D_nbr_walk`, `_build_D_walk`, `_build_D_buckets`) | `augment_graph/policy_nbr_walk.py`, `policy_walk.py`, `policy_buckets.py` |
| 10. `class Fodiwalk` | `fodiwalk/fodiwalk.py` | `fodiwalk/model.py` |
| 13. the shell laws | `fodiwalk.py` | `augment_graph/planes.py` (repointed 2026-08-21, see section 20; was `embed/planes.py`) |
| 17. the walk policies | `fodiwalk.py` (`_build_D_walk`, `_build_D_nbr_walk`) | `augment_graph/policy_walk.py`, `policy_nbr_walk.py` |
| 18. the non-walk removal | `fodiwalk.py` | `model.py` |

Section 16.5 keeps its link to `fodiwalk.py` (`_build_D_ball`,
`_build_D_sampled`). That code was REMOVED by section 18 and no module
holds it, thus there is nothing to repoint to; the link records where the
removed code was.

---

**Adopted at.** The fodiwalk package after the walk-only reduction of
section 18, on the commit that lands the split. The tree before it is the
tag `fodiwalk-pre-refactor` (89abaf2), which is the revert point.

**Provenance.** [config.py](../config.py), [model.py](../model.py),
[augment_graph/result.py](../augment_graph/result.py),
[augment_graph/policies.py](../augment_graph/policies.py),
[augment_graph/policy_walk.py](../augment_graph/policy_walk.py),
[augment_graph/policy_buckets.py](../augment_graph/policy_buckets.py),
[augment_graph/policy_nbr_walk.py](../augment_graph/policy_nbr_walk.py),
[augment_graph/merge.py](../augment_graph/merge.py),
[augment_graph/planes.py](../augment_graph/planes.py) (repointed
2026-08-21 from `embed/`, see section 20),
[augment_graph/degrees.py](../augment_graph/degrees.py) (same repointing),
[embed/planner.py](../embed/planner.py),
[tests/golden.py](../tests/golden.py),
[tests/check_api.py](../tests/check_api.py),
[dev-docs/REFACTOR.md](REFACTOR.md)


---

## 20. UPDATE 2026-08-21 -- the stage boundary moves; the RECIPE, named

**Reason for the update.** Section 19's split put the plane builder and the
degree resolver in `embed/` -- stage 3. `dev-docs/fodiwalk-module.md` was
then sharpened with one sentence about stage 3 (embedding): "This stage
shall not do any graph analysis or data preparation. It must only consume
the data. Its main goal is to apply the force function on the input data
using the best implementation to optimize resource utilization." A plane
is a choice about WHICH VALUES a law needs; a degree is a choice about
WHICH ROW would freeze without an explicit one. Both are choices about
DATA, decided by the pair policy and the law together, and neither is a
property of the kernel that later applies them. Building them in `embed/`
put data preparation inside the consumption stage.

**What moved, and what did not.** `embed/planes.py` -> new
`augment_graph/planes.py` (`ForceSpec`, `PLANE_BUILDERS`, `build_planes`,
`force_params`). `embed/degrees.py` -> new `augment_graph/degrees.py`
(`resolve_degrees`). No line of a function BODY changed; the error
messages, the plane registry, the three-source order of a degree are all
verbatim. `embed/planner.py` (`PlanSpec`, `PlanSet`, `build_plans`, the
`jax.jit`/`device_put` plumbing) did NOT move: the plan build and the
jitted step consume a finished recipe, and stay stage 3.

**What widened.** `augment_graph.result.Augmentation` (19.1) gained three
fields, `planes`, `degrees` and `params`, all defaulting to `None`: a
policy's own `build()` does not know the force law, thus it cannot fill
them, and `Fodiwalk.augment_graph` fills them once the law is known --
still calling `augment_graph` code, immediately after `policies.build`
returns. `augment_graph`'s allowed imports widen from "numpy, scipy only"
to "numpy, scipy, `core`" (`REFACTOR.md` section 3): the moved code reads
`core.forces.planes_of`/`fuse` and `core.plan_contract.PLANE_CHECKS` to
know a law's plane contract, and knowing the contract is itself part of
preparing that law's data. `embed/` narrows the other way: it still reads
`core` (for `make_plan`, `step`, `plan_contract.check_plan`), and now
builds no plane and resolves no degree at all.

### 20.1 The RECIPE

**One augmentation policy plus one force law, plus the data they exchange,
is a RECIPE.** The pairing is not incidental: `_build_planes` in `19.1`'s
predecessor already read `self.freq`, which only a MATCHING policy had
set, and `plan_contract.check` exists because a mismatched pair is a
silent wrong-physics run and not an error. A recipe names that pairing on
purpose, at the place that assembles it.

Baseline Fodiwalk's recipe: pairs `nbr_walk` (or `walk` / `walk_edges`)
feeding the law `fdlinear`'s two planes `(h, freq)`, or `fdlinear_fused`'s
one fused plane `(w,)`. A different augmentation prepares a different
data set for the SAME law -- `walk`/`buckets`/`far` builds a different
`freq` than `nbr_walk` does, for instance -- and a different law reads a
different plane set from the SAME augmentation. `augment_graph` is where
one full recipe is assembled, end to end, from the walk to the force
params; `embed` is the one engine every recipe shares, and it never learns
which recipe it is running -- it reads planes, a degree array and a params
dict, by position and by name, and nothing else.

```python
# fodiwalk/model.py -- one recipe, assembled
aug = policies.build(G, n, self.aug_spec, self.rng)   # the pairs: augment_graph
planes  = build_planes(self.law, D, self.freq, ...)   # the law's data: augment_graph
degrees = resolve_degrees(D, G, self.force_spec, ...)  # augment_graph
params  = force_params(self.force_spec)                # augment_graph
ps = build_plans(D, planes, degrees, self.plan_spec, self.force_fn)  # embed: CONSUMES
```

**Why the kernel must never learn the recipe.** `core.sell_c_sigma.step`
unpacks the plane tuple POSITIONALLY (trap 3, `REFACTOR.md` section 4): it
has no name for what it reads, only an order. That is what makes ONE
engine serve every recipe -- and it is also why the plane contract
(`plan_contract.check`) must run BEFORE the kernel ever sees the data: a
recipe mismatch caught at the seam is a raised `PlaneContractError`; a
recipe mismatch that reaches the kernel is a silent wrong-physics run
under the right law's name (section 1.4, the 2026-08-18 defect this whole
package exists to stop repeating).

### 20.2 What did NOT change

The physics, the force laws, the numbers. `Config`'s 39 fields, names and
defaults. The public surface of `Fodiwalk`
(`tests/check_api.py` reports `API OK: 39 Config fields, 13 methods, 17
attributes, 24 keys`, unchanged). `_build_planes(D)` and `_build_degrees(D,
G)` as callable model methods -- `tests/golden.py` still calls both by
those names.

### 20.3 The gates, re-run after the move

| gate | command | result |
| --- | --- | --- |
| G1 | `python -m fodiwalk.tests.golden --check` | `GOLDEN OK (both)` -- BYTE EXACT |
| G2 | `pytest fodiwalk/tests -q -m "not parity and not big"` | `66 passed, 2 skipped, 9 deselected` |
| G3 | `python -m fodiwalk.tests.check_api` | `API OK`, exit 0 |
| G4-G8 | `pytest fodiwalk/tests/test_structure.py` | `13 passed`, one contract widened (`test_augment_graph_imports_no_embed_no_model`, now allows `core`) |

Independent checks, run against the moved code and not merely re-reported:
`augment_graph()` still calls the augmentation exactly ONCE per `embed()`
call (trap 12, instrumented count); `set_D` still bypasses the walk and
keeps the handed matrix; a missing `freq` still raises `PlaneContractError`
naming the plane and the law. Two negative controls, planted in scratch
copies and never the live tree: `embed/planner.py` importing
`augment_graph.planes` fails `test_embed_imports_only_core_no_augment_graph_no_model`,
naming the file; `augment_graph/policies.py` importing `embed.planes` fails
`test_augment_graph_imports_no_embed_no_model`, naming the file.

**Adopted at.** The fodiwalk package immediately after section 19's split,
same session, before the tree was committed.

**Provenance.** [augment_graph/planes.py](../augment_graph/planes.py),
[augment_graph/degrees.py](../augment_graph/degrees.py),
[augment_graph/result.py](../augment_graph/result.py),
[augment_graph/__init__.py](../augment_graph/__init__.py),
[embed/__init__.py](../embed/__init__.py),
[embed/planner.py](../embed/planner.py),
[model.py](../model.py), [config.py](../config.py),
[dev-docs/fodiwalk-module.md](fodiwalk-module.md),
[dev-docs/REFACTOR.md](REFACTOR.md) section 8.


---

## 21. UPDATE 2026-08-21 -- the engine, the pipeline and the model, split into three classes

**Reason for the update.** On request: `ForceDirected` mixed two concerns
under one name -- a domain-agnostic RELAXATION ENGINE (the epoch loop, the
batching, the callbacks, the `Z` update) and the Fodiwalk PROJECT's own
pipeline stubs (`make_graph`, `augment_graph`, `fit`). A class named for
embedding should not declare methods about how a graph is built or
augmented; those are pipeline concerns, and a future non-Fodiwalk model
built on the same engine would inherit stubs it has no use for.

**The three-tier hierarchy.**

```
core.ForceDirectedEmbedding   the engine. embed(D, epochs, ...) relaxes Z
        |                     against a GIVEN D. forces() is its one hook.
        |                     NO make_graph, NO augment_graph, NO fit.
        v
fodiwalk.Fodiwalk_base        the pipeline CONTRACT. make_graph, graph_walk,
        |                     augment_graph, embed(G, ...) -- all RAISE.
        |                     `forces` is NOT re-declared: it is already
        |                     abstract on the engine, and every concrete
        |                     subclass must supply it too.
        v
fodiwalk.Fodiwalk             the concrete, baseline model. Implements
                              every one of the five.
```

`ForceDirected` -> `ForceDirectedEmbedding` is a RENAME, and
`make_graph`/`augment_graph`/`fit` are REMOVED from it, not deprecated: no
alias survives, and every caller in the package was updated in the same
change (`core/__init__.py`, `README.md`, the tests).

### 21.1 `Fodiwalk.embed` -- the orchestration, made explicit

Before this split, `ForceDirected.embed(G, ...)` did two things at once: it
called `self.augment_graph(G)` to get `D`, THEN ran the epoch loop against
it. Now that the engine takes `D` directly, `Fodiwalk.embed` does the
first half explicitly and hands the result to the engine by NAME (not
`super()`, which would resolve to `Fodiwalk_base.embed` -- the abstract
stub one layer up -- and raise):

```python
def embed(self, G, epochs=1000, lr=None, Z=None, batch_count=1,
          epsilon=None, **kwargs):
    """`augment_graph(G)` ONCE (trap 12), then the engine's loop on `D`."""
    self.G = G
    D = self.augment_graph(G, **kwargs)
    return ForceDirectedEmbedding.embed(
        self, D, epochs=epochs, lr=lr, Z=Z, batch_count=batch_count,
        epsilon=epsilon, **kwargs)
```

Trap 12 (`dev-docs/REFACTOR.md` section 4, item 12) is unchanged by
construction: `augment_graph` is called exactly once, right here, and the
engine's own `embed` never calls it -- it no longer CAN, it does not know
the method exists.

### 21.2 `fit()` -- removed, not deprecated

`ForceDirected.fit(data, epochs)` was `self.embed(self.make_graph(data),
epochs=epochs)`. `make_graph` is a STUB in every concrete model of this
project (`fodiwalk/make_graph/datasets.py` is the real stage-1 reader),
thus `fit` promised a stage that was never real, and `Fodiwalk.fit` always
raised `NotImplementedError`. Removing it outright, rather than keeping
the raise, means `hasattr(fw, "fit")` is `False` -- a caller gets an
`AttributeError` naming the missing method, not a raised exception whose
message has to be read to learn the same fact.

### 21.3 What did NOT change

The physics, the numbers, `Config`'s 39 fields. The PUBLIC signature of
`Fodiwalk.embed(G, epochs=..., lr=..., Z=..., batch_count=..., epsilon=...)`
-- every existing caller (`tests/harness.py`,
`experiments/fodiwalk/bench_fodiwalk.py`) is unchanged. `check_api.py`
reports `API OK` with 12 methods (`fit` dropped from the recorded 13).

### 21.4 The gates, re-run after the split

| gate | result |
| --- | --- |
| `python -m fodiwalk.tests.golden --check` | `GOLDEN OK (both)` -- BYTE EXACT |
| `pytest fodiwalk/tests -q -m "not parity and not big"` | `66 passed, 2 skipped, 9 deselected` |
| `python -m fodiwalk.tests.check_api` | `API OK`, 12 methods, exit 0 |
| `pytest fodiwalk/tests/test_structure.py` | `13 passed` (the `core/force_directed.py` size pin lowered 432 -> 421) |
| GPU parity P1 | re-run after the split, see `dev-docs/REFACTOR.md` |

Independent checks: the MRO is
`Fodiwalk -> Fodiwalk_base -> ForceDirectedEmbedding -> object`; `fit` is
absent from all three; `ForceDirectedEmbedding` has neither `make_graph`
nor `augment_graph`; `augment_graph` is still called exactly once per
`embed`; `Fodiwalk_base`'s five abstracts (`make_graph`, `graph_walk`,
`augment_graph`, `embed`, and the inherited `forces`) all raise
`NotImplementedError`; `ForceDirectedEmbedding.embed(D, epochs=1)` runs
STANDALONE given a bare `D` and a borrowed `forces` function, proving the
engine needs nothing else.

**Adopted at.** The fodiwalk package immediately after section 20's seam
move, same session.

**Provenance.** [core/force_directed.py](../core/force_directed.py),
[base.py](../base.py), [model.py](../model.py).


---

## 22. FINDING 2026-08-21 -- `task_hop`'s `vector`/`fodined` claim was wrong

**Found by** a peer session building `evaluator/`, cross-checking
`fodiwalk/misc/evaluation.py`'s docstring against the live
`fodined/modular.py`. Confirmed independently here, from THREE sources
that agree with each other and not with the docstring:

1. `fodined/modular.py:632`, live: `sp_X = np.linalg.norm(Z[u_of[...]] -
   Z[v_of[...]], axis=1)[:, None]` -- ONE column. Two `vector`-shaped
   lines sit COMMENTED OUT directly above it.
2. `fdmap_backup/fodined/modular.py` (mtime 2026-08-15 09:55) has the
   IDENTICAL three-line block. `experiments/modular-graphs/modular_cora_khop3.log`
   (mtime 2026-08-15 00:29, the log `dev-docs/PLAN.md`/`FINDINGS.md` cite
   for the recorded `r2 = 0.257`) was generated in that same window --
   the published number is a ONE-feature number, not `n_dim`.
3. `experiments/other-ge/bench_other_ge.py:419`, independently: "This is
   the same feature that `fodined/modular.py` uses NOW, thus the numbers
   are comparable" -- describing its own one-column
   `np.linalg.norm(Z[u]-Z[v])[:, None]`.

`dev-docs/CATALOG.md` section 16.1 (2026-08-19) asserted the opposite --
`"the hop features | task_hop(feature="vector") IS modular.py's
|Z[u]-Z[v]|"` -- four days AFTER the code and the published log already
used `distance`. That entry is not corrected in place (nothing is ever
removed from this catalog); this entry is the correction.

**Repaired.** `fodiwalk/misc/evaluation.py::task_hop`: the docstring now
names `distance` as `fodined/modular.py`'s protocol (shared with
`other-ge/bench_other_ge.py`, on that file's own word), and `vector` as
NOT corresponding to any baseline recorded in this repository. The
DEFAULT changed `"vector"` -> `"distance"`: every caller in this
repository (`experiments/fodiwalk/bench_fodiwalk.py`,
`experiments/fdwalk/bench_fdwalk.py`, `tests/harness.py`) passes
`feature` explicitly in a loop over both, thus the default was never
read and this is behaviour-NEUTRAL for every existing run -- verified by
grep before the change, and by the golden gate and the full suite after
it. The `200`/`(128, 64)`/`early_stopping=True` sentence beside it, about
the MODEL hyperparameters and not the feature width, was already correct
and is unchanged.

**A second claim, checked and found FALSE.** The same peer session
initially reported that `edge_features` L2-normalizes the Hadamard
product. It does not, in either `fodiwalk/misc/evaluation.py` or
`fodined/link_prediction.py` -- both are the bare elementwise product,
byte-identical, confirmed by a `grep` for normalization across all four
evaluation sources in the repository returning nothing. The peer
retracted this independently after re-checking; recorded here only so a
future session does not have to re-derive that there was never anything
to fix.

**A related finding, not a defect of THIS package.** `hop_sample` (same
file) does not filter hop-1 pairs itself -- its only guard is `d > 0`
(self-pairs, unreachable pairs). Every caller in this repository filters
`d >= hop_min` afterward (`tests/harness.py`); `hop_sample` itself does
not enforce it. No number in this repository is wrong because of this --
every existing caller already filters -- but a FUTURE caller that forgets
would silently score an easier problem, the same class of defect as an
unfiltered NCBI star (section 16.4). Repaired with a docstring warning
only, naming the guard and the convention; the function's BEHAVIOUR does
not change, because every recorded number already depends on today's
unfiltered output being filtered by the caller, not by this function.

**Adopted at.** The fodiwalk package immediately after section 21's class
split, same session.

**Provenance.** [misc/evaluation.py](../misc/evaluation.py),
[../../fodined/modular.py](../../fodined/modular.py) line 632,
[../../fdmap_backup/fodined/modular.py](../../../fdmap_backup/fodined/modular.py),
[../../experiments/other-ge/bench_other_ge.py](../../experiments/other-ge/bench_other_ge.py)
line 419.

---

## 23. UPDATE 2026-08-25 -- the SELL-C-sigma algorithm becomes ONE package, `sellcsigma`

**Reason for the update.** The repository held the algorithm FIVE times:
`fodined/embedding/sell_c_sigma.py` (547 lines),
`fodiwalk/core/sell_c_sigma.py` (558 lines), and three more under
`archive/` -- `fdge_jax_sell_c_sigma/embedding/sell_c_sigma.py`, the
ancestor `fdmap_bucketed_bench_jax.py`, and a torch port of the same plan
builder, `fdmap_bucketed_bench_torch.py`. The two live copies had NOT
diverged in the algorithm, but a copy that nobody diffs is a defect that
waits for the first fix applied to one of the two.

**What it is.** A new top-level package holding the ONE implementation:

```
sellcsigma/
  __init__.py         the public surface
  sell_c_sigma.py     build_ladder, make_plan, step, PlanCache, to_csr
  PARITY.md           the evidence, dated 2026-08-25
  tests/
    m1_old_vs_new.py  the one-window old-against-new script
    test_parity.py    the permanent gates
```

**Why a NEW package and not a home inside one of the two.** Both other
directions are wrong. `core` declares itself self-contained (this package's
`core/__init__.py`, plus `test_contracts.test_core_imports_only_core`),
thus `core` must not read `fodined`. And `fodined` reading `fodiwalk` makes
the older package depend on the newer one, which the four benchmark scripts
under `experiments/` then inherit. `sellcsigma` reads NO package of this
repository, thus every dependency points at it and a cycle is impossible.

**The common argument set, which is why this was a move and not a rewrite.**
All SEVEN live call sites of `make_plan` pass the same set, thus the shared
signature is the one that already existed and nothing was generalized:

```python
plan, inv_deg_ext, stats = make_plan(
    D, planes, degrees=degrees, b_cells=..., k_max=..., ladder_base=...)
kernel = jax.jit(functools.partial(step, n=n, force_fn=force_fn))
dZ = kernel(Z, plan, inv_deg_ext, params)
```

`planes` is a 2-tuple at every live site. The two chunked callers
(`embed/planner.py`, `experiments/fdwalk/bench_fdwalk.py`) slice `D` and the
planes BEFORE the call and pass GLOBAL `degrees`; that is a caller concern
and needs nothing from the package.

**`to_csr` -- the ONE name added.** `fodined/embedding/shell_force.py` line
69 imports `_to_csr`, and gate D6 (`tests/test_structure.py`) forbids a
plain module to import a private name of another module. A shared package
must therefore offer a public one. `_to_csr` stays as an alias, the same
pattern `step`/`_step` already uses. No other name changed.

**`core/sell_c_sigma.py` is now a FORWARDER**, 46 lines, and it is part of
the design and not debt. It holds no logic. It imports the public names by
name -- never `import *`, which would skip `_step` and `_to_csr` unless
`__all__` named them -- and binds the two private aliases by ASSIGNMENT,
thus gate D6 stays satisfied.

**The one new external dependency.** This file, alone in `core`, also
imports `sellcsigma`. Every other module of `core` still imports numpy,
scipy, jax and `core` only. The structure gates read `fodiwalk.*` imports
only and are therefore silent on it, thus it is recorded here and in
`core/__init__.py` rather than left for a reader to discover.

**The evidence.** [forcedirected/PARITY.md](../../forcedirected/PARITY.md)
(moved there 2026-08-26, section 24). The
copy was byte identical
(sha256 `cfdbe266...318d`) before any edit, and against the `fodined` copy
the algorithm region is identical under `ast.unparse` with docstrings
stripped. `build_ladder` exact over 45 cases; `make_plan` exact on every
array, dtype and stat over four graphs; `step` bit-exact on Cora and on a
path graph, and 1.4e-09 to 3.2e-09 relative where a split hub row makes the
scatter nondeterministic -- which is section 15's mechanism and not this
change.

**NOT claimed.** No bit equality of a long run. Section 15 and
`experiments/fdwalk/FINDINGS.md` lines 1100-1182 show a 2000-epoch feedback
loop amplifies one 1.0 ULP scatter to about 5.4e-06 relative with no change
to the code at all.

**Adopted at.** 2026-08-25, repository at `89abaf2`. The consumers were NOT
rewired: they keep their import paths through the two forwarders. A rewire
is a separate decision and was deliberately not taken here.

**Provenance.** `sellcsigma/sell_c_sigma.py` and `sellcsigma/PARITY.md`,
both ABSORBED into `forcedirected/` on 2026-08-26 (section 24) --
[forcedirected/sell_c_sigma.py](../../forcedirected/sell_c_sigma.py),
[forcedirected/PARITY.md](../../forcedirected/PARITY.md);
[core/sell_c_sigma.py](../core/sell_c_sigma.py),
[../../fodined/embedding/sell_c_sigma.py](../../fodined/embedding/sell_c_sigma.py).

---

## 24. UPDATE 2026-08-26 -- the engine becomes the root package `forcedirected`, and `sellcsigma` is absorbed into it

**Reason for the update.** Two of them, and they are one decision.

1. `ForceDirectedEmbedding` goes back to **`ForceDirected`**. The 2026-08-21
   rename (section 21) named the class for what it stopped doing. The class
   is the force-directed relaxation engine, and that is what the name says.
2. `sellcsigma` (section 23) was a ROOT package for ONE kernel with ONE
   caller. The reason it sat at the root -- `fodined` reads it and must not
   be made to depend on `fodiwalk` -- applies to the ENGINE as well, and the
   engine already reads the kernel. A separate package for the kernel alone
   was therefore overkill: the same import rule, one package instead of two.

**What it is.**

```
forcedirected/            the shared engine. numpy, scipy, jax ONLY.
  __init__.py             the public surface
  force_directed.py       class ForceDirected, class Callback_Base
  sell_c_sigma.py         build_ladder, make_plan, step, PlanCache, to_csr
  csr.py                  row_of, n_rows
  optim.py                RULES, STATE_ARRAYS, state_arrays
  tests/                  test_parity.py, reconstruct_pre_unification.py,
                          m1_old_vs_new.py
  PARITY.md               the kernel evidence, now with section 9
```

**THE IMPORT RULE, and it is the whole reason for the shape.**
`forcedirected` imports numpy, scipy, jax and its own modules, and NOTHING
of this repository. `fodined` reads it, `fodiwalk` reads it, and neither
reads the other. Every dependency points AT the engine, thus a cycle is
impossible. This is section 23's rule, unchanged, applied to a bigger unit.

**Why `csr.py` and `optim.py` moved too, and it is not scope creep.**
`force_directed.py` does `from .csr import n_rows` at module level and
`from ..misc import optim` inside `set_rule`. At the root the second points
outside the package, and `updateZ` dispatches through those rules, thus the
engine cannot run without them. Both moved; both old paths forward.

**The forwarders.** Four, all of them logic-free:

| path | forwards to |
| --- | --- |
| [`core/force_directed.py`](../core/force_directed.py) | `forcedirected.force_directed` |
| [`core/sell_c_sigma.py`](../core/sell_c_sigma.py) | `forcedirected.sell_c_sigma` (was `sellcsigma`) |
| [`core/csr.py`](../core/csr.py) | `forcedirected.csr` |
| [`misc/optim.py`](../misc/optim.py) | `forcedirected.optim` |

Plus [`fodined/embedding/sell_c_sigma.py`](../../fodined/embedding/sell_c_sigma.py),
repointed from `sellcsigma` to `forcedirected`. `fodined` still imports no
`fodiwalk`, and it must never be made to.

Each forwarder imports the public names BY NAME -- never `import *` -- and
binds `_step` / `_to_csr` by ASSIGNMENT, thus gate D6 stays satisfied. The
`optim` forwarder re-exports the SAME function objects, which is forced:
`tests/test_smoke.py` asserts `fw.rule is optim.RULES["plain"]` with `is`,
and a forwarder that rebuilt the dict would fail it. That failure would be
correct, and the fix would be the forwarder.

**NO `ForceDirectedEmbedding` alias is kept, anywhere.** The user asked for
a rename, and a lingering alias is how two names for one class survive
forever. A caller that still says the old name gets an `ImportError`. The
name is recorded here, in section 21 and in the module docstrings as
HISTORY, and nowhere as an import.

**The kernel moved with sha256 UNCHANGED**, `0406d2ec...bebe`, before and
after. That matters beyond tidiness:
[`forcedirected/tests/reconstruct_pre_unification.py`](../../forcedirected/tests/reconstruct_pre_unification.py)
is the only route back to the pre-unification `core/sell_c_sigma.py`, it
works by reversing enumerated edits, and it pins that exact sha. One line
changed in it -- `SOURCE` -- and it still prints `cfdbe266...318d`, `MATCH`,
exit 0 from the new home. `sellcsigma/` was untracked, thus its removal was
permanent and the sha check ran BEFORE it.

**One stale name survives on purpose.** `forcedirected/sell_c_sigma.py`
line 152 still reads `Parity record: sellcsigma/PARITY.md`. Its sha256 is
the anchor of the reconstruction above; an editorial fix inside that file
would break the route back. `PARITY.md` section 9 states the correction.

**The gates.** No number changed, and that was the requirement. Golden
`GOLDEN OK (both)` with the augment half BYTE EXACT; `check_api` 39 Config
fields, 12 methods, 17 attributes, 24 keys; the `fodiwalk` suite filtered
`66 passed, 2 skipped, 9 deselected`, the same two structural skips as
before; `forcedirected/tests` `9 passed`. Identical to the baselines
measured before the move.

**The structure gates were NOT weakened.** They read `fodiwalk.*` import
targets only, thus a root package was always invisible to them, and the
`sellcsigma` forwarder of section 23 already relied on that. The silence is
now NAMED: `ALLOWED_ROOT_PKGS` in
[`tests/test_structure.py`](../tests/test_structure.py) states which root
package is allowed and why, and the two size exceptions record that both
files are forwarders now. No rule was loosened and no test was deleted.

**Adopted at.** 2026-08-26, repository at `89abaf2` plus the uncommitted
working tree.

**Provenance.**
[forcedirected/__init__.py](../../forcedirected/__init__.py),
[forcedirected/force_directed.py](../../forcedirected/force_directed.py),
[forcedirected/PARITY.md](../../forcedirected/PARITY.md) section 9,
[core/__init__.py](../core/__init__.py).

---

## 25. UPDATE 2026-08-27 -- the four forwarders are DELETED; `fodiwalk` reads `forcedirected` directly

**Reason for the update.** Section 24 left four logic-free modules in
`fodiwalk` -- `core/force_directed.py`, `core/sell_c_sigma.py`,
`core/csr.py`, `misc/optim.py` -- so that no caller had to change on the day
the engine moved. That was the right call for the move and the wrong shape
to keep. A forwarder is a SECOND NAME for one thing: a reader who opens
`fodiwalk/core/sell_c_sigma.py` to find the kernel finds an empty file and a
pointer, and must open a second file to learn there was never anything in
the first. The move is done, thus the scaffolding goes.

**What changed.** Every importer names `forcedirected` now:

| file | was | is |
| --- | --- | --- |
| [`base.py`](../base.py) | `from .core.force_directed import ForceDirected` | `from forcedirected import ForceDirected` |
| [`model.py`](../model.py) | `from .core.force_directed import ForceDirected` | `from forcedirected import ForceDirected` |
| [`core/forces.py`](../core/forces.py) | `from .csr import row_of` | `from forcedirected import row_of` |
| [`core/__init__.py`](../core/__init__.py) | three `from .<forwarder> import ...` | one `from forcedirected import ...` |
| [`misc/__init__.py`](../misc/__init__.py) | `from .optim import RULES, ...` | `from forcedirected import RULES, ...` |
| [`embed/planner.py`](../embed/planner.py) | `from ..core.sell_c_sigma import make_plan, step` | `from forcedirected import make_plan, step` |

Then the four files were deleted. `experiments/fodiwalk/bench_fodiwalk.py`
and two test modules followed the one name that had no other route,
`optim`: `from fodiwalk.misc import optim` is now
`from forcedirected import optim`.

**WHAT STILL WORKS, and what does not.** A NAME that `core/__init__.py` or
`misc/__init__.py` re-exports is untouched: `from fodiwalk.core import
ForceDirected, make_plan, step, row_of` and `from fodiwalk.misc import
RULES, state_arrays` all resolve exactly as before. A MODULE PATH into a
deleted file does not: `fodiwalk.core.sell_c_sigma`, `fodiwalk.core.csr`,
`fodiwalk.core.force_directed` and `fodiwalk.misc.optim` are gone, and an
import of one raises `ModuleNotFoundError` instead of resolving in silence.
That is the intent -- one route to the engine, and it is named.

`core/` keeps its own physics and nothing else: `forces.py` (the force
laws) and `plan_contract.py` (the asserter of the plane contract). Neither
was ever a forwarder.

**THE IMPORT RULE IS UNCHANGED, and it is now checked.** `forcedirected`
imports numpy, scipy, jax and its own modules, and NOTHING of this
repository; `fodined` and `fodiwalk` both read it and neither reads the
other. Before this update the rule had NO gate of its own -- the closest
thing was `test_b1_csr_imports_nothing_of_the_package`, which read the
`fodiwalk` forwarder and would have been deleted with it. It became
[`test_b1_engine_package_imports_nothing_of_this_repository`](../tests/test_contracts.py):
same criterion B1.2, widened from one file to every module of
`forcedirected/`, and it reads the repository root from the FILESYSTEM, thus
a package added tomorrow is judged too. `forcedirected/tests/` is excluded,
because `m1_old_vs_new.py` names both callers on purpose.

**`test_structure.py` was not weakened.** `ALLOWED_ROOT_PKGS` and
`test_fodiwalk_imports_no_root_package_but_the_engine` are unchanged; only
the comments that described the forwarders were corrected. The two
`SIZE_EXCEPTIONS` entries for `core/sell_c_sigma.py` (558) and
`core/force_directed.py` (421) were REMOVED with the files they pinned: a
cap on a file that does not exist asserts nothing. `augment_graph/walks.py`
(422) is the one exception left.

**One dead record was repointed.**
[`forcedirected/tests/m1_old_vs_new.py`](../../forcedirected/tests/m1_old_vs_new.py)
loaded `fodiwalk.core.sell_c_sigma` by name. Its window closed 2026-08-25
(its own docstring says the comparison becomes a module against itself), no
module imports it and pytest does not collect it, thus the line now names
`forcedirected.sell_c_sigma` with a comment saying why. A record that
crashes on import is worse than a record that repeats itself.

**The gates.** No number changed, and that was the requirement.
`A/B OK -- every score matches side A` on all four cases;
`GOLDEN OK (both)`; `API OK: 39 Config fields, 12 methods, 17 attributes,
24 keys`; `pytest fodiwalk/tests forcedirected/tests -m "not big"` gives
`82 passed, 2 skipped, 3 deselected`, the same as the baseline measured
before the change; and `fodined.embedding.sell_c_sigma.make_plan` still
resolves -- `fodined` reaches the same kernel through its own forwarder,
which was NOT touched and must never import `fodiwalk`.

**Adopted at.** 2026-08-27, repository at `89abaf2` plus the uncommitted
working tree.

**Provenance.** [core/__init__.py](../core/__init__.py),
[misc/__init__.py](../misc/__init__.py), [base.py](../base.py),
[embed/planner.py](../embed/planner.py),
[tests/test_contracts.py](../tests/test_contracts.py),
[tests/test_structure.py](../tests/test_structure.py).

---

## 26. UPDATE 2026-08-28 -- the three stages, and nothing else: `core/` is deleted and the LAW moves into `embed/`

**Read this before an earlier provenance link.** Every link above into
`../core/forces.py`, `../core/plan_contract.py`, `../augment_graph/planes.py`
or `../augment_graph/degrees.py` now resolves under `../embed/`, with the
same file name. Nothing above is removed; the table in this section is the
mapping.

**Reason for the update.** Two reasons, and the second REVERSES section 20.

The first is `fodiwalk/core/`. Section 25 deleted its four forwarders and
left two files, `forces.py` and `plan_contract.py`.
`dev-docs/fodiwalk-module.md` names THREE stages -- graph construction,
graph augmentation, embedding -- and `core/` was a fourth home that no
stage boundary describes. A force law is the FUNCTION stage 3 applies, thus
it belongs to stage 3.

The second is the stage contract itself, stated by the project owner:

> Stage 2 prepares the material for stage 3. There is absolutely no
> interaction of any sort between stages other than producing and consuming
> per a predefined contract. The contract being the types and shape of the
> data being handed over between the two stages. Each stage will have its
> own functions and data structures.

Section 20 moved `planes.py` and `degrees.py` into `augment_graph/` on the
reasoning that a plane and a degree are "data preparation of the recipe".
That reasoning is now rejected, and the evidence is in the imports it
forced. To BUILD a plane, `augment_graph/planes.py` had to ask the law what
it reads (`planes_of`, `fuse`, `PLANE_CHECKS`), and `augment_graph/
degrees.py` had to call `degrees_from_D`. Asking is an import. Stage 2 was
CALLING INTO the physics, which is an interaction and not a hand-over. A
plane is defined by the law that unpacks it, a degree divisor by the law
that needs it; both are stage-3 things, and the recipe idea of 20.1 was
what disguised that.

**What moved.**

| from | to |
| --- | --- |
| `core/forces.py` | [`embed/forces.py`](../embed/forces.py) |
| `core/plan_contract.py` | [`embed/plan_contract.py`](../embed/plan_contract.py) |
| `augment_graph/planes.py` | [`embed/planes.py`](../embed/planes.py) |
| `augment_graph/degrees.py` | [`embed/degrees.py`](../embed/degrees.py) |
| `core/__init__.py` | DELETED. `fodiwalk/core/` no longer exists |

`plan_contract.py` had to move WITH `forces.py`: it reads `FORCE_PLANES`
and `planes_of`, thus leaving it in `core/` would have made `core` import
`embed` and turned the dependency upside down.

**No line of a function body changed.** `degrees_from_D`, `fdlinear`,
`fdlinear_fused`, `fuse`, `planes_of`, `force_fn`, `FORCE_PLANES`,
`FORCE_FN`, `ForceSpec`, `PLANE_BUILDERS`, `build_planes`, `force_params`,
`resolve_degrees` and every function of `plan_contract.py` keep their
behaviour, their names and their error messages exactly. `embed/planner.py`
did not move and did not change, except that `from ..core import
plan_contract` became `from . import plan_contract`.

**What NARROWED: the seam, back to DATA.** Section 20 widened
`augment_graph.result.Augmentation` (19.1) with `planes`, `degrees` and
`params`. All three are REMOVED. No code ever filled them or read them --
`Fodiwalk.augment_graph` builds the three into local variables and hands
them straight to `build_plans` -- thus the removal changes no behaviour and
takes three stage-3 names out of the stage-2 dataclass. The seam is again
exactly four fields:

    D: sp.csr_matrix        the weighted matrix; `D.data` is the `h` values
    freq: np.ndarray|None   (nnz,) aligned to `D.indices`, or None
    stats: dict             the walk statistics
    info: dict              the counts and timings a log prints

**The import rules, as they now stand.**

    forcedirected/  imports nothing of this repository. The engine.
    make_graph/     numpy, scipy.
    augment_graph/  numpy, scipy and its own modules. NOTHING else.
    embed/          forcedirected and its own modules. NO augment_graph.
    model.py        all of the above, and it is the ONLY place they meet.

`model.py` is the composition root and belongs to neither stage; carrying
an `Augmentation` from stage 2 to stage 3 is its job. Section 20's widened
rule for `augment_graph` ("numpy, scipy, `core`") is withdrawn.

**The gates that enforce them.** Both directions, and at FUNCTION level as
well as module level, because a leak hidden in a function body is defect D6:

  * [`test_augment_graph_imports_no_other_stage`](../tests/test_structure.py)
    -- allowed set EMPTY. It was `test_augment_graph_imports_no_embed_no_model`
    with `allowed={"core"}`.
  * [`test_augment_graph_imports_no_root_package_at_all`](../tests/test_structure.py)
    -- NEW. `ALLOWED_ROOT_PKGS` lets any stage reach `forcedirected`, which
    is right for `embed` and wrong for stage 2: stage 2 drives no kernel.
  * [`test_embed_imports_no_augment_graph_no_model`](../tests/test_structure.py)
    -- allowed set EMPTY. It was
    `test_embed_imports_only_core_no_augment_graph_no_model`.
  * [`test_the_core_package_stays_deleted`](../tests/test_structure.py)
    -- NEW, beside the god-class gate: a file put back under `core/` is a
    fourth home for the physics that no stage boundary describes.
  * [`test_embed_imports_only_embed`](../tests/test_contracts.py)
    -- the module-level form, renamed from `test_core_imports_only_core`.
  * `test_core_imports_no_embed` is DELETED with the package it judged.

Five negative controls were planted in a scratch COPY of the tree, never in
the live one, and every one was caught: stage 2 reaching `embed` at module
level and inside a function body, stage 2 reaching `forcedirected`, and
stage 3 reaching `augment_graph` at module level and inside a function body.

**The gates.** No number changed, and that was the requirement.
`A/B OK -- every score matches side A` on all four cases;
`GOLDEN OK (both)`; `API OK`; `pytest fodiwalk/tests forcedirected/tests
-m "not big"` gives `82 passed, 2 skipped, 3 deselected`, the same as the
baseline; and `fodined.embedding.sell_c_sigma.make_plan` still resolves.

**Adopted at.** 2026-08-28, repository at `89abaf2` plus the uncommitted
working tree.

**Provenance.** [embed/__init__.py](../embed/__init__.py),
[embed/forces.py](../embed/forces.py),
[embed/plan_contract.py](../embed/plan_contract.py),
[embed/planes.py](../embed/planes.py),
[embed/degrees.py](../embed/degrees.py),
[augment_graph/__init__.py](../augment_graph/__init__.py),
[augment_graph/result.py](../augment_graph/result.py),
[model.py](../model.py), [config.py](../config.py),
[__init__.py](../__init__.py), [README.md](../README.md),
[tests/test_structure.py](../tests/test_structure.py),
[tests/test_contracts.py](../tests/test_contracts.py).

---

## 27. UPDATE 2026-08-28 -- the stage contract, named as an entity

**What it is.** The rule that governs every boundary between the three
stages of `fodiwalk` -- `make_graph`, `augment_graph`, `embed`. A stage
produces material for the next one and consumes material from the one
before, and that is the ONLY thing that happens between two stages. No
stage imports another stage, calls a function of another stage, or reads
another stage's registry, table or constant. What crosses a boundary is
DATA, and the contract is the TYPE and the SHAPE of that data.

**How it works.** Each boundary names one data type, fixed in shape:

```
make_graph     --[ A: symmetric CSR, zero diagonal, sorted indices ]-->
augment_graph  --[ Augmentation: D, freq, stats, info ]-->
embed          --[ Z: (n, n_dim) ]-->
```

`model.py` is the ONLY place two stages meet: it takes what one stage
produced and hands it to the next. It belongs to neither stage.

**Reason for the update.** Given by the project owner, in answer to a claim
that "the stage that prepares the values has to ask the force law which
values it needs" -- REPUDIATED:

> Stage 2 prepares the material for stage 3. There is absolutely no
> interaction of any sort between stages other than producing and consuming
> per a predefined contract. The contract being the types and shape of the
> data being handed over between the two stages. Each stage will have its
> own functions and data structures.

Section 20 and section 8 of `REFACTOR.md` argued the repudiated position:
that a plane and a degree divisor are stage-2 "data preparation", which
built `augment_graph/planes.py` and `augment_graph/degrees.py` importing
`core.forces.planes_of`/`fuse` to ASK the law what it reads. That import is
the interaction the ruling forbids. Section 26 above reverses it: the two
files moved to `embed/`, which owns the law, and `core/` is deleted.
`REFACTOR.md` sections 3, 3.1 and 8 are marked superseded and kept for
history; they are not corrected in place.

**The test the ruling gives.** If a stage needs to know the name of a force
law, a policy of another stage, or the shape of another stage's internals,
it is doing work that belongs elsewhere.

```python
D: sp.csr_matrix        # the weighted matrix; D.data holds h
freq: np.ndarray | None # (nnz,) aligned to D.indices, or None
stats: dict              # the walk statistics
info: dict               # the counts and timings a log prints
# this is ALL of `Augmentation` -- the whole seam, stage 2 to stage 3
```

**Adopted at.** 2026-08-27 (the ruling), written into the tree 2026-08-28
with section 26.

**Provenance.** [dev-docs/fodiwalk-module.md](fodiwalk-module.md) section
"The stage contract (2026-08-27)", [augment_graph/result.py](../augment_graph/result.py),
[dev-docs/CATALOG.md](CATALOG.md) section 26,
[dev-docs/REFACTOR.md](REFACTOR.md) sections 3, 3.1, 8.

## 28. UPDATE 2026-08-28 -- `RowStats` and `RowCSR`, the row-blocked carrier of `nbr_walk`

**What it is.** The task: `nbr_walk`'s augmentation peaked at 3067 MB to
build a 495 MB result at 150,000 nodes of com_youtube (6.2x), which killed
the run at the real size, 1,134,890 nodes, on a 15 GB machine. Two new
types replace the one-array carrier the pair (`key = row * n + col`) and
the `D`/`freq` container (`scipy.sparse.csr_matrix`) used before:

  `RowStats`  a ROW-BLOCKED carrier -- `indptr`/`col`/`mn`/`cnt`, the
              layout a CSR already keeps in `indptr`/`indices`. Row `u`'s
              partners are `col[indptr[u]:indptr[u+1]]`, ascending, no
              duplicate. `col` is int32, `mn` is int16 (`h` fits
              `1 .. walk_len - 1`), `cnt` is int32. `key` is NOT stored;
              a reader that still wants it (`golden.py`'s digest,
              `check_api.py`) gets it through `stats["key"]`, built ONLY
              on that ask.
  `RowCSR`    a plain object holding `D`/`freq`'s `indptr`/`indices`/
              `data`/`shape`, and not a `scipy.sparse.csr_matrix`. Every
              real consumer -- `forcedirected.sell_c_sigma.make_plan`
              (through a chunk-local rebuild), `embed.degrees.
              resolve_degrees`, `embed.plan_contract.check` -- touches
              only those attributes and `.nnz`; nothing calls a scipy
              MATRIX method (`.dot`, `@`, `.T`, `.getrow()`) on `D`.
              `.tocoo()` wraps into a real `sp.csr_matrix` for the ONE
              reader that needs scipy's own duplicate-summing COO build
              (`merge.add_far_pairs`, `far > 0` only).

**How it works.** Three changes, each one value-preserving (same numbers,
less memory to hold them at once):

1. `walk_rows` emits `RowStats` directly, pre-sizing its output once the
   walk is done instead of `np.concatenate`-ing a list of per-block
   arrays (that concatenate held the whole list AND the new array at
   once: +414 MB at 150,000 nodes).
2. `row_merge.py`'s `_merge_row_edges` merges `RowStats` with `A`'s edges
   ROW BLOCK BY ROW BLOCK, with `np.searchsorted` finding a hit and
   placing every kept entry at its final rank -- a merge of two ALREADY
   SORTED sequences (walk output is row/col sorted by construction; a
   CSR's `indices` are sorted inside a row) needs no `argsort`. The old
   code's ONE global `np.argsort` over 26M keys cost +1707 MB, almost all
   of it the sort's permutation array and the four-array gather it forced.
3. `rows.to_csr_directed_rows` builds `D` (and then `freq`, from `D`'s
   OWN `indptr`/`indices` -- the SAME array objects) with
   `RowCSR(indptr, col, val, shape)` directly: no COO, no sort, no second
   copy of the sparsity structure (99 MB at 150,000 nodes; ~750 MB at
   1.13M, avoided).

```python
# rows.py
class RowStats:
    __slots__ = ("indptr", "col", "mn", "cnt", "n", "extra")
    # indptr: int64 (n+1); col: int32 (nnz); mn: int16 (nnz); cnt: int32 (nnz)

class RowCSR:
    __slots__ = ("indptr", "indices", "data", "shape")
    @property
    def nnz(self): return self.data.size
    def tocoo(self): ...  # the one scipy crossing, for far > 0
```

**Result, measured at 150,000 nodes of com_youtube, default `nbr_walk`
(`walks=10, walk_len=20`), the production path
(`augment_graph.policies.build`, sampled the same way the pre-fix number
was):**

| measure | before | after |
| --- | --- | --- |
| peak RSS | 3067 MB | 1154-1158 MB |
| resting (`D` + `freq.data`) | 495 MB | 495 MB (unchanged) |
| peak / resting | 6.2x | 2.33x |

**The stretch goal, met and not merely attempted.** The real reason this
mattered: at the full size, 1,134,890 nodes, the pre-fix peak
extrapolated to ~23 GB against 15 GB of RAM, and the process died before
it stored anything. After the fix, the full com_youtube augmentation
(default `nbr_walk`, `walks=10, walk_len=20`) completes in 82.5 s, `D.nnz
= 145,222,742`, peak RSS **4626 MB** -- comfortably inside a 6 GB budget.

`golden.py`'s digest hashes an array's dtype along with its bytes, thus
two representation choices needed care to stay byte-exact rather than
becoming a re-record:

  * `RowStats["mn"]`/`["cnt"]` widen back to the pre-fix dtype (`int32`,
    `int64`) ONLY on `__getitem__` -- a compatibility view for
    `golden.py`/`check_api.py`, never taken on the hot path, since the
    STORED object holds the narrow dtype.
  * `to_csr_directed_rows` narrows `indptr` to `int32` when `nnz` and `n`
    both fit (true at every size this project reaches) to match what
    `sp.csr_matrix((data, (row, col)))` already chose -- `RowStats.indptr`
    itself stays `int64` throughout the merge.

A dead end worth recording: a redundant `counts.astype(np.float64)` in
`policy_nbr_walk.build` (present before this task, invisible beside the
COO build it used to sit next to) became the single largest remaining
cost once the COO build and the global sort were gone -- +198 MB, a
second copy of an array `freq_pair`/`freq_node` already returned as
`float64`. Fixed with `copy=False`.

**Scope.** `nbr_walk` only (`policy_nbr_walk.py`, `row_merge.py`,
`rows.py`, and `pairs.row_cap`). `walk`/`walk_edges`
(`policy_walk.py`, `walk_pair_stats`, `pairs.to_csr`/`to_csr_directed`)
keep the flat `key` and a real `scipy.sparse.csr_matrix`, untouched. The
far-pair path (`merge.add_far_pairs`, `far > 0`) is untouched: it depends
on `scipy.sparse` summing a duplicate COO coordinate, and `RowCSR.tocoo()`
is exactly the one crossing that lets it stay that way.

**Adopted at.** 2026-08-28, `agentic-log/10.mem-agent/`.

**Provenance.** [augment_graph/rows.py](../augment_graph/rows.py),
[augment_graph/row_merge.py](../augment_graph/row_merge.py),
[augment_graph/pairs.py](../augment_graph/pairs.py) (`row_cap`),
[augment_graph/walks.py](../augment_graph/walks.py) (`walk_rows`),
[augment_graph/policy_nbr_walk.py](../augment_graph/policy_nbr_walk.py)
(`build`), [embed/plan_contract.py](../embed/plan_contract.py) (the
`sp.issparse` -> `hasattr(D, "nnz")` dispatch fix `RowCSR` needed),
[embed/planner.py](../embed/planner.py) (`build_plans`'s reused per-chunk
scratch buffer), [tests/mem.py](../tests/mem.py) (the memory gate),
`agentic-log/10.mem-agent/`.
