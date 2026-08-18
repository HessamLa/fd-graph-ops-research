# fdwalk -- catalog of entities

Created 2026-08-17T22:53:32-07:00.

**Why this file starts late.** CLAUDE.md asks for a catalog of the named
entities of the project, and no such file existed. The entities below were
built between 2026-08-15 and 2026-08-17 and were described only in
`PLAN.md`, `FINDINGS.md`, and the logs. This file collects them. It is an
INCREMENTING document: nothing is ever removed from it. An update states
its reason, gives the new version, and names the step at which the update
is adopted.

Each entry holds: what the entity is, how it works, a code example where
one applies, and the provenance -- a link to the code.

---

## 1. Force functions

### 1.1 `shell_force` (also `v1`) -- the law of the package

**What it is.** The force law that `fodined` shipped with. It reads two
coefficient planes, `(shell_coeff, h)`.

**How it works.**

```
guard = where(h > 1, 0, 1)
Fa    = guard * k1 * shell_coeff * x * exp(-k2 * (h - h_shift))
Fr    = -k3 * h * exp(-k4 * x)
F     = Fa + Fr
```

`x` is the distance `||z_v - z_u||`. `shell_coeff = 1 / |S_h(u)|`, thus at
`h = 1` it is `1 / deg(u)`.

**Two properties, both measured.**

1. **The attraction exists at `h = 1` ONLY.** `guard` is 0 for every
   `h > 1`. Thus a pair that a walk found, but that is not an edge, gives
   repulsion and nothing else.
2. **`k2` is a dead parameter.** Because `guard` kills every `h > 1`, the
   exponent is always `exp(-k2 * (1 - h_shift))`, one constant. Verified
   numerically identical for `k2` in {0.01, 1.0, 100}.

**Provenance.** [fodined/embedding/shell_force.py](../../fodined/embedding/shell_force.py)

---

### 1.2 `shell_force_v2` (`v2`) -- the off-branch law

**What it is.** The law of the off-branch experiment of 2026-08-16.

**How it works.**

```
h == 1:  F = Fa + Fr,  Fr = -k3 * h * exp(-k4 * x)
h >= 2:  F = Fr,       Fr = -k3 * h            (no exponent, thus no decay)
```

The far repulsion does not fall with the distance. It is constant per
shell.

**Provenance.** [force_offbranch.py](../force_offbranch.py),
report in [offbranch/REPORT.md](../offbranch/REPORT.md)

---

### 1.3 `fdlinear` -- the linear-attraction law

**Adopted at:** the mix-and-match step, 2026-08-17.

**What it is.** The law that the user specified on 2026-08-17. The
attraction is LINEAR in the distance, which gives the name, and it carries
NO `shell_coeff`, thus the pull of an edge does not fall when the degree
grows.

**How it works.**

```
h == 1:  Fa = k1 * x
         Fr = -kr * exp(-k4 * x)
h >= 2:  Fa = 0
         Fr = -(h / freq) * exp(-k4 * x)
```

`freq` is how often the pair was found in the walks. It DIVIDES the
repulsion, thus a node that a row reached many times is pushed away less.
That is the opposite direction from the far-pair bias, which pushes the
hubs more.

**Two readings the specification left open, and the decision for each.**

1. The specification writes `exp(z_v - z_u)`, which reads as `exp(+x)` and
   GROWS with the distance; no layout can settle. The default is
   `exp(-k4 * x)`. `--fdlinear-sign +1` gives the literal reading.
2. "How often node v appeared" has a per-PAIR and a per-NODE reading. The
   per-pair reading is the default, because it is the multiplicity that
   node2vec optimizes. `--freq-mode node` gives the other.

**The engine still divides the row sum by `deg1(u)`** unless the caller
passes `--no-deg-norm`. The grid of 2026-08-17 shows that `degnorm on` is
necessary: with it off, `lr` 1.0 and 0.1 both diverge.

**Provenance.** [force_fdlinear.py](../force_fdlinear.py)

#### UPDATE 2026-08-17T22:45 -- the plane count

**Reason.** `fdlinear` bound a `shell_coeff` plane and never read it. At
`n = 1,134,890` and `nnz = 23,163,843` that dead plane cost a 92.7 MB host
array, a 109.5 MB packed tile, and about 460 MB of transient inside
`shell_counts` -- for a value the kernel drops. This is a defect, not a
design.

**The updated version.** `fdlinear` unpacks TWO planes, `(h, freq)`. The
physics is unchanged, line for line. The earlier three-plane form is
RENAMED to `fdlinear_3plane` and stays callable, as the reference that
every new form must reproduce number for number. Nothing is removed.

**Adopted at:** the memory step of 2026-08-17.

**Provenance.** [force_fdlinear.py:120](../force_fdlinear.py#L120),
the reference at [force_fdlinear.py:59](../force_fdlinear.py#L59)

---

### 1.4 `fdlinear_fused` -- the one-plane form of `fdlinear`

**Adopted at:** the memory step of 2026-08-17, behind `--fuse-planes`. It
is NOT the default.

**What it is.** The same law as `fdlinear`, from ONE coefficient plane.
`h` and `freq` never appear apart in this law, thus one number carries
both.

**How it works.**

```
w = -1.0        h == 1     (a SENTINEL; the h == 1 branch reads no value)
w = h / freq    h >= 2     (always > 0, since h >= 2 and freq >= 1)
w = 0.0         a pad cell
```

```python
def fdlinear_fused(x, planes, params):
    w, = planes
    live = w != 0
    near = w < 0
    ex = jnp.exp(params["sign"] * params["k4"] * x)
    Fa = jnp.where(near, params["k1"] * x, 0.0)
    coeff = jnp.where(near, params["kr"], w)
    Fr = jnp.where(live, -coeff * ex, 0.0)
    return Fa + Fr
```

**Why the sentinel is load-bearing.** `h` has three uses, not one:
`live = h > 0`, `near = h <= 1`, and `h / freq`. A plain `w = h / freq`
keeps the pad mask but DESTROYS the branch: `h=1, freq=3` and `h=2, freq=6`
both give 0.333, thus the attraction fires on the wrong cells and nothing
raises an error. The `h == 1` branch reads neither `h` nor `freq` -- its
coefficient is the constant `kr` -- thus it needs a flag and no value, and
the sign carries the flag at no cost. The three regions are disjoint.

**`degrees_from_D` must be read BEFORE the collapse**, because it counts
the `h == 1` entries and `w` keeps only their sign.

**The cost.** `h / freq` becomes a BUILD-time quantity. `k1`, `k4` and `kr`
stay traced scalars and sweep for free, but a law of the form
`h / freq**beta` could not sweep `beta` without a rebuild of `D`. This is
why the fused form is a flag and not the default: the law is still moving.

**Verification, Cora, `nbr_walk / min_gap`, dim 64, 200 epochs, lr 1.0,
seed 42.** Every embedding-derived metric is identical to `fdlinear`:

| | 3-plane ref | fused |
| --- | --- | --- |
| final \|\|dZ\|\| | 0.5016 | 0.5016 |
| accuracy | 0.9754 | 0.9754 |
| F1 | 0.9752 | 0.9752 |
| AUC | 0.9962 | 0.9962 |
| hop R2 | 0.253 | 0.253 |
| hop MAE | 1.291 | 1.291 |
| t_aug / t_embed | 0.1 / 3.8 s | 0.1 / 3.7 s |
| peak RSS | 1010 MB | 1004 MB |

(`r2_vec` moved 0.606 -> 0.612. That metric is fitted by an sklearn MLP
with its own seed, and the embedding itself is identical.)

**Provenance.** [force_fdlinear.py:133](../force_fdlinear.py#L133),
the host-side builder `fuse` at [force_fdlinear.py:146](../force_fdlinear.py#L146)

**Measured at 1,134,890 nodes, 2026-08-17T23:10** (com_youtube,
`nbr_walk/both`, dim 64, 500 epochs, 8 host chunks, seed 42):
peak RSS **4374 MB against 4799 MB, -8.9%**; `||dZ||` 0.049724 against
0.049724; accuracy, F1, AUC and hop R2 all identical. The saving is
smaller than the 780 MB predicted, because a freed CSR does not lower a
high-water RSS; only the memory never allocated counts. The 9.6% fall in
the embed time is NOT claimed: `t_aug` fell 23.5% on an identical code
path, thus the machine variance covers it.

---

## 2. Augmentation policies

### 2.1 `ball` -- the k-hop ball

**Status: ABANDONED at 2026-08-16, by instruction.** Kept in the catalog
because the measurement is the reason no later policy uses it.

**What it is.** Every node within `k` hops enters the row.

**Why it cannot scale.** It is a property of the hub degree, and no budget
reaches it. On `com_youtube`: 2.5 G pairs at `k = 2` (50 GB), 53.2 G at
`k = 3` (1 TB). On `roadNet-CA`, the same `k` gives 9 and 17 pairs per
node. The policy is not slow; it is unusable on a graph with hubs.

---

### 2.2 `walk` -- the pair set of the random walk

**What it is.** The pair-finding rule of DeepWalk and node2vec at
`p = q = 1`. From every node, `n_walks` walks of `walk_len` steps. Any two
nodes inside one window of the walk make a pair.

**Provenance.** [walks.py](../walks.py), `uniform_walks`, `walk_pair_stats`

---

### 2.3 `walk_edges` -- the walk, plus every edge

**What it is.** `walk` with the capped window policy, and then every edge
of `A` is added at `h = 1`.

**Why the addition is needed.** The attraction lives at `h = 1` only. A
capped walk policy can drop an edge, and a dropped edge is a lost
attraction. Adding the edges restores 100% adjacency coverage.

**Cost at 1.13 M:** the augmentation takes 571.8 s, because the added
edges must be merged into an existing accumulator.

---

### 2.4 `nbr_walk` -- the neighbours, and the walk of the row

**Adopted at:** 2026-08-17, and it is the current default.

**What it is. This is the contract the user stated on 2026-08-17:** "the
embedding of each node is updated based on its neighbouring nodes AND all
the nodes found in its walk". Row `u` holds every neighbour of `u` at
`h = 1`, and every node that a walk STARTING AT `u` reached, at `h` = the
step of first arrival.

**How it works, and why it is fast.** The rows are built in blocks that are
ROW-DISJOINT. Thus no pair must be merged across blocks, there is no global
accumulator, and there is no prune. The prune of `walk_pair_stats` was an
approximation; `nbr_walk` has none.

**Measured at 1.13 M:** the augmentation takes 28.9 s against 571.8 s for
`walk_edges`, a factor of **20**, with 100% adjacency and no approximation.
The cost is a higher peak RSS, 4799 MB against 3451 MB, partly because the
`freq` plane is a second CSR (see 1.4).

**Provenance.** [walks.py](../walks.py), `walk_rows`, `with_all_neighbours`

---

### 2.5 `low_deg` -- the asymmetric edge rule

**Adopted at:** 2026-08-17, as `--edge-rule low_deg`. It is NOT the
default; see the measurement.

**What it is.** For immediate neighbours `u, v`: `v` enters the row of `u`
only if `deg(v) >= deg(u)`. The walk list of `u` is added as usual. The
goal is memory.

**A trap, and the engine hides it.** A hub can now hold NO entry at
`h = 1`. `degrees_from_D` counts exactly those entries, thus it returns 0,
and `inv_deg_ext` turns a 0 into 0.0. The kernel multiplies the WHOLE row
force by that number: every force of the hub becomes zero, the repulsion
too, and the node never moves. Nothing raises an error.
`--deg-source auto` therefore gives `make_plan` the degree of `A`.

**Measured at 1.13 M:** it saves 11.3% of `D.nnz`, 6.1% of the RSS and 5.3%
of the time, and it costs **64% of the hop R2** (0.436 -> 0.157) and 6.2%
of the accuracy. The reason: the attraction lives at `h = 1` only, and the
adjacency coverage falls to 54.6%.

**Provenance.** [walks.py](../walks.py), `with_neighbours_low_deg`

---

### 2.6 Far pairs, with a degree bias

**What it is.** `n * log10(n)` pairs that are NOT stored in `D` are added
at a large weight (`--far-weight 100`), to give the layout a long-range
term. `--far-bias 0.75` draws them proportional to `deg^0.75`, which is the
negative-sampling distribution of word2vec.

**Why it matters.** The long-range term drives the hop R2, and this is
shown four independent ways: removing the bucket policy costs 73%, changing
the landmark range costs 68%, a far weight of 1000 costs 61%, and restoring
the far pairs to `nbr_walk` gains 48.5%.

**A defect, fixed 2026-08-16.** With a DIRECTED `D`, `sample_far_pairs`
rejected on the directed key, thus a pair stored as `(v, u)` did not reject
`(u, v)`, and the CSR build SUMMED the two into weights of 101..119. About
0.18% of the entries. Far pairs are now filtered against BOTH directions.

**Provenance.** [fodined/graph_augmentation.py](../../fodined/graph_augmentation.py),
`sample_far_pairs`, `degree_table`

---

## 3. Edge weights

`weights.py` holds four rules. **All are rounded to an integer** and
clipped to `[1, window]`.

**Why the rounding is not optional.** A continuous weight gives every pair
its own shell, thus `degrees_from_D` returns 0 for every row,
`inv_deg_ext` is 0.0, and the whole force vanishes: the runs gave AUC
0.55/0.70 with `||dZ|| = 0.000`. The invalid rows are kept as
`results/g1_cora_floatweights_INVALID.tsv`.

| name | rule |
| --- | --- |
| `flat` | every pair at `h = 1` |
| `min_gap` | the smallest step gap seen. An UPPER BOUND on the hop distance, provably never under. **The default.** |
| `mean_gap` | the mean step gap, rounded |
| `pmi` | the pointwise mutual information, rescaled and rounded |

**Provenance.** [weights.py](../weights.py)

---

## 4. Gradient update schedules and optimizers

### 4.1 `plain` -- the schedule in use

**What it is.** `Z = Z + lr * dZ`, full-batch, every row every epoch.
`dZ` is the sum of the forces of the row, divided by `deg1(u)` unless
`--no-deg-norm`.

**Status: this is the ONLY rule that any reported result used.**

### 4.2 `momentum`, `nesterov`, `adam`, `fa2`

**Status: WRITTEN, NEVER RUN.** They exist in `optim.py` and are reachable
through `--optim`, and no result in `FINDINGS.md` uses them. CLAUDE.md asks
for the optimizer comparison; it is still open.

**Provenance.** [optim.py](../optim.py)

---

## 5. The layout

### 5.1 SELL-C-sigma plan

**What it is.** Hub-split, width-sort, ladder-quantize, cell-budget pack,
into padded `(nb, R, k)` tiles that a `lax.scan` kernel reads.

**The padding contract.** A pad cell carries EVERY plane at zero, and its
neighbour index is the row's own node id, thus `x = 0` exactly. A force law
built from the planes must vanish there. The engine guards it a second
time.

**The plane contract.** `planes` is a sequence of `(D.nnz,)` arrays,
aligned 1:1 with `D.indices` in `D`'s own pre-split CSR order. `make_plan`
never interprets a plane, thus the plane set is a FORCE-LAW question and
the caller decides it. This is why the fused plane of 1.4 does not violate
the contract.

**Provenance.** [fodined/embedding/sell_c_sigma.py](../../fodined/embedding/sell_c_sigma.py),
`make_plan`, `_step`

### 5.2 Chunking, axis D1

**What it is.** `--chunks k` cuts the plan into `k` ROW RANGES.
`--chunk-host` keeps them in the host memory and moves one at a time.

**Why a row range, and not any set of pairs.** The kernel writes
`dZ.at[rows].add(...)`. A row range gives DISJOINT rows to each chunk, thus
nothing must be summed across chunks, and the chunk and the batch become
the same object.

**Why it was needed.** At 1.13 M and dim 64, `Z`, `dZ`, the transient of
`_step` and the new `Z` are 290 MB each, and the card has 2 GB.

---

## 6. Evaluation protocol

**Link prediction.** At most **50,000** unique node pairs, half positive
and half negative. Negatives come from `sample_far_pairs`, which rejects
any pair stored in `A`. The classifier is a random forest, Hadamard product
`Z[u] * Z[v]`. Ours uses 200 trees; the node2vec baseline uses 100.

**Hop distance.** Pairs from a set of BFS sources, filtered to `hop > 1`.
The feature is the single scalar `||Z[u] - Z[v]||`; the models are a mean
baseline, a random forest and an MLP. **Identical on both sides of the
node2vec comparison.**

**The 2% rule.** A gain counts only if it passes BOTH `delta/old > 2%` AND
`delta > the spread across seeds`. Anything else is noise.

**Every experiment reports the peak RSS and the time of each stage.**

**Provenance.** [fodined/link_prediction.py](../../fodined/link_prediction.py)
