# fdwalk -- TODOs

Populated 2026-08-18T00:06:49-07:00. Before this the file held the
placeholders `a`, `b`, `c`, `d` of its own template; they are replaced, and
the replacement is recorded in `log/2026-08-17T225332.md`.

**How to read this file.** Every item carries WHAT, WHY, HOW, and a GATE --
the number that decides whether the item passed. An item does not move to
**Done** until its gate is measured and its finding is in `FINDINGS.md`.
A new named entity goes in `CATALOG.md` at the same time.

**Standing rules for every item below**, from `CLAUDE.md`:
report peak RSS and the runtime OF EACH STAGE; a gain of 1% or less is
noise, and a gain above 1% is still noise when it is under the seed spread;
verify on a small graph first, then report on a large one.

**Priority.** P0 blocks other work or is a correctness debt. P1 is a
requested experiment. P2 is planned and not blocking.

---

## TODOs

### P0 -- correctness and record integrity

- [ ] **Finish the 1% re-check of the three unnumbered verdicts.**
  WHY: the noise floor moved from 2% to 1% on 2026-08-17. Three verdicts
  were recorded as "inside 2%" with NO number beside them, thus the new
  floor cannot be applied to them from the document. A verdict without its
  number does not survive a change of threshold; this is a record defect,
  not a new experiment.
  HOW: arithmetic on the runs already in `results/`. Two are recovered and
  both still read as noise on AUC: `v1`+buckets vs `v2`+buckets is +0.60%,
  `v2`+cap vs `v2`+buckets is +0.07%. Both are ONE seed, thus neither is
  written to `FINDINGS.md` yet.
  REMAINING: `fdlinear` at `k4 = 1.0` against `k4 = 0.01`, and the three
  seeds for all of them.
  GATE: every verdict in `FINDINGS.md` carries the number that produced it.

- [ ] **Decide whether `--fuse-planes` becomes the default.**
  WHY: it is verified identical at 2,708 and 1,134,890 nodes and saves 8.9%
  of peak RSS, but it makes `h/freq` a BUILD-time quantity, thus a law of
  the form `h/freq**beta` could not sweep `beta` without rebuilding `D`.
  BLOCKED BY: the force law must stop moving first.

- [ ] **Commit, and resolve the blocked push.**
  Three commits are local. The push is blocked by 3.7 GB across five files
  over GitHub's 100 MB limit, introduced by six earlier unpushed commits
  that this branch did not create. Needs a decision between rewriting the
  unpushed history and cherry-picking the fdwalk commits onto a fresh
  branch. **No history is rewritten without that decision.**

### P1 -- the gradient optimizer roster (marked IMPORTANT in CLAUDE.md)

**This is the largest open item of the branch.** `optim.py` holds five
rules and **only `plain` has ever produced a reported number.** Every
result in `FINDINGS.md` is a `plain` result.

- [ ] **Write the two rules that do not exist.**
  - `velocity`: `v = eta*dZ + (1-eta)*v0` ; `Z = Z + lr*v` ; `v0 = v`.
    NOTE: this is an exponential moving average, thus it is NOT the
    `momentum` already in `optim.py`, which is `m = beta*m + dZ` and grows
    the step by `1/(1-beta)`. The two must be separate entries.
  - `sqn`: Stochastic Quasi-Newton. Needs a decision on the form
    (L-BFGS two-loop with a memory of `m` pairs, or a diagonal
    approximation). **`dZ` here is a FORCE and not the gradient of a loss**,
    thus there is no objective for a curvature pair `(s, y)` to approximate.
    Record the choice and its justification in `CATALOG.md` BEFORE running.
    MEMORY: L-BFGS with memory `m` costs `2*m` state arrays of `(n, d)`,
    which is 580 MB for each pair at 1.13M nodes and dim 64. At `m = 5`
    that is 2.9 GB, thus it may not fit beside `D`. Arithmetic first.
- [ ] **Run the roster on Cora and PubMed**, 3 seeds, `fdlinear` and `v1`,
  on the best augmentation. GATE: does any rule beat `plain` by more than
  1% AND more than the seed spread, on AUC or hop R2?
- [ ] **Run the survivors at 1.13M.** GATE: hypothesis H6 says the state
  arrays are the limit. Adam needs 2 arrays = 1.16 GB at dim 128. Measure
  it, do not assume it.
- [ ] **Settle H5**: "momentum helps, Adam does not". The reason to expect
  it: `dZ` is a FORCE, thus its magnitude means "this node must move far",
  and Adam divides that information away.

### P1 -- `cap` and `buckets` stay in the experiment set (2026-08-18)

Decision of the user, 2026-08-18: **both `--policy` values remain live
methods of fdwalk.** An earlier statement in the session called `buckets`
"not on the live path"; that was wrong, and `CATALOG.md` 2.7 carries the
correction. Every grid from here on varies `--policy` as an axis, and does
not fix it at `cap`.

Remember what they are: `cap` runs ALWAYS, and `buckets` is a SECOND stage
on top of it, thus the comparison is "cap only" against "cap, then
buckets". `buckets` has no far pairs by construction, thus read it as a
MEMORY method first (`D.nnz`, peak RSS) and a quality method second.

- [ ] **Implement `buckets` for `--pairs nbr_walk`.**
  WHY: `nbr_walk` is the current best augmentation AND it is the one path
  where `buckets` does not exist. `build_D` branches early and returns
  before the bucket sampler. Until 2026-08-18 that combination silently ran
  `cap` and recorded `policy=buckets`; a guard now refuses the run.
  HOW: the bucket rule is global over `h >= 2`, and `walk_rows` produces
  rows directly, thus the rule must be restated row-wise or applied to the
  concatenated row output before the CSR build.
  GATE: at the same `D.nnz` as `cap`, does it match on AUC and hop R2?

- [ ] **Sweep `--bucket-total`.** The default is `n*log10(n)`, which is the
  size the specification asked for and has never been varied. The 73% hop
  R2 cost may be a budget effect and not a policy effect, and one sweep
  separates them.

- [ ] **Give `buckets` a far-pair option.** The hypothesis that follows
  from every measurement of this branch: `buckets` loses the hop R2 because
  it has NO long-range term, not because its stratification is wrong. Add
  `n*log10(n)` far pairs at `deg^0.75` on top of the bucket budget and
  re-measure. If the R2 returns, the finding is about far pairs and the
  bucket policy is exonerated. **This is the cheapest test of the branch's
  central claim.**

### P1 -- asymmetric `D` for `cap` and `buckets` (requested 2026-08-18)

- [ ] **`cap` with a directed `D`.**
  WHY: `nbr_walk` already uses `to_csr_directed` and keeps its quality, thus
  the asymmetry is cheap where the pipeline is directed. `cap` has never
  been tried this way.
  **A trap that makes this NOT a one-line change:** `_pairs_of` collapses
  direction at the source with the key `min(u,v)*n + max(u,v)`, so that the
  two directions reduce into one group. A directed `cap` therefore cannot
  simply be routed through `to_csr_directed` -- the direction is already
  gone by the time `cap_per_node` runs. Two ways out:
  1. Keep directed keys (`u*n + v`) through `walk_pair_stats`. This DOUBLES
     the pair accumulator, which is the exact structure that stopped the
     first three G3 attempts at 1.13M nodes. Measure the accumulator before
     running the large graph.
  2. **Preferred:** cap the output of `walk_rows`, which is already
     row-disjoint and directed by construction, and needs no accumulator.
     This is the cheaper path and it reuses a tested function.
  GATE: AUC within 1% of symmetric `cap`, with a measured fall in `D.nnz`
  and peak RSS.

- [ ] **`buckets` with a directed `D`.**
  Same trap. Note additionally that `buckets` has NO far pairs by
  construction, and the far pairs are what carries the hop R2 -- symmetric
  `buckets` already costs 73% of it (0.616 -> 0.169 on Cora, force held at
  `v1`). Thus expect this variant to be a memory result and not a quality
  result, and say so before running it.
  GATE: the same. Report `D.nnz`, `pad_frac`, `n_split` and peak RSS.

- [ ] **The 2x2 of both against their symmetric forms**, on Cora first, then
  PubMed, then 1.13M. Three seeds. Report the four cells, not two.
  Cells: {cap, buckets} x {symmetric, directed}. `--policy` is an AXIS of
  this grid and it is not fixed at `cap`.

### P1 -- the second-order walk of node2vec (requested 2026-08-18)

- [ ] **Implement the real node2vec walk (`p`, `q`), not `p = q = 1`.**
  WHY: `uniform_walks` is the DeepWalk walk, which equals node2vec ONLY at
  `p = q = 1`. Every fdwalk result to date uses it. The `p` (return) and
  `q` (in-out) parameters are what let node2vec trade BFS-like structural
  similarity against DFS-like community similarity, and that trade is
  exactly the axis between our two metrics: link prediction and hop
  distance.
  Provenance of the current walk: [walks.py:29](walks.py#L29)
  **The scalability trap, and it is decisive:** the standard node2vec
  preprocessing builds an alias table for every EDGE, at a cost of
  `sum(deg^2)`. com_youtube has a node of degree 28,754, thus that node
  alone contributes 827 million entries. **Alias tables are not an option
  at 1.13M nodes.** Use rejection sampling instead (Yang et al., "Efficient
  and effective"), which needs no per-edge table, and add the reference to
  `REFERENCES.md`.
  GATE: at `p = q = 1` the new walk must reproduce the uniform walk's
  metrics within the noise floor. That is the correctness test, and it runs
  on Cora before any sweep.
- [ ] **Sweep `p` and `q`** on Cora and PubMed: at least
  `(1,1)`, `(1,0.5)`, `(1,2)`, `(0.5,1)`, `(2,1)`. Three seeds.
  GATE: does any `(p,q)` beat `(1,1)` by more than 1% on AUC or hop R2?
  HYPOTHESIS to record before the run: a low `q` (DFS-like, wider
  exploration) should help the hop R2, because the hop R2 is carried by the
  long-range term; a high `q` (BFS-like) should help link prediction.
- [ ] **Run the winner at 1.13M**, and compare against the node2vec
  baseline, which uses `p = q = 1` and would then differ from us on the
  walk as well as the embedder. State that as a confound if it happens.

### P1 -- `edge_sampling`, edge sampling during embedding (requested 2026-08-18)

**The idea, as stated by the user.** At the start of the embedding, use
every `h = 1` and `h = 2` pair plus a random subset of the longer pairs,
resampled EACH EPOCH, thus the early layout works on the local topology. As
the embedding proceeds, the portion of long pairs RISES and the portion of
local pairs FALLS. Expected: lower memory and lower runtime, with small
changes in the other metrics.

**The name: `edge_sampling`, "edge sampling during embedding".**
Decision of the user, 2026-08-18.

**A rename, with its reason, because the first name was wrong.** I first
proposed `shell_anneal`, on the argument that `shell` is the project's word
for the level set `S_h(u)`. **The user rejected it, and the rejection is
correct:** fdwalk is a WALK-based method and it has diverged from the
shell/k-ball formulation by miles. `h` here is a walk gap, not a shell
index, and `fdlinear` reads no `shell_coeff` at all -- the plane was
removed on 2026-08-17 precisely because the law never used it. A name
carrying `shell` would point at the abandoned formulation. The earlier name
is kept in this record and is used nowhere in the code.

**What the name says.** The sampling happens during the EMBEDDING, and not
during the augmentation. That is the whole distinction: every policy of
this branch so far (`cap`, `buckets`, `low_deg`, `row_cap`) decides the
edge set ONE time, before the first epoch. This one decides it again at
every epoch.

**Where it belongs.** This is axis D3 of PLAN.md generalised: "D1 for the
near pairs, and fresh negative pairs for each epoch, as LargeVis and UMAP
do". D3 has a FIXED mixing ratio; `edge_sampling` puts a schedule on it,
and adds a per-band budget.

#### Trap 1 -- FATAL as stated: the attraction lives ONLY at `h = 1`

**"local edges decrease" cannot include `h = 1`.** In every force law of
this project the attraction is zero for `h > 1`:

* `shell_force`: `guard = where(h > 1, 0, 1)`, and `Fa` is multiplied by it.
  Provenance: [shell_force.py:179](../../fodined/embedding/shell_force.py#L179)
* `fdlinear`: `Fa = where(near & live, k1*x, 0)` with `near = h <= 1`.
  Provenance: [force_fdlinear.py:127](../force_fdlinear.py#L127)

Thus a schedule that removes `h = 1` pairs removes ALL attraction, and a
layout with repulsion only expands without a limit.

**And it fails SILENTLY, which is worse.** `degrees_from_D` counts the
`h == 1` entries of a row; a row that loses them returns 0, `inv_deg_ext`
turns that into 0.0, and the kernel multiplies the WHOLE row force by it.
The node freezes where it stands and nothing raises an error. This is the
identical trap that `low_deg` hit on 2026-08-17.
Provenance: [shell_force.py:135](../../fodined/embedding/shell_force.py#L135)

**Therefore the schedule anneals `h >= 2` ONLY.** `h = 1` is resident for
every epoch. The `h = 2` share may fall, the long share may rise, and the
adjacency never moves. Any variant that drops `h = 1` must first change the
force law, and that is a different experiment.

#### Trap 2 -- "all h=2" must mean the WALK GAP, not the true 2-hop ball

If `h = 2` means the true 2-hop neighbourhood, this is the k-ball at
`k = 2`, which is **the policy this branch abandoned**: 2.5 G pairs and
50 GB on com_youtube, because the ball follows the hub degree.

In fdwalk `h` is the MINIMUM WALK GAP, thus the set `h <= 2` is bounded by
the walk budget `n_walks * walk_len` and is a SUBSET of the true 2-ball.
That is feasible. **The item is only feasible under the walk-gap reading,
and the implementation must assert which one it uses.**

#### The design: a per-band edge budget, fixed to the SELL-C-sigma slots

Requirement of the user, 2026-08-18: **respect an allowed number of edges
for each set, so that the SELL-C-sigma memory optimisation holds.** This is
the right constraint, and it resolves trap 3 below rather than merely
avoiding it.

A fixed row width is what the layout wants. `row_cap` already measures it:
a cap of `m` gives a maximum row width of exactly `m` and exactly `n * m`
entries, thus `n_split` falls to 0, `pad_frac` falls from 0.154 toward 0,
and the batches balance. Provenance: [walks.py](../walks.py), `row_cap`.

**But a fixed row width and trap 1 are in direct conflict, and the conflict
must be resolved in the design and not discovered in a run.** Trap 1 says
every `h = 1` entry stays resident. A hub of com_youtube has degree 28,754.
No fixed row width `m` of a useful size holds that.

**The resolution: TWO plans, both built one time.**

    D_near      every h = 1 pair. Resident, never resampled, built once.
                5,975,248 entries at 1.13M nodes -- the adjacency, and the
                cheap part.
    D_sampled   a FIXED budget of `m` slots for each row, refilled each
                epoch from the h >= 2 pairs by the schedule.
                n * m entries, exactly, by construction.

    dZ = dZ_near + dZ_sampled

Both plans have a FIXED geometry, thus neither is ever rebuilt: only the
`nbrs` indices and the plane values of `D_sampled` change per epoch, and
their SHAPES do not. The force law is a sum over the stored pairs of a row,
thus splitting the sum across two plans is exact -- the same property that
permits the chunking of axis D1.

**What it saves, in arithmetic, at 1.13M nodes.** Today `nbr_walk/both`
holds 23,163,843 entries. The adjacency is 5,975,248 of them. At `m = 8`:

    5,975,248 + 1,134,890 * 8 = 15,054,368 entries, a fall of 35%

and, more importantly, **the `h >= 2` set is never materialised in full**,
which is the part that matters because a freed object does not lower a
high-water RSS.

**One consequence to carry:** with `D_near` and `D_sampled` separate, the
force-law degree must come from `A` and not from a count over one of the
two matrices. `--deg-source A` already does exactly this, and it exists for
the `low_deg` trap. Reuse it; do not re-derive it.

#### Trap 3 -- a naive implementation would RAISE the runtime

Resampling the long pairs each epoch changes the sparsity pattern, and the
SELL-C-sigma plan BAKES IN the neighbour indices. A naive form rebuilds the
plan every epoch, which would RAISE the runtime and contradict the
expectation. The plan build is a large part of the 3583 -> 4374 MB step at
1.13M nodes.

**The form that can deliver both claims:** build the plan ONE time with the
resident `h <= 2` pairs plus a FIXED slot budget for the long pairs, and
each epoch refill only those slots -- same `nnz`, same tile shapes, same
plan, only the `nbrs` and the plane values change in the far slots. Then
the peak never holds the whole long-pair set, and no replanning happens.

**A lesson that applies directly**, from 2026-08-17: freeing an object does
NOT lower a peak RSS, because RSS is a high-water mark. **Only memory never
allocated counts.** Thus the long pairs must never be materialised in full,
not merely freed after use.

#### The items

- [ ] **Fix the reading of `h` and the resident set.** Assert walk-gap
  semantics; `h = 1` resident for every epoch.
- [ ] **Implement the two-plan form** above: `D_near` resident, `D_sampled`
  at a fixed `n * m` budget, both planned one time, `dZ` summed.
- [ ] **Choose the per-band split of `m`.** How many of the `m` slots go to
  `h = 2`, to `h = 3`, and to the long pairs, and how that split MOVES with
  the epoch. This is the schedule, and it is the experiment.
- [ ] **Choose the schedule shape** and record it in `CATALOG.md` before
  running: linear, cosine, or step, over what fraction of the epochs, and
  whether the TOTAL pair count is constant (a crossfade) or falls.
  A constant total is the form that keeps the memory flat and makes the
  saving come from never materialising the long set.
- [ ] **Guard, required, from G1 finding 1:** log the count of `h = 1`
  entries in EVERY epoch. If it falls, stop. `mean_gap` collapsed from a
  version of this defect and the run reported an AUC rather than an error.
- [ ] **Cora, then PubMed, 3 seeds**, against `plain` at a fixed pair set.
  GATE: AUC and hop R2 within the 1.5% floor of the fixed-set run, with a
  MEASURED fall in peak RSS and in the runtime of each stage.
- [ ] **1.13M.** This is where the claim is worth something. GATE: peak RSS
  below the 4374 MB of the fused-plane run, and the wall clock below 11:20.
- [ ] **Measure `pad_frac` and `n_split` of `D_sampled`.** The fixed row
  width predicts `n_split = 0` and `pad_frac` near 0, against 0.154 today.
  If that does not happen, the SELL-C-sigma benefit is not being taken.
- [ ] **Record it as a `gradient update schedule` entity** in `CATALOG.md`,
  which is the category `CLAUDE.md` names for it.

### P1 -- attribution, which is owed from 2026-08-17

- [ ] **Decompose the four-change combination at 1.13M.**
  WHY: the run that took AUC from 0.8988 to 0.9954 changed FOUR things at
  once: `walk_edges` (100% adjacency), `fdlinear` at `k4 = 1.0`, `lr = 0.1`,
  and far pairs at `deg^0.75`. The parts measured alone sum to about 4%,
  and the combination gives 10.7%. **The attribution is not established.**
  HOW: four runs, each holding the other three fixed.
  GATE: the parts account for the whole, or the interaction is named.

- [ ] **The clean `v1` + `buckets` run at 1.13M.**
  WHY: `FINDINGS.md` states twice that the cause of the 0.90 AUC is
  UNKNOWN, and names this run as the clean test. It has never run.

### P2 -- Axis E, the remaining forms of asymmetry

The three forms must not be mixed in one run. See `PLAN.md` Axis E.

- [ ] **(E-B) Structural asymmetry from the cap** -- row `u` holds exactly
  its own best `m`, thus EVERY row has the same width.
  WHY it is the best ratio of payoff to work: no hub split
  (`n_split` -> 0), little padding (`pad_frac` is 0.154 today), balanced
  batches, and an exact bound of `n*m` cells.
  NOTE: this overlaps the directed-`cap` item above; do them together and
  report the two effects apart.
- [ ] **(E-A) Upper-triangle storage** -- store one direction and let the
  kernel scatter `+F` to `u` and `-F` to `v`. Halves the cells, about 10M
  at 1.13M nodes, the largest item after `Z`. Changes NO semantics.
  BLOCKED BY: one question with no answer yet -- a shared force needs ONE
  normalisation, and the two rows divide by different degrees today, thus
  "whose degree" must be decided first. It also needs a kernel change:
  `_step` writes one endpoint (`dZ.at[rows].add`).
- [ ] **(E-C) Directed semantics** -- Cora and PubMed ARE citation graphs
  and the loader symmetrises them. Keeping the direction changes the TASK
  and not only the engine, thus it must not ride with (A) or (B).
- [ ] **Watch for a non-converging update cycle** in every asymmetric run.
  The user's condition: a cycle is addressed only IF the system fails to
  converge, by a cycle detector or a decreasing learning rate. The signal
  to watch is final `||dZ||`.

### P2 -- Axis D, bounding the memory

Peak RSS is the ONE gate G3 still fails: 4374 MB against a 2600 MB gate,
and the augmentation stage alone holds 3583 MB before the plan is built.
**No change to the plane count reaches the rest** -- it is in `D` and in
the walk stage.

- [ ] **(D2) No `D` at all** -- the pairs of each epoch come from fresh
  walks. Bounds the plan, the device, AND the pair accumulator.
  THE RISK, from G1 finding 1: the weight 1 must mean adjacency. A fresh
  sample gives the gap of THAT sample, not the minimum over all walks, thus
  a pair can arrive at gap 3 in one epoch and gap 1 in another. `mean_gap`
  collapsed for a version of this defect.
  REQUIRED GUARD: measure the count of weight-1 entries in every batch, and
  stop if it falls.
- [ ] **(D3) D1 for the near pairs, fresh negatives each epoch**, as
  LargeVis and UMAP do. The same bound as D2 with a stable near set, thus it
  is the safer form and should be tried first.
- [ ] **Attack the augmentation stage directly** -- 3583 MB, the largest
  single term. Not yet analysed at the allocation level.
  A LESSON that applies here, learned 2026-08-17: freeing an object does
  NOT lower a peak RSS, because RSS is a high-water mark and a Python free
  returns pages to the allocator and not to the OS. **Only memory that is
  never allocated counts.** This is why the fused-plane saving came in at
  425 MB against a 780 MB prediction.

### P2 -- diagnostics that are owed

- [ ] **`--save-z`**, so that an embedding can be examined after a run.
  BLOCKS the two items below.
- [ ] **(D1) Degree-stratified AUC** -- is the residual error concentrated
  on the hubs? This is the standing hypothesis for the 0.90 AUC.
- [ ] **(D2) Adjacency recovery** -- for each node, how many of its true
  neighbours are among its nearest embedded neighbours.

### P2 -- coverage gaps in what has already been run

- [ ] **The mix-and-match grid on PubMed.** Only Cora and 1.13M have run,
  thus the middle size is missing for the current best configuration.
- [ ] **Three seeds at 1.13M.** Every 1.13M number in `FINDINGS.md` is seed
  42 alone, thus no 1.13M result has a measured spread, and the 1% rule
  needs one.
- [ ] **Re-run the node2vec dim-64 baseline for a time measurement.** Its
  dim-64 training took 854.7 s against 732.5 s at dim 128, which is
  backwards. The machine carried at least 23% of timing variance that day
  (`t_aug` moved 23.5% on a byte-identical code path). **No wall-clock claim
  against node2vec is safe until this is repeated.**

### P2 -- open hypotheses from PLAN.md

- [ ] **H5** momentum helps, Adam does not -- blocked by the optimizer roster.
- [ ] **H6** the optimizer state is the limit at 1M -- arithmetic says yes,
  and it is not measured.
- [ ] **H7** the far weight is not sensitive -- never swept.

### P3 -- the elaborate experiment (GATED, do not start early)

`CLAUDE.md`: run this **only if a model has passed all gates and performs
well on one very large-scale graph.** G3 still fails on memory, thus this
is NOT started.

- [ ] Methods: node2vec, and the best fodined configurations.
- [ ] Dimensions **in this order: 128, 16, 32, 64**.
- [ ] Several very large graphs, not only com_youtube.
- [ ] Every cell reports peak RSS and the runtime of each stage.

---

## Done

- [x] **G1, Cora** -- 10 variants x 3 seeds. `min_gap` passes; `mean_gap`
  and `pmi` fail, and G1 finding 1 gives the mechanism: a continuous weight
  makes every pair its own shell, `degrees_from_D` returns 0, and the whole
  force vanishes. 2026-08-15.
- [x] **G2, PubMed** -- walk/min_gap, ball/min_gap, walk/flat. All pass. 2026-08-15.
- [x] **The k-hop ball is abandoned** -- 2.5 G pairs at k=2 on com_youtube
  (50 GB) and 53.2 G at k=3 (1 TB), against 9 and 17 per node on
  roadNet-CA. The size follows the hub degree, thus no budget reaches it.
  Decision of the user, 2026-08-16.
- [x] **H2 confirmed** -- the minimum walk gap bounds the hop distance:
  98.1% exact, 0.0% under. 2026-08-15.
- [x] **H3 refuted** -- the pair set decides link prediction and the weight
  decides the geometry. 2026-08-15.
- [x] **Far pairs at `deg^0.75`** -- +3.9% AUC, +5.0% accuracy, `||dZ||` 5x
  lower, for one array of `n` float64 (9 MB). Mean endpoint degree moves
  14.09 -> 209.85. 2026-08-17.
- [x] **Landmarks rejected** -- no metric gains on Cora or PubMed. The
  finding survives the move to the 1% floor. 2026-08-16.
- [x] **The `nbr_walk` augmentation** -- the contract the user stated: row
  `u` holds every neighbour of `u` AND everything `u`'s walk reached. Rows
  are built in row-disjoint blocks, thus no accumulator, no prune, and no
  approximation. 28.9 s against 571.8 s at 1.13M, a factor of 20, at 100%
  adjacency. 2026-08-17.
- [x] **`low_deg` measured and rejected** -- saves 11.3% of `D.nnz`, 6.1%
  RSS, 5.3% time; costs 64% of the hop R2 and 6.2% accuracy, because
  attraction lives at `h=1` only and coverage falls to 54.6%. 2026-08-17.
- [x] **The `fdlinear` force law**, and its `--no-deg-norm` finding: the
  engine's degree division is REQUIRED, and without it `lr` 1.0 and 0.1
  both diverge. 2026-08-17.
- [x] **The dead `shell_coeff` plane removed from `fdlinear`.** It was
  built for every run and never read. 2026-08-17.
- [x] **`fdlinear_fused`, one plane with a sign sentinel** -- verified
  identical at 2,708 and 1,134,890 nodes (`||dZ||` 0.049724 both), peak RSS
  4374 MB against 4799 MB, -8.9%. 2026-08-17.
- [x] **The node2vec baseline re-run at an equal dim 64** -- link
  prediction is a TIE (0.9978 against 0.9964, +0.14%, noise), the hop R2
  holds at 10.4x (0.436 against 0.042), and memory is the only axis
  node2vec clearly wins (1382 MB against 4374 MB). 2026-08-17.
- [x] **`CATALOG.md` created** -- it was required from the start and did
  not exist. 2026-08-17.
