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
