# METROLOGY: how a method is measured in fd-graph-ops-research

Status: DRAFT v0.1, 2026-09-02. Owner: project owner. Enters the repository
at `evaluator/dev-docs/METROLOGY.md` after approval. `evaluator/` has one
owner session (fdmap-68); a metric in section 6 marked PLANNED enters
through that session and nowhere else.

This document says WHAT is measured, ON WHAT graph, AT WHAT dimension, HOW
MANY times, and WHEN a difference counts. A number that does not name all
five is not a result.

---

## 1. The one rule

A result record carries, always:

```
method  graph  tier  class  n_dim  epochs  seeds  protocol  protocol_modified
lr  effective_lr  update_rule  strategy  device  peak_RSS_MB  wall_s
```

plus the scores. Anything missing makes the row unquotable. This extends
`evaluator`'s rule of 2026-08-26 ("a number is comparable to a number with
the same protocol name, and to no other") from the protocol to the whole
measurement context.

Two numbers are never reported alone:

- A link-prediction score without a distance score. `roadnet_ca` reaches
  AUC 1.0000 with a NEGATIVE hop R2 (streaming campaign, 2026-08-28). The
  two tasks measure different things.
- A quality score without its cost. A frozen embedding wins every stability
  score and loses every quality score; a 2000-epoch run wins hop R2 and
  loses link prediction (`REPORT_1M.md`, peak at epoch 200).

---

## 2. Vocabulary

| Term | Meaning |
| --- | --- |
| method | a `(pair policy, weight, force law, update rule, strategy)` tuple and its parameters. A change to any element is a new method. |
| strategy | `precomputed` (walk once, store `D`) or `streaming` (walk per epoch, discard). Not interchangeable for schedules: decay REVERSES between them (CATALOG 29.4). |
| protocol | a frozen set of scorer settings, by name (`evaluator/config.py:PROTOCOLS`). |
| tier | a size band of graphs, section 3. |
| class | a structural band of graphs, section 4. |
| regime | a dimension band, section 5. |
| level | an evaluation stage, L0 to L6, section 7. Each level has fixed tier, regime, seed count, metrics and a pass rule. |
| noise floor | the smallest difference that counts, section 8. |
| gate | a pre-registered pass rule for one level on one graph. Numbers are fixed BEFORE the run. |

---

## 3. Graph tiers, by size

| Tier | Nodes | Current graphs | Used for |
| --- | --- | --- | --- |
| T0 synthetic | 10^2 to 10^4 | SBM, ring lattice, grid, balanced tree, Barabasi-Albert, Watts-Strogatz (to add; none in `datasets.py`) | ground truth that no real graph gives: known hop distances, known communities, known planarity, known dynamics |
| T1 small | < 10^4 | `cora` (2,708) | proof of concept, a brand-new idea, the first smoke run, optimiser screening |
| T2 medium | 10^4 to 10^5 | `pubmed` (19,717); `wordnet` (82,115); `com_youtube` cut to 150,000 by BFS ball | memory and time. This is the tier where the augmentation cost was found and fixed (6.17x -> 2.41x peak/resting at 150k). Small enough for many runs, large enough that data beats the runtime floor. |
| T3 large | > 10^6 | `com_youtube` (1.13M), `as_skitter` (1.70M), `roadnet_ca` (1.97M), `ncbi_taxonomy` (2.94M) | the scale claim. One or two runs per method. Never used to screen. |

Rules for a tier:

- A truncated graph names its cut: `graph@150k/bfs` or `graph@150k/subtree`.
  A BFS ball on a tree and a subtree walk on a graph both score perfectly on
  an empty problem (CATALOG 4.11). The cut is part of the graph name.
- T2 is NOT a proxy for T3 on memory. Streaming loses to precomputed at
  150,000 nodes (1,523 MB against 1,467 MB) and wins by 3.5x at 1.13M. A
  memory claim is made at the tier it applies to.
- T1 is NOT a proxy for anything but itself on hop distance. `cora` hop R2
  runs 0.22 to 0.44 where `pubmed` runs 0.40 to 0.62; cora has 78 connected
  components.

---

## 4. Graph classes, by structure

The 2026-08-28 campaign found that hop-distance quality splits by AVERAGE
DEGREE near 3, sharply, with nothing in between. Size does not predict it.
So a graph has a class, and the class decides the regime (section 5) and
which distance metric is primary (section 6).

| Class | Definition | Current graphs | Distance metric that applies |
| --- | --- | --- | --- |
| H, hub | avg degree > 3, heavy-tailed degrees, small diameter | `cora` 3.90, `pubmed` 4.50, `com_youtube` 5.27, `as_skitter` 13.08 | hop R2 and rank correlation both readable. Hop distances take 5 to 8 values; tie fraction is high. |
| S, sparse / tree / planar | avg degree <= 3, or a tree, or planar | `wordnet` 2.06, `ncbi_taxonomy` 2.00 (trees); `roadnet_ca` 2.82 (planar, max degree 12, diameter in the hundreds) | hop R2 is NOT the metric. It reads 0.00 to 0.02 at every epoch count. Use per-shell rank correlation up to a resolution horizon `h*`, and stress-1. Report the horizon. |
| B, bipartite | two node sets, no edge inside a set | none loaded. DyGLib Wikipedia, Reddit, MOOC, LastFM are candidates | degree normalisation must be tested first; every 2-hop pair is same-side. |
| L, labelled | node labels available | `cora`, `pubmed` carry labels; not loaded today | label-free structure scores (section 6.6) stay primary. Node classification is out of scope by decision. Labels are used only for NMI/ARI against communities. |
| W, weighted | edge weights available | none loaded | the true distance becomes the weighted shortest path. `evaluator/hops.py` reads unweighted BFS today; a weighted protocol is a new named protocol, not an override. |
| D, directed | `A` not symmetric | none loaded (`datasets.load` symmetrises) | out of scope until a directed force law exists. A directed `D` from a symmetric `A` is a strategy choice, not a class. |
| C, multi-component | more than one connected component | `cora` (78 components) | cross-component pairs have infinite hop distance. The scorer must say what it does with them: drop, or cap at `diameter + 1`. Today: `hops.py` draws inside BFS rows, so they are dropped silently. Name it. |
| T, temporal | edges carry time | none loaded. HEP-Th, CollegeMsg, AS-733 are tier-1 candidates (see `fodi dynamic`) | section 6.7 |

A graph is in the class its structure gives it, not the class its dataset
is usually filed under. `com_youtube` is a social graph and class H;
`roadnet_ca` is class S. A new graph is classified by `avg degree`,
`max degree`, `diameter estimate`, `component count` and `degree
assortativity`, and these five numbers are printed in every report header.

---

## 5. Dimension regimes

The dimension is a measurement setting, not a method parameter. It is set
by the class of the graph and the level of the evaluation.

| Regime | `n_dim` | When |
| --- | --- | --- |
| R-quick | 32 or 64 | L1 smoke, L3 optimiser comparison, anything where turnaround matters. 64 is the campaign default and the dimension of every recorded fodiwalk number. |
| R-baseline-H | 128, and 256 where memory allows | class H at L4 and L6. This is the native dimension of node2vec and DeepWalk. A comparison against them is made HERE first, then at lower dimensions. |
| R-robust-H | 64, 32 | class H at L5, after R-baseline-H. The claim is "the method holds at a quarter of the baseline dimension". It is a second claim, and it is reported beside the first, never instead of it. |
| R-low-S | 8, 4, 3, 2 | class S. A road network is near-planar and a tree has intrinsic dimension near 1 to 2; 64 Euclidean dimensions hold neither (campaign finding). At 2 and 3 the layout is also a drawing, and stress-1 against the graph-theoretic distance is the native score of the graph-drawing literature. |
| R-stress | a sweep, 2 to 256 in powers of 2 | L5 only, one graph per class, to find where each score collapses. This gives the paper's dimension-versus-quality figure. |

Rules:

- A method is compared to a baseline at the BASELINE's regime first. A win
  at 64 against node2vec at 128 is a win on two axes at once, and the
  reader cannot separate them. Run node2vec at 128, then at 64; run ours at
  128, then at 64; report four cells.
- `Z` at 128 dimensions on a T3 graph is 960 to 1,434 MB (`roadnet_ca`,
  `ncbi_taxonomy`). On a 2 GB card this forces the CPU backend. That is a
  permitted deviation, recorded in the row (`device = cpu`), and it changes
  time but not quality.
- An optimiser with state arrays multiplies `Z` by `1 + state arrays`
  (`momentum` x2, `adam` x3, `sqn` x9 at the default memory). The regime
  for L3 stays at 32 or 64 so that every rule fits the card and the
  comparison is of the rule, not of what fitted.

---

## 6. Metric catalogue

Metrics are ranked by how much sits between the layout and the number. A
model-free score is primary. A model-based score is secondary and is only
comparable inside one protocol name.

Status: IN = in `evaluator/` today. PLANNED = named in this document, not
implemented. A PLANNED metric is adopted by the procedure of section 10.

### 6.1 Distance fidelity, model-free (PRIMARY for hop distance)

The claim of the project is that Euclidean distance in `Z` is monotone in
graph distance. These scores test that claim with nothing in between.

| Metric | Definition | Signifies | Ties | Status |
| --- | --- | --- | --- | --- |
| Spearman ρ | Pearson correlation of the midranks of `||Z_u - Z_v||` and `d_G(u, v)` | strength of the monotone relation, on a variance scale. ρ = 1: the layout orders every pair as the graph does. | midranks, not the `1 - 6Σd²/n(n²-1)` form | PLANNED |
| Kendall τ-b | `(C - D) / sqrt((C + D + T_x)(C + D + T_y))` over pairs of pairs | `P(agree) - P(disagree)` for two sampled pairs. Reads as a probability difference. Robust to a few badly placed pairs. | tied comparisons leave the denominator one variable at a time. Cannot reach 1.0 when `y` is heavily tied and `x` is continuous. | PLANNED |
| Somers' D_yx | `(C - D) / (C + D + T_y)` | τ conditioned on ties in the response only. The cleaner choice when `y` is an ordinal target (hop count) and `x` is continuous (Euclidean distance), which is exactly this case. | by construction | PLANNED, headline candidate |
| per-source ρ | ρ computed inside one BFS row, then the mean and the spread over sources | matches how a distance oracle is used. Removes the effect of which sources were sampled. | midranks | PLANNED |
| per-shell R2 and per-shell ρ | the score restricted to pairs at hop `h`, for each `h` | the shell at which fidelity collapses IS the resolution horizon `h*`. This is the cleanest discriminating number against node2vec, and it is the only distance metric that means something on class S. | none inside a shell for `y`; report `n` per shell | PLANNED |
| tie fraction of `y` | fraction of pair-of-pair comparisons tied in hop distance | without it, neither ρ nor τ is readable | -- | PLANNED, mandatory beside any rank score |
| attainable max τ-b | τ-b of a perfect layout under the observed `y` tie pattern | the ceiling. Report τ-b beside it, or report `τ-b / max`. | -- | PLANNED |

Rule: a rank correlation is reported with its tie fraction, its pair count,
and its bootstrap spread (section 8). ρ and τ are never compared to each
other. τ is roughly two thirds of ρ on well-behaved data; that gap is not a
finding.

### 6.2 Distance fidelity, model-free, magnitude (PRIMARY for class S and for the drawing regime)

Rank correlation is invariant to any monotone map, so it says nothing about
whether the map is smooth or where it saturates. Stress does.

| Metric | Definition | Signifies | Status |
| --- | --- | --- | --- |
| Kruskal stress-1, scaled | `sqrt(Σ(s·x_uv - y_uv)² / Σ y_uv²)` at the optimal scalar `s` | the native quality measure of a force-directed layout. Scale-free, since the layout has no fixed scale. | PLANNED |
| Shepard diagram | scatter of `x_uv` against `y_uv`, with the isotonic fit | the calibration map itself, and where it goes flat. The flat region is `h*` read a second way. | PLANNED, figure not number |
| average distortion, worst-case distortion | `mean(max(x/y, y/x))` after scaling; and the max | the numbers of metric-embedding theory (Bourgain, Thorup-Zwick, Blasius 2021). They place the method on the space-stretch frontier the paper claims. | PLANNED |
| hop MAE, MRE, RMSE, R2, exact | regression of a model on a feature of `Z` | SECONDARY. Comparable inside one protocol only. Kept because every recorded number uses them and because `exact` reads plainly for an integer target. | IN |

Rule for class S: hop R2 is reported because it is in every table, and the
report says in the same row that it is not the metric for this class. The
primary number for class S is per-shell ρ up to `h*`, and stress-1 at the
R-low-S regime.

### 6.3 Neighbourhood retrieval, model-free (PRIMARY for local structure)

Link prediction with no classifier. This is what `TODOs.md` calls
"adjacency recovery" (P2, owed).

| Metric | Definition | Signifies | Status |
| --- | --- | --- | --- |
| precision@k, recall@k | of the `k` nearest points to `u` in `Z`, the fraction that are true neighbours; and the converse | are the neighbours where they should be | PLANNED |
| MAP | mean over `u` of the average precision of its ranked neighbour list | the standard number of the Poincare and node2vec literature; comparable across papers | PLANNED |
| MRR, Hits@k | rank of the first true neighbour; whether one is inside `k` | the oracle view: "how far down do I look" | PLANNED |
| kNN overlap | Jaccard of the `k`-nearest set in `Z` and the `k`-nearest set by hop distance | the one number that does not privilege `h = 1` | PLANNED |
| trustworthiness, continuity | the two errors of dimensionality reduction: false neighbours entering, true neighbours leaving | reads the direction of the failure | PLANNED |
| graph reconstruction | of the `|E|` nearest pairs in `Z`, the fraction that are edges | one number, no negatives, no training | PLANNED |

`k` is fixed per protocol. Default `k = 10` for class H, `k = degree(u)`
for a per-node variant. Both are named.

### 6.4 Link prediction (SECONDARY, kept for comparability)

| Metric | Definition | Status |
| --- | --- | --- |
| accuracy, precision, recall, F1, AUC | random forest on `hadamard(Z_u, Z_v)`, balanced negatives, protocol-fixed | IN |
| distance-AUC, distance-AP | score a pair by `-||Z_u - Z_v||`, no classifier | PLANNED, the model-free version |
| AP at a realistic negative ratio | positives to negatives at 1:100 or at the graph's true sparsity | PLANNED. The balanced AUC of 0.99 overstates the result on a sparse graph; AP under the true ratio is the number that means something. |

Rule: AUC saturates near 0.99 for every method on every class H graph on
record. It discriminates nothing. It is reported because it is expected,
and it is never the axis on which a verdict rests.

### 6.5 Cost (always reported, never alone)

| Metric | Definition | Status |
| --- | --- | --- |
| peak RSS, MB | the high-water mark of the process. Only memory never allocated helps. | IN, guarded |
| peak RSS by stage | load, augment, embed, score, each | PLANNED. The 2026-08-28 fix was found by this split. |
| wall time by stage | `t_load, t_embed, t_lp, t_da` | IN |
| epochs to a target | first epoch at which the primary score reaches a fixed fraction (0.95) of its 200-epoch value | PLANNED. The 2026-08-28 campaign found nothing converged at 200 epochs; this number says how much of the curve a run bought. |
| `D.nnz / n`, or pairs per epoch for streaming | the working set of the force | IN (reported), PLANNED as a gate axis |
| rows touched per update | for a dynamic graph only | PLANNED |

### 6.6 Structure without labels (SECONDARY, for class H and L)

| Metric | Definition | Status |
| --- | --- | --- |
| NMI, ARI: Louvain on `A` against k-means on `Z` | community structure survives the embedding | PLANNED |
| conductance of `Z`-clusters on `A` | the clusters found in the layout are real cuts | PLANNED |
| Spearman of `||Z_u - center||` against degree, and against a centrality rank | the scale-free-layout prediction (Blasius 2021): hubs at the centre | PLANNED |

### 6.7 Stability (mandatory at L5)

| Metric | Definition | Status |
| --- | --- | --- |
| seed spread | max minus min of each primary score over `>= 3` seeds | IN by practice, not by tool. It is HALF of the noise-floor rule. |
| Procrustes residual across seeds | RMS distance after optimal rotation, reflection and scale | PLANNED |
| kNN Jaccard across seeds | mean Jaccard of the `k`-nearest sets between two seeds | PLANNED, the model-free stability number |
| drift, for temporal graphs | mean `||Z_t[u] - Z_(t-1)[u]||` over unchanged nodes, raw and after Procrustes | PLANNED |

### 6.8 Sanity guards (L0, before any score)

`evaluator/guards.py` today: `check_shape`, `check_finite`,
`check_zero_rows`, `check_metric`, `check_class_balance`,
`check_target_spread`, `check_target_degeneracy`. Add: `check_dz_scale`
(`||dZ||` in `[1e-3, 1e3]`; the generalised-momentum blow-up reached
`1.11e9` with NO non-finite value, so `check_finite` misses it), and
`check_degrees` from `plan_contract` (a row with no `h = 1` entry freezes).

---

## 7. Evaluation levels

Every method passes the levels in order. A level has a fixed tier, regime,
seed count, epoch count, metric set and pass rule. A method that fails a
level stops there, and the report says at which level and by which number.
Tuning is bounded: three attempts per level (PLAN.md section 8), then stop.

| Level | Question | Tier | Regime | Seeds | Epochs | Metrics | Pass rule |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L0 sanity | does it run and produce a layout | T1 | R-quick 32 | 1 | 5 | guards of 6.8; `||dZ||` trace; `max abs Z` | every guard passes; `||dZ||` falls or plateaus; no non-finite value |
| L1 smoke | is the idea alive | T1, and one T0 graph with known distances | R-quick 64 | 1 | 50 | AUC; hop R2; ρ; peak RSS; wall | inside 1.5% of the best current method on AUC; not worse than it on ρ; time and `nnz` inside the current G1 bounds (5 s, 400,000) |
| L2 cost | what does it cost | T2, both trees of the same graph | R-quick 64 | 1 | 5 and 50 | peak RSS by stage; wall by stage; `nnz/n`; peak/resting ratio | reported, not gated. A cost claim is a difference on the same graph, same cut, same script, same machine. A ratio above 3x peak/resting is a defect to name. |
| L3 optimiser | which update rule and schedule | T1 and one T2 | R-quick 32 or 64 | 3 | 50 and 200 | primary score at 50 and at 200; `||dZ||`; effective lr; epochs-to-target; state arrays | ranked at fixed EFFECTIVE lr, never at fixed nominal lr. Any run at effective lr >= 1.0 is excluded, not reported as a loss. A rule wins only above the noise floor AND above the seed spread. |
| L4 quality | how good is it, against the baselines | T2, one graph per class | class regime: R-baseline-H then R-robust-H; R-low-S | 3 | 200 | full 6.1 to 6.4, 6.6; both LP and distance | the pre-registered gate for that graph; a claim against a baseline is made at the baseline's dimension first |
| L5 robustness | where does it break | one graph per class | R-stress sweep; and the class regime | 3 | 200 | primary score against `n_dim`; seed spread; Procrustes; kNN Jaccard; per-shell ρ against `h` | reported as curves. The collapse dimension and `h*` are the findings. No pass rule. |
| L6 scale | does the claim hold at T3 | T3, one class H and one class S | class regime; R-quick 64 where the card forces it | 1, then 3 for the paper table | 50, then 200 | the L4 set; peak RSS; wall; against node2vec at its regime | the pre-registered G3-style gate: finishes; time <= 2x node2vec; RSS at the gate; AUC >= 0.95; primary distance score better than node2vec on the same walks |

Notes:

- L1 uses 50 epochs, not 5. "Five epochs undersells this method" (streaming
  report, section 3): pubmed triples its hop R2 from 5 to 50 for 11 s.
- L3 is where the effective-lr law binds. `momentum` at nominal `lr = 0.1`
  and `plain` at `lr = 1.0` have the same effective rate. Comparing them at
  the same NOMINAL rate compares nothing. The law: `effective lr = lr x DC
  gain`; gain 1 for `plain`, `sgd`, `velocity`; `1/(1-beta)` for
  `momentum`, `nesterov`; up to 10 for `fa2`; adaptive for `adam`, `sqn`.
  Standing rule (2026-08-29): no effective lr is ever 1.0.
- L3 is repeated per strategy. A schedule proven on `precomputed` is not
  proven on `streaming` (CATALOG 29.4).
- L4 at T2 and L6 at T3 use the SAME protocol name, or the comparison is
  between protocols and not between tiers.

---

## 8. When a difference counts: the noise floor

Two conditions, both required (PLAN.md, 2026-08-18T01:30, in force):

1. `delta / old > 1.5%`
2. `delta > the spread across seeds`

Additions this document makes:

- The 1.5% floor was set for AUC and R2. It does NOT transfer to a rank
  correlation or to stress. Each PLANNED metric gets its own floor by
  bootstrap over pairs (1,000 resamples of the scored pair set) on `cora`
  and `pubmed` at 3 seeds, BEFORE any verdict uses it. The floor is
  recorded in `evaluator/config.py` beside the metric, with the date.
- Seed spread is measured at 3 seeds minimum for any number in a paper
  table. A single-seed number carries `seeds = 1` in the row and is quoted
  as a measurement, never as a comparison.
- A difference inside the floor is a TIE. It is recorded as a tie, with the
  number, so a later change of the floor can be applied from the record.
  "Inside 2%" with no number recorded is the defect of 2026-08-17.
- A gate number is fixed before the run. A change of the seed, of the
  metric, or of the gate after the run is not permitted.

---

## 9. Fair comparison against a baseline

- Same walks. node2vec and DeepWalk at `p = q = 1` consume the same walk
  sample as `fodiwalk`; give them the same `walks x walk_len` and seed. The
  claim "the same walk information, read by a force law, holds more of the
  global structure" is only a claim when the walks are the same.
- Same dimension, at the baseline's regime first (section 5).
- Frozen embedding, one downstream protocol for every method. A method that
  trains its own decoder (TGN, GNN baselines) is scored on its frozen
  encoder output through `evaluator`, or it is not in the table.
- Same protocol name, and `protocol_modified` printed. The streaming report
  overrides `max_pairs` to 50,000 to match an archived node2vec run; both
  rows carry `protocol_modified = True`, and they are comparable to each
  other and to nothing else.
- Same machine, same cut, same script for a cost number. A cost number from
  a different machine is a different measurement.
- The baseline's own defects stay. The `fodined` protocol keeps its 2,000
  pairs and one-column feature on purpose.

---

## 10. Adopting a new metric

1. A definition with its tie rule, in this document, section 6.
2. A reference implementation against `scipy.stats` or `sklearn` where one
   exists, tested on the four-pair worked example of the rank-criteria
   note (`τ-b = 0.55`) and on a perfect layout (score = its attainable max).
3. A noise floor by bootstrap, section 8, on `cora` and `pubmed`.
4. Entry through the `evaluator` owner session, as a new score in
   `scoring.py` with a new column in the record. No existing column changes
   meaning.
5. A re-run of the current best method with the new score, so the record
   has a baseline for it from day one.

Order of adoption: ρ with tie fraction; Somers' D; per-shell ρ; stress-1;
precision@k and MAP; distance-AUC and AP at a realistic ratio; then the
rest. The first three make every recorded hop R2 readable without the
model.

---

## 11. Report template

Header, printed by the driver:

```
graph=<name>[@cut]  n=  m=  avg_deg=  max_deg=  diam_est=  components=
assortativity=  class=<H|S|B|L|W|D|C|T>  tier=<T0..T3>
method=<policy/weight/law/rule/strategy>  n_dim=  regime=  epochs=  seeds=
lr=  effective_lr=  protocol=<name>  protocol_modified=<bool>
device=  machine=  date=
```

Body, one row per `(method, graph, n_dim, seed)`, and one summary row per
`(method, graph, n_dim)` with mean and spread. Columns in this order: cost
(peak RSS, wall), distance model-free (ρ, tie fraction, D_yx, stress-1),
distance model-based (hop R2, MAE), retrieval (MAP, p@10), link prediction
(AUC, AP), stability (seed spread, kNN Jaccard).

Every verdict sentence in a report names the metric, the number, the floor
it was judged against, and the seed spread.

---

## 12. Traps that change a measurement

From the record. Each caused a wrong number.

- A float stored weight makes every degree 0, the force vanishes, AUC reads
  0.55, and no error appears.
- The order of the COO triples changes the last bit. Do not sort.
- A split hub row makes the GPU step non-reproducible. No bit-equality test
  on `pubmed` or larger.
- The wrong negative sampler moves the RNG: 0.9777 against the correct
  0.9744 on `fodined`. Do not quote 0.9777.
- H2 (walk gap against true distance) is a tightness diagnostic, not a
  quality score. `walk` is exact on 98.1% of stored pairs, `nbr_walk` on
  7.3%, and `nbr_walk` wins every score.
- `check_finite` passes a run that blew up by eight orders of magnitude.
  Watch `||dZ||`.
- A killed pytest piped to `tail` reads as a pass.
- BFS-ball truncation of a tree, and subtree truncation of a graph, both
  give a perfect score on an empty problem.
- A 5-epoch number undersells every method on record.
- A cost measured at T2 does not predict T3 in either direction.

---

## 13. Open decisions

Decisions this document needs from the owner before it is adopted:

1. The primary distance headline: Somers' D_yx, or τ-b with its attainable
   max, or ρ. This document proposes D_yx as headline and ρ beside it.
2. Cross-component pairs on class C graphs: drop, or cap at `diameter + 1`.
3. `k` for retrieval: 10, or `degree(u)`, or both.
4. The T0 synthetic set: which generators, which sizes, and whether known
   `h*` is planted (a ring lattice gives a known horizon).
5. Which class S graph gets the R-low-S sweep first: `roadnet_ca` at 2, 3,
   4, 8 is the obvious one, and it is also the one that fits the card at
   low dimension.
6. Whether hop R2 stays in the headline table for class S at all, or moves
   to an appendix with the note of section 6.2.