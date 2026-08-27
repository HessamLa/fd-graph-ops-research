# fdwalk: findings

Created: 2026-08-15. Last updated: 2026-08-16 23:25 PDT.
Every gate section carries the time of its measurement. An `UPDATE` block
carries the time of the update, and it never replaces the text above it.

This document holds the RESULTS. PLAN.md holds the design and the criteria,
and the criteria do not change after a run.

Rule of this document: a run that fails goes in, with its numbers. A run
that we stop goes in, with the reason. A number that came from a deviation
carries the deviation beside it.

## Status

| Date | Gate | Variant | Result |
| --- | --- | --- | --- |
| 2026-08-15 | G1, Cora | 10 variants, 3 seeds, 30 runs | closed. `min_gap` passes, `mean_gap` and `pmi` fail. |
| 2026-08-15 | G2, PubMed | walk/min_gap, ball/min_gap, walk/flat | closed. All three pass. |
| 2026-08-15 | G3, com_youtube | walk/min_gap | failed, 3 attempts, all inside the augmentation |
| 2026-08-16 | G3, com_youtube | walk/min_gap, `prune_factor 1` | **the run FINISHES**, and the gate FAILS: AUC 0.8988 (needs 0.95) and 2858 MB (needs 2600). The hop R2 is 0.433 against 0.043 for node2vec. |
| 2026-08-16 | landmarks | far pairs with a real distance | **rejected** by the rule of 2%. No metric gains on Cora or PubMed. |

## The baselines, before fdwalk starts

Measured in this repository. The logs are the source, and this table only
collects them.

| Graph | n | Method | AUC | hop R2 | Time | Peak RSS | Log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cora | 2,708 | fodined `k_hop` k=3 | 0.9986 | 0.257 | 17.4 s | small | `../modular-graphs/modular_cora_khop3.log` |
| Cora | 2,708 | fodined `k_hop` k=2 | 0.9953 | degenerate | 9.7 s | small | this session |
| PubMed | 19,717 | fodined `sampled_pairs` | -- | 0.340 | -- | -- | `../modular-graphs/modular_pubmed.log` |
| roadNet-CA | 200,000 | fodined `k_hop` k=2 | 1.0000 | degenerate | 209.7 s | 1867 MB | `../modular-graphs/modular_roadnet_ca_200k.log` |
| com_youtube | 1,134,890 | node2vec | 0.9984 | 0.043 | 764.6 s | 2212 MB | `../large-graph-node2vec/node2vec_com_youtube_1M.log` |
| com_youtube | 1,134,890 | fodined, any policy | -- | -- | did not run | 2465 MB | `../modular-graphs/modular_com_youtube_1M.log` |

"degenerate" means that the augmentation stored only 2 distinct distances,
thus the regression had one target value and the R2 has no meaning. See
PLAN.md section 11, item 4.

## Why the augmentation must change

A probe of the k-hop ball, from a sample of 2,000 rows, 2026-08-15:

| Graph | max degree | pairs per node k=2 | whole graph k=2 | pairs per node k=3 | whole graph k=3 |
| --- | --- | --- | --- | --- | --- |
| com_youtube | 28,754 | 2,200 | 2.50 G (50 GB) | 46,874 | 53.2 G (1,064 GB) |
| roadNet-CA | 12 | 9 | 0.02 G (0.4 GB) | 17 | 0.03 G (0.7 GB) |

One policy, two graphs of the same size, and a factor of 1,500 between the
two results. The size of the ball follows the degree of the hubs, and no
budget controls it. That is the finding that starts this experiment.

Three defects of the old path were repaired on the way to that probe, and
they hold for fdwalk too:

1. `sample_far_pairs` drew the whole request in one batch. 24.7M pairs and
   about nine int64 temporaries need 2.2 GB. A batch of 2M holds the peak
   near 180 MB at any count.
2. `A[_keep][:, _keep]` overflows the int32 that scipy uses for the count of
   the result, on a graph of 2M nodes. An O(nnz) mask and renumber replaces
   it.
3. The truncation to `MAX_NODES` walked the edge list as "parent to child".
   That is correct for a tree and wrong for a road network: it returned
   500,000 nodes with 504 edges, and every score was then 1.0000 on an empty
   problem. A graph that is not a tree now takes a BFS ball.

Defect 3 is a warning for every measurement of this directory: a perfect
score is a reason to look at the graph, and not a reason to celebrate.

## Gate G1 -- Cora

Measured 2026-08-15, about 13:30 to 15:40 PDT. 30 runs: 10 variants at the seeds 42, 56, and 88. The table gives the mean
and the standard deviation over the three seeds, from
`results/g1_cora.tsv`, through `summarize.py`. No number here is typed by
hand.

| pairs | weight | D.nnz | aug s | embed s | AUC | R2 dist | R2 vec | final dZ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| walk_edges | min_gap | 75389 | 0.5 | 8.1 | **0.9972 ±.0005** | **0.658 ±.026** | 0.815 ±.024 | 0.356 |
| walk | min_gap | 75386 | 0.5 | 8.1 | 0.9970 ±.0009 | 0.655 ±.028 | 0.813 ±.021 | 0.356 |
| ball | min_gap | 79512 | 0.1 | 8.3 | 0.9967 ±.0009 | 0.630 ±.016 | 0.801 ±.008 | 0.185 |
| walk | flat | 75386 | 0.5 | 8.2 | 0.9915 ±.0007 | 0.485 ±.020 | 0.623 ±.002 | 0.060 |
| walk_edges | flat | 75389 | 0.5 | 8.3 | 0.9912 ±.0006 | 0.485 ±.020 | 0.625 ±.011 | 0.060 |
| ball | flat | 79512 | 0.1 | 8.4 | 0.9881 ±.0007 | 0.501 ±.022 | 0.685 ±.026 | 0.055 |
| walk | pmi | 75386 | 0.5 | 8.0 | 0.8417 ±.0034 | 0.015 ±.008 | 0.178 ±.036 | 0.050 |
| walk_edges | pmi | 75389 | 0.4 | 8.1 | 0.8381 ±.0028 | 0.016 ±.009 | 0.200 ±.039 | 0.050 |
| walk | mean_gap | 75386 | 0.5 | 8.0 | 0.5509 ±.0104 | 0.013 ±.002 | 0.010 ±.012 | 0.000 |
| walk_edges | mean_gap | 75389 | 0.5 | 8.1 | 0.5487 ±.0125 | 0.016 ±.003 | 0.009 ±.013 | 0.000 |

`R2 dist` uses ONE feature, the distance of the pair, and it drops the
pairs at hop 1. That is the protocol of `../other-ge/`, thus it compares
directly with node2vec, which reaches **0.324** on Cora. `R2 vec` uses the
128 numbers of `|Z[u] - Z[v]|`, thus it is a different question and the two
columns must not be mixed.

### G1: which variants pass

| Variant | AUC >= 0.99 | R2 >= 0.25 | aug < 5 s | nnz <= 400k | Pass |
| --- | --- | --- | --- | --- | --- |
| walk/min_gap, walk_edges/min_gap, ball/min_gap | yes | yes | yes | yes | **pass** |
| walk/flat, walk_edges/flat, ball/flat | yes | yes | yes | yes | pass |
| walk/pmi, walk_edges/pmi | no | no | yes | yes | **fail** |
| walk/mean_gap, walk_edges/mean_gap | no | no | yes | yes | **fail** |

A note on the honesty of the R2 gate: PLAN.md fixed "R2 >= 0.25" against
the 0.257 of `modular.py`. That number came from a DIFFERENT protocol (the
stored pairs of D, and the vector features), thus the threshold is not the
comparison that it claims to be. The gate stands as written, because a gate
does not move after a run. The comparison that carries the weight is the
one with the same protocol: node2vec at 0.324, which every passing variant
beats, and `min_gap` beats by two times.

### G1 finding 1: the weight must keep "weight 1 means adjacency"

`mean_gap` and `pmi` did not fail slowly. They collapsed:

| weight rule | entries at weight 1 | AUC | final `||dZ||` |
| --- | --- | --- | --- |
| min_gap | 10,550 | 0.9970 | 0.356 |
| pmi | 2,244 | 0.8417 | 0.050 |
| mean_gap | 142 | 0.5509 | 0.000 |

Cora has 5,278 edges, thus 10,556 directed entries. `min_gap` gives the
weight 1 to EXACTLY the true adjacency: 10,550 of 10,556, and the walks
missed only 6 edges.

`degrees_from_D` counts the entries that are exactly 1 in each row, and
that count is the force-law degree. A walk crosses one edge at many gaps,
thus the MEAN of those gaps is near 3, and only 142 entries stay at 1.
Nearly every row then has the degree 0, the attraction goes with it, and
`||dZ||` falls to 0.000.

Thus a weight rule for this force law has a hard requirement: **the
weight 1 must be the adjacency of the graph.** A rule that only ORDERS the
pairs correctly is not enough. This holds for any future rule, and it is
the reason to expect a hitting time or a PPR rule to need the same
treatment.

### G1 finding 2: the weight carries the global structure (H3 refuted)

The flat control keeps the pairs and it throws the weight away. It loses
only 0.005 of AUC, and it loses 0.17 of R2.

H3 predicted "within 0.01 of AUC and clearly worse R2", and it asked for
the conclusion "the pair set matters more". The AUC part is right, and the
conclusion is wrong. Link prediction is a LOCAL question, and the pair set
answers it alone. The hop distance is a GLOBAL question, and only the
weight answers it. The correct statement is: the pair set decides the link
prediction, and the weight decides the geometry.

### G1 finding 3: the walk and the capped ball tie on Cora (H4 open)

At the same budget of `M = 16`: AUC 0.9970 against 0.9967, and R2 0.655
against 0.630. The AUC difference is inside one standard deviation over the
seeds. The R2 difference (0.025) is about the size of the seed spread
(±0.028), thus it is at the edge of the noise, and it is not a result.

H4 predicted that the walk wins both. It does not, on Cora. The stop rule
of PLAN.md section 8 says that a win by the capped ball ends the walk work.
The ball did not win, thus the work continues, but the reason to prefer the
walk is now ONLY the scale: a ball cannot be built at all on com_youtube
(2.5 G pairs at k=2), and a walk has a budget. G3 is therefore not a
formality, it is where H4 is decided.

### G1 finding 4: the walks already hold the edges (B2 = B1)

`walk_edges` adds the original edges that the walks did not give. On Cora
at `r = 10, l = 20` it added **3** edges, and every metric moved less than
one standard deviation. By the stop rule "two variants inside the noise are
one variant, keep the cheaper one", `walk_edges` and `walk` are the same
variant, and `walk` continues.

This can change on a graph with more low-degree nodes, thus the option
stays in the code.

### G1 finding 5: H2 is confirmed, and by a wide margin

The walk gap against the exact BFS distance, on 4,190 stored pairs of Cora:

| measure | value |
| --- | --- |
| exact | 98.1% |
| over by 1 or more | 1.9% |
| **under** | **0.0%** |
| MAE | 0.019 |

H2 predicted more than 80% exact at `r = 10, l = 20`, and it asked for a
refutation below 60%. The measured value is 98.1%.

The 0.0% is the important digit, and it is not a measurement, it is a
proof: a walk that goes from `u` to `v` in `t` steps SHOWS a path of `t`
steps, thus the gap can never be smaller than the true distance. A value
above 0.0% would mean a defect in this code. The ball gives 100.0%, as it
must, because it computes the distance directly.

### G1: two defects found, and what they cost

1. **A float weight destroys the force law.** The first `mean_gap` and
   `pmi` gave a continuous weight. `shell_counts` makes a table with one
   column for each DISTINCT value of `D.data`, thus every pair took its own
   shell of size 1, and `degrees_from_D` found no entry that is exactly 1.
   Both variants collapsed to an AUC near 0.55 and 0.70 with `||dZ||` at
   0.000. Every rule now returns an integer in `[1, window]`. The invalid
   numbers stay in `results/g1_cora_floatweights_INVALID.tsv`.
   The rounding repaired `pmi` (0.70 -> 0.84), and it did NOT repair
   `mean_gap` (0.55 -> 0.55), because the second defect of `mean_gap` is
   the one that finding 1 describes.
2. **`walk_edges` stopped on every run.** The added edges grew `key` and
   the weight, and not `mn`, `sm`, and `cnt`. The H2 matrix then had arrays
   of two lengths. 12 runs were lost and repeated.

## Gate G2 -- PubMed

Measured 2026-08-15, about 16:10 PDT. 19,717 nodes, 44,324 undirected edges. 9 runs: 3 variants at the seeds 42,
56, and 88. Source: `results/g2_pubmed.tsv`.

| pairs | weight | D.nnz | aug s | embed s | AUC | R2 dist | R2 vec | final dZ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| walk | min_gap | 653776 | 4.8 | 43.2 | 0.9953 ±.0002 | **0.549 ±.032** | 0.515 ±.051 | 0.230 |
| ball | min_gap | 678178 | 2.5 | 46.2 | **0.9958 ±.0007** | 0.531 ±.017 | 0.511 ±.023 | 0.202 |
| walk | flat | 653776 | 4.8 | 43.2 | 0.9909 ±.0004 | 0.336 ±.009 | 0.339 ±.022 | 0.071 |

### G2: which variants pass

| Test | walk/min_gap | ball/min_gap | walk/flat |
| --- | --- | --- | --- |
| AUC inside 0.01 of its own Cora number | 0.0017 yes | 0.0009 yes | 0.0006 yes |
| R2 >= 0.30 | 0.549 yes | 0.531 yes | 0.336 yes |
| augmentation < 60 s | 4.8 s yes | 2.5 s yes | 4.8 s yes |
| peak RSS < 1.5 GB | 1447 MB yes | 1343 MB yes | 1447 MB yes |
| | **pass** | **pass** | **pass** |

The peak RSS holds the JAX and CUDA context too, which is about 780 MB
before any work of this experiment starts. The augmentation itself is a
small part of that number.

### G2 finding 1: the result holds at 7 times the size

Cora to PubMed is 2,708 to 19,717 nodes. The AUC of walk/min_gap moves from
0.9970 to 0.9953, and the R2 from 0.655 to 0.549. Both stay far above the
baselines, and the augmentation grows from 0.5 s to 4.8 s, thus about
linearly with `n`.

### G2 finding 2: the margin over node2vec grows

The same protocol, and the same graph:

| Graph | node2vec R2 | Poincaré R2 | fdwalk walk/min_gap | fdwalk is better by |
| --- | --- | --- | --- | --- |
| Cora | 0.324 | 0.123 | **0.655** | 2.0x |
| PubMed | 0.080 | 0.094 | **0.549** | 6.9x |

node2vec is weakest exactly where the force law is strongest. Both methods
read the SAME walks: node2vec sends them to word2vec, and fdwalk sends them
to a force law with the gap as a distance. The difference is what the
method does with the gap, and not what the walk saw. This is the central
claim of the design, and PubMed is the first graph where it is large.

The link prediction tells the opposite story, and it must be said: node2vec
reaches 0.9972 on PubMed and fdwalk reaches 0.9953. Both are at the ceiling
of that metric, and the difference is not meaningful, but nothing here
shows fdwalk winning the local question.

### G2 finding 3: the ball still ties (H4 still open)

AUC 0.9958 against 0.9953, and R2 0.531 against 0.549. Both differences are
inside the seed spread. Two graphs now, and no separation. The ball is even
FASTER than the walk (2.5 s against 4.8 s) at this size.

H4 rests entirely on G3. If the ball cannot be built at 1.13M nodes and the
walk can, the walk wins for the only reason that ever mattered. If the ball
also runs, the walk has no case, and the correct conclusion is that the fix
for the old policy is a cap and not a walk.

## Gate G3 -- com_youtube, 1.13M nodes

Measured 2026-08-15, about 21:00 to 22:40 PDT.

**FAILED.** Three attempts, and the run never left the augmentation. The
variant stops, by the rule of PLAN.md section 8.

Log: `results/g3_com_youtube_walk_min_gap_cap8_d64_e500.log` (the third
attempt overwrote the two before it; the numbers of all three are here).

| Attempt | Config | Died at | Peak RSS | The defect |
| --- | --- | --- | --- | --- |
| 1 | cap 8, dim 64, 500 epochs, far = n·log10(n) | 20 s | 3081 MB | the pair accumulator had no bound |
| 2 | the same, far = 2n | 32 s | 2739 MB | `cap_per_node` held six int64 arrays of 16M entries, about 1.4 GB |
| 3 | the same, int32 and a smaller block | 152 s | 2913 MB | the prune removed nothing (below) |

### G3: why the third attempt failed

The prune of `walk_pair_stats` starts when the accumulator passes
`prune_max` (4M) entries, and it then keeps `prune_factor * cap` pairs for
each node. That target is `4 * 8 = 32` pairs for each node, thus:

```
32 pairs/node * 1,134,890 nodes = 36.3M pairs = 726 MB at 20 bytes each
```

The accumulator is therefore NOT bounded by `prune_max`. It is bounded by
`prune_factor * cap * n`, which is nine times larger than the threshold that
starts the prune. At 1.13M nodes almost no node yet had 32 partners, thus
every prune sorted the whole accumulator and removed nothing. That is also
why the third attempt was SLOWER (152 s against 32 s): it did the work of a
prune 114 times for no result.

The correct bound is one line: the prune target must give an accumulator
that is smaller than the memory, thus `prune_factor = 1` (9M pairs, 180 MB)
or a `prune_max` above `prune_factor * cap * n`.

**This is not run.** The rule of PLAN.md section 8 gives the budget one
reduction, and the run had it (the far pairs, 6.87M to 2.27M). Three
crashes are enough. A fourth attempt with a corrected constant belongs to a
new decision, and not to this gate.

### G3: what the failure does and does not say

It does NOT say that the idea does not scale. Every defect of the three
attempts is in the code of this experiment, and not in the design:

| Attempt | The defect | Is it a property of the method? |
| --- | --- | --- |
| 1 | no bound on the accumulator | no, the plan claimed a bound that the code did not have |
| 2 | int64 where int32 is enough | no |
| 3 | the prune target is 9 times the threshold | no, one constant |

It DOES say that the claim of PLAN.md section 2 was too strong. The cap
bounds `D`, which is the RESULT. It does not bound the work before it. The
walks of com_youtube give 306M raw pairs, and the set of DISTINCT pairs
among them is the quantity that must be bounded, which the plan never
stated. **A budget on the output is not a budget on the pipeline.**

### G3: the honest comparison with the ball

Neither method embedded com_youtube on this machine. The two reasons are
not the same, and the difference is the whole point of H4:

| Method | What it needs at 1.13M nodes | Can a constant repair it? |
| --- | --- | --- |
| k-hop ball, k=2 | 2.5 G pairs, about 50 GB | **no.** The size follows the degree of the hubs. |
| k-hop ball, k=3 | 53.2 G pairs, about 1 TB | no |
| fdwalk, cap 8 | 18M pairs in `D`, and an accumulator of 726 MB that one constant makes 180 MB | **yes** |

A method that needs 50 GB needs a different machine. A method that needs
726 MB where it should need 180 MB needs a corrected line. Thus H4 is not
decided by G3 as planned, but the evidence still separates the two: the
ball fails for a reason that no budget can reach, and the walk fails for a
reason that a budget can reach.

That statement is weaker than the one that G3 was designed to give, and it
must stay weaker until a run finishes.

### G3, the fourth attempt: it FINISHES, and it does not pass

Measured 2026-08-16 00:35 PDT. Log: `results/g3_yt_alloc_platform.log`.

Config: `walk/min_gap`, cap 8, dim 64, 500 epochs, far 2.27M, 8 chunks,
`prune_factor 1`, no landmarks, `XLA_PYTHON_CLIENT_ALLOCATOR=platform`.

| Stage | Time | Peak RSS |
| --- | --- | --- |
| augmentation | 538.0 s | 2291 MB |
| embedding, 500 epochs | 443.1 s (1.1 epochs/s) | 1995 MB |
| evaluation | about 180 s | 2858 MB |
| **total** | **about 19.4 min** | **2858 MB** |

| Measure | Value | node2vec, same graph | Gate | Pass |
| --- | --- | --- | --- | --- |
| finishes | **yes** | yes | must | **yes** |
| total time | 19.4 min | 12.7 min | <= 30 min | yes |
| peak RSS | 2858 MB | 2212 MB | <= 2600 MB | **no** |
| AUC | 0.8988 | 0.9984 | >= 0.95 | **no** |
| hop R2, distance | **0.433** | 0.043 | > 0.043 | **yes, by 10 times** |

**G3 FAILS**, on the AUC and on the memory. Two of the five tests do not
pass, thus the gate does not pass. The variant does not enter `fodined/`.

### G3: what the result says

**The claim of the design survives at 1.13M nodes.** The hop R2 is 0.433
against 0.043, thus ten times better than node2vec on the SAME walks. The
gate asked for "better than 0.043", and the measurement is far above it.
The force law reads the global structure out of the walks, and word2vec
does not. That was the reason to build fdwalk, and it holds at a million
nodes.

**The link prediction is bad, and the cause was visible before the run.**
0.8988 against 0.9984. The augmentation gave the weight 1 to 4,036,102
entries, and the graph has 5,975,248 directed edges, thus **33% of the
adjacency never entered `D`**. G1 finding 1 says that the weight-1 shell IS
the local signal. A third of it is missing, thus the local metric falls.
The cause is the budget: `r=5, l=20, window=3, cap 8` here, against
`r=10, l=20, window=5, cap 16` on Cora, which recovered 99.9%.

This is a budget failure, and not a failure of the method. It is also NOT a
free repair: a larger budget needs more memory, and the memory is what the
gate already fails.

> **UPDATE 2026-08-16. The paragraph above is kept, and it is WRONG.**
>
> Reason for the update: the off-branch experiment
> (`offbranch/REPORT.md`) adds EVERY original edge by construction, thus
> its 1.13M run holds 100% of the adjacency at `h = 1` (histogram
> `{1: 5,975,248, ...}`). If the missing 33% were the cause, the AUC had to
> rise. It moved from 0.8988 to 0.9026, which is 0.4%, thus noise by the
> rule of 2%.
>
> Rationale for keeping the text: the reasoning above is the reasoning that
> the run was read with, and a reader of the G3 section must see the claim
> that the later measurement removed. A deleted claim cannot be checked.
>
> The state now: the comparison is confounded, because the force law and
> the pair policy also changed. The clean test is the force law of the
> package with the bucket policy at 1.13M nodes, and it has not run.
> **The cause of the 0.90 AUC is unknown.**
>
> An open candidate, from the analysis of 2026-08-16: the far pairs are
> drawn UNIFORMLY, and node2vec draws its negative samples in proportion to
> `degree^0.75`. On a graph with a maximum degree of 28,754 a hub is about
> 636 times more likely to be a negative sample for node2vec than for a
> uniform draw. A hub in fdwalk therefore receives almost no repulsion,
> while its attraction is divided by `deg^2`. Not measured. This document first said that the missing
adjacency was the cause. The off-branch experiment
(`offbranch/REPORT.md`) tests it: its policy adds EVERY original edge, thus
its 1.13M run holds 100% of the adjacency at `h = 1`. The AUC moved from
0.8988 to 0.9026, which is 0.4%, thus noise by the rule of 2%.

Restoring the adjacency in full does NOT repair the link prediction at this
size. The comparison is confounded, because the force law and the pair
policy also changed, thus the clean test is the force law of the package
with the bucket policy at 1.13M nodes, and it has not run. **The cause of
the 0.90 AUC is unknown.**

**The memory failure sits in the EVALUATION, and not in the method.** The
embedding peaked at 1995 MB. The evaluation harness (the BFS blocks, the
50,000 link-prediction pairs, and the random forest over 64 features)
pushed it to 2858 MB. A deployment does not run that harness. The gate
measures the whole script, thus the failure stands as written, and the
reader must know where the memory went.

**H2 holds at scale.** 96.9% exact on 4,783 stored pairs, 0.0% under. Cora
gave 98.1%. A walk gap is a distance at any size.

### G3: the deviations of this run

All of them are in the log, and they are not permitted to be forgotten:

| Deviation | Registered? |
| --- | --- |
| dim 64 instead of 128 | NO. PLAN.md permitted 64 only for a variant with an optimizer state. |
| epochs 500 instead of 2000 | yes, PLAN.md section 9 |
| cap 8 instead of 16 | yes, PLAN.md section 9 |
| far 2.27M instead of 6.87M | yes, the one permitted budget reduction |
| `XLA_PYTHON_CLIENT_ALLOCATOR=platform` | not a budget. It changes the allocator, and not the experiment. |

### G3: what a next attempt needs

Not part of this gate. Recorded so that the numbers are not lost:

1. `prune_factor = 1`, thus an accumulator of `cap * n` = 9M pairs,
   180 MB.
2. `dim 64` was already an UNREGISTERED deviation. PLAN.md section 7
   permitted 64 only for a variant with an optimizer state. `Z` and `dZ` at
   128 dimensions are 1.16 GB on a 2 GB card, thus no variant fits at 128,
   and the deviation is honest but it was not planned.
3. The far pairs came down from 6.87M to 2.27M. That was the one permitted
   budget reduction, and it is used.

## Roster item 1 -- a far pair drawn at `deg^0.75`

Measured 2026-08-16 23:34 to 2026-08-17 00:45 PDT. Six runs, three seeds,
on a BFS ball of com_youtube of 150,000 nodes, which keeps the maximum
degree of 28,754. Cora and PubMed cannot show this effect: their maximum
degrees are 168 and about 171.

| alpha | AUC | accuracy | R2 dist | final \|\|dZ\|\| |
| --- | --- | --- | --- | --- |
| 0.0 (uniform, what fodined always did) | 0.9146 ±.0079 | 0.8410 ±.0123 | 0.160 ±.041 | 2.016 ±.188 |
| 0.75 (word2vec and node2vec [4, 5]) | **0.9506 ±.0029** | **0.8830 ±.0039** | 0.128 ±.019 | **0.396 ±.041** |

| Metric | Change | > 2% | > spread | Verdict |
| --- | --- | --- | --- | --- |
| AUC | +3.9% | yes | yes | **real** |
| accuracy | +5.0% | yes | yes | **real** |
| R2 dist | -20% | yes | no | noise |
| final \|\|dZ\|\| | 5x lower | yes | yes | **real** |

The mean degree of an endpoint of a far pair moves from 14.09, the mean of
the graph, to 209.85. Thus the draw really does reach the hubs.

**This is the first change of this branch that improves the link
prediction.** The landmarks, the bucket policy, and the `v2` force law all
moved the AUC by less than 2%, or they moved it down.

It costs one array of `n` float64, which is 9 MB at 1.13M nodes.

**A caution about the graph.** The ball of 150,000 nodes has an average
degree of 14.09, and the whole graph has 5.27, thus a ball around a hub
region is denser than the graph. The A/B comparison holds, because both
sides use the same graph. The ABSOLUTE numbers do not transfer, and the
0.90 AUC of the whole graph is not yet re-measured.

**A caution about the reading.** The first report of this result used the
seed 42 alone, and it said that the R2 rose by 48%. Three seeds show the
opposite direction, inside the noise. See the log of 2026-08-17 00:45 PDT.

## 2026-08-17 -- com_youtube at 1.13M nodes, with every validated change

Measured 2026-08-17 03:10 to 03:40 PDT. One seed (42).
Log: `results/fdlinear/fdl_yt1M_walkedges.log`.

The run combines four changes, each of which was measured on its own first:
`walk_edges` (every original edge at `h = 1`), the `fdlinear` force law at
`k4 = 1.0`, `lr = 0.1`, and the far pairs drawn at `deg^0.75`.

| Measure | G3, 2026-08-16 | this run | node2vec |
| --- | --- | --- | --- |
| AUC | 0.8988 | **0.9954** | 0.9984 |
| accuracy | 0.8241 | **0.9658** | -- |
| F1 | 0.8029 | **0.9652** | 0.9780 |
| R2 dist | 0.433 | 0.420 | 0.043 |
| final \|\|dZ\|\| | 1.532 | **0.0442** | -- |
| entries at `h = 1` | 4,036,102 (68%) | **5,975,248 (100%)** | -- |
| peak RSS | 2858 MB | 3451 MB | 2212 MB |
| total time | 19.4 min | about 18.5 min | 12.7 min |

**The link prediction gap against node2vec closes from 0.0996 to 0.0030.**
The AUC rises by 10.7%, the accuracy by 17.2%, and `||dZ||` falls by 35
times, thus the layout also settles far better.

**The hop R2 holds at 10 times node2vec.** 0.420 against 0.433 is a fall of
3%, from one seed and with no spread, thus it is not a measured loss. The
claim that carries is: the ten-fold advantage over node2vec survives every
change of this run.

### The gate G3, measured again

| Test | Value | Gate | Pass |
| --- | --- | --- | --- |
| finishes | yes | must | yes |
| total time | 18.5 min | <= 30 min | yes |
| AUC | 0.9954 | >= 0.95 | **yes** |
| R2 dist | 0.420 | > 0.043 | yes |
| peak RSS | 3451 MB | <= 2600 MB | **no** |

**Four of the five tests pass, and the memory does not.** The gate does not
pass. The AUC test, which failed on 2026-08-16, now passes with room.

### What this does NOT establish

The run changes four things at the same time, thus it says what the SET is
worth and not what any part is worth. The parts, measured alone, do not add
up to this:

| Change | measured alone | on which graph |
| --- | --- | --- |
| 100% adjacency | +0.4% AUC, thus noise | com_youtube, off-branch |
| far pairs at `deg^0.75` | +3.9% AUC | a 150k ball with hubs |
| `fdlinear` at `k4 = 1.0` | AUC inside 2%, R2 +7% | Cora and PubMed |
| `lr = 0.1` | part of the same runs | Cora and PubMed |

The sum of the measured parts is about 4%, and the combination gives 10.7%.
Thus either the parts interact, or a part behaves differently at 1.13M
nodes than on the graph where it was measured. **The attribution is not
established**, and one run for each of the four, holding the others fixed,
is what would establish it.

## Hypotheses: the score

| Id | Statement | Result |
| --- | --- | --- |
| H1 | a walk budget bounds the augmentation | **refuted as written**: the cap bounds `D`, and it does not bound the accumulator of distinct pairs before it. The augmentation of com_youtube did not finish, and it passed 2.6 GB. The bound exists (`prune_factor * cap * n`), and the code set it 9 times too high. |
| H2 | the minimum walk gap estimates the hop distance | **confirmed**: 98.1% exact, 0.0% under, against a prediction of 80% |
| H3 | the pair set matters more than the weight | **refuted**: the pair set decides the link prediction, and the weight decides the geometry (0.005 AUC against 0.17 R2) |
| H4 | walks beat a capped ball as a selector | open, and G3 did not decide it. They TIE on Cora and on PubMed, inside the noise. At 1.13M nodes neither ran, and the reasons differ: the ball needs 50 GB, which no constant repairs, and the walk needs 726 MB where one constant gives 180 MB. |
| H5 | momentum helps, Adam does not | open |
| H6 | the state of the optimizer is the limit at 1M | open (arithmetic says yes) |
| H7 | the far weight is not sensitive | open |

## Decisions

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-08-15 | The k-hop policy stays in `fodined/` while fdwalk runs. | It works up to 200,000 nodes on a graph without hubs, and it is the control of H4. |
| 2026-08-15 | Nothing moves into `fodined/` before a variant passes G3. | `fodined/` holds one policy that we trust. `experiments/` holds the ones that we test. |
| 2026-08-15 | `mean_gap` and `pmi` stop after G1. | Both fail the AUC gate, and finding 1 gives the mechanism. Three attempts are permitted, and a fourth is not: the rule breaks the "weight 1 is adjacency" requirement, thus a hyperparameter cannot repair it. |
| 2026-08-15 | `walk_edges` stops, and `walk` continues. | They are one variant inside the noise, and `walk` is the cheaper one. The option stays in the code for a graph with more low-degree nodes. |
| 2026-08-15 | G2 runs walk/min_gap, ball/min_gap, and walk/flat. | The winner, the control of H4, and the control of H3. |
| 2026-08-15 | The ball stops, at every gate. Only the walk continues. | Decision of the user. The ball needs 50 GB at k=2 on com_youtube and 1 TB at k=3, and the size follows the degree of the hubs, thus no budget reaches it. H4 closes as "not measured, and not needed": the ball is out for the reason that H4 was written to test. `--pairs ball` stays in the code as the control of the small graphs. |

---

# UPDATE 2026-08-17T22:53:32-07:00 -- the node2vec baseline, at an equal dimension

**Reason for the update.** Every comparison against node2vec in this
document above used the dim-128 baseline, because that was the only one in
hand, while our runs were at dim 64. The difference was stated as a caveat
only, and a caveat is not sufficient: the comparison was not at an equal
dimension, thus it was not a comparison. **Nothing above is removed.** The
statements above stay, with this entry as their correction.

**Adopted at:** the mix-and-match step, com_youtube, 1,134,890 nodes.

The baseline is re-run at dim 64, holding everything else: 5 walks x 20
steps, window 5, 3 epochs, seed 42, and `--lp-pairs 25000` so that both
sides score 50,000 link-prediction pairs.

| stage / metric | node2vec d64 | nbr_walk/both d64 | walk_edges d64 |
| --- | --- | --- | --- |
| load | 2.1 s | 3.1 s | 3.1 s |
| walk / augmentation | 28.6 s | 28.9 s | 571.8 s |
| train / embed | 854.7 s | 512.1 s | 465.1 s |
| **total wall** | **17:01** | **12:20** | ~18:30 |
| **peak RSS** | **1382 MB** | 4799 MB | 3451 MB |
| accuracy | 0.9750 | 0.9764 | 0.9658 |
| F1 | 0.9753 | 0.9760 | 0.9652 |
| AUC | 0.9964 | **0.9978** | 0.9954 |
| hop R2 | 0.042 | **0.436** | 0.420 |

**What changes.**

1. **Link prediction is a TIE.** 0.9978 against 0.9964 is +0.14%, inside
   the 2% floor. The direction also flipped: at dim 128 node2vec led with
   0.9984, at dim 64 we lead. Neither difference is meaningful. The correct
   statement is "indistinguishable", and it REPLACES "node2vec wins link
   prediction" wherever that appears above.
2. **The hop distance holds, and it is the one result with no
   reservation.** 0.436 against 0.042 is a factor of 10.4, on an identical
   protocol: the same sources, the same pairs, the same `hop > 1` filter,
   the same single scalar feature.
3. **Memory is now the only axis where node2vec clearly wins**, 1382 MB
   against 4799 MB, a factor of 3.5. Its corpus streams from the disk; our
   `D` stays in the RAM.

**A number that must NOT be quoted as precise.** node2vec's dim-64 training
took 854.7 s against 732.5 s at dim 128. That is backwards, because fewer
dimensions is less work for each token. Both are single runs on a machine
that was busy. I read it as variance, thus the 27% advantage in the wall
clock rests on one noisy baseline.

---

# UPDATE 2026-08-17T22:55 -- the plane count of fdlinear

**Reason.** The memory gap of 3.5x above is the open item, and the first
part of it is waste, not physics. `fdlinear` bound a `shell_coeff` plane
and never read it. At `nnz = 23,163,843` and `pad_frac = 0.154` (27.38 M
padded slots) one plane costs 92.7 MB as a host array and 109.5 MB as a
packed tile, and `shell_coeff_data` also allocates about 460 MB of nnz-
sized int64 transient inside `shell_counts`, `row_of` and `searchsorted`.

**Two changes, both at 2026-08-17.**

1. `shell_coeff_data` is built ONLY for a law that reads it. `fdlinear`
   now takes `(h, freq)`.
2. `fdlinear_fused`, behind `--fuse-planes`, takes ONE plane
   `w = h/freq`, with `w = -1` as the sentinel for `h = 1`. See
   `CATALOG.md` 1.4 for why the sentinel is load-bearing: a plain
   `h/freq` cannot separate `h=1, freq=3` from `h=2, freq=6`, and the
   attraction would fire on the wrong cells with no error raised.

**Predicted saving at 1.13 M, steady: 1167 MB -> 387 MB**, plus about
460 MB of transient peak. Against a measured peak of 4799 MB that is about
20%. **It does not close the gap to node2vec**: the augmentation stage
alone holds 3512 MB before the plan is built.

**Verified identical on Cora** (`nbr_walk/min_gap`, dim 64, 200 epochs,
lr 1.0, seed 42): `||dZ||` 0.5016, accuracy 0.9754, F1 0.9752, AUC 0.9962,
hop R2 0.253, hop MAE 1.291 -- every embedding-derived metric matches the
3-plane reference exactly. Peak RSS 1004 MB against 1010 MB; Cora is too
small for the saving to show.

**The 1.13 M measurement is running at the time of this entry.** The result
is appended below when it lands, and not predicted here.

## The 1.13 M measurement, 2026-08-17T23:10

com_youtube, 1,134,890 nodes, `nbr_walk/both`, `min_gap`, `fdlinear`,
dim 64, 500 epochs, lr 0.1, k4 1.0, kr 1.0, far 2.27 M at `deg^0.75`,
8 chunks on the host, seed 42. The configuration is the baseline's, and
only `--fuse-planes` is added.

| | 3 planes | fused | change |
| --- | --- | --- | --- |
| t_aug | 28.9 s | 22.1 s | -23.5% |
| t_embed | 512.1 s | 463.0 s | -9.6% |
| **wall clock** | 12:20 | **11:20** | -8.1% |
| **peak RSS** | 4799 MB | **4374 MB** | **-425 MB, -8.9%** |
| final \|\|dZ\|\| | 0.049724 | 0.049724 | identical |
| accuracy | 0.9764 | 0.9764 | identical |
| F1 | 0.9760 | 0.9760 | identical |
| AUC | 0.9978 | 0.9978 | identical |
| hop R2 | 0.436 | 0.436 | identical |
| hop MAE | 0.755 | 0.755 | identical |

**Correctness: settled.** `||dZ||` agrees to six decimals and every
embedding-derived metric is identical at both 2,708 and 1,134,890 nodes.
The sentinel is sound.

**Memory: real, and I over-predicted it by 1.8x.** The prediction in the
entry above was about 780 MB of steady saving; the measurement is 425 MB.
The prediction assumed that freeing the `freq` CSR lowers the RSS. It does
not, reliably: the RSS is a HIGH-WATER mark, and a Python free returns
memory to the allocator and not always to the OS. What the measurement
does confirm is the part that is never allocated at all -- the dead
`shell_coeff` plane, its tile, and the transients of `shell_counts`.

**Speed: NOT established, and it must not be quoted.** The embed time fell
9.6%, which passes the 2% floor. But `t_aug` fell 23.5% on a code path
that is byte-identical between the two runs. Thus the machine carried at
least 23% of timing variance today, and the 9.6% sits inside it. A
mechanism exists -- `--chunk-host` moves the plan to the device for every
use, thus 1 plane moves a third of what 3 planes move -- and one run at one
seed does not measure it. It stays open.

**What it does not do.** node2vec holds 1382 MB. We now hold 4374 MB, a
factor of 3.2 against 3.5. The augmentation stage alone reports 3583 MB
before the plan is built, thus the remaining memory is in `D` and in the
walk stage, and no change to the plane count reaches it.

**Adopted:** `--fuse-planes` is verified and available. It is NOT the
default, for the reason in `CATALOG.md` 1.4: `h/freq` becomes a build-time
quantity, and the law is still moving.

---

# The mix-and-match grid, 2026-08-17 (entered 2026-08-17T23:45)

**Why this section is entered late.** The grid ran on 2026-08-17 from about
04:30, and it was reported to the user and never written here. The rule of
`experiments/fdwalk/CLAUDE.md` is to update this document with each new and
meaningful finding, thus the omission is a defect. Nothing above is
changed. Source: `results/mixmatch/`, and the log
[log/2026-08-17T043000.md](log/2026-08-17T043000.md).

The grid is a CROSS of the axes, by instruction: 2 force laws x 3
configurations x 3 seeds = 18 runs. Cora, dim 64, 2000 epochs, far pairs at
`deg^0.75`. The spread is the half-range over the seeds.

| force | pairs | rule | AUC | accuracy | R2 dist |
| --- | --- | --- | --- | --- | --- |
| fdlinear | nbr_walk | both | **0.9989 ±.0002** | **0.9853 ±.0021** | 0.668 ±.012 |
| v1 | nbr_walk | both | 0.9979 ±.0001 | 0.9785 ±.0031 | 0.572 ±.046 |
| fdlinear | walk_edges | both | 0.9967 ±.0005 | 0.9770 ±.0040 | **0.677 ±.012** |
| v1 | walk_edges | both | 0.9969 ±.0006 | 0.9749 ±.0007 | 0.445 ±.019 |
| fdlinear | nbr_walk | low_deg | 0.9969 ±.0007 | 0.9724 ±.0007 | 0.345 ±.064 |
| v1 | nbr_walk | low_deg | 0.9948 ±.0008 | 0.9683 ±.0024 | 0.395 ±.077 |

## Finding: `fdlinear` buys the geometry, and not the link prediction

Every `both` row sits in AUC 0.9967..0.9989, a band of 0.22%. That is noise
at the floor of 2%, and it stays noise at the floor of 1%. **The force law
does not decide the link prediction on Cora.**

The hop R2 separates, and the test is the delta against the SEED SPREAD and
not the percentage alone:

| policy | fdlinear | v1 | delta | against the larger spread | verdict |
| --- | --- | --- | --- | --- | --- |
| `walk_edges/both` | 0.677 | 0.445 | +52.2% | 0.232 against 0.037, a factor of 6 | **real** |
| `nbr_walk/both` | 0.668 | 0.572 | +16.7% | 0.096 against 0.091 | **real, MARGINAL** |
| `nbr_walk/low_deg` | 0.345 | 0.395 | -12.8% | 0.051 against 0.154 | noise |

The `walk_edges` result is decisive. The `nbr_walk/both` result passes both
halves of the rule, and it passes the second half by a hair: it must be
quoted with its spread, and never as "+17%" alone.

## Finding: `low_deg` also DESTABILISES the result

Beside the loss of the mean, the seed spread of the hop R2 GROWS under the
rule: ±.012 to ±.064 for `fdlinear`, and ±.046 to ±.077 for `v1`. The
result starts to depend on the seed. This was not stated when the rule was
proposed, and it is a second reason to reject it.

## `low_deg` at 1.13M nodes: the trade, measured

com_youtube, dim 64, 500 epochs, 8 host chunks, seed 42.
Source: `results/mixmatch/mm_yt1M_*.log`, `time_both.txt`, `time_low_deg.txt`.

| | `nbr_walk/both` | `nbr_walk/low_deg` | change |
| --- | --- | --- | --- |
| `D.nnz` | 23,163,843 | 20,555,488 | -11.3% |
| t_aug | 28.9 s | 23.8 s | -17.6% |
| t_embed | 512.1 s | 490.8 s | -4.2% |
| wall clock | 12:20.62 | 11:41.03 | -5.3% |
| **peak RSS** | 4799 MB | 4505 MB | **-6.1%** |
| accuracy | 0.9764 | 0.9157 | -6.2% |
| F1 | 0.9760 | 0.9103 | -6.7% |
| AUC | 0.9978 | 0.9821 | -1.6% |
| **hop R2** | **0.436** | 0.157 | **-64.0%** |

**`low_deg` is rejected.** It buys 6.1% of the memory and it sells 64% of
the geometry. The adjacency coverage falls to 54.6%, and the attraction
lives at `h = 1` only.

**What survives the rejection, and it matters.** The asymmetry itself is
NOT the problem. Every 1.13M run above uses a directed `D` (axis E-C), and
`nbr_walk/both` reaches AUC 0.9978 with it. What fails is dropping an
ATTRACTION term. A rule that drops FAR entries instead of `h = 1` entries
carries none of this cost, and it has not been tested.

---

# 2026-08-18 -- the Cora and PubMed campaign, 137 runs

Measured 2026-08-18, roughly 02:00 to 09:30 PDT. Cora 88 runs, PubMed 49.
Four grids and a learning-rate ladder, at `dim 64`, 2000 epochs, weight
`min_gap`, far pairs at `deg^0.75` and weight 100.
Source: `RESULTS.md`, regenerated by `update_results.py` from the RESULT
line of every log. No number below is typed by hand.

**A caution that governs every memory number here.** On Cora and PubMed the
peak RSS is dominated by a fixed ~1 GB of JAX and Python. Across runs where
`D.nnz` varies 13 times, the peak moves 27 MB. **`D.nnz` is the size proxy
at this scale, and RSS is not a measurement.** Real memory numbers exist
only at com_youtube.

## Finding 1: `plain` is not a good update rule, and every earlier result used it

The largest gap of the branch, and it was never tested until now. `optim.py`
held five rules since 2026-08-15 and **only `plain` had ever produced a
reported number.**

Hop R2, the discriminating metric, at each rule's best schedule:

| rule | Cora | PubMed |
| --- | --- | --- |
| adam + linear decay | **0.616** | **0.517** |
| nesterov, const 0.1 | 0.554 | 0.513 |
| momentum, const 0.1 | 0.561 | 0.512 |
| sqn, const 0.999 | 0.556 | 0.508 |
| velocity, const 0.999 | 0.557 | 0.505 |
| plain + decay | 0.513 | 0.494 |
| **plain, const** | 0.523 | **0.288** |
| sgd, const | 0.464 | 0.275 |
| fa2 | 0.172 | 0.235 |
| adam, const | 0.414 | 0.219 |

**On PubMed `plain` is 44% below the leaders.** Both graphs agree on the top
group (momentum, nesterov, sqn, velocity) and on the bottom (fa2, and adam
at a constant rate).

## Finding 2: the learning rate dominates the choice of rule

`plain` alone moves from 0.134 to 0.523 across the rate, a factor of 3.9 --
**larger than any difference between rules.** Thus every statement about an
optimizer that was made at one learning rate is worth nothing, including
the ones this project made before today.

The ladder of the user (0.999, then 0.9, then 0.1, every rate below 1.0)
gives: `momentum` and `nesterov` diverge at BOTH 0.999 and 0.9 and survive
only at 0.1; the other six run at 0.999.

## Finding 3: linear decay is a CONVERGENCE fix, not a general improvement

The mechanism is visible in one column. Sorting the PubMed rules by their
constant-rate final `||dZ||`:

| constant-rate `\|\|dZ\|\|` | rules | what decay does to the hop R2 |
| --- | --- | --- |
| above 1, thus NOT settled | adam 1.59, plain 1.22, sgd 1.17 | **+136%, +72%, +80%** |
| about 0.02, thus settled | sqn, velocity | -3%, -1% |

**Decay helps exactly the runs that failed to settle, and buys nothing
where the layout already settled.** `adam` goes 0.219 -> 0.517 on PubMed and
0.414 -> 0.616 on Cora, and its AUC repairs from 0.9677 -- the only
sub-0.99 AUC of the grid -- to 0.9990.

**A near miss, recorded because it nearly cost the campaign its best
result.** The PubMed grid first carried only the CONSTANT-rate survivors of
Cora, thus `adam + decay` -- the best point on Cora -- was not run on
PubMed at all. Flagging `adam` on its constant-rate number would have
excluded the leading rule from com_youtube. The user caught this. The rule
that follows: **a configuration that survives on one graph must be carried
to the next, and a schedule is part of the configuration.**

Two configurations fail under decay and are flagged: `fa2` under either
schedule (0.212..0.235, last on both graphs), and `nesterov` decaying from
0.999, which nearly diverges (`||dZ||` 175, AUC 0.879) because the decay
STARTS at the rate that breaks it. `momentum` diverges outright from both
0.999 and 0.9.

## Finding 4: the bucket policy is EXONERATED; the far pairs were the cause

**This overturns a verdict recorded in this document.** The G3-era finding
said "`cap -> buckets` costs 73% of the hop R2", and attributed the loss to
the bucket policy. That attribution is wrong, and the reason it was made:
`buckets` has NO far pairs BY CONSTRUCTION, because its size formula
`|E| + |V|*log10(|V|)` leaves no room for them. The policy and the
long-range term were confounded in every earlier run.

`--far-with-buckets` separates them, at 3 seeds on both graphs:

| | Cora hop R2 | PubMed hop R2 |
| --- | --- | --- |
| buckets, no far | 0.434 ±0.013 | 0.346 ±0.008 |
| buckets + far | **0.636 ±0.024** | **0.446 ±0.009** |
| gain | **+46.5%** | **+28.8%** |

On Cora `buckets + far` reaches 0.636 against `cap_sym`'s 0.639 -- a 0.5%
difference, noise -- **at 62% of the entries** (46,880 against 75,390).

**The earlier text is NOT removed**, per the rule of this document. It
stands with this entry as its correction. The corrected statement: the
bucket policy is competitive with the cap once it carries a long-range
term, and it is cheaper.

## Finding 5: a directed `D` buys size and speed and costs geometry

The 2x2 the user asked for, 3 seeds, both graphs. PubMed shown with its
sizes and times, because the ordering is identical on Cora:

| variant | D.nnz | aug s | embed s | AUC | hop R2 |
| --- | --- | --- | --- | --- | --- |
| cap_sym | 653,900 | 6.7 | 23.3 | 0.9960 | **0.427** |
| cap_dir | 360,700 | **1.9** | 12.0 | **0.9989** | 0.353 |
| buckets_sym | 228,400 | 7.2 | 11.3 | 0.9968 | 0.346 |
| buckets_dir | **173,300** | 2.5 | **9.7** | 0.9988 | 0.243 |

`cap_dir` holds 45% fewer entries, augments 3.5 times faster and embeds
1.9 times faster, and has the BEST AUC -- and loses 17% of the hop R2.
`buckets_dir` is 3.8 times smaller than `cap_sym` and loses 43%.

**`buckets_dir` is NOT flagged despite being last on both graphs.** The one
gate com_youtube still fails is memory, 4374 MB against 2600. The variant
with the worst geometry may be the one that passes the gate that blocks the
branch, thus it goes to com_youtube on its size and not on its R2.

## Finding 6: the second-order walk -- `p` is a clean tradeoff axis, and AUC is a bystander

The real node2vec walk, by rejection sampling. The ordering
`p2.0 > q0.5 > control > q2.0 > p0.5` reproduces EXACTLY on both graphs,
which is the cleanest replication of the campaign.

Against the `p = q = 1` control:

| | Cora D.nnz | Cora embed | Cora R2 | PubMed D.nnz | PubMed embed | PubMed R2 |
| --- | --- | --- | --- | --- | --- | --- |
| p = 0.5 | **-17.9%** | **-8.5%** | -15.2% | **-12.0%** | **-7.0%** | -8.2% |
| p = 2.0 | +14.1% | +4.5% | **+16.9%** | +8.6% | +8.3% | **+11.5%** |

**`p` trades size and speed against geometry in opposite directions**, on
both graphs. A lower `p` returns to the previous node more often, thus the
walk covers less ground: a smaller `D`, a faster embedding, a worse hop R2.
A higher `p` does the reverse.

**A separation test that changes the reading, and it disagrees with a first
report of this result.** Only the GAIN direction is established:

| | R2 gap | seed spread | separated? |
| --- | --- | --- | --- |
| Cora p = 2.0 | 0.065 | 0.041 | **yes** |
| PubMed p = 2.0 | 0.050 | 0.019 | **yes** |
| Cora p = 0.5 | 0.059 | 0.058 | marginal |
| PubMed p = 0.5 | 0.036 | 0.042 | **NO** |

Thus `p = 2.0` really does buy hop R2. **`p = 0.5`'s hop R2 LOSS is not
established on PubMed**, while its 12..18% saving in `D.nnz` is
deterministic and not a sampled quantity. That makes `p = 0.5` a better
candidate than the raw percentages suggest: a real size saving for a cost
that the seeds cannot resolve.

**AUC never discriminates in this sweep, and that is itself a finding.**
Every AUC and accuracy delta across the whole `p` sweep is 15 to 50 times
UNDER the 1.5% floor and inside the seed spread, on both graphs: AUC moves
by at most 0.063%. Thus the choice of `p` is a two-way tradeoff of memory
and time against geometry, with link prediction as a bystander -- not a
three-way tradeoff. Link prediction cannot be used to choose `p`, and a
report that quotes AUC to justify a `p` is quoting noise.

## What this campaign does NOT establish

1. **Nothing here is a memory result.** Every peak RSS is inside a ~1 GB
   fixed cost. The G3 gate is a com_youtube question and it is untouched.
2. **The optimizer rows are one seed.** Grids B, C and D carry three seeds;
   grid A and the ladder carry one. The top four rules sit inside 0.008 of
   each other on PubMed, which one seed cannot separate.
3. **`fdlinear` only.** Every grid above holds the force law fixed. Whether
   these rankings survive the `v1` law is not measured.

---

# UPDATE 2026-08-24 -- `fdwalk/modular.py`, and two limits of this record

**What was built.** `fdwalk/modular.py`, the walk augmentation in the
modular form of `fodined/modular.py`: the file holds the configuration,
the ORDER of the steps and the messages, and nothing else. The engine, the
force law (`v1`) and the layout are imported from `fodined` unchanged.
`fdwalk/walks.py` and `fdwalk/weights.py` are VERBATIM copies of this
directory, thus `diff` is the parity argument. Every score comes from the
`evaluator` package; the script measures nothing itself.

**Nothing above is changed.** The two entries below are limits of THIS
record that the reproduction exposed, and they are appended, not applied.

## Finding 1: this directory no longer reproduces its own G2 table

The G2 PubMed table was measured 2026-08-15. Re-run TODAY, unchanged, with
the command of `run_g2.sh`, `bench_fdwalk.py` gives:

| | recorded 2026-08-15 | `bench_fdwalk.py`, 2026-08-24 |
| --- | --- | --- |
| raw pairs | 15,447,428 | 15,449,427 |
| unique | 2,554,345 | 2,557,515 |
| after the cap | 242,252 | 242,414 |
| `D.nnz` | 653,866 | **654,190** |
| final \|\|dZ\|\| | 0.2223 | **0.3553** |
| AUC | 0.9951 | 0.9950 |
| hop R2 | 0.522 | **0.513** |

**The cause is `walk_pair_stats(block=)`, and it is not a memory knob.**
`uniform_walks` takes ONE `rng.random(block)` for each STEP of the walk,
thus the partition of the start nodes decides which numbers of the stream
reach which walk. Two block sizes give two different walk sets from one
seed. Measured on PubMed at the seed 42:

| block | raw | unique | after the cap |
| --- | --- | --- | --- |
| 30,000 | 15,451,785 | 2,560,240 | 242,375 |
| 50,000 (the value in `walks.py` today) | 15,449,427 | 2,557,515 | 242,414 |
| **100,000** | **15,447,428** | **2,554,345** | **242,252** |
| 200,000 | 15,447,798 | 2,555,755 | 242,239 |

**The G1 and G2 tables were made at `block = 100,000`.** Cora is not
affected at any value above 27,080 -- `n * n_walks` is 27,080 there, thus
one block -- which is why G1 reproduces and G2 does not. `fdwalk/modular.py`
therefore carries `PAIR_BLOCK = 100_000` and it names the difference.

The divergence begins at the RAW pair count, thus it is upstream of the
cap, the weight, the far pairs and the physics. It is a change of the
INPUT and not of the method, and no number of this document moves because
of it.

## Finding 2: four recorded fields have a floor, and it is not 4 decimals

Four runs of ONE configuration (PubMed, seed 42), with a bit-identical `D`
and a bit-identical `Z` -- `final ||dZ|| = 0.3553` and `r2_dist = 0.513`
in all four:

| field | the four runs | spread |
| --- | --- | --- |
| `dz`, `auc`, `r2_dist`, `mae_dist`, `dnnz` | identical | **0** |
| `acc` | 0.9676, 0.9676, 0.9675, 0.9673 | 3e-4 |
| `r2_vec` | 0.485, 0.473, 0.480, 0.474 | **0.012** |
| `mae_vec` | 0.836, 0.848, 0.845, 0.846 | 0.012 |

The cause is the multi-threaded estimator and not the embedding.
`RandomForest(n_jobs=-1)` sums its trees in thread order, thus a
borderline sample flips; the MLP of the `vector` feature reads 128 columns
and its matmuls accumulate the same way, and 300 iterations amplify it.
The `distance` feature reads ONE column and it is stable.

**Thus `r2_vec` must not be quoted to three decimals, and a difference
below about 0.012 in it is not a result.** The 1.5% floor of `CLAUDE.md`
gives 0.007 at `r2_vec = 0.47`, which is BELOW this spread: for this one
field the estimator noise, and not the floor, is the test. `r2_dist` is
unaffected and it stays the discriminating metric, as every earlier entry
of this document already treats it.

## What reproduces

Cora and PubMed, three seeds each, `walk/min_gap/plain`, through
`fdwalk/modular.py` at `PAIR_BLOCK = 100_000`:

* EXACT on every deterministic field -- `D.nnz`, final `||dZ||`, `auc`,
  `r2_dist`, `mae_dist`.
* Inside the spread of Finding 2 on `acc`, `f1`, `r2_vec` and `mae_vec`.
* NOT reproduced: `h2_exact` and `h2_pairs`. `evaluator` holds no H2 task,
  and rebuilding one inside the script would defeat the reason to call the
  package. H2 measures the augmentation and not the embedding.

## A defect of `evaluator`, found by this work and repaired

`evaluator`'s `fodined` and `fodiwalk_*` protocols drew their negative
pairs with the rejection sampler of `bench_other_ge.py`. Both files they
are NAMED for call `sample_far_pairs` instead
(`fodined/link_prediction.py:80`, `fodiwalk/misc/evaluation.py:80`), which
collapses the direction with `np.unique`, sorts the batch, and spends a
SECOND `rng.choice` when the batch overfills. Thus it returns other pairs
AND it leaves the generator in another state, and every draw after it --
the whole hop sample -- differs. Measured on Cora: `acc` 0.9777 against
0.9744, and the hop R2 that followed the shifted generator 0.630 against
0.616.

The repair is a third compatibility knob beside the `pos_draw` and
`pair_draw` that `evaluator/config.py` already carries: `LPCfg.neg_draw`,
with the draw sequence transcribed into `evaluator/pairs.py`. Verified:
`otherge` is unchanged (0.000e+00 on all five scores) and `fodined` now
reproduces `fodined/link_prediction.py` bit-exactly.

### UPDATE 2026-08-24 -- Finding 2 named the WRONG mechanism

**The text of Finding 2 above is kept, and its MECHANISM is wrong.** It
says the spread of `acc`, `f1`, `r2_vec` and `mae_vec` comes from
multi-threaded sklearn estimators. It does not. Session `fdmap-b6`
investigated and refuted it; this session re-ran the two load-bearing
tests and confirms them.

**The scoring stack is BIT-DETERMINISTIC.** Three repeats of the whole
evaluation on ONE fixed `Z` give identical scores to six decimals. The
four repeat runs of Finding 2 varied because each one recomputed `Z`.

**`Z` itself is not reproducible on this GPU.** Two runs with a
bit-identical `D` give `||A-B|| / ||A|| = 3.10e-06`, and 96.4% of the
cells differ.

**The cause is a scatter-add in the kernel.**
`sell_c_sigma.py:436` is `dZ.at[rows].add(F, mode="drop")`. A hub row that
the plan SPLITS becomes several virtual rows that carry the SAME owner id,
thus several adds land on one address. Float32 addition is commutative but
NOT associative, thus the order decides the last bits.

**The threshold is exactly 3, and it explains the size asymmetry.**
Measured here, `jnp.zeros(...).at[idx].add(F)` with `k` duplicate indices,
8 calls each, on this GPU:

| duplicates | distinct results of 8 calls |
| --- | --- |
| 1 | 1 -- deterministic |
| 2 | 1 -- deterministic (`a + b == b + a`) |
| 3 | 6 -- NONDETERMINISTIC |
| 4 | 8 -- NONDETERMINISTIC |

Two addends are safe; three are not. The maximum owner-id multiplicity
INSIDE ONE BATCH, measured on the plans of this experiment:

| graph | `n_split` | batches by max in-batch multiplicity | reaches 3+ |
| --- | --- | --- | --- |
| Cora | 3 | `{1: 15, 2: 1}` | **no** |
| PubMed | 17 | `{1: 54, 4: 1}` | **yes** |

**Thus Cora is bit-reproducible and PubMed is not, and it is not about the
size of the graph.** It is whether one hub row splits into three or more
virtual rows inside ONE batch. Cora reaches 2 and stops there.

**Why only those four fields.** `dz` is a mean over `n * 128` cells, `auc`
is a rank statistic, and `r2_dist` reads ONE feature and converges in 17
iterations -- all three absorb a 3e-6 shift. The `vector` MLP reads 128
features, stops on an Adam loss plateau (`n_iter_no_change`), and sits deep
in overfit; a 3e-6 shift moves WHEN the stop fires (`n_iter_` 66 against
64), and the test R2 moves with it. `acc` and `f1` threshold at `p = 0.5`,
thus about 7 of 10,000 test pairs flip, which is the 7e-4 seen.
The hypothesis that the MLP hits `max_iter` is REFUTED: it converges at
64..66 of 300.

**The record cannot be reproduced by ANY code in this repository.** The
ORIGINAL `bench_fdwalk.py`, unmodified, at `block = 100_000`, misses the
recorded table on exactly the same four fields and on no other, and the
direction of the miss is MIXED across the seeds (s42 and s56 high, s88
low) -- the signature of a scatter and not of a pipeline difference.
Library drift is ruled out: numpy 1.26.4 and sklearn 1.7.2 were both
installed BEFORE the 2026-08-15 record, and jax 0.6.2 matches its log.

**What follows for this document.** `acc`, `f1`, `r2_vec` and `mae_vec`
are not reproducible to four decimals on this machine, for any run. Quote
them with a spread. `r2_dist`, `auc`, `dz` and `D.nnz` are exact and stay
the fields a claim rests on.

**Two open items, neither applied here.** `k_max` above the widest row
gives `n_split = 0` and makes `_step` bit-identical (measured), at the
cost of padding -- a decision for the engine and not for this document.
And `sell_c_sigma.py:11-22` tells the reader the layout has "no scatter,
no atomics"; that is false whenever a hub row splits, and the next reader
of that file will believe it.

#### A note on the two multiplicity tables, 2026-08-24

`fdmap-b6` reported this mechanism with a table of its own, and the two do
NOT line up cell by cell. They count DIFFERENT statistics, and neither is
a failed reproduction of the other:

| source | what one cell counts | Cora | PubMed |
| --- | --- | --- | --- |
| `fdmap-b6` | the per-OWNER multiplicities above 1, pooled over every batch | `{2: 1}` | `{2: 1, 3: 1, 4: 1}` |
| the table above | the per-BATCH MAXIMUM multiplicity, the 1s included | `{1: 15, 2: 1}` | `{1: 54, 4: 1}` |

Both say the one thing that matters: **Cora tops out at 2 addends on one
address and PubMed reaches 4.** A reader who diffs the two tables without
this note will look for a defect that is not there.

---

# 2026-08-20 -- an outside check of one campaign number

Two other sessions checked our scoring today. Recording it because until now
every number here had been produced and checked by one session only.

**What was checked, and against what.** Session `c5` re-scored the Cora run
`results/g1_cora_walk_min_gap_plain_s42.log` (walk pairs, `min_gap` weight,
`plain`, seed 42) and reproduced nine fields exactly, including:

    r2_dist  = 0.616
    mae_dist = 0.930

Both values are in `results/g1_cora.tsv` and this file confirms them. This
is the first check of a campaign number by anyone outside this session, and
it landed on the hop-distance score -- the number we lean on hardest, since
it is the one that beats node2vec about ten times over.

**A second check, and what it does NOT cover.** Session `fdmap-68` also
found its own scoring package bit-exact (`max abs diff = 0.000e+00` across
15 values) against a frozen copy of `fodiwalk/misc/evaluation.py`'s
`task_hop` and `hop_sample`. That copy was taken 2026-08-22. It shows their
package agrees with the frozen copy. **It says nothing about whether our
live `bench_fdwalk.py` still matches that copy.** The `c5` run is the one
that covers our live code, and it is the one to cite for this campaign.

**One thing worth stating plainly, because it looks wrong and is not.**
`bench_fdwalk.py:848` passes `gap_csr` -- the walk gap of every stored pair
-- into `hop_sample`. A reader could fairly assume the stored guess becomes
the thing we predict. It does not. The target comes from a real
shortest-path run on the graph:

    bench_fdwalk.py:630
    hop = shortest_path(A, method="D", unweighted=True, indices=blk)

`gap_csr` feeds only the `h2_exact` column, which reports how often the
walk gap equalled the true hop distance. That is how H2 was confirmed at
98.1% exact and 0.0% under. It never touches the regression.

This matters because `fodined/modular.py` does have the defect this looks
like: it reads its hop target from `D.data`, the stored weight, which is an
upper bound from sampled pairs and not the graph's true distance. A hop
score out of `modular.py` cannot be compared with one out of this campaign.
