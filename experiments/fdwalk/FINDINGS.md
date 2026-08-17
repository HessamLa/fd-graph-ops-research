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
