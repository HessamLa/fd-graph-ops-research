Q: 
 
Please consume this repo www.github.com/hessamla/fd-graph-ops-research
 
---
 
A:
 
I read the repository. Here is what it contains.
 
# fd-graph-ops-research
 
Force-directed graph embedding, driven by random walks. JAX on GPU. 22 commits, last one 2026-08-27. There is no root README, so the entry points are `so-far.md`, `CLAUDE.md` and `AGENTS.md`.
 
## The idea
 
Put each node of a graph at a point in `d` dimensions. Two forces move each point. An attractive force pulls a node to its neighbors. A repulsive force pushes it from other nodes. The layout at equilibrium is the embedding. No loss function.
 
The contribution is in stage 2. Earlier work collected every node in the k-hop ball of a node. That set follows the hub degree, so it does not scale: 2.5 G pairs at k=2 on com_youtube, 53 G at k=3. Walks replace it. A budget of `walks x walk_len` steps per node does not grow when a hub appears.
 
## Three stages, and one rule
 
1. **Graph construction** (`make_graph/`). Load or build the graph.
2. **Graph augmentation** (`augment_graph/`). Walk the graph. Build `D`, a sparse matrix of node pairs with an integer weight `h`.
3. **Embedding** (`forcedirected/`, `embed/`). Apply the force law to `D`.
 
The rule: stage 3 consumes only. It does no graph analysis and no data preparation. Planes, degrees and force parameters belong to stage 2.
 
## The packages
 
| Path | Role |
| --- | --- |
| `forcedirected/` | The one engine. `ForceDirected` (epoch loop, row batching, `Z` update), the SELL-C-sigma kernel, 8 update rules, CSR helpers. It imports numpy, scipy, jax and itself only. Nothing else in the repo. |
| `fodiwalk/` | The current method. Config, the three stages, the walk policies, the force laws. |
| `fodined/` | The earlier k-ball method. Kept as the reference. It forwards to `forcedirected/` for the kernel. |
| `evaluator/` | The one scorer. Two tasks: `link_prediction` and `dist_approx`. Six named protocols, so a number states which settings made it. |
| `experiments/` | The campaigns, the run scripts and the records. |
| `fdwalk/`, `archive/` | Older code. `fdwalk` is the former name of `fodiwalk`. |
 
Two packages have a named owner session in `AGENTS.md`. Other sessions read and run them, but do not edit them: `evaluator/` (session fdmap-68) and `forcedirected/` (session fdmap-b3).
 
## Force law: `fdlinear`
 
Let `z_uv = z_v - z_u` and `x = ||z_uv||`.
 
```
h == 1:  Fa = k1 * x
         Fr = -kr * exp(-k4 * x)
h >= 2:  Fa = 0
         Fr = -(h / freq) * exp(-k4 * x)
```
 
`h` is the minimum walk gap, an integer in `[1, window]`. `freq` is the count of times `v` came up in the walks of `u`. `fdlinear_fused` puts `h/freq` into one plane and saves 8.9% of peak memory.
 
## State of the work
 
137 completed runs on Cora (2,708 nodes) and PubMed (19,717 nodes). Gates G1 and G2 pass. Gate G3, com_youtube at 1,134,890 nodes, fails on memory: 4374 MB against a 2600 MB limit.
 
Against node2vec on Cora at dimension 64:
 
| Metric | node2vec | nbr_walk |
| --- | --- | --- |
| accuracy | 0.9750 | 0.9764 |
| AUC | 0.9964 | 0.9978 |
| hop R2 | 0.042 | 0.436 |
| peak RSS | 1382 MB | 4799 MB |
 
Link prediction is a tie. Hop distance is 10.4 times better. Memory is the one axis where node2vec wins clearly. Memory is thus the open problem of the project.
 
Other results on record: the learning rate decides which update rule wins, so no rule is good on its own. `plain` is weak, and every result before 2026-08-18 used it. `momentum` and `nesterov` diverge at lr 1.0. Linear decay fixes convergence but gives no general gain. A directed `D` saves size and time and costs geometry.
 
## Traps the documents mark
 
These caused silent defects, so I list them.
 
- A stored weight must be an integer, and 1 must mean adjacency. A float weight makes every degree 0, the whole force goes to zero, and no error appears. `mean_gap` and `pmi` have this defect.
- A row with no `h = 1` entry freezes for the whole run.
- The order of the COO triples is part of the result. Do not sort them.
- `step` is not bit-reproducible when a hub row splits. Do not write a bit-equality test on PubMed.
- For `fodined` link prediction, accuracy 0.9744 and hop R2 0.616 are correct. 0.9777 and 0.630 are the defect.
- Do not pipe a test run to `tail`. A killed pytest then reads as a pass.
 
## Largest open items
 
- The memory of the augmentation stage: 3583 MB of the 4374 MB peak. Only memory that is never allocated helps, because RSS is a high-water mark.
- `edge_sampling`: two plans, one resident set of `h = 1` pairs and one refilled budget of far pairs. This is the main design on the list.
- Attribution of the 1.13M run that moved AUC from 0.8988 to 0.9954. Four things changed at once.
- Three commits are local. The push is blocked by 3.7 GB in five files above the GitHub limit of 100 MB.
- A paper draft exists at `fodiwalk/dev-docs/PAPER.md`, dated 2026-08-23.
 
I did not run any code. Tell me what you want next: a written orientation document, a review of one package, or work on one of the open items.
 
---
 
Q:
 
Let's talk about evaluation metrics used therein.
 
Link prediction and R^2 is used. Another metric that could have been used is \rho. What other metrics could have been used to evaluate fodiwalk? (excluding node classification)
 
---
 
A:
 
None of these appear anywhere in the repo, so all of the following are new.
 
# What the repo measures today
 
| Task | Score | How it is produced |
| --- | --- | --- |
| `link_prediction` | accuracy, precision, recall, F1, AUC | random forest on the Hadamard product `Z[u]*Z[v]`, balanced positives and negatives |
| `dist_approx` | mae, mre, rmse, r2, exact | baseline, random forest and MLP on either 1 column (Euclidean distance) or `n_dim` columns (`l1`) |
 
Both go through a trained model. That is the weak point. A model with capacity can rescue a poor layout, and the feature width is part of the protocol, so a number is only comparable inside one protocol. This is exactly why `evaluator` had to freeze six named protocols.
 
Your `rho` fixes that for the distance side. It needs no model, no train and test split, and no feature choice. Note one detail: the hop distance is a small integer with many ties, so use Spearman with midranks, or Kendall tau-b, and report the tie fraction beside it.
 
# Metrics that fit this project
 
## 1. Model-free correlation, the direct replacement for hop R2
 
Spearman rho, Kendall tau-b, and Pearson r between the embedded distance and the true hop distance. Report rho both globally and per source node. The per-source form matches how a distance oracle is used, and it removes the effect of the source sample.
 
Value here: high. It states the claim of the project, that the layout holds the order of graph distance, with nothing between the layout and the number.
 
## 2. Stress
 
A force-directed layout is a stress minimizer, so stress is the native quality measure of the method. Report Kruskal stress-1 after an optimal scale factor, because the layout has no fixed scale. Also report the average distortion and the worst-case distortion of metric embedding theory, which are the numbers the Bläsius paper in `papers/` uses.
 
Value here: high, and no other metric in the set measures the magnitude of the distance error in a scale-free way.
 
## 3. Neighborhood retrieval, model free
 
For each node, take its `k` nearest points in the embedding and compare them against its true neighbors or its true `k`-nearest nodes by hop distance:
 
- precision@k and recall@k
- mean average precision over each node's neighbor list
- Hits@k and mean reciprocal rank
- kNN overlap, the Jaccard index of the two neighbor sets
- trustworthiness and continuity, the two errors of the dimensionality reduction field
 
`TODOs.md` already owes "adjacency recovery" as a P2 diagnostic. That item is precision@k under another name.
 
Value here: high. It is link prediction with no classifier, and MAP with mean rank is the standard comparison in the Poincare and node2vec literature that this repo already benchmarks against.
 
## 4. Graph reconstruction
 
Take the `|E|` nearest pairs in the embedding and measure how many are real edges. One number, no training, no negative sampling.
 
## 5. Ranking scores for link prediction
 
Score a pair by the distance alone, then take AUC and average precision. Two gains:
 
- No random forest, so the number is a property of the layout.
- Average precision is the correct score under the true class ratio. Today the negatives are balanced, and a real graph is sparse, so the AUC of 0.99 overstates the result. Add precision@k at a realistic negative ratio.
 
## 6. Structure preservation without labels
 
- Run Louvain on the graph, run k-means on the embedding, then report NMI or ARI between the two partitions.
- Take the clusters found in the embedding and measure their conductance on the graph.
- Correlate the vector norm, or the distance to the layout center, with node degree or with a centrality rank. The scale-free layout paper predicts this relation.
 
None of these use node labels, so they stay outside node classification.
 
## 7. Stability
 
Run two seeds, align the two embeddings with Procrustes, then report the residual. Or report the rank correlation of the nearest-neighbor sets across seeds. `FINDINGS.md` states that every 1.13M number comes from seed 42 alone, so a stability score is owed.
 
## 8. Cost, which the project already treats as a metric
 
Peak RSS and the runtime of each stage are required by `CLAUDE.md`. Two more are cheap and useful: the epochs to reach a target quality, and `D.nnz` per node.
 
# What I would add first
 
1. Spearman rho and Kendall tau-b on the `dist_approx` pairs. Lowest cost, highest relevance, and it makes every recorded hop R2 readable without the model.
2. Normalized stress with optimal scaling.
3. precision@k and MAP for neighborhood retrieval.
 
Together these give one model-free number for order, one for magnitude, and one for local structure. The current metrics then become the downstream check, not the primary evidence.
 
Two constraints to respect. `evaluator/` has one owner session, so a new score enters through it. And a new score needs its own noise floor, because the 1% rule of `CLAUDE.md` was set for AUC and R2 and does not transfer to a rank correlation without measurement.
 
Do you want me to write this as a proposal for the evaluator package, with the definition and the tie rule for each metric?
 
---
 
Q:
 
Elaborate the following. Why did the evaluator had to freeze six named protocols? To achieve what?
"A model with capacity can rescue a poor layout, and the feature width is part of the protocol, so a number is only comparable inside one protocol. This is exactly why `evaluator` had to freeze six named protocols." 
 
 
use "ρ" instead of "rho"
 
Breakdown Spearman ρ and Kendall τ-b. What are they? What do they signify?
 
---
 
A:
 
# Part 1: why `evaluator` froze six named protocols
 
## The situation before the package
 
Four files in this repository each scored an embedding, and each one was written separately:
 
- `experiments/other-ge/bench_other_ge.py`
- `experiments/large-graph-node2vec/bench_node2vec_1M.py`
- `fodined/modular.py` with `fodined/link_prediction.py`
- `fodiwalk/misc/evaluation.py`
 
They disagree on almost every setting that moves a score:
 
| Axis | The values found in the four files |
| --- | --- |
| pair budget, link prediction | 80,000 against 50,000 |
| pair budget, hop distance | 20,000 against 2,000 |
| forest size | 100 trees against 200 trees |
| forest leaf size | `min_leaf` 1 against 25 |
| MLP size | (64, 32), (128, 64), (256, 128) |
| early stopping | on against off |
| feature width | 1 column (Euclidean distance) against `n_dim` columns (`l1`) |
| source of the truth | exact PLL, block BFS, or the stored weight `D.data` |
| negative sampler | rejection against `far_pairs` |
| positive draw | `rng.choice` only over the cap, against always |
| hop pair draw | reject every edge, against draw inside one BFS row |
 
Nothing in the output said which of these produced a number. A reader saw two AUC values and assumed one comparison.
 
## Why this is fatal, not untidy
 
Two mechanisms, and both are measured in this repo.
 
**Capacity is part of the score.** The score of `dist_approx` is the score of a regressor that reads a feature of the embedding. A wider feature or a larger model raises that score with no change to the layout. `fodiwalk/misc/evaluation.task_hop` states it directly: a model with 128 features can win only because it has more of them. So a difference between two rows can come from the layout, from the feature width, or from the model size, and the number alone cannot say which.
 
**The sampler moves the random generator.** Two samplers that draw the same count of pairs still leave the generator in a different state, so every later draw differs. This is not theory. On Cora with one embedding, the wrong negative sampler gave accuracy 0.9777 and hop R2 0.630 against the correct 0.9744 and 0.616. That defect was live in the record until 2026-08-24, and `AGENTS.md` now carries a correction stating that 0.9777 must not be quoted as a baseline.
 
## What the freeze achieves
 
A protocol is a frozen set of settings, copied from the file that produced a published number. The six are `otherge`, `n2v1m`, `fodined`, `fodiwalk_dist`, `fodiwalk_vec`, and `default`. Six exist because five recorded baselines exist, plus one entry that claims no baseline at all. `fodiwalk` gets two names because its callers sweep both feature widths, so one baseline produced two rows.
 
The freeze buys five things:
 
1. **Comparability becomes explicit.** Every result record carries the protocol name. A number is comparable to a number with the same name, and to no other. The incomparability is still there, but it is now visible instead of silent.
2. **Old numbers stay reproducible.** Verified at a maximum difference of 0.000e+00 for `otherge` link prediction, and on all five scores and all six counts for `fodined`.
3. **Defects are preserved on purpose.** The `fodined` protocol keeps its 2,000 pairs and its one-column feature. A repair here would make every recorded `fodined` number unreproducible. The docstring says this: a protocol records a baseline including its poor choices.
4. **A modified run declares itself.** `override` sets `protocol_modified` when the caller passes any keyword, even one that matches the current value. The flag reports intent, so a later edit of the table cannot turn a hand-tuned row back into a baseline.
5. **No silent fallback.** An unknown protocol name raises and lists the valid names. A typo cannot produce a number that claims a baseline it never used.
 
## The limit of the freeze, and the link to ρ
 
Freezing does not remove the model from the measurement. It only holds the model constant, so that the layout is the one thing that varies inside one protocol. The cost is that six baselines can never be put in one table.
 
A model-free score has no such cost. ρ needs no train and test split, no feature width, no forest size, and no early stopping rule. There is nothing left to freeze, so one number compares across every run and every baseline. That is why it is the first thing I would add.
 
# Part 2: Spearman ρ and Kendall τ-b
 
Both are rank correlation coefficients. Both take a set of pairs, where each pair has two values, and both return a number in `[-1, 1]`. Here the two values are:
 
- `x`, the distance between the two nodes in the embedding
- `y`, the true hop distance between the same two nodes in the graph
 
Both answer one question: **when the graph says one pair is farther apart than another, does the layout agree?**
 
The property that matters for this project: both are invariant under any increasing transform of `x` or of `y`. A force-directed layout has an arbitrary scale, and the force law here is exponential, so the map from embedded distance to hop count is not linear. R2 punishes that curvature. ρ and τ do not see it at all.
 
## Spearman ρ
 
Replace each value by its rank, then compute the ordinary Pearson correlation of the two rank columns.
 
With no ties, this simplifies to:
 
```
ρ = 1 - 6 * Σ d_i² / (n(n² - 1))        d_i = rank(x_i) - rank(y_i)
```
 
With ties, which is the case here, the ranks must be midranks and the simplified form is wrong. Use the Pearson form on the midranks.
 
**What it signifies.** The strength of the monotone relation, measured the same way Pearson r measures a linear relation, but on ranks. ρ = 1 means the layout puts the pairs in exactly the graph order. ρ = 0 means no monotone relation. ρ = -1 means the reverse order.
 
**Its character.** It squares the rank differences, so one badly misplaced pair costs much more than several slightly misplaced ones. Read ρ as "how far, on average, is a pair from its correct rank position".
 
## Kendall τ-b
 
Count over every pair of pairs, meaning every `(i, j)` from the sample. Classify each one:
 
| Class | Condition |
| --- | --- |
| concordant, `C` | the two agree on the order in both `x` and `y` |
| discordant, `D` | they disagree |
| tied in `y` only, `T_y` | the graph gives the same hop distance |
| tied in `x` only, `T_x` | the layout gives the same distance |
 
```
τ-b = (C - D) / sqrt((C + D + T_x) * (C + D + T_y))
```
 
**What it signifies.** τ has a direct reading as a probability:
 
```
τ ≈ P(the two agree) - P(the two disagree)
```
 
τ-b = 0.6 says that for two pairs picked at random, agreement is 60 percentage points more likely than disagreement. There is no such plain statement for ρ.
 
**A worked example.** Four sampled pairs:
 
| pair | embedded distance `x` | hop distance `y` |
| --- | --- | --- |
| a | 0.8 | 2 |
| b | 1.5 | 3 |
| c | 1.2 | 4 |
| d | 2.9 | 4 |
 
Six comparisons: (a,b) agree, (a,c) agree, (a,d) agree, (b,c) disagree because `x` rises and `y` rises but in the opposite order for these two, (b,d) agree, (c,d) is tied in `y` only. So `C` = 4, `D` = 1, `T_y` = 1, `T_x` = 0, and τ-b = 3 / sqrt(5 x 6) = 0.55.
 
## Why "-b" and not plain τ
 
The plain form, τ-a, puts `n(n-1)/2` in the denominator and counts a tie as a failure. In this project `y` is a hop distance, so after the `min_hop = 2` filter it takes maybe five or six distinct values across 20,000 pairs. The tie count is then enormous and τ-a is crushed toward zero for reasons that say nothing about the layout. τ-b removes the tied comparisons from the denominator, one variable at a time.
 
One caveat to record if you adopt it. When `y` is heavily tied and `x` is continuous, τ-b cannot reach 1.0 even for a perfect layout, because the tie patterns do not match. Two answers:
 
- Compute the attainable maximum for the given `y` distribution and report τ-b beside it.
- Or use Somers' D with `y` as the response, `D_yx = (C - D) / (C + D + T_y)`, which conditions on ties in the response only. This is the cleaner choice when one variable is an ordinal target and the other is continuous, which is exactly the situation here.
 
## Which to use
 
| | Spearman ρ | Kendall τ-b |
| --- | --- | --- |
| reading | strength of the monotone relation, on a variance scale | difference of two probabilities |
| ties | needs midranks | handled in the denominator |
| outliers | sensitive, squares the rank gaps | robust, counts only the direction |
| typical size | larger | smaller, roughly two thirds of ρ for well behaved data |
| cost | `O(n log n)` | `O(n log n)` with a merge sort count |
 
They are not interchangeable numbers. Never compare a τ against a ρ. Report both, or pick one and hold it.
 
For this project I would report τ-b or Somers' D as the headline, because the tie structure of a hop distance is severe and τ-b states what it does about it. I would report ρ beside it, because it is the number most readers of a paper expect.
 
Three practical notes for the record:
 
- Report the tie fraction of `y` next to the coefficient. Without it neither number is readable.
- Report ρ per source node as well as globally. The per-source form matches how a distance oracle is used and it removes the effect of which sources were sampled.
- Set a new noise floor. The 1% rule of `CLAUDE.md` was set for AUC and R2. A rank correlation on 20,000 pairs has a much smaller standard error, so use a bootstrap over the pairs to find the real spread before you adopt a threshold.