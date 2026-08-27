Wed Aug 26 10:27:01 PM PDT 2026

The progress so far is the following

This research is a walk-based force-directed graph embedding. We don't use k-ball or shell sets here. We walk graph walk methods, such as those employed in DeepWalk and node2vec, to find the set of candid nodes for each target node.

NOTE: fodiwalk is same as fdwalk. From now on we use fodiwalk (*Fo*rce *Di*rected *Walk*) as a more descriptive, and somewhat unique name.

## Graph augmentation:

After loading the graph, we perform walks, and for each node u, create a set of other nodes W(u): the set of nodes visited on the walk windows of u.
The walk methods and their parameters are listed at the bottom of this file.

## Graph embedding:

We have separated the graph embedding into a package in ``forcedirected/``. All sessions must import this package, instead of rewriting the whole class.

The, dz_u (gradient of node embedding of u) will be calculated with respect to the node embedding of others nodes v ∈ N(u) ∪ W(u)

To calculate dz_u, we need to define (1) force function, (2) optimizer, (3) parameters such as learning rate (constant vs. diminishing) or 

1. Force functions are in close tandem with walk methods and in general with graph augmentation policies. All the parameters and values that are consumed in a force function are generated during graph augmentation, such as the W(.) sets and walk-related values per each node in node in a W(.) set.

2. Optimizer functions determine how quick an embedding converges. Some optimizers work better with some force functions.
The optimizer functions are listed at the bottom of this file.

3. Some parameters control how the embedding algorithm functions and converges. These parametes are of varying nature and use. For example, a constant or decaying learning rate can affect the rate and quality of convergence. Another example is the drop strategy, which determines which rows of columns of the gradient calculation to avoid. One drop strategies work row-wise, skipping a random set of nodes per epoch. Another strategy zeros out random (row,column) values in the gradient block. Another strategy which is **pending an experiment** skips a random set of columns during each epoch.

Another integral part of graph embedding is the sell-C-σ method, used for memory optimization. This method is a core functionality of the current ForceDirected implementation and, therefore, is placed under ``forcedirected/`` package.

## Evaluation:

To have a unified evaluation, we are using the ``evaluator/`` package. All agents must import this module for evaluation tasks. If an update is required, or if there is bug report, it must be relayed to the ``evaluator`` agent in ``.claude/agents/evaluator.md``.


## Experiments summary
137 completed runs. Cora and PubMed. com_youtube has no completed grid.

Full tables: `experiments/fdwalk/RESULTS.md`. Reasoning: `experiments/fdwalk/FINDINGS.md`.

**The learning rate dominates the choice of update rule.** Rank order changes with the rate. A rule is not good or bad on its own.

**`plain` is weak.** Every result before 2026-08-18 used it.

**`momentum` and `nesterov` diverge at lr 1.0.** Also at decay 0.9 and 0.999.

**Linear decay fixes convergence. It is not a general gain.** It rescues `adam` at lr 0.999: hop R2 0.414 -> 0.616.

**The bucket policy is exonerated.** The far pairs caused the earlier loss, not the buckets.

**A directed `D` buys size and speed and costs geometry.** Cora: hop R2 0.29 directed against 0.43 symmetric.

**`p` is a clean tradeoff axis. AUC does not respond.** Cora, AUC stays 0.9983-0.9990 across all five `(p, q)` points while hop R2 moves 0.33 -> 0.45.

### Against node2vec, cora at equal dimension 64

| metric | node2vec | nbr_walk |
| --- | --- | --- |
| accuracy | 0.9750 | 0.9764 |
| AUC | 0.9964 | 0.9978 |
| hop R2 | 0.042 | **0.436** |
| peak RSS | **1382 MB** | 4799 MB |

Link prediction is a tie. Hop distance is 10.4x, on an identical protocol. Memory is the one axis where node2vec clearly wins.

---

## Walk methods

Set with `--pairs`. Each builds `D`, the sparse matrix of pairs and their hop weight `h`.

| method | row `u` holds | window | cap | `D` symmetric |
| --- | --- | --- | --- | --- |
| `walk` | any two nodes inside one window | yes | yes | yes |
| `walk_edges` | as `walk`, plus every edge of `u` | yes | yes | yes |
| `nbr_walk` | every neighbour of `u` at h=1, plus every node a walk **from** `u` reached, at h = the first step | no | no | **no** |
| `ball` | every node within `k` hops | n/a | n/a | yes |

`nbr_walk` anchors each pair at its row, so blocks of start nodes never merge. This removed the accumulator that stopped the 1.13M-node run three times.

### Walk generators

| generator | rule |
| --- | --- |
| uniform | one uniform random walk from each start. The walk of DeepWalk, and of node2vec at `p=q=1`. |
| second-order | node2vec, by rejection sampling. Weight `1/p` to return, `1` to a neighbour of the previous node, `1/q` otherwise. |

`q > 1` keeps the walk near its previous node, which is BFS-like. `q < 1` sends it away, which is DFS-like.

### Walk parameters

| flag | default | meaning |
| --- | --- | --- |
| `--walks` | 10 | walks from each node |
| `--len` | 20 | steps in one walk |
| `--window` | 5 | window width for pairing |
| `--cap` | 16 | pairs kept per node |
| `--p` | 1.0 | node2vec return parameter |
| `--q` | 1.0 | node2vec in-out parameter |
| `--k` | 3 | ball radius, for `--pairs ball` |
| `--far` | 0 | far pairs; 0 means `n*log10(n)` |
| `--far-weight` | 100.0 | `h` given to a far pair |
| `--far-bias` | 0.0 | draw a far pair at `deg^alpha` |
| `--prune-max` | 4,000,000 | accumulator ceiling |
| `--prune-factor` | 4 | prune ratio |

### Edge rule

Set with `--edge-rule`.

| rule | an edge `(u, v)` |
| --- | --- |
| `both` | enters both rows |
| `low_deg` | enters the row of `u` only when `deg(v) >= deg(u)` |

`low_deg` halves the edge cost: 2,987,624 entries against 5,975,248 at 1.13M nodes. It also destabilises the result.

### Weight functions

Set with `--weight`. Each maps a pair's walk statistics to `h`.

| function | `h` |
| --- | --- |
| `flat` | 1.0 for every pair |
| `min_gap` | the smallest step gap, an integer in `[1, window]` |
| `mean_gap` | the mean step gap, rounded into `[1, window]` |
| `pmi` | from `log(c_uv * T / (c_u * c_v))`, mapped into `[1, window]`. A large PMI becomes a small `h`. |

### Row policy

Set with `--policy`.

| policy | keeps |
| --- | --- |
| `cap` | the first `--cap` pairs of each row |
| `buckets` | a fixed fraction from each hop distance |

---

## Optimizer functions

Set with `--optim`. Each takes the force `dZ` and returns the next `Z`. State arrays are counted in units of the size of `Z`, which decides what fits on the card at 1M nodes.

| rule | step | state arrays |
| --- | --- | --- |
| `plain` | `Z + lr*dZ` | 0 |
| `momentum` | `m = 0.9*m + dZ` ; `Z + lr*m` | 1 |
| `nesterov` | look-ahead: `Z + lr*(dZ + 0.9*m_new)` | 1 |
| `adam` | bias-corrected first and second moment | 2 |
| `fa2` | ForceAtlas2 local speed, per node | 1 |
| `velocity` | `v = eta*dZ + (1-eta)*v` ; `Z + lr*v` | 1 |
| `sgd` | update a random `--sgd-frac` of the rows | 0 |
| `sqn` | L-BFGS two-loop recursion | `2*memory + 2`, so 8 at the default |

**`momentum` and `velocity` are not the same rule.** Their steady-state gain differs: `momentum` reaches `dZ/(1-beta)`, a gain of 10 at `beta=0.9`; `velocity` reaches `dZ`, a gain of 1. A learning rate tuned for one is wrong for the other.

**`adam` divides by the size of `dZ`.** A large force means "this node must move far", and `adam` throws that away. `fa2` keeps it: a node that swings gets a small step, a node that travels one way gets a large one.

**`sqn` has no objective to approximate.** `dZ` is a force, not the gradient of a loss, so a curvature pair `(s, y)` approximates nothing. It is recorded as an interpretation.

### Optimizer parameters

| flag | default | applies to |
| --- | --- | --- |
| `--lr` | 1.0 | all |
| `--lr-decay` | `const` | all; `const` or `linear` |
| `--eta` | 0.3 | `velocity` |
| `--sgd-frac` | 0.5 | `sgd` |
| `--sqn-memory` | 3 | `sqn` |
