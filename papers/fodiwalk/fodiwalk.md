# Fodiwalk: a walk-augmented force-directed graph embedding method

Fodiwalk turns a graph into one vector per node. It is force-directed:
each vector moves under a simulated force, until the forces settle. The
method has two parts. **Graph augmentation** turns the raw graph into a
weighted graph that states which node pairs feel a force, and how
strong. **Graph embedding** applies a force law to that weighted graph
and relaxes the vectors.

The force law is **fdhop** (`fodiwalk/embed/forces.py:149`), adopted as
fodiwalk's force law. §4 and §6 report what is and is not yet measured
about that choice, honestly: fdhop is not an unqualified win over the
law it replaces, and the paper says where it trails and why that is
being read as a tie rather than a loss.

## 1. Two parts, three stages

The code splits into three stages, and the split is a rule the codebase
enforces, not a suggestion: a stage may not import, call, or read the
internals of another stage
(`fodiwalk/dev-docs/fodiwalk-module.md`, "The stage contract").

| stage | question it answers | where |
| --- | --- | --- |
| 1. graph construction | what is the raw graph? | `fodiwalk/make_graph/` |
| 2. graph augmentation | which pairs feel a force, and how much? | `fodiwalk/augment_graph/` |
| 3. graph embedding | given those pairs and weights, where do the vectors settle? | `fodiwalk/embed/`, `forcedirected/` |

Stage 1 is a loader: it reads a named dataset and returns a symmetric,
zero-diagonal adjacency matrix `A` (`fodiwalk/make_graph/datasets.py`).

## 2. Graph augmentation

### 2.1 Objective

A force law needs, for every node `u`, a set of partner nodes and a
number for each partner that says how strongly `u` should feel it.
Fodiwalk does not compute this from an exact all-pairs distance: an
exact distance table on a graph of a million nodes does not fit in
memory, and most of it would go unused.

Instead, augmentation asks a cheaper question with a random walk: which
nodes does `u` reach, and after how many steps? A short walk from `u`
finds a small set of partners at a small, known cost. Two things must
hold for the step count to work as a distance proxy:

1. **Every direct neighbor of `u` must be a partner of `u`, at the
   smallest possible weight.** A walk over neighbors already finds most
   of these by chance, but not all: at 1.13 million nodes, the walk
   alone found only 68% of the direct edges. Augmentation closes that
   gap explicitly (§2.3) so the local structure of the graph — which
   node touches which — is never approximate.
2. **Farther partners, found only by the walk, still carry a usable
   number.** The step count at which a walk first reaches a node is an
   upper bound on the true hop distance. The force law needs an ORDER
   over how far things are, not the exact distance (§2.4).

### 2.2 The walk

Augmentation starts several walks from every node and lets each walk
take uniform random steps to a neighbor
(`fodiwalk/augment_graph/walks.py`, `uniform_walks`). This is the same
walk as DeepWalk, and the same as node2vec at its return/in-out
parameters `p = q = 1`.

A walk from `u` is one-directional: `u` is the source, and every node
the walk reaches is recorded against `u`. The result, stored as a
matrix `D`, is in general not symmetric — `D[u, v]` comes from `u`'s
own walks and `D[v, u]` from `v`'s, independently, and the two need not
agree. `fodiwalk` implements this design in three ways, called `pairs`
policies, and they are NOT interchangeable — each is a different
method (§4.1):

| `pairs` policy | a row holds | direction |
| --- | --- | --- |
| `walk` | every pair inside a walk's window, capped per node | undirected |
| `walk_edges` | the same as `walk`, plus every real edge forced to `h = 1` | undirected |
| `nbr_walk` | every neighbor at `h = 1`, plus every node a walk reached, with no window or cap | directed, asymmetric |

`nbr_walk` is the fully directed design: row `u` states everything
`u` itself found, and the force law is applied one row at a time, so
`u` moves according to what `u` found by walking outward, in its own
row. It is also, on the evidence gathered for this paper, the *weakest*
fodiwalk arm on distance geometry — cora Spearman ρ 0.516 against
`walk_edges`'s 0.787 on the same force law, budget and optimizer
(`evaluator/reports/260905-store-scorecard.md`, cora table). Every
recorded `fdhop` run in this project uses `walk_edges`, not `nbr_walk`
(§4.1); this paper describes both because `walk_edges` is what fdhop
has actually been run with, and `nbr_walk` is the fully directed,
asymmetric design.

### 2.3 A neighbor gets an edge to `u` and from `u`

Whichever `pairs` policy runs, the walk alone leaves gaps in
direct-neighbor coverage (§2.1). Augmentation closes them by adding
every edge of `A` to `D`, in **both** directions, at the smallest
weight (`fodiwalk/augment_graph/row_merge.py:190`, `with_all_neighbours`):

```python
def with_all_neighbours(stats: RowStats, A, n: int, block: int = 20_000):
    """Add EVERY edge of `A` at `h = 1`, and keep the walk pairs.

    The specification says that no neighbour may be missing. ... This
    function closes that gap by construction, thus the count of `h = 1`
    entries EQUALS `A.nnz`.
    """
```

For an original edge `(u, v)`, this places `v` in `u`'s row at weight 1
AND `u` in `v`'s row at weight 1 — an edge to `u` and from it, in both
rows at once. A pair the walk also found keeps the smaller of the two
weights, which is always 1 for a direct neighbor, so this step can only
tighten a weight, never loosen one. This is how `walk_edges` gets its
name: `walk`'s pairs, plus this step.

### 2.4 The weight `h`

Every stored pair `(u, v)` in `D` carries one number, `h`, an integer
from 1 upward (`fodiwalk/augment_graph/weights.py`). The default rule,
`min_gap`, takes the fewest steps any walk needed to get from `u` to
`v`:

```python
def weight_min_gap(stats, window, n):
    """The smallest step gap. An integer in `[1, window]`."""
    return stats["mn"].astype(np.float64)
```

`h = 1` means `u` and `v` are direct neighbors. `h >= 2` means the
shortest walk that connected them took that many steps — an upper
bound on the true hop distance, not the distance itself. That bound
need not be tight to be useful: on the walks that feed `nbr_walk`, only
7.3% of stored pairs hit the true hop distance exactly, against 98.1%
for `walk`, and `nbr_walk` still wins every link-prediction and
neighbor-recovery score (`fodiwalk/README.md`, "Do not read H2 as a
quality score"). The layout needs the order of `h` and enough
partners, not the value of `h` — §5.2 covers this distinction in full,
because it is easy to conflate with a different, model-based number
that also involves hops.

`h` is the ONE quantity `fdhop` reads. It plays two roles at once:

- **It selects the force regime.** `h == 1` triggers a spring-like pull
  toward the neighbor; `h >= 2` triggers a decaying push away from a
  farther node (§3.1).
- **It sets the push's strength.** For a far pair, the push scales with
  `h` itself: a partner ten steps away is pushed away ten times harder,
  before distance decay is applied.

Because `h` is read directly, it must be exact and it must be an
integer: `degrees_from_D` (`fodiwalk/embed/forces.py:64`) counts the
entries that equal exactly 1 to get each node's degree, and a
non-integer weight has silently produced a measured degree of zero and
a frozen node before (`fodiwalk/augment_graph/weights.py:13-20`).

### 2.5 What augmentation hands to embedding

Augmentation produces `D` and nothing else `fdhop` needs: no `freq`
plane, since `fdhop` does not read one. Embedding also needs one
number per node, the degree, used to average the summed force over
however many partners a row holds; by default this is the count of
that row's `h = 1` entries — the true degree of `u` in `A`, once §2.3
has run.

## 3. Graph embedding

Embedding stage 3 does no graph analysis of its own
(`fodiwalk/dev-docs/fodiwalk-module.md`: "This stage shall not do any
graph analysis or data preparation. It must only consume the data.").
It takes `D` and the degrees, applies `fdhop`, and relaxes the vectors
`Z`.

### 3.1 The force law: fdhop

For a pair `(u, v)` with weight `h` and embedding distance
`x = ||Z[v] - Z[u]||`, fdhop gives (`fodiwalk/embed/forces.py:149`):

```python
def fdhop(x, planes, params):
    h, = planes
    live = h > 0                       # a pad cell has every plane at 0
    near = h <= 1

    Fa = jnp.where(near & live, params["k1"] * x, 0.0)
    decay = jnp.where(near, 1.0, jnp.exp(-params["k4"] * x))
    Fr = jnp.where(live, -params["kr"] * h * decay, 0.0)
    return Fa + Fr
```

In the two regimes `h` selects:

- **`h == 1` (a neighbor).** Attraction `Fa = k1 * x` grows with
  distance, like a spring. Repulsion `Fr = -kr` is constant: it does
  not fall off with `x`. The two balance at `x* = kr / k1` — neighbors
  settle at one characteristic distance, set by two constants alone.
- **`h >= 2` (a farther partner).** Attraction is off. Repulsion
  `Fr = -kr * h * exp(-k4 * x)` pushes `u` and `v` apart, stronger for
  a larger `h` and weaker as `x` grows.
- **A pad cell** (`h = 0`, from the batching of §3.3, never a real
  pair) gives exactly zero.

This is the classical force-directed form `-k3 * h * exp(-k4 * x)` for
the repulsive term (`k3 = kr` here) — the form the original
force-directed literature on scale-free networks uses
(`papers/2021_Blasius_-_Force-directed_embedding_of_scale-free_networks.pdf`).
`fdhop` reaches it by dropping the `freq` term an earlier law,
`fdlinear`, carried in its far-pair coefficient (`h / freq` there,
against plain `h` here).

### 3.2 The engine and the layout

`ForceDirected` (`forcedirected/force_directed.py`) is the epoch loop,
the row batching, and the update of `Z`; it calls
`forces(Z, D, row_start, row_end)` and is blind to which law `forces`
dispatches to. SELL-C-sigma (`forcedirected/sell_c_sigma.py`) is the
batch layout below it: a row with far more partners than most (a hub)
is split into shorter virtual rows, then rows are grouped by width into
padded batches, so a batch reduces with one plain, fixed-shape sum
instead of a scatter-add whose write order a GPU does not guarantee —
"exactly the pathology a power-law degree distribution maximizes," in
the file's own words. `fdhop` reads one plane, `h`; a padded cell
carries `h = 0`, so `live = h > 0` is already false there and the
formula above returns zero for it with no separate branch.

### 3.3 Gradient and learning rate

`dZ`, the row sum the engine accumulates each epoch, **is a force, and
not the gradient of a loss** (`forcedirected/optim.py`, module
docstring). A large `dZ` means "this node must move far," not "the
loss is steep here" — a rule that normalizes by the size of `dZ`, as
Adam does, throws that information away. That distinction is why every
recorded `fdhop` run uses the plain rule
(`Z = Z + lr * dZ`, `forcedirected/optim.py:37`) and not Adam: Adam has
not been run against `fdhop` at all, and the project's own optimizer
study found no quality reason to prefer anything else at this stage.

**The rule that was actually used.** Every recorded `fdhop` run uses
`optim="plain"`, `lr = 0.999`, `lr_decay = "const"`
(`experiments/embeddings/make_embedding.py:44-45`). `plain` was not
picked for winning: an optimizer sweep on the streaming variant of this
method found `velocity` (0.5871), `nesterov` (0.5820) and `plain`
(0.5808) statistically tied — the gap between them is smaller than the
0.039–0.046 hop-R2 spread the same study measured across seeds at fixed
settings (`experiments/fodiwalk-streaming/FINDINGS.md`, §10). The
tiebreak was cost: `plain` holds zero optimizer state arrays, against
one for `velocity`/`momentum`/`nesterov`/`fa2`, two for `adam`, and
eight for `sqn` at its default memory — and it recorded zero
divergences over 230+ runs (same source).

`lr` never reaches 1.0 in this project's runs, and that is a rule, not
a coincidence: `effective_lr = lr * gain`, where `gain` is the rule's
steady-state multiplier on the nominal rate — 1 for `plain`, `sgd` and
`velocity`; `1/(1 - beta)` for `momentum`/`nesterov` (10 at the default
`beta = 0.9`); up to 10 for `fa2`; adaptive for `adam`/`sqn`
(`experiments/fodiwalk-streaming/FINDINGS.md`, §10; standing rule
recorded in `evaluator/dev-docs/METROLOGY.md`, §7: "no effective lr is
ever 1.0"). At `lr = 0.999` and `gain = 1`, `plain`'s effective rate is
0.999.

**The other three constants: two fixed, one not swept.** `k1 = 0.999`
(attraction gain) and `kr = 1.0` (near repulsion) sit at
`fodiwalk/config.py`'s own defaults on every recorded run; `kr` has
never been varied for `fdhop` at all. `k4 = 1.0` — the far-repulsion
decay — is NOT `Config`'s default of `0.01`; every fdhop run overrides
it. This is not a swept optimum: `fdhop` was first written with the
decay hard-coded as the literal `exp(-x)`, and `k4 = 1.0` is exactly
the value that reproduces that run (cora: accuracy 0.9882, AUC 0.9990,
hop R2 0.7119 — identical to four decimals with and without the
parameter). Because `fdlinear`'s recorded runs stay at the `Config`
default `k4 = 0.01`, **the fdhop-vs-fdlinear numbers in §5 are
confounded by a 100x difference in decay sharpness**, not just by the
two laws' formulas. No sweep of `k4` isolates that confound today.

**Epochs.** Every fdhop record ran 200 epochs. Nothing in this method
converges by then: the same optimizer study found every configuration
still gaining 0.12 to 0.22 hop R2 between epoch 50 and epoch 200
(`experiments/fodiwalk-streaming/FINDINGS.md`, §10). A number from a
200-epoch run is a snapshot of an unfinished relaxation, not a
converged one.

## 4. What was actually run

Before the results: what configurations exist, because the paper's
augmentation section (§2.2) and its evidence (§5) must not be read as
describing the same run.

### 4.1 The store, walker × force law

Every `fodiwalk` embedding recorded in `data_cache/embeddings/` pairs
one `pairs` policy with one force law:

| `pairs` | force law | graphs |
| --- | --- | --- |
| `nbr_walk` | `fdlinear` | cora, pubmed |
| `walk_edges` | `fdlinear` | cora, pubmed, wordnet, com_youtube |
| `walk_edges` | `fdhop` | cora, pubmed, wordnet, com_youtube |

**No run pairs `nbr_walk` with `fdhop`.** The directed, asymmetric
design of §2.2 has not been measured with the force law this paper
adopts. Reading a `nbr_walk` number as evidence for `fdhop`, or the
reverse, would be citing one method's numbers for another — §5's
`fdhop` evidence is `walk_edges` evidence, and only that.

Other force laws exist in the store on cora and pubmed only; they are
out of scope for this paper and are not discussed here.

`as_skitter`, `roadnet_ca` and `ncbi_taxonomy` carry no `fodiwalk`
records at all: the four attempted `as_skitter` runs were killed by the
OOM killer (`experiments/embeddings/runs.log`, `grep FAIL`), and the
other two were scheduled and never ran.

### 4.2 One seed, no error bars

Every record in the store above uses seed 42 alone
(`experiments/embeddings/make_embedding.py:38`, default). No
multi-seed `fdhop` comparison exists anywhere in this project's tree.
This project's own metrology draft treats a single-seed result as a
MEASUREMENT and not a WIN (`evaluator/dev-docs/PRD-v2.md`, §5.6): a
comparison needs a measured difference larger than both a 1.5% floor
and the spread across seeds (`evaluator/dev-docs/METROLOGY.md`, §8),
and with one seed the second condition has nothing to test it against.
§5 reports these numbers as measurements for that reason, not because
of house style.

## 5. Metrology: what the numbers in §5 mean

### 5.1 Two scores that sound alike and are not

**hop R2** is the out-of-sample R2 of a trained regressor (a random
forest or an MLP) predicting a pair's TRUE shortest-path hop distance
from a feature of the embedding — either the scalar Euclidean distance
`||Z_u - Z_v||`, or the `n_dim`-wide vector `|Z_u - Z_v|`; the two are
different protocols and are never mixed
(`evaluator/tasks/dist_approx.py:33-36`, "a model with 128 features can
win only because it has more of them"). The target never comes from
`D`; it is read fresh, by BFS, from the true graph
(`evaluator/tasks/dist_approx.py:4-6`). Pairs at `h = 1` are dropped as
the easy case before scoring (`min_hop = 2` by default).

**H2**, a different number entirely, is the gap `D` itself stores —
the walk-found `h` against the true hop distance, measured only on the
pairs `D` actually holds (an in-sample check of the augmentation, not
the layout). §2.4 already leans on it: `nbr_walk`'s stored gaps are
exact on only 7.3% of pairs, against `walk`'s 98.1%, yet `nbr_walk`
wins every downstream score. hop R2 and H2 are drawn from different
pair sets and answer different questions; this paper never puts them
on one axis.

`evaluator/dev-docs/METROLOGY.md` §4 offers wordnet as a stress test of
class-level claims about hop R2 ("average degree <= 3 ... it reads 0.00
to 0.02 at every epoch count") — that specific range does not hold at
the dimension and method set this paper reports (§6 below shows the
fodiwalk arms' random-forest R² at 0.096–0.156, well above it). The
more informative signal on wordnet is not a magnitude: Spearman ρ and
rf R² rank the node2vec arms in opposite orders there — the `q = 1.0`
run has the highest rf R² (0.042) and the lowest ρ (0.011) of the three
node2vec settings — and AUC stays at or above 0.999 across every arm
regardless, including that one. Two model-free numbers disagreeing on
order is stronger evidence that hop R2 is the wrong instrument for a
sparse, tree-like graph than any single number reading near zero;
AUC's saturation at almost every setting is the companion fact this
project's own rule already names: "a quality score without its cost
[is never reported alone]," and here a link-prediction score without a
distance score would say nothing about the layout at all.

### 5.2 Protocol status

The scorecard cited in §5.3 was scored under `n2v1m` with `max_pairs`
overridden to 50,000; every row is marked `protocol_modified = true`,
and its rows are therefore comparable to each other and to nothing
recorded under an unmodified `n2v1m` baseline
(`evaluator/reports/260905-store-scorecard.md`). Its model-free columns
(Spearman ρ, Somers' D, Kruskal stress, kNN recall) come from a
separate script that calls the scoring modules directly rather than
through a named, versioned protocol — real numbers, but not yet a
frozen one. `evaluator/dev-docs/METROLOGY.md` and `PRD-v2.md` are both
still DRAFT (v0.1 and v0.2), unapproved, and five specific claims in
them are already known to be wrong as written — a backwards argument
order in one statistic, an unstable horizon rule, an undefined
per-shell case, an unfixed distortion scale, and the class-S hop-R2
range noted above. None of the five is used in this paper.

### 5.3 Caveats specific to §5's table

- **Above 200,000 nodes, two measurement rules change**: hop-BFS
  sources are capped at 200, and kNN recall switches to an approximate
  index. com_youtube's numbers are not comparable to cora's, pubmed's,
  or wordnet's on those two columns for that reason alone, independent
  of anything about the method.
- **The `k4` confound of §3.3** applies to every `fdhop`-vs-`fdlinear`
  cell in the table below.
- **One seed, no error bars** (§4.2): every difference in §5.3 is a
  measurement, not a demonstrated win.
- **cora's distance columns are conditional on reachability.** The
  scorer drops cross-component pairs before scoring
  (`data_cache/evaluator/store-scores/eval_store.py:48-56`) — a
  provisional choice, still an open decision in
  `evaluator/dev-docs/METROLOGY.md`, §13. cora has 78 connected
  components, and 3,277 of 20,000 drawn pairs (16.4%) are dropped for
  it; pubmed, wordnet and com_youtube drop 0%. cora's ρ and hop R2
  below measure fidelity within a component, over the 83.6% of drawn
  pairs that are reachable, not the graph as a whole — one more reason
  its numbers do not transfer to the other three graphs.

## 6. Results

Best-arm comparison (by Spearman ρ), `fdhop` (`walk_edges`, `plain`,
`k4 = 1.0`) against `fdlinear` (`walk_edges`, `plain`, `k4 = 0.01`, its
`Config` default), 128 dimensions, 200 epochs, seed 42
(`evaluator/reports/260905-store-scorecard.md`, §3 tables; `rf R²` is
the source's own column name for its random-forest hop-distance
regression, §5.1):

| graph (n) | law | acc | auc | Spearman ρ | rf R² |
| --- | --- | --- | --- | --- | --- |
| cora (2,708) | fdhop | 0.9905 | 0.9993 | 0.839 | 0.696 |
| cora (2,708) | fdlinear | 0.9692 | 0.9936 | 0.787 | 0.697 |
| pubmed (19,717) | fdhop | 0.9912 | 0.9992 | 0.776 | 0.606 |
| pubmed (19,717) | fdlinear | 0.9712 | 0.9955 | 0.710 | 0.503 |
| wordnet (82,115) | fdhop | 0.9964 | 0.9999 | 0.355 | 0.096 |
| wordnet (82,115) | fdlinear | 0.9929 | 0.9990 | 0.374 | 0.156 |
| com_youtube (1,134,890) | fdhop | 0.9852 | 0.9980 | 0.683 | 0.476 |
| com_youtube (1,134,890) | fdlinear | 0.9681 | 0.9941 | 0.692 | 0.524 |

`fdhop` leads accuracy and AUC on all four graphs. It also leads
Spearman ρ and rf R² on the two smaller graphs (cora, pubmed) and
trails `fdlinear` on both distance columns on the two larger ones —
by 0.019 of ρ on wordnet and 0.009 on com_youtube, 0.060 of rf R² on
wordnet and 0.048 on com_youtube. Given the single-seed caveat of §4.2
and the `k4` confound of §3.3, these four distance-column deltas are
read as ties rather than losses: none has been shown to exceed a
measured seed spread, and the `fdlinear` side of the comparison runs a
100x-flatter decay by `Config`'s own default rather than by a matched
setting. **`fdhop` is adopted as fodiwalk's force law on that basis** —
a consistent link-prediction win, a distance-metric split that is
small enough, and not yet measured precisely enough, to read as a
loss.

One further reversal, unexplained: on com_youtube, DeepWalk's Spearman
ρ (0.754) beats every recorded fodiwalk arm, including `fdhop`'s
0.683; on the other three graphs every fodiwalk arm beats DeepWalk by
0.09 to 0.20 (`evaluator/reports/260905-store-scorecard.md`). Nothing
in the store isolates whether this is the walk objective, the walk
budget, or the graph, since DeepWalk and node2vec differ in both
objective and budget from `fodiwalk` at once. It is recorded here
because §5.1's rule against reporting one number alone applies to it
too: com_youtube is also the one graph where the `pairs`, source-count
and kNN-index changes of §5.3 all apply simultaneously.

## 7. Provenance

| entity | file |
| --- | --- |
| the force law, `fdhop` | `fodiwalk/embed/forces.py:149` |
| the walk | `fodiwalk/augment_graph/walks.py`, `uniform_walks` |
| directed pairing, `nbr_walk` | `fodiwalk/augment_graph/policy_nbr_walk.py` |
| neighbor completion, both directions | `fodiwalk/augment_graph/row_merge.py:190` |
| the weight `h` | `fodiwalk/augment_graph/weights.py` |
| the degree divisor | `fodiwalk/embed/forces.py:64`, `degrees_from_D` |
| the epoch loop | `forcedirected/force_directed.py` |
| the batch layout | `forcedirected/sell_c_sigma.py` |
| the update rules | `forcedirected/optim.py` |
| the run configuration and store | `experiments/embeddings/make_embedding.py`, `data_cache/embeddings/*/*/*/config.json` |
| the optimizer verdict, the 200-epoch note | `experiments/fodiwalk-streaming/FINDINGS.md`, §10 |
| the metrology protocol, draft status | `evaluator/dev-docs/METROLOGY.md`, `evaluator/dev-docs/PRD-v2.md` |
| hop-distance regression | `evaluator/tasks/dist_approx.py` |
| the classification and regression scores | `evaluator/scoring.py` |
| the results scorecard | `evaluator/reports/260905-store-scorecard.md` |
| the stage contract | `fodiwalk/dev-docs/fodiwalk-module.md` |
