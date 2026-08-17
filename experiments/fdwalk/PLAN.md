# fdwalk: random walks as the source of the augmented graph

Created: 2026-08-15 (time not recorded; the stamps below start when the
practice of stamping did)
Last updated: 2026-08-16 23:25 PDT

Status: design. No experiment ran yet.

> **UPDATE 2026-08-16 23:25 PDT.** The status line above is kept, and it is
> out of date. Reason for the update: the rule of this project is that a
> line is not deleted, thus a reader sees what the document said when the
> work started. The state now: G1 and G2 are closed and G3 failed. See
> FINDINGS.md, and the `Status` table at its top.
>
> Sections added after the first write carry their own stamp. A section
> with no stamp belongs to the first write of 2026-08-15.

## 1. The problem

The augmentation of fodined gives the matrix `D`. A stored entry
`D[u, v] = h` is a distance, and the force law reads that `h`. Two policies
exist today, and both fail on a large graph:

| Policy | How it makes the pairs | Where it fails |
| --- | --- | --- |
| `sampled_pairs` | random pairs, then one BFS for each source | 114 hours at 1.13M nodes; the Python sets need 3 GB |
| `k_hop` | the ball of k hops around each node | the BALL is too large: 2.5 G pairs at k=2 on com_youtube, 53 G at k=3 |

The two failures have one cause: the size of `D` follows a property of the
GRAPH, and not a budget that we choose. A hub with 28,754 neighbours puts
2,200 pairs in the 2-hop ball of an average node. We cannot control that
number, thus we cannot control the memory.

Reference for the measurements: `../large-graph-node2vec/REPORT.md` and the
probe in section 9 of this document.

## 2. The idea

node2vec and DeepWalk run on a graph of a million nodes on this machine.
They finish in 12.7 minutes [3, 4, and `../large-graph-node2vec/`]. They
scale because a random walk is LOCAL and because its cost is a budget:
`r` walks of `l` steps from each node, thus `n * r * l` steps, whatever the
degree of a hub is.

fdwalk takes the same generator, but it uses the walks for a different
purpose. node2vec sends the walks to word2vec. fdwalk sends the walks to the
augmentation:

1. Two nodes that appear in the same walk, inside a window of `w` steps,
   become a pair of `D`.
2. A property of the walk gives the WEIGHT of that pair.
3. Random far pairs get a large weight, as they do today.
4. A cap of `M` pairs for each node bounds `D.nnz` at `2 * n * M`.

Step 4 is the point of the whole design. The size of `D` becomes a number
that we choose, and it does not follow the degree of a hub.

The literature supports step 2. Levy and Goldberg showed that word2vec with
negative sampling factors a shifted PMI matrix [5]. Qiu et al. showed that
DeepWalk and node2vec factor a matrix of walk co-occurrence [6]. Thus the
co-occurrence counts of a walk already hold the information that node2vec
uses. fdwalk gives those counts to a force law instead of to a
factorization. LargeVis and UMAP make the same move for a different reason:
they build a sparse neighbour graph, and then they apply attraction to the
neighbours and repulsion to random pairs [8, 9].

## 3. Hypotheses

Each hypothesis has a prediction, and a measurement that can refute it.

**H1 (scale).** A walk budget bounds the augmentation. The time and the
memory of the augmentation grow with `n * r * l` and with `n * M`, and not
with the maximum degree.
*Prediction:* the augmentation of com_youtube (1.13M nodes, max degree
28,754) finishes in less than 3 minutes and less than 1.5 GB of RSS.
*Refuted if:* the time or the memory of com_youtube divided by the same
number of PubMed is more than 3 times the ratio of `n`.

**H2 (the min gap is a distance).** In a walk, the number of steps between
two nodes is an upper bound of their hop distance. The MINIMUM of that
number over many walks converges to the true hop distance.
*Prediction:* for pairs within 3 hops, the min gap equals the true hop
distance for more than 80% of the pairs, at `r = 10`, `l = 20`.
*Refuted if:* less than 60%, or if the error grows with the degree of the
node in a way that a larger `r` does not repair.

**H3 (the pair set matters more than the weight).** A force-directed layout
mostly needs to know WHICH pairs attract. The exact value of `h` is a second
order effect.
*Prediction:* the control with all weights at 1.0 reaches an AUC within 0.01
of the best weight variant, and a clearly worse hop R2.
*Refuted if:* the flat weight loses more than 0.02 of AUC. That result would
mean the opposite: the weight carries the signal.

**H4 (walks beat the ball as a selector).** At the same budget `M`, the pairs
of a walk are more useful than the `M` nearest pairs of the k-hop ball,
because a walk visits a node in proportion to its relevance and it crosses a
hub only sometimes.
*Prediction:* fdwalk with `M = 16` beats the capped ball with `M = 16` on
Cora, on both the AUC and the hop R2.
*Refuted if:* the capped ball wins both. Then the correct fix for the ball is
a cap, and not a walk, and that is a much smaller change.

**H5 (momentum helps, Adam does not).** The step of fodined is a physical
force, and its size has a meaning. Momentum keeps that meaning and it only
adds inertia [13]. Adam divides by the size of the gradient of each
coordinate [11], thus it removes the meaning of a strong force.
*Prediction:* momentum reaches the same `||dZ||` in fewer epochs than the
plain step; Adam converges fast at the start and it gives an equal or worse
AUC.
*Refuted if:* Adam wins on both the epochs and the AUC. That result is
interesting and we then follow it.

**H6 (the memory of the optimizer is the limit at 1M).** Momentum holds one
state array of the size of `Z`. Adam holds two.
*Prediction:* at 1.13M nodes and 128 dimensions, `Z` and `dZ` are 1.16 GB
together. Adam adds 1.16 GB more, thus it cannot run on a 2 GB card. This is
arithmetic, and the experiment only confirms it.
*Action if true:* Adam runs at 1M only at a smaller `n_dim`, and we report
that as a deviation.

**H7 (the far weight is not sensitive).** The weight of a far pair is a
constant that separates "far" from "near". Its exact value does not change
the result much, inside one order of magnitude.
*Prediction:* the AUC on Cora moves less than 0.01 between a far weight of
10, 100, and 1000.
*Refuted if:* it moves more. Then the far weight is a hyperparameter, and
every other comparison must hold it constant, or tune it.

## 4. The variants

Three axes. The experiment does NOT take the full product of them: axis A
and axis B first, at the default optimizer, and then axis C on the winner
only.

### Axis A -- how a pair gets its weight

| Id | Weight `h` of a pair | Note |
| --- | --- | --- |
| A0 | 1.0 for every pair | the control of H3 |
| A1 | the MINIMUM step gap in a walk | an upper bound of the hop distance, H2 |
| A2 | the MEAN step gap | smoother, and it uses every visit |
| A3 | from the co-occurrence count: `h = 1 + a * log(c_max / c_uv)` | the PMI form [5, 6], mapped so that "more visits" means "nearer" |

`h` stays in `[1, w]` for A1 and A2 by construction. A3 needs the scale `a`,
and we fix it so that the maximum `h` is `w`, thus the four variants have
the same range and the force law sees the same numbers.

### Axis B -- where the pairs come from

| Id | Pair source |
| --- | --- |
| B0 | the k-hop ball, capped at `M` for each node (the control of H4) |
| B1 | the walks, capped at `M` for each node, by the count |
| B2 | the walks, plus the first-order edges always kept |

B2 exists because a walk can miss an edge of a low-degree node. The original
edges are cheap to add, and they are the pairs that we trust most.

Every variant adds far pairs: `F * n` random pairs at the weight
`FAR_WEIGHT = 100`, which is the value that the current policy uses. The far
pairs come from `sample_far_pairs`, which rejects any pair that the walk
matrix holds.

### Axis D -- how many pairs exist at one time

Added 2026-08-15 22:50 PDT, after G3 failed. Every variant above builds the WHOLE `D`
and one plan, thus the memory holds every pair at the same time. Axis D
removes that.

| Id | Form | What it bounds |
| --- | --- | --- |
| D0 | one `D`, one plan (today) | nothing |
| D1 | `D` in `P` chunks of pairs; one plan for each chunk; `dZ` sums over the chunks | the plan and the device memory |
| D2 | the pairs of each epoch come from fresh walks, and no `D` exists | the plan, the device, AND the pair accumulator |
| D3 | D1 for the near pairs, and fresh negative pairs for each epoch, as LargeVis and UMAP do [8, 9] | the same as D2, with a stable near set |

The force law permits D1: `Fa` and `Fr` are SUMS over the stored pairs of a
row, thus a chunk gives a part of the sum and the parts add.

One quantity does not chunk, and it must be computed one time from the
whole pair set: the shell coefficient `1 / |S_h(u)|`, and the force-law
degree, which is the count of the hop-1 entries of a row. Both come from a
table of `n` rows and `window` columns, which is 45 MB at 1.13M nodes and 5
columns. Thus the table is global and cheap, and only the PAIRS chunk.

The risk of D2 sits in G1 finding 1. The weight 1 must be the adjacency. A
fresh sample gives the gap of THAT sample, and not the minimum over all the
walks, thus a pair can arrive at the gap 3 in one epoch and at the gap 1 in
another. `mean_gap` collapsed for a version of this defect. D2 must
therefore measure the count of the weight-1 entries in each batch, and stop
if that count falls.

### Axis E -- a matrix `D` that is not symmetric

Added 2026-08-16 02:30 PDT, from a question of the user. NOT measured yet. Three
different changes carry this name, and they have different payoffs. They
must not be mixed in one run.

**(A) Storage asymmetry, and symmetric physics.** Store the upper triangle
only, and let the kernel scatter `+F` to `u` and `-F` to `v`. It halves the
cells of the plan -- about 10M at 1.13M nodes, which is the largest item
after `Z`. It changes NO semantics. It needs a change of the kernel, which
today writes one endpoint (`dZ.at[rows].add`). One question must be
answered first: a shared force needs one normalisation, and the two rows
divide by different degrees today, thus "whose degree" has no answer yet.

**(B) Structural asymmetry from the cap.** Today `cap_per_node` keeps a
pair if EITHER node wants it, as LargeVis and UMAP do [8, 9], thus a row
can hold more than `m` pairs and a hub collects many. If instead row `u`
holds exactly its own best `m`, then EVERY row has the same width. For
SELL-C-sigma that is a large win: no hub split (`n_split` -> 0), little
padding (`pad_frac` is 0.16 today), balanced batches, and an exact bound of
`n * m` cells. It also bounds the row of a hub. This is the variant with
the best ratio of payoff to work.

**(C) Directed semantics.** Cora and PubMed ARE citation graphs, and the
loader symmetrises them. Keeping the direction makes the hop distance
itself asymmetric, thus it changes the task and not only the engine. It is
an experiment about the DATA, and it must not ride with (A) or (B).

A note on the objection that (A), (B), and (C) break Newton's third law:
**the third law is already broken.** `shell_coeff = 1/|S_h(u)|` is a
per-ROW normalisation, thus the force from `v` on `u` is `k1*x/deg(u)^2`
and the force from `u` on `v` is `k1*x/deg(v)^2`. For an edge between a hub
and a leaf those differ by six orders of magnitude. Only the SPARSITY
PATTERN of `D` is symmetric today, and the dynamics are not.

A second note, on the argument from `../drop_strategies/REPORT.md`: that
experiment drops half of the rows of each epoch and it loses no AUC, thus
the system tolerates a non-reciprocal UPDATE. That is evidence, and it is
not proof, because the drop is RANDOM and unbiased over the epochs, thus
every row is updated in expectation. A `D` that is not symmetric omits the
same direction in EVERY epoch. Tolerance of a stochastic asymmetry does not
give tolerance of a systematic one.

The user notes that a cycle of updates may then not converge. If that
happens, the answer is a detector for the cycle, or a learning rate that
decreases. This is examined only if the system fails to converge.

### Axis C -- the update rule

| Id | Rule | State arrays |
| --- | --- | --- |
| C0 | `Z = Z + lr * dZ` with `drop_steady_rate` (today) | 0 |
| C1 | heavy-ball momentum, `beta = 0.9` [13] | 1 |
| C2 | Nesterov momentum [13] | 1 |
| C3 | Adam, `b1 = 0.9`, `b2 = 0.999` [11] | 2 |
| C4 | the local speed of ForceAtlas2: a per-node step from the "swinging" of that node [10] | 1 |

C4 is the physical answer to the same problem that Adam solves. It belongs
in the comparison because it comes from the force-directed literature, and
not from deep learning.

## 5. What we measure

The same harness for every variant. It already exists, and we reuse it:

| Measurement | How | Source |
| --- | --- | --- |
| Link prediction | 50,000 capped pairs, random forest, accuracy, precision, recall, F1, AUC | `fodined/link_prediction.py` |
| Hop distance | exact BFS from 200 random sources, 20,000 pairs, MAE, MRE, RMSE, R2, exact % | `../other-ge/bench_other_ge.py` |
| Speed | seconds for each stage: walks, pairs, augmentation, embedding | the log |
| Memory | peak RSS (`VmHWM`) and peak VM (`VmPeak`) | `../modular-graphs/run_guarded.sh` |
| Stability | `||dZ||` at the end, and `isfinite(Z)` | the log |
| Size | `D.nnz`, and the pad fraction of the plan | the log |

**The rule of 2%** (decision of the user, added 2026-08-16 00:20 PDT). A gain of 2% or
less on ANY metric is noise, and it is not a result. It is a floor, and not
a test: a difference above 2% is still noise if it is smaller than the
spread over the seeds. Thus a result must pass BOTH:

    (new - old) / old > 0.02      AND      (new - old) > std over the seeds

This rule closes an argument before it starts. A method that gives 0.9970
against 0.9967 does not "win", and a method that gives 0.664 against 0.655
does not "win" either.

Rules of the measurement:

* The exact hop distances come from a SAMPLE of sources. A full matrix is
  `n * n` and it never fits. `../bench_shortest_path_gemsec.py` measured the
  cost of the sampled form.
* Seeds 42, 56, and 88 on Cora and on PubMed. We report the mean and the
  standard deviation. A difference smaller than the standard deviation over
  the seeds is not a result.
* One seed only at 1M nodes, because of the cost. A 1M result is therefore a
  demonstration of the scale, and not a comparison of quality.
* Every variant uses the same `N_DIM`, `EPOCHS`, and force constants
  `K1..K4` as `fodined/modular.py` today, except where a gate says
  otherwise. A deviation goes into the log AND into FINDINGS.md.

## 6. The baselines to beat

These numbers are measured, and they are in the logs of this repository.

| Graph | Method | AUC | hop R2 | Time | Peak RSS |
| --- | --- | --- | --- | --- | --- |
| Cora | fodined, `k_hop`, k=2 | 0.9953 | see note | 9.7 s | small |
| Cora | fodined, `k_hop`, k=3 | 0.9986 | 0.257 | 17.4 s | small |
| PubMed | fodined, `sampled_pairs` | to measure | 0.340 | to measure | to measure |
| com_youtube 1.13M | node2vec | 0.9984 | 0.043 | 764.6 s | 2212 MB |
| com_youtube 1.13M | fodined, both policies | did not run | did not run | died at 20 s | 2465 MB |
| roadNet-CA 200k | fodined, `k_hop`, k=2 | 1.0000 | degenerate | 209.7 s | 1867 MB |

Note: the hop R2 of a `k_hop` run at k=2 has ONE target value, thus that
number has no meaning. A bounded ball gives a bounded set of distances. The
walk gap of fdwalk gives `w` values, which repairs the measurement. This is
one more reason for the change.

The roadNet-CA AUC of 1.0000 is not a strong result. A road network has an
average degree of 2.75, thus link prediction on it is easy.

## 7. The gates

The work goes Cora, then PubMed, then 1M. A variant passes a gate, or it
stops. The numbers are fixed BEFORE the run.

**Gate G1, Cora (2,708 nodes).**
Pass if all of:
* AUC >= 0.99, and inside 0.01 of the best current policy (0.9986 at k=3).
* Hop R2 >= 0.25, thus not worse than the k=3 policy.
* The augmentation finishes in less than 5 s.
* `D.nnz` <= 400,000, thus not more than the current k=3 policy.

**Gate G2, PubMed (19,717 nodes).**
Pass if all of:
* AUC inside 0.01 of the same variant on Cora, thus the method does not
  degrade with the size.
* Hop R2 >= 0.30, thus at least the old `sampled_pairs` number.
* The augmentation finishes in less than 60 s and 1.5 GB.

**Gate G3, com_youtube (1,134,890 nodes).** This gate is about the scale.
Pass if all of:
* The whole script FINISHES. Nothing in this repository has done that yet.
* Total time <= 30 minutes, thus more than 2 times the node2vec time is a
  failure.
* Peak RSS <= 2.6 GB, which the guard enforces.
* AUC >= 0.95.
* Hop R2 > 0.043, thus better than node2vec on the same walks. This is the
  scientific claim of the whole idea: the same walk information, read by a
  force law, holds more of the global structure.

A pre-registered deviation for G3: at 128 dimensions `Z` and `dZ` are
1.16 GB, thus a variant with an optimizer state may need `N_DIM = 64`. That
is a permitted deviation, and it goes into the report. A change of the seed,
of the metric, or of the gate itself is NOT permitted.

## 8. When to stop a path

* A variant that fails G1 stops. We do not tune it more than 3 times. The 3
  attempts and their numbers go into FINDINGS.md, whatever the result is.
* A variant that fails G2 after passing G1 stops, and the report says which
  property of the larger graph broke it.
* Two variants that differ by less than the standard deviation over the
  seeds are ONE variant. We keep the cheaper one and we say so.
* An axis where every variant ties (H3, if true) closes. We then fix that
  axis at its cheapest value and we do not measure it again.
* If A0, the flat weight, passes every gate, the weight work stops. That
  result would say that the walk gives the pairs and that the force law does
  the rest.
* If B0, the capped ball, wins G1, the walk work stops. The fix is then a
  cap on the ball, which is a much smaller change than a walk sampler.
* A crash from the memory is not a failure of the idea. It is a failure of
  the budget, and the budget (`M`, `r`, `l`, `N_DIM`) comes down one time.
  A second crash at the smaller budget stops the variant.

## 9. What we already know about the numbers

A probe measured the ball of the two large graphs from a sample of 2,000
rows (`scratchpad/probe_ball.py`, 2026-08-15):

| Graph | max degree | pairs per node, k=2 | k=3 |
| --- | --- | --- | --- |
| com_youtube 1.13M | 28,754 | 2,200 | 46,874 |
| roadNet-CA 1.97M | 12 | 9 | 17 |

The same walk budget on both graphs gives the same `n * M`. That contrast is
the strongest argument for the design, and the report must show it.

Predictions of the cost at 1.13M nodes, from measured numbers:

* The walks: our node2vec run made 5.67M walks of 20 steps in 27.9 s.
* The pairs: `n * r * (l - w) * w` at `r = 2, l = 10, w = 3` gives about
  47M raw pairs, and the top-`M` cut brings them to `2 * n * M` = 36M at
  `M = 16`.
* `D` at 36M entries: about 290 MB as CSR, about 340 MB as the padded plan.
* The embedding: the 200k run did 3.6M cells at 9.5 epochs/s. 36M cells thus
  give about 1 epoch/s, and 2000 epochs need 33 minutes. **Therefore G3 runs
  at `EPOCHS = 500`, or at `M = 8`.** This is a pre-registered deviation,
  and the report gives the epochs of every run.

## 10. The files

```
experiments/fdwalk/
  PLAN.md          this document
  HYPOTHESES.md    -> section 3 of this document is the source; no copy
  FINDINGS.md      the results, one section for each gate. Written as we go.
  REFERENCES.md    the citations, with a note on what each one gives us
  walks.py         the walk generator and the pair extraction (pure functions)
  weights.py       axis A: the four weight rules
  optim.py         axis C: the five update rules
  bench_fdwalk.py  the harness: it runs one variant on one graph, and it
                   prints the same lines as modular.py
  run.sh           the guard, the seeds, and the log names
  results_*.log    one log for each run, never edited by hand
```

The code follows the style of `fodined/graph_augmentation.py`: pure
functions, no printing inside them, and the random generator arrives as an
argument.

If a variant passes every gate, it moves into `fodined/` as a policy beside
`k_hop`, and `AUGMENT` selects it. Nothing moves into `fodined/` before
that.

## 11. Threats to the validity

1. **A hub is a partner of many rows.** The cap `M` bounds the pairs of a
   ROW. After the symmetrisation, a hub can hold many more. We must measure
   `D.nnz` after the symmetrisation, and cap again if it grows.
2. **The walk gap is biased high.** A short gap needs the walk to take the
   short path. Thus the min gap is an upper bound, and it is exact only
   often enough. H2 measures that directly, and the result decides whether
   A1 is honest.
3. **The AUC saturates.** Cora is at 0.9986 and roadNet-CA at 1.0000. A
   metric at its ceiling cannot separate two variants. The hop R2 is the
   metric that still moves, and the gates weight it accordingly.
4. **The target of the hop regression must have a range.** A policy that
   stores only 2 distinct distances makes that measurement meaningless, and
   we saw exactly that at k=2. Every fdwalk run must report the histogram of
   the weights of `D`, as `modular.py` already does.
5. **One machine.** 3 GB of free RAM and a 2 GB card. A negative result at
   1M is a result about THIS machine, and the report must say the size that
   the method needs, and not only that it did not fit.
