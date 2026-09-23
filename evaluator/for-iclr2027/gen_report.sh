#!/bin/bash
# Rebuild evaluator/reports/260917-iclr2027-preliminary.md from cells.csv.
# Run summarize.py first; every number here comes from the tables, not by hand.
cd /home/h/gnn/fd-graph-embedding/fdmap
# S is where these scripts live (in git). D is where the data lives
# (data_cache, which git ignores). Keeping them apart lets the committed
# copy run without calling the ignored one.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
D=data_cache/evaluator/for-iclr2027
T=".venv/bin/python $S/report_tables.py"
R=evaluator/reports/260917-iclr2027-preliminary.md
MADE=$(ls -d $D/../../../experiments/for-iclr2027/embeddings/*/*/*/ | wc -l)
# Count what the TABLES hold, not what is on disk: a run scored after the
# last summarize pass is not in cells.csv and must not be claimed here.
SCORED=$(.venv/bin/python -c "import csv;print(sum(int(r['seeds']) for r in csv.DictReader(open('$D/cells.csv'))))")
STAMP=$(date -u +%Y-%m-%dT%H:%MZ)
{
cat <<'HEAD'
# ICLR 2027 runs: preliminary results

**Built:** __STAMP__. **Status:** preliminary. Runner has made __MADE__ run
folders; __SCORED__ of them are in the tables below. The count passed the
2,277 of the original plan because the owner added cells after it was
written: the matched-budget baselines of section 1.1, the wordnet
dimensions, and the deferred large-graph q values. The planned total is
stale, not overrun. __SCORED__ is the number of runs inside `cells.csv`,
which is what every table aggregates; runs scored after the last pass are
on disk but not counted here. cora and pubmed are complete
at all four dimensions for the main cells, and so is wordnet. The large
graphs have started: com_youtube and roadnet_ca runs exist but are NOT yet
scored, because the scorer keeps a large graph out of memory while Runner
times a large embedding.

**Store:** `experiments/for-iclr2027/embeddings/<graph>/<dim>/<run>/`. Each
run holds `Z.npy`, `config.json` and `evaluation.json`. Cell table:
`data_cache/evaluator/for-iclr2027/cells.csv`. This file is generated:
`data_cache/evaluator/for-iclr2027/gen_report.sh`.

## How to read these numbers

- Every value is **mean +/- std over seeds**. The seed is the EMBEDDING
  seed. The evaluation seed is fixed at 42, so every run of one graph is
  scored on the same link-prediction split, the same hop pairs and the same
  query nodes. The spread here is method spread, not sample spread.
- The owner's rule is 11 seeds for a small or medium graph. A cell that
  shows fewer seeds is still running.
- Protocol `n2v1m`, lp_max_pairs 50,000. Comparable only inside this report.
- `rho` is Spearman between embedding distance and hop distance, on pairs
  with hop >= 2. `hop R2` is the random forest on one distance column.
  `recall@10` is the share of a node's true neighbours among its 10 nearest
  nodes. It is NOT a held-out Hits@10.
- **No verdict word appears in this report, and none is available.**
  `rho` has no measured noise floor: `evaluator/config.py:605` sets it to
  `_UNSET`, and only auc, accuracy, f1_score, r2 and mae carry one. Under
  PRD-v2 5.6 a metric without a floor gives a MEASUREMENT, never a win, at
  any size and any seed count. So 0.231 on pubmed and 0.002 on some other
  row are in the SAME verdict class. Read every number here as "measures
  higher", not "wins" (Metrologist, 2026-09-21).
- **"gap / seed spread" is NOT a test statistic.** The evaluation seed is
  fixed at 42, so that ratio is built from spread over embedding seeds
  only. It says nothing about whether a result survives a different split,
  and it is not a p-value. The honest form is a paired interval or a TOST
  equivalence test over VARIED evaluation seeds against a margin fixed in
  advance; this project has not run one.
- **How many seeds a number needs.** The owner's rule is 11 embedding
  seeds for a small or medium graph, and 3 for a very large graph. cora,
  pubmed and wordnet are medium, so 11 applies to all three.
  A cell below that count is still a measurement, and it may be quoted.
  Quote it only with its seed count and its spread beside it. It can show a
  direction, and it can rule a direction out. It cannot settle an order
  between two cells.
  Section 7 holds 3 seeds in most of its cells. Read it as a map of
  directions. Two cells in it are apart only when the gap is much larger
  than both spreads, and the metric there is rho, which has no measured
  floor. So even a large gap in that table is a measurement, not a win.
  (Wording: Metrologist, 2026-09-23.)
- **LP AUC is saturated.** It sits between 0.971 and 0.9996 over every cell
  in this report, while rho spans 0.27 to 0.87. Rank the methods on
  geometry, not on AUC.

## 1. Main comparison, dim 128

HEAD
$T main
cat <<'T2'
fdhop with the walk_edges policy leads on cora and pubmed, on both geometry
measures. It measures 0.101 higher than deepwalk on cora and 0.115 higher
on pubmed, which is 15 and 33 times the seed spread of the difference. node2vec is monotone in q,
favouring q=0.5, as measured before.

**wordnet breaks this order.** At 11 seeds, fdlinear reaches
rho 0.365 +/- 0.007 against fdhop at 0.348 +/- 0.010. That gap is 1.5
times the seed spread of the difference, which does not separate them. Every
method is weak on wordnet: deepwalk reaches 0.163 and node2vec q=0.5 only
0.023, while all of them keep an AUC above 0.999. The collapse of the walk
baselines is far outside the seed spread and holds now. The best wordnet
cell of all is fdhop with the FLAT weight, at 0.393 +/- 0.010; see the next
section.

`fdhop nbr_walk` is the exception: it has the best recall@10 and the best
AUC of all, and the worst rho of the fodiwalk cells. It puts neighbours
close and loses the longer distances.

## 1.1 The baselines at a matched walk budget

Runner ran deepwalk and node2vec q=1 over the same seven budgets at window
5. At q=1 both use the SAME first-order walk generator, so a row here
differs only in the output layer: deepwalk uses hierarchical softmax,
node2vec uses negative sampling with 5 negatives.

__BUDGET__

**The budget is not the reason node2vec scores lower.** deepwalk wins every
row, by 0.21 to 0.34 rho on cora and by 0.21 to 0.66 on pubmed, and the gap
does not close as the budget grows. On pubmed node2vec stays at or below
zero rho until 40x20. The cause is the output layer, not the walk count.
An earlier note of mine said the budget was the likely cause; this table
replaces it.

Both also gain from a wider window: node2vec q=1 at 10x80 reaches 0.699
with window 10 against 0.491 with window 5 on cora. The window is the
larger lever for node2vec than the walk count.

The headline table above keeps each baseline at its own published budget,
deepwalk 80x40 window 10 and node2vec 10x80 window 10.

## 2. Dimension (Spearman rho)

T2
$T dims
cat <<'T3'
A seed count in brackets marks a cell that is still running.

On cora and pubmed, fdhop walk_edges rises with the dimension and is best
at 128. It still falls below deepwalk at dim 16 on cora (0.767 against
0.785). The walk baselines barely move from 16 to 128 on those two graphs.

**On wordnet the two families move in opposite directions.** fdhop rises
with the dimension, as on the other graphs (0.239 at 16, 0.348 at 128).
deepwalk instead falls hard, and only here (0.462 at 16, 0.163 at 128). So
at dim 16 deepwalk beats fdhop on wordnet by 0.223, and at dim 128 fdhop
leads by 0.185. A one-dimension comparison on a
tree-like graph picks its own winner. The paper must state the dimension
with every wordnet claim.

## 3. Force law, dim 128

T3
$T laws
cat <<'T4'
`fdhop_min` and `fdhop_all_freq` are clearly worse on geometry. Note
`fdhop_all_freq`: recall@10 0.996 and f1_score 0.9964 on cora, the best of
any cell, with rho 0.474. Local structure and global geometry come apart.

## 4. Pair support and walk budget, cora dim 128

T4
$T support
cat <<'T5'
Two results here:

- **The weight rule matters, the walk policy does not.** On cora, min_gap
  beats flat by 0.187 rho. walk_edges, walk and walk_edges_pq are within
  0.002 of each other.
- **The walk budget does nothing.** From 5x20 to 40x20 walks, and from
  10x10 to 10x80 length, rho stays at 0.842-0.844. The spread over the
  whole budget sweep is smaller than the seed spread of one cell.

### 4.1 The weight rule reverses on wordnet

Spearman rho, `fdhop walk_edges 10x20 k4=1 kr=1`, dim 128. The seed count
is in brackets.

__WEIGHTS__

This is the paper's central claim under test, and the answer is not the
same on every graph. On cora and pubmed the observed walk gap is worth
0.187 and 0.231 rho over a flat weight, at 26 and 62 times the seed spread
of the difference. On wordnet the flat weight measures 0.045 HIGHER, at 3.3
times that spread.

So walk-gap weighting is not a general improvement. It is large on the two
citation graphs and it is negative on the hierarchy. Do not write the
mechanism claim without this row. wordnet is a tree-like graph, which is
the regime the paper plan already names as a risk.

## 5. Force parameters, dim 128

T5
$T force_params
cat <<'T6'
k4=1 and kr=1 are near the top on both graphs. kr=10 is a failure
(rho 0.272 on cora). kr=2 buys recall@10 (0.991) and costs rho.

The fdlinear runs at k4=0.01 have a mean row norm near 234 on cora against
5.6 for fdhop at k4=1. This is scale, not instability: weak decay spreads
the layout. Do not read it as a blow-up.

## 6. Optimizer study, dim 64

T6
$T optim
cat <<'T7'
sqn at lr 0.999 and adam with linear decay lead on cora. On pubmed the top
five rules are within 0.003 rho, which is under one standard deviation:
that group is not separated by these data.

## 7. Learning-rate map, dim 64 (Spearman rho +/- std, seeds in brackets,
const decay)

T7
$T lrmap
cat <<'T8'
**Most cells here hold 3 seeds, not 11.** momentum at lr 0.099 has all
eleven; the rest of that rule has seeds 7, 42 and 56 only. Read every
difference in this map as a measurement on three runs. plain 0.831 against
velocity 0.836 is not a separation.

"x" marks a cell whose median row norm is more than 10x the reference, the
same graph, dim and force setting under plain / const / lr 0.999. A ratio
is printed wherever it reaches 2x, so any other threshold can be applied
without asking me. Ratios below 2x exist and are simply not printed: a
blank is not a missing measurement.

The threshold is FROZEN at 10x as of 2026-09-21 and will not be moved.
Three things about it, all needed to read the marks honestly:
- **Unit: per cell, not per seed.** The ratio is the cell's median row norm
  over its seeds. A cell can therefore sit under the line while one of its
  seeds sits over it; where that happens it is named under the table.
- **It was set on cora, where it is safe.** At dim 64 the largest healthy
  ratio there is 2.08x and the smallest blown one 34.6x, so any cut from 3x
  to 30x gives the same answer. **Do not call the split
  threshold-insensitive in general.** On pubmed it is not: sqn at lr 0.02
  reaches 18.9x on seed 56 and momentum at lr 0.5 reaches 26 to 30x, so 10x
  and 20x disagree there.
- **It is frozen rather than tuned because the pubmed norms were read on
  2026-09-17, before any cut-off existed.** Nobody can now pick one blind
  on that graph, so picking it at all would be a data-driven choice.
  Measurements and ruling: Metrologist, 2026-09-21.

The mark is a LABEL, not a filter. Every marked cell keeps its row and its
rho here, and no number anywhere in this report drops a marked run. If one
is ever excluded, the threshold becomes a data-selection knob and needs
pre-registration and the effect shown both ways.

Under it, momentum is marked at lr 0.5 and 0.999, and nesterov at lr 0.999.
No run produced non-finite values, so `diverged.keys` is empty and the word
"diverged" is not used for any of these.

sqn at low lr is a different failure: rho 0.096 with a normal norm. It did
not blow up; it failed to learn. Keep the two apart.

## 8. Defects found, and what was done

1. **nbr_walk ignores the weight rule.** `policy_nbr_walk.py:119` sets
   `h = stats.mn` and never calls `weights.RULES`; only
   `policy_walk.py:144` does. So every nbr_walk run used min_gap whatever
   its `config.json` recorded. Proof: on cora dim 128 the sha256 of `Z.npy`
   was identical across flat, min_gap, mean_gap and pmi at seeds 42, 7, 56
   and 88, while walk_edges differs between flat and min_gap. Runner
   reproduced both halves.
   **Done:** the 30 runs with a wrong weight record were deleted, listed in
   `experiments/for-iclr2027/removed-nbr-walk-weights.tsv`. Each was a
   byte-copy of its min_gap twin, so no embedding was lost. The schedule
   now has ONE nbr_walk cell. Every nbr_walk run left in the store records
   min_gap. Verified here after the deletion. The fodiwalk fix itself is with that
   package's owner; a fix changes every nbr_walk Z and needs a rerun.
2. **mean_gap and pmi are not usable with walk_edges.** All 16 such runs
   failed with PlaneContractError I5 (a real edge gets h > 1, so a row gets
   degree 0). They have no folder. In a table they are a CONTRACT FAILURE,
   not a divergence. With the nbr_walk copies deleted, **no mean_gap or pmi
   result exists at all.** The weight-rule comparison is min_gap against
   flat only.
3. **Single-protocol numbers.** Nothing here is a held-out link prediction.
   The embeddings saw every edge, so the LP panel is edge reconstruction.
4. **Not a convergence claim.** These are 200-epoch runs. The checkpoint
   blocks at epochs 50/100/150 exist for the optimizer study only.

## 8.1 Note for the cost and scale tables (section 5.6)

Every large-graph runtime and peak-memory figure was measured while one
nice-19 scoring thread could be running beside it, on 8 cores. The effect
is small but it must be stated with the times. No large graph is ever
loaded by the scorer while a large embedding is in flight: the machine has
15 GB, the desktop holds about 10 GB, and a com_youtube fodiwalk run peaks
at 5.3 GB.

## 9. What is missing

- A cell with fewer than 11 seeds is still running. Its real count is in
  every table.
- No large-graph result yet. com_youtube and roadnet_ca runs exist but are
  unscored; ncbi_taxonomy and as_skitter have not started. So there is no
  large-graph number and no cost or memory number in this report.
- Force2Vec and rForce2Vec, the closest competitors in the paper plan, are
  not in this run set.
T8
} > $R
.venv/bin/python $S/report_tables.py budget > /tmp/b.$$ &&
  sed -i -e "/__BUDGET__/r /tmp/b.$$" -e "/__BUDGET__/d" $R && rm -f /tmp/b.$$
.venv/bin/python $S/report_tables.py weights > /tmp/w.$$ &&
  sed -i -e "/__WEIGHTS__/r /tmp/w.$$" -e "/__WEIGHTS__/d" $R && rm -f /tmp/w.$$
sed -i "s/__STAMP__/$STAMP/; s/__MADE__/$MADE/; s/__SCORED__/$SCORED/" $R
echo "wrote $R ($MADE made, $SCORED scored)"
