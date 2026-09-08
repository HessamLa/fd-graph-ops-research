# Stored embeddings: fodiwalk against node2vec, dim 128

Ran 2026-09-02, 16:24 to 16:34 UTC. 10 embeddings, 0 failures. Every one is
under `data_cache/embeddings/<graph>/128/`, with the `config.json` that
`data_cache/embeddings/README.md` defines.

Driver: `experiments/embeddings/make_embedding.py`. Matrix:
`experiments/embeddings/run_matrix.sh`. Attempt log:
`experiments/embeddings/runs.log`. Run output: `matrix.log`.

## What was picked, and on what evidence

**Force law: `fdlinear`.** There is no contest to run. `fdlinear_fused` is
the SAME law from one fused plane, and `fdwalk/CATALOG.md:173` records
every embedding metric as identical to `fdlinear`. `fdlinear_3plane` is the
deprecated form that bound a plane it never read.

**Optimisers: `plain` and `velocity`.** From `fodiwalk-streaming
/FINDINGS.md` section 10, the top three are `velocity` 0.5871, `nesterov`
0.5820, `plain` 0.5808. The spread across SEEDS alone is 0.039 to 0.046, so
those three are one group and not a ranking. `velocity` is the top score.
`plain` is taken over `nesterov` because it is tied, holds no state array,
and has zero divergences in 230+ recorded runs. Swapping `nesterov` in is a
one-word change to the matrix.

**Walk methods: `walk_edges` and `nbr_walk`.** `walk_edges` is the top
scorer of the only ranked comparison (`fdwalk/FINDINGS.md` gate G1, cora,
three seeds): AUC 0.9972, R2 0.658. `walk` is second at 0.9970 / 0.655 --
inside the error bars, so choosing between the two adds nothing.
`nbr_walk` is the second arm because it is the policy of record for all
work since 2026-08-28 and it uses about 20x fewer pairs. **`nbr_walk` has
no score in that ranked table**, so it is here as the current policy and
not as a measured top-two finisher.

Weight rule `min_gap` throughout: G1 puts it far above `flat` (0.658
against 0.485) and `pmi` and `mean_gap` both fail the gate.

## Settings

Shared: dim 128, seed 42, 10 walks x 20 steps, window 5, protocol `n2v1m`,
`lp_max_pairs` 50,000, CPU.

fodiwalk: 200 epochs, lr 0.999 constant, `fdlinear`, `min_gap`, the
precomputed strategy (stage 2 builds the pair data once).
node2vec: gensim skip-gram, 5 epochs, 5 negative samples.

Both read the graph through `fodiwalk.make_graph.load`, so row `i` of `Z`
is the same node in both and the two sets compare directly.

## Results

| graph | method | walker | optim | acc | f1 | auc | hop R2 (mlp) | hop R2 (rf) | s | MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cora | node2vec | uniform | word2vec | 0.9744 | 0.9741 | **0.9976** | 0.2305 | 0.2066 | 5.5 | 275 |
| cora | fodiwalk | walk_edges | plain | 0.9692 | 0.9691 | 0.9936 | 0.7073 | 0.6970 | 13.9 | 935 |
| cora | fodiwalk | walk_edges | velocity | 0.9697 | 0.9695 | 0.9941 | **0.7103** | **0.7021** | 6.2 | 890 |
| cora | fodiwalk | nbr_walk | plain | 0.9730 | 0.9726 | 0.9954 | 0.2537 | 0.2229 | 19.8 | 907 |
| cora | fodiwalk | nbr_walk | velocity | 0.9711 | 0.9706 | 0.9949 | 0.2604 | 0.2373 | 13.3 | 918 |
| pubmed | node2vec | uniform | word2vec | 0.9811 | 0.9811 | 0.9973 | 0.0481 | 0.0332 | 20.4 | 289 |
| pubmed | fodiwalk | walk_edges | plain | 0.9712 | 0.9710 | 0.9955 | **0.5170** | **0.5027** | 45.5 | 1088 |
| pubmed | fodiwalk | walk_edges | velocity | 0.9709 | 0.9707 | 0.9949 | 0.5197 | 0.5021 | 39.6 | 1092 |
| pubmed | fodiwalk | nbr_walk | plain | **0.9851** | **0.9850** | **0.9986** | 0.4268 | 0.4081 | 114.8 | 1108 |
| pubmed | fodiwalk | nbr_walk | velocity | 0.9824 | 0.9822 | 0.9986 | 0.4371 | 0.4185 | 114.9 | 1163 |

`hop R2 (mlp)` and `hop R2 (rf)` are the two regressors `evaluator`
reports for hop-distance approximation. `MB` is peak RSS of the whole
process, embedding and scoring together, and it includes the JAX runtime
for the fodiwalk rows -- that floor is most of the gap to node2vec at this
size, not the embedding itself.

## What the numbers say

**Hop distance is where fodiwalk wins, and the margin is large.** cora:
0.7103 against node2vec's 0.2305, a factor of 3.1. pubmed: 0.5197 against
0.0481, a factor of 10.8. This repeats the 1M-node result (0.428 against
0.043) at a size where both methods run in seconds.

**Link prediction is close, and node2vec is not behind.** node2vec takes
cora on AUC (0.9976, the best cell in the table). fodiwalk with `nbr_walk`
takes pubmed (0.9851 accuracy, 0.9986 AUC). No arm separates by more than
0.016 accuracy.

**`plain` and `velocity` are indistinguishable, again.** The largest gap
between them in eight paired cells is 0.0103 hop R2 and 0.0027 accuracy.
This matches section 10 and is a third independent confirmation. Neither
rule is worth choosing on quality; `plain` is free in memory.

**The two walk methods split by TASK, and they disagree with each other.**
`walk_edges` wins hop distance on both graphs -- and on cora it is not
close (0.7103 against 0.2604, a factor of 2.7). `nbr_walk` wins link
prediction on pubmed by 0.014 accuracy. So the ranked G1 evidence holds for
hop distance, and `nbr_walk`'s advantage is elsewhere. A method chosen for
hop-distance quality should be `walk_edges`; `nbr_walk` keeps its place for
scale, where `walk_edges` cannot follow.

## Limits of this run

- Two graphs, both small and both above the average-degree 3 line where
  hop R2 is known to work at all. Nothing here tests the low-degree case.
- One seed. Section 10 measured seed spreads of 0.039 to 0.046 in hop R2,
  which is wider than most gaps in this table. **Read the walk-method gap
  as real and the optimiser gap as noise.**
- 200 epochs. Nothing converged at 200 in the recorded grid; every config
  there still gained +0.12 to +0.22 hop R2 from 50 to 200.
- node2vec runs 5 word2vec epochs against fodiwalk's 200 force steps. The
  walk budget is matched (10 x 20); the training budget is not, and no fair
  common unit exists between the two.
