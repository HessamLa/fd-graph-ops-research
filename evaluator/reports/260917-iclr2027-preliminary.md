# ICLR 2027 runs: preliminary results

**Built:** 2026-09-19T18:27Z. **Status:** preliminary. Runner has made 2365 of
the 2,277 planned runs; 2052 are scored. cora and pubmed are complete
at all four dimensions for the main cells; wordnet dim 128 has started; the
large graphs have not.

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
- **LP AUC is saturated.** It sits between 0.971 and 0.9996 over every cell
  in this report, while rho spans 0.27 to 0.87. Rank the methods on
  geometry, not on AUC.

## 1. Main comparison, dim 128

**cora, dim 128**

| cell | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score |
|---|---|---|---|---|---|---|
| fdhop walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.843 ± 0.006 | 0.710 ± 0.016 | 0.960 ± 0.003 | 0.9989 ± 0.0003 | 0.9880 ± 0.0015 |
| fdlinear walk_edges min_gap 10x20 k4=0.01 kr=1 | 11 | 0.779 ± 0.005 | 0.685 ± 0.008 | 0.721 ± 0.006 | 0.9942 ± 0.0010 | 0.9678 ± 0.0020 |
| node2vec q=0.5 | 11 | 0.736 ± 0.003 | 0.510 ± 0.012 | 0.852 ± 0.002 | 0.9963 ± 0.0005 | 0.9675 ± 0.0028 |
| deepwalk | 34 | 0.684 ± 0.071 | 0.497 ± 0.104 | 0.913 ± 0.021 | 0.9986 ± 0.0005 | 0.9794 ± 0.0032 |
| node2vec q=2.0 | 11 | 0.633 ± 0.006 | 0.371 ± 0.018 | 0.866 ± 0.003 | 0.9973 ± 0.0006 | 0.9724 ± 0.0018 |
| fdhop nbr_walk min_gap 10x20 k4=1 kr=1 | 11 | 0.630 ± 0.008 | 0.392 ± 0.020 | 0.972 ± 0.002 | 0.9995 ± 0.0001 | 0.9895 ± 0.0010 |
| node2vec q=1.0 | 34 | 0.497 ± 0.178 | 0.282 ± 0.135 | 0.843 ± 0.041 | 0.9973 ± 0.0009 | 0.9765 ± 0.0055 |

**pubmed, dim 128**

| cell | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score |
|---|---|---|---|---|---|---|
| fdhop walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.779 ± 0.003 | 0.613 ± 0.010 | 0.779 ± 0.005 | 0.9990 ± 0.0002 | 0.9910 ± 0.0006 |
| fdlinear walk_edges min_gap 10x20 k4=0.01 kr=1 | 11 | 0.713 ± 0.002 | 0.510 ± 0.006 | 0.473 ± 0.007 | 0.9949 ± 0.0003 | 0.9715 ± 0.0009 |
| deepwalk | 11 | 0.664 ± 0.002 | 0.517 ± 0.005 | 0.729 ± 0.004 | 0.9979 ± 0.0002 | 0.9735 ± 0.0015 |
| fdhop nbr_walk min_gap 10x20 k4=1 kr=1 | 11 | 0.565 ± 0.005 | 0.342 ± 0.016 | 0.863 ± 0.004 | 0.9995 ± 0.0001 | 0.9920 ± 0.0008 |
| node2vec q=0.5 | 11 | 0.415 ± 0.004 | 0.220 ± 0.009 | 0.635 ± 0.004 | 0.9959 ± 0.0003 | 0.9681 ± 0.0017 |
| node2vec q=1.0 | 11 | 0.363 ± 0.004 | 0.182 ± 0.011 | 0.627 ± 0.002 | 0.9960 ± 0.0002 | 0.9679 ± 0.0018 |
| node2vec q=2.0 | 11 | 0.276 ± 0.004 | 0.135 ± 0.011 | 0.607 ± 0.005 | 0.9962 ± 0.0002 | 0.9701 ± 0.0015 |

**wordnet, dim 128**

| cell | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score |
|---|---|---|---|---|---|---|
| fdlinear walk_edges min_gap 10x20 k4=0.01 kr=1 | 11 | 0.365 ± 0.007 | 0.166 ± 0.006 | 0.812 ± 0.006 | 0.9992 ± 0.0001 | 0.9927 ± 0.0004 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.348 ± 0.010 | 0.113 ± 0.013 | 0.992 ± 0.002 | 0.9998 ± 0.0001 | 0.9963 ± 0.0005 |
| deepwalk | 11 | 0.163 ± 0.007 | 0.028 ± 0.006 | 0.683 ± 0.004 | 0.9992 ± 0.0001 | 0.9891 ± 0.0006 |
| fdhop nbr_walk min_gap 10x20 k4=1 kr=1 | 11 | 0.062 ± 0.006 | 0.011 ± 0.007 | 0.905 ± 0.006 | 0.9943 ± 0.0005 | 0.9560 ± 0.0022 |
| node2vec q=0.5 | 11 | 0.023 ± 0.005 | 0.043 ± 0.007 | 0.471 ± 0.002 | 0.9994 ± 0.0001 | 0.9934 ± 0.0006 |
| node2vec q=1.0 | 11 | 0.015 ± 0.005 | 0.032 ± 0.010 | 0.482 ± 0.006 | 0.9994 ± 0.0000 | 0.9939 ± 0.0007 |
| node2vec q=2.0 | 11 | 0.005 ± 0.005 | 0.031 ± 0.008 | 0.496 ± 0.004 | 0.9994 ± 0.0001 | 0.9943 ± 0.0005 |

fdhop with the walk_edges policy leads on cora and pubmed, on both geometry
measures. The gap over deepwalk is 0.101 on cora and 0.115 on pubmed, about
15 and 33 standard deviations of the difference. node2vec is monotone in q,
favouring q=0.5, as measured before.

**wordnet breaks this order.** At 11 seeds, fdlinear reaches
rho 0.365 +/- 0.007 against fdhop at 0.348 +/- 0.010. That gap is 1.4
standard deviations of the difference, so the two are not separated. Every
method is weak on wordnet: deepwalk reaches 0.163 and node2vec q=0.5 only
0.023, while all of them keep an AUC above 0.999. The collapse of the walk
baselines is far outside the seed spread and holds now. The best wordnet
cell of all is fdhop with the FLAT weight, at 0.393 +/- 0.010; see the next
section.

`fdhop nbr_walk` is the exception: it has the best recall@10 and the best
AUC of all, and the worst rho of the fodiwalk cells. It puts neighbours
close and loses the longer distances.

## 2. Dimension (Spearman rho)

**cora**

| dim | deepwalk | node2vec q=0.5 | fdhop walk_edges | fdhop nbr_walk |
|---|---|---|---|---|
| 128 | 0.684 ± 0.071 (34) | 0.736 ± 0.003 | 0.843 ± 0.006 | 0.630 ± 0.008 |
| 64 | 0.779 ± 0.002 | 0.770 ± 0.005 | 0.831 ± 0.007 | 0.578 ± 0.008 |
| 32 | 0.793 ± 0.003 | 0.789 ± 0.002 | 0.809 ± 0.007 | 0.521 ± 0.010 |
| 16 | 0.785 ± 0.003 | 0.783 ± 0.004 | 0.767 ± 0.011 | 0.454 ± 0.016 |

**pubmed**

| dim | deepwalk | node2vec q=0.5 | fdhop walk_edges | fdhop nbr_walk |
|---|---|---|---|---|
| 128 | 0.664 ± 0.002 | 0.415 ± 0.004 | 0.779 ± 0.003 | 0.565 ± 0.005 |
| 64 | 0.670 ± 0.002 | 0.565 ± 0.004 | 0.769 ± 0.004 | 0.542 ± 0.003 |
| 32 | 0.673 ± 0.003 | 0.639 ± 0.003 | 0.746 ± 0.005 | 0.502 ± 0.005 |
| 16 | 0.670 ± 0.004 | 0.649 ± 0.002 | 0.700 ± 0.005 | 0.437 ± 0.009 |

**wordnet**

| dim | deepwalk | node2vec q=0.5 | fdhop walk_edges | fdhop nbr_walk |
|---|---|---|---|---|
| 128 | 0.163 ± 0.007 | 0.023 ± 0.005 | 0.348 ± 0.010 | 0.062 ± 0.006 |
| 64 | 0.295 ± 0.006 (9) | 0.034 ± 0.005 (9) | 0.318 ± 0.008 (9) | 0.038 ± 0.004 (9) |
| 32 | 0.400 ± 0.008 (9) | 0.054 ± 0.005 (9) | 0.283 ± 0.007 (9) | 0.021 ± 0.005 (9) |
| 16 | 0.459 ± 0.011 (8) | 0.063 ± 0.008 (9) | 0.240 ± 0.008 (9) | 0.016 ± 0.003 (9) |

A seed count in brackets marks a cell that is still running.

On cora and pubmed, fdhop walk_edges rises with the dimension and is best
at 128. It still falls below deepwalk at dim 16 on cora (0.767 against
0.785). The walk baselines barely move from 16 to 128 on those two graphs.

**On wordnet the two families move in opposite directions.** fdhop rises
with the dimension, as on the other graphs (0.240 at 16, 0.348 at 128).
deepwalk instead falls hard, and only here (0.459 at 16, 0.163 at 128). So at dim 16 deepwalk beats fdhop on wordnet by 0.219,
and at dim 128 fdhop leads by 0.185. A one-dimension comparison on a
tree-like graph picks its own winner. The paper must state the dimension
with every wordnet claim.

## 3. Force law, dim 128

**cora, dim 128, walk_edges min_gap, plain**

| cell | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score |
|---|---|---|---|---|---|---|
| fdhop walk_edges_pq min_gap 10x20 k4=1 kr=1 q=2 | 11 | 0.843 ± 0.005 | 0.713 ± 0.013 | 0.960 ± 0.001 | 0.9990 ± 0.0003 | 0.9889 ± 0.0022 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.843 ± 0.006 | 0.710 ± 0.016 | 0.960 ± 0.003 | 0.9989 ± 0.0003 | 0.9880 ± 0.0015 |
| fdhop walk_edges_pq min_gap 10x20 k4=1 kr=1 q=0.5 | 11 | 0.842 ± 0.006 | 0.709 ± 0.016 | 0.960 ± 0.002 | 0.9990 ± 0.0003 | 0.9884 ± 0.0019 |
| fdhop walk min_gap 10x20 k4=1 kr=1 | 11 | 0.841 ± 0.006 | 0.707 ± 0.016 | 0.960 ± 0.002 | 0.9987 ± 0.0002 | 0.9876 ± 0.0013 |
| fdhop2 walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.830 ± 0.004 | 0.687 ± 0.013 | 0.921 ± 0.004 | 0.9984 ± 0.0004 | 0.9847 ± 0.0015 |
| fdhop_all walk_edges min_gap 10x20 k2=2 k4=1 kr=1 | 11 | 0.779 ± 0.008 | 0.623 ± 0.019 | 0.981 ± 0.001 | 0.9991 ± 0.0003 | 0.9888 ± 0.0024 |
| fdhop walk_edges flat 10x20 k4=1 kr=1 | 11 | 0.656 ± 0.005 | 0.510 ± 0.010 | 0.661 ± 0.006 | 0.9932 ± 0.0013 | 0.9710 ± 0.0027 |
| fdhop walk flat 10x20 k4=1 kr=1 | 11 | 0.656 ± 0.005 | 0.511 ± 0.010 | 0.661 ± 0.006 | 0.9935 ± 0.0013 | 0.9719 ± 0.0021 |
| fdhop nbr_walk min_gap 10x20 k4=1 kr=1 | 11 | 0.630 ± 0.008 | 0.392 ± 0.020 | 0.972 ± 0.002 | 0.9995 ± 0.0001 | 0.9895 ± 0.0010 |
| fdhop_all_freq walk_edges min_gap 10x20 k2=7 k4=1 kr=1 | 11 | 0.474 ± 0.006 | 0.252 ± 0.015 | 0.996 ± 0.001 | 0.9993 ± 0.0008 | 0.9964 ± 0.0013 |
| fdhop_min walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.360 ± 0.016 | 0.159 ± 0.011 | 0.577 ± 0.002 | 0.9906 ± 0.0005 | 0.9469 ± 0.0034 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=10 | 11 | 0.272 ± 0.008 | 0.062 ± 0.015 | 0.451 ± 0.014 | 0.9718 ± 0.0024 | 0.9119 ± 0.0054 |

**pubmed, dim 128, walk_edges min_gap, plain**

| cell | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score |
|---|---|---|---|---|---|---|
| fdhop walk_edges_pq min_gap 10x20 k4=1 kr=1 q=0.5 | 11 | 0.779 ± 0.004 | 0.611 ± 0.012 | 0.782 ± 0.005 | 0.9989 ± 0.0001 | 0.9908 ± 0.0007 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.779 ± 0.003 | 0.613 ± 0.010 | 0.779 ± 0.005 | 0.9990 ± 0.0002 | 0.9910 ± 0.0006 |
| fdhop walk_edges_pq min_gap 10x20 k4=1 kr=1 q=2 | 11 | 0.777 ± 0.003 | 0.607 ± 0.012 | 0.777 ± 0.004 | 0.9990 ± 0.0002 | 0.9906 ± 0.0012 |
| fdhop walk min_gap 10x20 k4=1 kr=1 | 11 | 0.770 ± 0.004 | 0.602 ± 0.011 | 0.783 ± 0.005 | 0.9987 ± 0.0002 | 0.9898 ± 0.0010 |
| fdhop_all walk_edges min_gap 10x20 k2=2 k4=1 kr=1 | 11 | 0.738 ± 0.005 | 0.573 ± 0.012 | 0.867 ± 0.004 | 0.9993 ± 0.0001 | 0.9905 ± 0.0007 |
| fdhop2 walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.640 ± 0.005 | 0.427 ± 0.019 | 0.711 ± 0.006 | 0.9977 ± 0.0003 | 0.9815 ± 0.0011 |
| fdhop nbr_walk min_gap 10x20 k4=1 kr=1 | 11 | 0.565 ± 0.005 | 0.342 ± 0.016 | 0.863 ± 0.004 | 0.9995 ± 0.0001 | 0.9920 ± 0.0008 |
| fdhop walk_edges flat 10x20 k4=1 kr=1 | 11 | 0.547 ± 0.002 | 0.275 ± 0.008 | 0.507 ± 0.005 | 0.9945 ± 0.0004 | 0.9703 ± 0.0013 |
| fdhop walk flat 10x20 k4=1 kr=1 | 11 | 0.545 ± 0.002 | 0.274 ± 0.010 | 0.507 ± 0.006 | 0.9943 ± 0.0003 | 0.9702 ± 0.0008 |
| fdhop_all_freq walk_edges min_gap 10x20 k2=7 k4=1 kr=1 | 11 | 0.512 ± 0.003 | 0.274 ± 0.008 | 0.735 ± 0.010 | 0.9991 ± 0.0001 | 0.9904 ± 0.0005 |
| fdhop_min walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.438 ± 0.003 | 0.235 ± 0.007 | 0.428 ± 0.002 | 0.9859 ± 0.0005 | 0.9303 ± 0.0009 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=10 | 11 | 0.366 ± 0.003 | 0.114 ± 0.007 | 0.223 ± 0.006 | 0.9750 ± 0.0010 | 0.9277 ± 0.0024 |

**wordnet, dim 128, walk_edges min_gap, plain**

| cell | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score |
|---|---|---|---|---|---|---|
| fdhop walk_edges flat 10x20 k4=1 kr=1 | 11 | 0.393 ± 0.010 | 0.219 ± 0.006 | 0.403 ± 0.008 | 0.9992 ± 0.0001 | 0.9959 ± 0.0003 |
| fdhop walk flat 10x20 k4=1 kr=1 | 11 | 0.393 ± 0.010 | 0.220 ± 0.006 | 0.403 ± 0.008 | 0.9992 ± 0.0001 | 0.9959 ± 0.0003 |
| fdhop walk_edges_pq min_gap 10x20 k4=1 kr=1 q=2 | 11 | 0.348 ± 0.008 | 0.115 ± 0.012 | 0.992 ± 0.001 | 0.9998 ± 0.0001 | 0.9962 ± 0.0003 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.348 ± 0.010 | 0.113 ± 0.013 | 0.992 ± 0.002 | 0.9998 ± 0.0001 | 0.9963 ± 0.0005 |
| fdhop walk_edges_pq min_gap 10x20 k4=1 kr=1 q=0.5 | 11 | 0.348 ± 0.009 | 0.115 ± 0.010 | 0.991 ± 0.002 | 0.9998 ± 0.0001 | 0.9960 ± 0.0004 |
| fdhop walk min_gap 10x20 k4=1 kr=1 | 11 | 0.346 ± 0.010 | 0.111 ± 0.012 | 0.991 ± 0.002 | 0.9998 ± 0.0001 | 0.9963 ± 0.0005 |
| fdhop_all walk_edges min_gap 10x20 k2=2 k4=1 kr=1 | 11 | 0.333 ± 0.005 | 0.093 ± 0.009 | 0.995 ± 0.001 | 0.9998 ± 0.0001 | 0.9963 ± 0.0004 |
| fdhop2 walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.248 ± 0.006 | 0.109 ± 0.014 | 0.885 ± 0.005 | 0.9993 ± 0.0001 | 0.9831 ± 0.0009 |
| fdhop_all_freq walk_edges min_gap 10x20 k2=7 k4=1 kr=1 | 11 | 0.201 ± 0.008 | 0.008 ± 0.010 | 0.903 ± 0.008 | 0.9983 ± 0.0006 | 0.9852 ± 0.0013 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=10 | 11 | 0.085 ± 0.005 | -0.035 ± 0.007 | 0.600 ± 0.013 | 0.9979 ± 0.0003 | 0.9818 ± 0.0009 |
| fdhop nbr_walk min_gap 10x20 k4=1 kr=1 | 11 | 0.062 ± 0.006 | 0.011 ± 0.007 | 0.905 ± 0.006 | 0.9943 ± 0.0005 | 0.9560 ± 0.0022 |
| fdhop_min walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.029 ± 0.009 | 0.012 ± 0.006 | 0.912 ± 0.005 | 0.9990 ± 0.0001 | 0.9774 ± 0.0009 |

`fdhop_min` and `fdhop_all_freq` are clearly worse on geometry. Note
`fdhop_all_freq`: recall@10 0.996 and f1_score 0.9964 on cora, the best of
any cell, with rho 0.474. Local structure and global geometry come apart.

## 4. Pair support and walk budget, cora dim 128

**cora, dim 128: walk policy and weight rule**

| cell | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score |
|---|---|---|---|---|---|---|
| fdhop walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.843 ± 0.006 | 0.710 ± 0.016 | 0.960 ± 0.003 | 0.9989 ± 0.0003 | 0.9880 ± 0.0015 |
| fdhop walk min_gap 10x20 k4=1 kr=1 | 11 | 0.841 ± 0.006 | 0.707 ± 0.016 | 0.960 ± 0.002 | 0.9987 ± 0.0002 | 0.9876 ± 0.0013 |
| fdhop walk_edges flat 10x20 k4=1 kr=1 | 11 | 0.656 ± 0.005 | 0.510 ± 0.010 | 0.661 ± 0.006 | 0.9932 ± 0.0013 | 0.9710 ± 0.0027 |
| fdhop walk flat 10x20 k4=1 kr=1 | 11 | 0.656 ± 0.005 | 0.511 ± 0.010 | 0.661 ± 0.006 | 0.9935 ± 0.0013 | 0.9719 ± 0.0021 |
| fdhop nbr_walk min_gap 10x20 k4=1 kr=1 | 11 | 0.630 ± 0.008 | 0.392 ± 0.020 | 0.972 ± 0.002 | 0.9995 ± 0.0001 | 0.9895 ± 0.0010 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=10 | 11 | 0.272 ± 0.008 | 0.062 ± 0.015 | 0.451 ± 0.014 | 0.9718 ± 0.0024 | 0.9119 ± 0.0054 |

**cora, dim 128: walk budget (walks x length), walk_edges min_gap**

| cell | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score |
|---|---|---|---|---|---|---|
| fdhop walk_edges min_gap 5x20 k4=1 kr=1 | 11 | 0.844 ± 0.003 | 0.708 ± 0.011 | 0.960 ± 0.003 | 0.9987 ± 0.0004 | 0.9884 ± 0.0016 |
| fdhop walk_edges min_gap 10x80 k4=1 kr=1 | 11 | 0.844 ± 0.006 | 0.711 ± 0.016 | 0.962 ± 0.001 | 0.9990 ± 0.0003 | 0.9890 ± 0.0019 |
| fdhop walk_edges min_gap 40x20 k4=1 kr=1 | 11 | 0.844 ± 0.007 | 0.716 ± 0.016 | 0.960 ± 0.003 | 0.9986 ± 0.0004 | 0.9882 ± 0.0015 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.843 ± 0.006 | 0.710 ± 0.016 | 0.960 ± 0.003 | 0.9989 ± 0.0003 | 0.9880 ± 0.0015 |
| fdhop walk_edges min_gap 10x40 k4=1 kr=1 | 11 | 0.843 ± 0.005 | 0.708 ± 0.010 | 0.960 ± 0.002 | 0.9990 ± 0.0003 | 0.9881 ± 0.0009 |
| fdhop walk_edges min_gap 20x20 k4=1 kr=1 | 11 | 0.843 ± 0.006 | 0.711 ± 0.016 | 0.960 ± 0.002 | 0.9988 ± 0.0004 | 0.9877 ± 0.0019 |
| fdhop walk_edges min_gap 10x10 k4=1 kr=1 | 11 | 0.842 ± 0.006 | 0.711 ± 0.013 | 0.958 ± 0.002 | 0.9989 ± 0.0002 | 0.9871 ± 0.0023 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=10 | 11 | 0.272 ± 0.008 | 0.062 ± 0.015 | 0.451 ± 0.014 | 0.9718 ± 0.0024 | 0.9119 ± 0.0054 |

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

| graph | min_gap | flat | difference | sd of the difference |
|---|---|---|---|---|
| cora | 0.843 ± 0.006 (11) | 0.656 ± 0.005 (11) | +0.187 | 25.7 |
| pubmed | 0.779 ± 0.003 (11) | 0.547 ± 0.002 (11) | +0.231 | 61.6 |
| wordnet | 0.348 ± 0.010 (11) | 0.393 ± 0.010 (11) | -0.045 | 3.3 |


This is the paper's central claim under test, and the answer is not the
same on every graph. On cora and pubmed the observed walk gap is worth
0.187 and 0.231 rho over a flat weight, at 26 and 62 standard deviations of
the difference. On wordnet the flat weight WINS by 0.045, at 3.3 standard
deviations.

So walk-gap weighting is not a general improvement. It is large on the two
citation graphs and it is negative on the hierarchy. Do not write the
mechanism claim without this row. wordnet is a tree-like graph, which is
the regime the paper plan already names as a risk.

## 5. Force parameters, dim 128

**cora, dim 128: k4 and kr, walk_edges min_gap 10x20**

| cell | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score |
|---|---|---|---|---|---|---|
| fdhop walk_edges min_gap 10x20 k4=1 kr=0.5 | 11 | 0.846 ± 0.005 | 0.729 ± 0.014 | 0.925 ± 0.003 | 0.9984 ± 0.0003 | 0.9841 ± 0.0025 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.843 ± 0.006 | 0.710 ± 0.016 | 0.960 ± 0.003 | 0.9989 ± 0.0003 | 0.9880 ± 0.0015 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=0.1 | 11 | 0.830 ± 0.005 | 0.726 ± 0.012 | 0.850 ± 0.003 | 0.9969 ± 0.0006 | 0.9777 ± 0.0013 |
| fdhop walk_edges min_gap 10x20 k4=0.1 kr=1 | 11 | 0.829 ± 0.004 | 0.723 ± 0.011 | 0.851 ± 0.003 | 0.9968 ± 0.0006 | 0.9768 ± 0.0022 |
| fdlinear walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.800 ± 0.009 | 0.659 ± 0.027 | 0.830 ± 0.005 | 0.9966 ± 0.0004 | 0.9760 ± 0.0021 |
| fdhop walk_edges min_gap 10x20 k4=0.01 kr=1 | 11 | 0.798 ± 0.005 | 0.705 ± 0.008 | 0.763 ± 0.005 | 0.9950 ± 0.0006 | 0.9699 ± 0.0020 |
| fdlinear walk_edges min_gap 10x20 k4=0.1 kr=1 | 11 | 0.797 ± 0.005 | 0.693 ± 0.010 | 0.768 ± 0.006 | 0.9951 ± 0.0010 | 0.9714 ± 0.0025 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=2 | 11 | 0.796 ± 0.009 | 0.634 ± 0.018 | 0.991 ± 0.001 | 0.9995 ± 0.0003 | 0.9928 ± 0.0016 |
| fdlinear walk_edges min_gap 10x20 k4=0.01 kr=1 | 11 | 0.779 ± 0.005 | 0.685 ± 0.008 | 0.721 ± 0.006 | 0.9942 ± 0.0010 | 0.9678 ± 0.0020 |
| fdhop walk_edges min_gap 10x20 k4=10 kr=1 | 11 | 0.736 ± 0.015 | 0.591 ± 0.021 | 0.962 ± 0.009 | 0.9995 ± 0.0003 | 0.9956 ± 0.0014 |
| fdlinear walk_edges min_gap 10x20 k4=10 kr=1 | 11 | 0.735 ± 0.008 | 0.581 ± 0.013 | 0.860 ± 0.005 | 0.9966 ± 0.0007 | 0.9747 ± 0.0034 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=10 | 11 | 0.272 ± 0.008 | 0.062 ± 0.015 | 0.451 ± 0.014 | 0.9718 ± 0.0024 | 0.9119 ± 0.0054 |

**pubmed, dim 128: k4 and kr, walk_edges min_gap 10x20**

| cell | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score |
|---|---|---|---|---|---|---|
| fdhop walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.779 ± 0.003 | 0.613 ± 0.010 | 0.779 ± 0.005 | 0.9990 ± 0.0002 | 0.9910 ± 0.0006 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=2 | 11 | 0.774 ± 0.004 | 0.618 ± 0.011 | 0.847 ± 0.004 | 0.9995 ± 0.0001 | 0.9951 ± 0.0006 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=0.5 | 11 | 0.758 ± 0.004 | 0.578 ± 0.012 | 0.729 ± 0.005 | 0.9984 ± 0.0002 | 0.9865 ± 0.0004 |
| fdhop walk_edges min_gap 10x20 k4=0.1 kr=1 | 11 | 0.750 ± 0.002 | 0.560 ± 0.009 | 0.626 ± 0.005 | 0.9971 ± 0.0003 | 0.9793 ± 0.0005 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=0.1 | 11 | 0.750 ± 0.003 | 0.559 ± 0.008 | 0.624 ± 0.004 | 0.9972 ± 0.0002 | 0.9796 ± 0.0007 |
| fdhop walk_edges min_gap 10x20 k4=0.01 kr=1 | 11 | 0.730 ± 0.002 | 0.536 ± 0.005 | 0.509 ± 0.005 | 0.9956 ± 0.0004 | 0.9738 ± 0.0012 |
| fdlinear walk_edges min_gap 10x20 k4=0.1 kr=1 | 11 | 0.717 ± 0.003 | 0.511 ± 0.008 | 0.536 ± 0.005 | 0.9959 ± 0.0002 | 0.9738 ± 0.0009 |
| fdlinear walk_edges min_gap 10x20 k4=0.01 kr=1 | 11 | 0.713 ± 0.002 | 0.510 ± 0.006 | 0.473 ± 0.007 | 0.9949 ± 0.0003 | 0.9715 ± 0.0009 |
| fdlinear walk_edges min_gap 10x20 k4=1 kr=1 | 11 | 0.662 ± 0.005 | 0.454 ± 0.007 | 0.620 ± 0.005 | 0.9966 ± 0.0003 | 0.9755 ± 0.0013 |
| fdhop walk_edges min_gap 10x20 k4=10 kr=1 | 11 | 0.637 ± 0.006 | 0.454 ± 0.013 | 0.805 ± 0.008 | 0.9997 ± 0.0001 | 0.9976 ± 0.0004 |
| fdlinear walk_edges min_gap 10x20 k4=10 kr=1 | 11 | 0.487 ± 0.007 | 0.294 ± 0.015 | 0.624 ± 0.008 | 0.9957 ± 0.0002 | 0.9699 ± 0.0010 |
| fdhop walk_edges min_gap 10x20 k4=1 kr=10 | 11 | 0.366 ± 0.003 | 0.114 ± 0.007 | 0.223 ± 0.006 | 0.9750 ± 0.0010 | 0.9277 ± 0.0024 |

k4=1 and kr=1 are near the top on both graphs. kr=10 is a failure
(rho 0.272 on cora). kr=2 buys recall@10 (0.991) and costs rho.

The fdlinear runs at k4=0.01 have a mean row norm near 234 on cora against
5.6 for fdhop at k4=1. This is scale, not instability: weak decay spreads
the layout. Do not read it as a blow-up.

## 6. Optimizer study, dim 64

**cora, dim 64, fdhop walk_edges min_gap, 11 seeds**

| optimizer | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score |
|---|---|---|---|---|---|---|
| sqn/const/lr=0.999 | 11 | 0.864 ± 0.005 | 0.751 ± 0.013 | 0.955 ± 0.003 | 0.9986 ± 0.0007 | 0.9885 ± 0.0021 |
| adam/linear/lr=0.999 | 11 | 0.857 ± 0.003 | 0.752 ± 0.012 | 0.954 ± 0.003 | 0.9986 ± 0.0004 | 0.9865 ± 0.0023 |
| velocity/const/lr=0.999 | 11 | 0.836 ± 0.006 | 0.698 ± 0.011 | 0.958 ± 0.002 | 0.9987 ± 0.0004 | 0.9880 ± 0.0019 |
| momentum/const/lr=0.099 | 11 | 0.832 ± 0.006 | 0.692 ± 0.017 | 0.958 ± 0.002 | 0.9987 ± 0.0004 | 0.9875 ± 0.0020 |
| plain/const/lr=0.999 | 11 | 0.831 ± 0.007 | 0.686 ± 0.018 | 0.958 ± 0.002 | 0.9988 ± 0.0004 | 0.9872 ± 0.0021 |
| nesterov/const/lr=0.099 | 11 | 0.831 ± 0.006 | 0.688 ± 0.016 | 0.958 ± 0.002 | 0.9990 ± 0.0003 | 0.9883 ± 0.0019 |
| sqn/linear/lr=0.999 | 11 | 0.822 ± 0.009 | 0.671 ± 0.022 | 0.956 ± 0.003 | 0.9987 ± 0.0006 | 0.9872 ± 0.0017 |
| adam/const/lr=0.999 | 11 | 0.810 ± 0.013 | 0.677 ± 0.023 | 0.839 ± 0.024 | 0.9953 ± 0.0011 | 0.9718 ± 0.0037 |
| velocity/linear/lr=0.999 | 11 | 0.793 ± 0.009 | 0.636 ± 0.019 | 0.962 ± 0.003 | 0.9989 ± 0.0004 | 0.9885 ± 0.0021 |
| plain/linear/lr=0.999 | 11 | 0.785 ± 0.011 | 0.622 ± 0.024 | 0.964 ± 0.002 | 0.9990 ± 0.0004 | 0.9879 ± 0.0016 |
| momentum/linear/lr=0.099 | 11 | 0.784 ± 0.009 | 0.627 ± 0.022 | 0.962 ± 0.002 | 0.9991 ± 0.0003 | 0.9874 ± 0.0019 |
| nesterov/linear/lr=0.099 | 11 | 0.781 ± 0.008 | 0.623 ± 0.022 | 0.963 ± 0.002 | 0.9990 ± 0.0004 | 0.9882 ± 0.0023 |

**pubmed, dim 64, fdhop walk_edges min_gap, 11 seeds**

| optimizer | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score |
|---|---|---|---|---|---|---|
| sqn/const/lr=0.999 | 11 | 0.785 ± 0.003 | 0.618 ± 0.011 | 0.759 ± 0.005 | 0.9987 ± 0.0003 | 0.9892 ± 0.0010 |
| velocity/const/lr=0.999 | 11 | 0.785 ± 0.004 | 0.622 ± 0.014 | 0.768 ± 0.004 | 0.9988 ± 0.0002 | 0.9890 ± 0.0007 |
| momentum/const/lr=0.099 | 11 | 0.783 ± 0.005 | 0.618 ± 0.013 | 0.766 ± 0.005 | 0.9987 ± 0.0002 | 0.9893 ± 0.0008 |
| nesterov/const/lr=0.099 | 11 | 0.782 ± 0.005 | 0.617 ± 0.012 | 0.768 ± 0.004 | 0.9987 ± 0.0002 | 0.9892 ± 0.0007 |
| adam/linear/lr=0.999 | 11 | 0.782 ± 0.003 | 0.615 ± 0.010 | 0.748 ± 0.004 | 0.9986 ± 0.0002 | 0.9889 ± 0.0006 |
| plain/const/lr=0.999 | 11 | 0.769 ± 0.004 | 0.594 ± 0.012 | 0.771 ± 0.003 | 0.9988 ± 0.0002 | 0.9887 ± 0.0010 |
| plain/linear/lr=0.999 | 11 | 0.750 ± 0.005 | 0.576 ± 0.015 | 0.796 ± 0.005 | 0.9990 ± 0.0001 | 0.9891 ± 0.0010 |
| velocity/linear/lr=0.999 | 11 | 0.749 ± 0.008 | 0.581 ± 0.015 | 0.795 ± 0.004 | 0.9989 ± 0.0002 | 0.9889 ± 0.0005 |
| sqn/linear/lr=0.999 | 11 | 0.743 ± 0.012 | 0.568 ± 0.016 | 0.791 ± 0.006 | 0.9989 ± 0.0002 | 0.9889 ± 0.0008 |
| momentum/linear/lr=0.099 | 11 | 0.738 ± 0.006 | 0.565 ± 0.016 | 0.795 ± 0.004 | 0.9988 ± 0.0002 | 0.9893 ± 0.0004 |
| nesterov/linear/lr=0.099 | 11 | 0.737 ± 0.006 | 0.563 ± 0.016 | 0.798 ± 0.004 | 0.9989 ± 0.0002 | 0.9894 ± 0.0006 |
| adam/const/lr=0.999 | 11 | 0.697 ± 0.005 | 0.484 ± 0.018 | 0.576 ± 0.009 | 0.9924 ± 0.0007 | 0.9614 ± 0.0021 |

sqn at lr 0.999 and adam with linear decay lead on cora. On pubmed the top
five rules are within 0.003 rho, which is under one standard deviation:
that group is not separated by these data.

## 7. Learning-rate map, dim 64 (Spearman rho, const decay)

**cora, dim 64: rho by rule and learning rate (const)**

| rule | lr 0.999 | lr 0.5 | lr 0.2 | lr 0.099 | lr 0.05 | lr 0.02 |
|---|---|---|---|---|---|---|
| plain | 0.831 | 0.787 | 0.692 | 0.600 | 0.511 | 0.410 |
| velocity | 0.836 | 0.789 | 0.691 | 0.600 | 0.512 | 0.410 |
| sqn | 0.864 | 0.816 | 0.641 | 0.333 | 0.103 | 0.096 |
| adam | 0.810 | 0.866 | 0.856 | 0.822 | 0.752 | 0.583 |
| momentum | 0.105 ✗ | 0.243 ✗ | 0.860 | 0.832 | 0.788 | 0.687 |
| nesterov | 0.199 ✗ | 0.872 | 0.861 | 0.831 | 0.787 | 0.687 |

**pubmed, dim 64: rho by rule and learning rate (const)**

| rule | lr 0.999 | lr 0.5 | lr 0.2 | lr 0.099 | lr 0.05 | lr 0.02 |
|---|---|---|---|---|---|---|
| plain | 0.769 | 0.750 | 0.630 | 0.565 | 0.519 | 0.451 |
| velocity | 0.785 | 0.751 | 0.631 | 0.565 | 0.519 | 0.451 |
| sqn | 0.785 | 0.725 | 0.427 | 0.362 | 0.181 | 0.116 |
| adam | 0.697 | 0.785 | 0.789 | 0.775 | 0.694 | 0.561 |
| momentum | 0.177 ✗ | 0.193 ✗ | 0.788 | 0.783 | 0.743 | 0.623 |
| nesterov | 0.256 ✗ | 0.788 | 0.790 | 0.782 | 0.743 | 0.624 |

"x" marks a cell whose mean row norm is more than 10x the same setting
under plain. The rule is a PROPOSAL; the owner has not chosen it. Under it,
momentum blows up at lr 0.5 and 0.999, and nesterov at lr 0.999. No run
produced non-finite values, so `diverged.keys` is empty.

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
- wordnet is at dim 128 only. com_youtube, roadnet_ca, ncbi_taxonomy and
  as_skitter have not started, so there is no large-graph result and no
  cost or memory result yet.
- Force2Vec and rForce2Vec, the closest competitors in the paper plan, are
  not in this run set.
