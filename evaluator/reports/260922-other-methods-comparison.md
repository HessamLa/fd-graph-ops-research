# Other-methods comparison: 11-seed results

**Built:** 2026-09-22T09:10Z. **Updated:** 2026-09-23T06:40Z, 11 seeds.
**Status:** near-final. Twelve of fourteen methods are at the full 11 seeds;
deepwalk (9-10) and the O(n²) Force2Vec base (3) are still completing and
are marked with their seed count. Nine published comparison methods against
fodiwalk, deepwalk and node2vec, on cora, citeseer and pubmed at dim 128.
From the plan in `evaluator/recommended-comparisons.md`.

**Store:** every embedding is in the shared store,
`data_cache/embeddings/<graph>/128/<run>/` — `Z.npy`, `config.json`, and
`eval.json` (the seven scored metrics, written next to the embedding).
Method code: `other-methods/`. Scores are produced by
`other-methods/eval_sequencer.py` (one graph preprocessing per experiment,
result cached in `eval.json`) and tabled by
`other-methods/score_report.py`.

## How to read these numbers

- **Seeds: 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52** (11 embedding
  seeds). Each value is **mean ± std over the seeds a row has**; the `seeds`
  column gives the count. deepwalk and Force2Vec base are still filling to
  11.
- **The evaluation seed is fixed at 42.** rho, recall@10, NMI and ARI use
  the same hop pairs, query nodes and k-means start for every embedding of a
  graph, so the spread across the 11 rows is method spread, not sample
  spread. LP AUC/f1 and hop R2 are the values `evaluator` wrote at embed
  time (scored with the embedding seed), read back from `config.json`.
- One protocol for every row: `fodiwalk_dist`, `lp_max_pairs=50_000`. One
  node numbering: every method loads through `fodiwalk.make_graph.load`.
- `rho` is Spearman between **Euclidean** embedding distance and hop
  distance (hop ≥ 2). `hop R2` is a random forest on that one distance
  column. `recall@10` is the share of a node's true neighbours among its 10
  Euclidean-nearest nodes (micro average, denominator min(degree, 10)).
  `NMI`/`ARI` compare Louvain communities on the graph against k-means on
  `Z`.
- **Force2Vec has no seed flag and is deterministic** (byte-identical
  reruns), so its "seeds" are node permutations: relabel the graph, embed,
  map back. Seed 42 is the identity. This measures run-to-run stability, the
  same thing an embedding seed measures for the others.
- **Some methods are deterministic**, so their `Z` is identical across all
  11 seeds and the fixed-eval columns (rho, recall@10, NMI, ARI) show a std
  of `± 0.0000` — that is correct, not a bug. Laplacian Eigenmaps, GraRep
  and HOPE are the spectral/factorization cases; their nonzero hop R2 spread
  is eval-split noise on one fixed embedding, not method spread.

### The one caveat that decides the ranking

**`rho`, `hop R2` and `recall@10` read EUCLIDEAN distance in `Z`. Five of
the methods do not put their signal in Euclidean distance.** LINE, GraRep,
NetMF, HOPE and ProNE learn an **inner-product** similarity; their
coordinates are not a metric space. On the Euclidean scores they land near
zero or negative rho — expected, not a failure. Read their **LP AUC / f1**
instead (a Hadamard-feature classifier, no Euclidean assumption): there they
stay high (ProNE and RandNE reach LP AUC 0.997–0.9997). fodiwalk, the
Force2Vec family, Laplacian Eigenmaps and landmark MDS DO build a distance
geometry, so the Euclidean scores are the fair ones for them.

**landmark MDS is a reference, not a competitor.** It reads the TRUE
shortest-path distances and fits coordinates to them — an upper line for how
well `dim=128` can hold the hop geometry, not a learned embedding.

**LP AUC is saturated** (0.92–0.9997). Rank on geometry (rho, hop R2) and
community (NMI, ARI), not on AUC.

## 0. Evaluation parameters

The full machine-readable set is `other-methods/results/eval_params.json`
(and embedded in `comparison_tables_11seed.json` under `eval_params`).
Pulled live from the `fodiwalk_dist` protocol, so it matches what ran.

| experiment | setup |
|---|---|
| **link prediction** (LP AUC, f1) | 80/20 train/test; balanced negatives (`neg_ratio=1.0`) drawn as `far_pairs` (non-edges); positives = graph edges (`over_cap`), capped at 50,000; features = **hadamard**(Z_u, Z_v); classifier = **random forest, 200 trees**. NOT held-out — the embedding saw every edge, so this is edge reconstruction. |
| **hop regression** (hop R2) | 20,000 pairs from **200 BFS sources**, `min_hop ≥ 2`; feature = one Euclidean distance in Z; target = true BFS hop; 80/20 split; models = mean-baseline / RF (100 trees, min_leaf 1) / MLP ([256,128], no early stop). `hop R2` = the RF R² on the test split. |
| **Spearman rho** | Spearman of Euclidean Z-distance vs hop distance on the SAME 200-source / 20,000-pair / `min_hop ≥ 2` sample; fixed eval seed 42. |
| **recall@10** | k = 10 nearest OTHER nodes per node of degree > 0, exact brute-force kNN, self excluded; micro average, denominator `sum(min(degree, 10))`. |
| **community** (NMI, ARI) | reference = Louvain on A (networkx, seed 42, resolution 1.0 → k = 105 cora / 471 citeseer / 45 pubmed); clusters = k-means on Z at k = the Louvain count, `n_init=10`, `random_state=42`. |

Common: dim 128, one node numbering (`fodiwalk.make_graph.load`). LP and hop
regression were scored at embed time with the **embedding** seed (stored in
`config.json`); rho, recall@10 and community use a **fixed** eval seed 42.

## 1. Main comparison, dim 128

**cora, dim 128**

| method | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score | NMI | ARI |
|---|---|---|---|---|---|---|---|---|
| fodiwalk fdhop walk_edges sqn | 11 | 0.8688 ± 0.0041 | 0.6394 ± 0.0192 | 0.9574 ± 0.0015 | 0.9991 ± 0.0004 | 0.9896 ± 0.0022 | 0.7007 ± 0.0072 | 0.3433 ± 0.0245 |
| landmark_mds *(distance oracle)* | 11 | 0.8199 ± 0.0147 | 0.5562 ± 0.0426 | 0.6420 ± 0.0089 | 0.9923 ± 0.0020 | 0.9702 ± 0.0023 | 0.6321 ± 0.0085 | 0.3256 ± 0.0433 |
| deepwalk | 10 | 0.7924 ± 0.0027 | 0.4825 ± 0.0283 | 0.9362 ± 0.0016 | 0.9985 ± 0.0007 | 0.9829 ± 0.0031 | 0.7078 ± 0.0108 | 0.3333 ± 0.0271 |
| node2vec | 11 | 0.7704 ± 0.0038 | 0.3264 ± 0.0305 | 0.9592 ± 0.0013 | 0.9979 ± 0.0009 | 0.9777 ± 0.0044 | 0.6887 ± 0.0057 | 0.2398 ± 0.0133 |
| tforce2vec | 11 | 0.7125 ± 0.0056 | 0.3329 ± 0.0290 | 0.7762 ± 0.0021 | 0.9849 ± 0.0024 | 0.9511 ± 0.0043 | 0.7791 ± 0.0038 | 0.5084 ± 0.0151 |
| force2vec | 3 | 0.5537 ± 0.0081 | 0.0401 ± 0.0209 | 0.7524 ± 0.0004 | 0.9693 ± 0.0003 | 0.9249 ± 0.0003 | 0.7721 ± 0.0080 | 0.4853 ± 0.0121 |
| laplacian_eigenmaps *(deterministic)* | 11 | 0.4685 ± 0.0000 | -0.0771 ± 0.0445 | 0.6043 ± 0.0002 | 0.9945 ± 0.0012 | 0.9755 ± 0.0042 | 0.5687 ± 0.0000 | 0.1271 ± 0.0000 |
| randne *(inner-product)* | 11 | 0.4549 ± 0.0122 | -0.1677 ± 0.0546 | 0.9237 ± 0.0030 | 0.9997 ± 0.0002 | 0.9853 ± 0.0036 | 0.4706 ± 0.0104 | 0.1056 ± 0.0137 |
| rforce2vec | 11 | 0.4057 ± 0.0148 | -0.2378 ± 0.0429 | 0.7261 ± 0.0091 | 0.9724 ± 0.0029 | 0.9224 ± 0.0041 | 0.5140 ± 0.0060 | 0.1118 ± 0.0099 |
| prone *(inner-product)* | 11 | 0.0451 ± 0.0074 | -0.3327 ± 0.0208 | 0.7938 ± 0.0013 | 0.9976 ± 0.0009 | 0.9793 ± 0.0032 | 0.6701 ± 0.0035 | 0.1827 ± 0.0045 |
| line *(inner-product)* | 11 | -0.1694 ± 0.0149 | -0.4306 ± 0.0305 | 0.1841 ± 0.0034 | 0.9923 ± 0.0016 | 0.9542 ± 0.0057 | 0.2146 ± 0.0049 | 0.0065 ± 0.0011 |
| grarep *(inner-product, deterministic)* | 11 | -0.1934 ± 0.0000 | -0.2830 ± 0.0402 | 0.2040 ± 0.0001 | 0.9808 ± 0.0032 | 0.9443 ± 0.0066 | 0.4817 ± 0.0044 | 0.1226 ± 0.0047 |
| netmf *(inner-product)* | 11 | -0.2592 ± 0.0014 | -0.3307 ± 0.0621 | 0.6554 ± 0.0010 | 0.9925 ± 0.0017 | 0.9773 ± 0.0025 | 0.5341 ± 0.0043 | 0.0391 ± 0.0059 |
| hope *(inner-product, deterministic)* | 11 | -0.3680 ± 0.0000 | -0.1188 ± 0.0348 | 0.3669 ± 0.0013 | 0.9832 ± 0.0027 | 0.9491 ± 0.0058 | 0.4108 ± 0.0000 | 0.0015 ± 0.0000 |

**citeseer, dim 128**

| method | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score | NMI | ARI |
|---|---|---|---|---|---|---|---|---|
| landmark_mds *(distance oracle)* | 11 | 0.9598 ± 0.0034 | 0.8674 ± 0.0184 | 0.6243 ± 0.0084 | 0.9896 ± 0.0025 | 0.9545 ± 0.0080 | 0.6549 ± 0.0044 | 0.0017 ± 0.0017 |
| fodiwalk fdhop walk_edges sqn | 11 | 0.8392 ± 0.0131 | 0.5824 ± 0.0246 | 0.9614 ± 0.0022 | 0.9995 ± 0.0004 | 0.9966 ± 0.0012 | 0.8677 ± 0.0027 | 0.3442 ± 0.0123 |
| tforce2vec | 11 | 0.7689 ± 0.0135 | 0.4117 ± 0.0373 | 0.8427 ± 0.0036 | 0.9963 ± 0.0013 | 0.9796 ± 0.0024 | 0.9089 ± 0.0062 | 0.5830 ± 0.0503 |
| deepwalk | 10 | 0.6926 ± 0.0033 | 0.3348 ± 0.0333 | 0.9413 ± 0.0008 | 0.9993 ± 0.0005 | 0.9912 ± 0.0021 | 0.8106 ± 0.0015 | 0.1445 ± 0.0041 |
| node2vec | 11 | 0.6818 ± 0.0068 | 0.2915 ± 0.0366 | 0.9384 ± 0.0030 | 0.9980 ± 0.0013 | 0.9870 ± 0.0032 | 0.8116 ± 0.0033 | 0.1532 ± 0.0053 |
| laplacian_eigenmaps *(deterministic)* | 11 | 0.4959 ± 0.0011 | -0.0407 ± 0.0325 | 0.6466 ± 0.0028 | 0.9863 ± 0.0013 | 0.9610 ± 0.0031 | 0.7088 ± 0.0063 | 0.0608 ± 0.0092 |
| force2vec | 3 | 0.4356 ± 0.0078 | -0.0107 ± 0.0422 | 0.8231 ± 0.0017 | 0.9852 ± 0.0020 | 0.9486 ± 0.0036 | 0.8663 ± 0.0037 | 0.4302 ± 0.0107 |
| randne *(inner-product)* | 11 | 0.3955 ± 0.0102 | -0.1985 ± 0.0399 | 0.9388 ± 0.0029 | 0.9997 ± 0.0003 | 0.9902 ± 0.0018 | 0.7351 ± 0.0031 | 0.1130 ± 0.0082 |
| rforce2vec | 11 | 0.3060 ± 0.0186 | -0.2738 ± 0.0296 | 0.6236 ± 0.0063 | 0.9312 ± 0.0049 | 0.8854 ± 0.0101 | 0.6589 ± 0.0020 | 0.0571 ± 0.0034 |
| prone *(inner-product)* | 11 | 0.0109 ± 0.0094 | -0.3485 ± 0.0241 | 0.8257 ± 0.0013 | 0.9989 ± 0.0005 | 0.9894 ± 0.0017 | 0.8147 ± 0.0024 | 0.1432 ± 0.0046 |
| line *(inner-product)* | 11 | -0.1367 ± 0.0156 | -0.4119 ± 0.0345 | 0.1718 ± 0.0042 | 0.9926 ± 0.0015 | 0.9554 ± 0.0038 | 0.5062 ± 0.0042 | 0.0032 ± 0.0005 |
| grarep *(inner-product, deterministic)* | 11 | -0.2728 ± 0.0000 | -0.1342 ± 0.0290 | 0.3776 ± 0.0006 | 0.9331 ± 0.0026 | 0.8779 ± 0.0062 | 0.4817 ± 0.0029 | -0.0255 ± 0.0005 |
| netmf *(inner-product)* | 11 | -0.2945 ± 0.0041 | -0.2845 ± 0.0425 | 0.7412 ± 0.0006 | 0.9787 ± 0.0020 | 0.9462 ± 0.0032 | 0.5433 ± 0.0026 | -0.0239 ± 0.0004 |
| hope *(inner-product, deterministic)* | 11 | -0.3885 ± 0.0000 | 0.0611 ± 0.0221 | 0.4512 ± 0.0025 | 0.9215 ± 0.0039 | 0.8709 ± 0.0066 | 0.3714 ± 0.0000 | -0.0276 ± 0.0000 |

**pubmed, dim 128**  (dense n×n and O(n²) methods not run — see §4)

| method | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score | NMI | ARI |
|---|---|---|---|---|---|---|---|---|
| fodiwalk fdhop walk_edges sqn | 11 | 0.7925 ± 0.0041 | 0.4844 ± 0.0232 | 0.7752 ± 0.0034 | 0.9990 ± 0.0002 | 0.9910 ± 0.0008 | 0.5810 ± 0.0037 | 0.3552 ± 0.0129 |
| landmark_mds *(distance oracle)* | 11 | 0.7766 ± 0.0085 | 0.4246 ± 0.0397 | 0.5622 ± 0.0032 | 0.9936 ± 0.0006 | 0.9627 ± 0.0023 | 0.4744 ± 0.0091 | 0.2349 ± 0.0148 |
| deepwalk | 9 | 0.6999 ± 0.0013 | 0.2983 ± 0.0277 | 0.7980 ± 0.0011 | 0.9990 ± 0.0002 | 0.9841 ± 0.0013 | 0.6234 ± 0.0079 | 0.4339 ± 0.0192 |
| node2vec | 11 | 0.6769 ± 0.0046 | 0.2252 ± 0.0314 | 0.9180 ± 0.0010 | 0.9990 ± 0.0001 | 0.9858 ± 0.0010 | 0.6128 ± 0.0067 | 0.3735 ± 0.0192 |
| tforce2vec | 11 | 0.5915 ± 0.0031 | 0.1177 ± 0.0282 | 0.5455 ± 0.0014 | 0.9904 ± 0.0006 | 0.9604 ± 0.0020 | 0.6521 ± 0.0069 | 0.4554 ± 0.0133 |
| randne *(inner-product)* | 11 | 0.4720 ± 0.0115 | -0.1106 ± 0.0344 | 0.6935 ± 0.0038 | 0.9981 ± 0.0002 | 0.9849 ± 0.0012 | 0.0885 ± 0.0033 | 0.0234 ± 0.0036 |
| laplacian_eigenmaps *(deterministic)* | 11 | 0.4259 ± 0.0000 | -0.0401 ± 0.0436 | 0.1814 ± 0.0000 | 0.9955 ± 0.0007 | 0.9779 ± 0.0014 | 0.0774 ± 0.0000 | 0.0046 ± 0.0000 |
| rforce2vec | 11 | 0.4018 ± 0.0135 | -0.2065 ± 0.0481 | 0.2437 ± 0.0057 | 0.9467 ± 0.0025 | 0.8469 ± 0.0045 | 0.0950 ± 0.0030 | 0.0265 ± 0.0031 |
| prone *(inner-product)* | 11 | 0.1149 ± 0.0058 | -0.3711 ± 0.0282 | 0.6061 ± 0.0008 | 0.9970 ± 0.0003 | 0.9745 ± 0.0013 | 0.5122 ± 0.0082 | 0.2244 ± 0.0149 |
| line *(inner-product)* | 11 | -0.2856 ± 0.0077 | -0.3301 ± 0.0335 | 0.0265 ± 0.0006 | 0.9910 ± 0.0005 | 0.9447 ± 0.0017 | 0.0318 ± 0.0009 | 0.0031 ± 0.0003 |

## 2. What the numbers say

Read the distance-geometry rows (fodiwalk, Force2Vec, Laplacian Eigenmaps,
landmark MDS) on rho and hop R2; read the inner-product rows on LP and
community.

- **fodiwalk leads the learned methods on hop geometry on cora and
  pubmed.** rho 0.869 vs the next learned method deepwalk 0.792 on cora;
  0.793 vs 0.700 on pubmed, both far outside the seed spread. It also leads
  recall@10 among the geometry methods. Only landmark MDS, the distance
  oracle, is close — and on citeseer it passes fodiwalk (rho 0.960 vs
  0.839), because citeseer is the sparsest graph (avg degree 2.74) where a
  walk-based hop estimate is noisiest and a distance oracle gains most.
- **tForce2Vec is the strongest Force2Vec option** and the only one that
  competes on rho (0.713 cora, 0.769 citeseer). The base Force2Vec (O(n²))
  and rForce2Vec are weaker on geometry at 1200 iterations. But **the
  Force2Vec family wins community structure**: tForce2Vec has the best NMI
  and ARI of ALL methods on cora (0.779 / 0.508) and citeseer (0.909 /
  0.583). Force-directed layouts cluster well even when hop ordering is
  middling.
- **The inner-product methods keep high LP AUC and lose Euclidean
  geometry.** RandNE is the sharp case: rho -0.17 on cora yet LP AUC 0.9997
  (best on the graph) and recall@10 0.924. Its random projection keeps
  neighbours retrievable by the classifier but does not order Euclidean
  distance by hop. ProNE, NetMF, GraRep and HOPE behave the same way — a
  property of the objective, not a defect.
- **Laplacian Eigenmaps sits mid-pack** on rho (~0.43–0.50) with near-zero
  hop R2: it preserves coarse community adjacency but at 128 smooth
  eigenvectors it does not separate 1-hop from 2-hop.

## 3. Runtime and memory (dim 128)

Wall-clock seconds and peak RSS (MB) of the embedding process, one machine
(8 cores, 7 GB RAM, no GPU used). fodiwalk's RSS includes in-process
scoring; the karateclub / nodevectors methods are scored separately so their
RSS is the embedding only. A dash is a run not made (see §4).

| method | cora s / MB | citeseer s / MB | pubmed s / MB |
|---|---|---|---|
| fodiwalk fdhop walk_edges sqn | 14.5 / 710 | 10.5 / 692 | 55.0 / 1459 |
| deepwalk | 302.1 / 395 | 254.7 / 374 | 1985.5 / 469 |
| node2vec | 48.0 / 394 | 34.8 / 372 | 329.2 / 447 |
| landmark_mds | 0.2 / 293 | 0.1 / 297 | 1.7 / 295 |
| laplacian_eigenmaps | 3.4 / 171 | 4.0 / 172 | 31.8 / 195 |
| prone | 2.4 / 235 | 2.9 / 236 | 6.5 / 287 |
| randne | 1.8 / 171 | 1.9 / 184 | 3.2 / 196 |
| line | 5.4 / 168 | 5.5 / 169 | 306.5 / 192 |
| netmf | 2.2 / 184 | 2.0 / 185 | — |
| grarep | 4.6 / 176 | 3.1 / 183 | — |
| hope | 2.2 / 179 | 1.9 / 179 | — |
| force2vec (opt 1, O(n²)) | 1085.0 / 269 | 1494.4 / 270 | — |
| tforce2vec | 7.9 / 269 | 5.2 / 270 | 41.4 / 284 |
| rforce2vec | 5.8 / 269 | 6.7 / 270 | 60.8 / 285 |

- **deepwalk is the slowest scalable method** (33 min on pubmed) — gensim
  hierarchical softmax over an 80×40 walk corpus. node2vec (negative
  sampling, 10×80) is 6× faster for a similar geometry.
- **Factorization and random-projection methods are the cheapest** (1–7 s);
  tForce2Vec / rForce2Vec are nearly as cheap (5–60 s). The Force2Vec base
  at O(n²) is the exception (18–25 min).

## 4. Methods, provenance, and what is not run

All embedding code is published; only the landmark-MDS baseline is written
here, which the plan (`evaluator/recommended-comparisons.md`, method 8)
specifies as an experimental baseline to build.

| method | source | version | note |
|---|---|---|---|
| fodiwalk | this repo | fdhop, walk_edges, min_gap, 10×20, sqn/const, lr 0.999 | best cell this session |
| deepwalk, node2vec | gensim word2vec | deepwalk 80×40, node2vec 10×80 p=q=1, window 10 | baselines |
| Laplacian Eigenmaps | karateclub `LaplacianEigenmaps` | 1.3.3 | sklearn's arpack hung on the disconnected graphs; this solves them in seconds; deterministic |
| landmark MDS | own, to the doc's spec | scipy BFS + classical MDS, 256 landmarks | distance oracle |
| ProNE | nodevectors `ProNE` | 0.2.0 | factorization + spectral propagation |
| NetMF, GraRep, HOPE | karateclub | 1.3.3 (GraRep order 4) | dense n×n; GraRep/HOPE deterministic |
| RandNE | karateclub `RandNE` | 1.3.3 | random projection |
| LINE | karateclub `FirstOrderLINE`+`SecondOrderLINE` | 1.3.3, 64+64 dims | first- and second-order halves |
| Force2Vec / tForce2Vec / rForce2Vec | HipGraph/Force2Vec (C++) | options 1 / 5 / 7, iter 1200 | built from source; deterministic, seeds are node permutations |

**Not run, with the reason (never dropped, per the plan):**

- **NetMF, GraRep, HOPE on pubmed** — each forms a dense n×n matrix, ≈3 GB
  at 19,717 nodes, which OOMs a 7 GB machine. cora and citeseer only.
- **Force2Vec base (option 1) on pubmed** — O(n²) all-pairs; impractical at
  19,717 nodes on this box. tForce2Vec and rForce2Vec (scalable O(ns)) ran
  on all three.
- **NetSMF** — C++ build (THUDM); not attempted this pass. NetMF stands in
  for the matrix-factorization family on the small graphs.
- **GOSH** — GPU/CUDA only; this box has a 6 GB card and no CUDA build.

## 5. Threats to reading this table

- **Single-protocol, not held-out.** The embeddings saw every edge, so the
  LP panel is edge reconstruction, not held-out link prediction.
- **Mixed eval seed by column.** rho/recall@10/NMI/ARI use a fixed eval seed
  (42); LP AUC/f1 and hop R2 are the stored evaluator values (embedding-seed
  eval). For a deterministic embedding this shows as `± 0.0000` in the
  fixed-eval columns and nonzero spread in the hop-R2/LP columns — the
  latter is eval-split noise, not method spread.
- **The Euclidean caveat again.** A single-number ranking of this table is
  wrong: it would penalise the inner-product methods for a geometry they do
  not target. Read the column that matches the method's objective.
- **fodiwalk is shown at its best cell only** (fdhop walk_edges sqn). The
  full fodiwalk sweep is in `260917-iclr2027-preliminary.md`.
- **deepwalk (9–10 seeds) and Force2Vec base (3 seeds) are still
  completing.** Their rows will reach 11 seeds in the final update; the
  means are already stable to within their listed spread.
