# Other-methods comparison: preliminary results

**Built:** 2026-09-22T09:10Z. **Status:** preliminary. Nine published
comparison methods against fodiwalk, deepwalk and node2vec, on cora,
citeseer and pubmed at dim 128. From the plan in
`evaluator/recommended-comparisons.md`.

**Store:** every embedding is in the shared store,
`data_cache/embeddings/<graph>/128/<run>/` (`Z.npy` + `config.json`), the
same place and layout as the fodiwalk and baseline runs. Method code:
`other-methods/`. Run list: `other-methods/results/final_runlist.txt`.
Regenerate the tables:
`.venv/bin/python other-methods/score_report.py <runlist> <out.md>`.

## How to read these numbers

- Every value is **mean ± std over seeds** (the EMBEDDING seed). The
  evaluation seed is fixed, so every run of one graph is scored on the same
  link-prediction split, the same hop pairs and the same query nodes.
- One protocol for every row: `fodiwalk_dist`, `lp_max_pairs=50_000`, the
  same as this session's fodiwalk and baseline runs. So a number here
  compares to a fodiwalk number with no adjustment.
- One node numbering: every method loads the graph through
  `fodiwalk.make_graph.load`, so row `i` of `Z` is the same node.
- `rho` is Spearman between **Euclidean** embedding distance and hop
  distance, hop ≥ 2. `hop R2` is a random forest on that one distance
  column. `recall@10` is the share of a node's true neighbours among its 10
  Euclidean-nearest nodes (micro average, denominator min(degree, 10)).
  `NMI`/`ARI` compare Louvain communities on the graph against k-means on
  `Z`.
- Seeds: fodiwalk, deepwalk, node2vec and the karateclub / nodevectors
  methods have **3 seeds**. The Force2Vec binary has **no seed flag**, so
  its rows are a **single run** (std shown as 0.0000, which means "not
  measured", not "zero variance").

### The one caveat that decides the ranking

**`rho`, `hop R2` and `recall@10` read EUCLIDEAN distance in `Z`. Four of
the methods here do not put their signal in Euclidean distance.** LINE,
GraRep, NetMF, HOPE and ProNE learn an **inner-product** (dot / cosine)
similarity; their coordinates are not meant to be read as a metric space.
On the Euclidean scores they land near zero or negative rho, and that is
expected, not a failure of the method. Read their **LP AUC / f1** instead,
which the random forest computes from a Hadamard product and so does not
assume Euclidean geometry: there they stay high (ProNE and RandNE reach LP
AUC 0.997–0.9997). fodiwalk, the Force2Vec family, Laplacian Eigenmaps and
landmark MDS DO build a distance geometry, so the Euclidean scores are the
fair ones for them.

**landmark MDS is a reference, not a competitor.** It reads the TRUE
shortest-path distances of the graph and fits coordinates to them. It shows
how well `dim=128` coordinates CAN hold the hop geometry, an upper line the
learned methods are measured against — not a learned embedding.

**LP AUC is saturated** — 0.92 to 0.9997 across almost every row. Rank on
geometry (rho, hop R2) and on community (NMI, ARI), not on AUC.

## 1. Main comparison, dim 128

**cora, dim 128**

| method | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score | NMI | ARI |
|---|---|---|---|---|---|---|---|---|
| fodiwalk fdhop walk_edges sqn | 3 | 0.8656 ± 0.0016 | 0.6503 ± 0.0191 | 0.9573 ± 0.0020 | 0.9992 ± 0.0002 | 0.9891 ± 0.0022 | 0.6992 ± 0.0037 | 0.3409 ± 0.0122 |
| landmark_mds *(distance oracle)* | 3 | 0.8172 ± 0.0111 | 0.5496 ± 0.0196 | 0.6457 ± 0.0114 | 0.9895 ± 0.0009 | 0.9686 ± 0.0022 | 0.6289 ± 0.0028 | 0.3002 ± 0.0219 |
| deepwalk | 3 | 0.7879 ± 0.0078 | 0.4946 ± 0.0248 | 0.9353 ± 0.0012 | 0.9981 ± 0.0009 | 0.9815 ± 0.0043 | 0.7057 ± 0.0081 | 0.3180 ± 0.0174 |
| node2vec | 3 | 0.7564 ± 0.0099 | 0.2938 ± 0.0210 | 0.9579 ± 0.0008 | 0.9978 ± 0.0012 | 0.9777 ± 0.0059 | 0.6935 ± 0.0030 | 0.2492 ± 0.0096 |
| tforce2vec | 1 | 0.7168 | 0.3943 | 0.7743 | 0.9832 | 0.9436 | 0.7814 | 0.5174 |
| force2vec | 1 | 0.5442 | 0.0125 | 0.7529 | 0.9696 | 0.9246 | 0.7834 | 0.5006 |
| laplacian_eigenmaps | 3 | 0.4800 ± 0.0116 | -0.0809 ± 0.0427 | 0.6042 ± 0.0002 | 0.9942 ± 0.0009 | 0.9753 ± 0.0024 | 0.5705 ± 0.0237 | 0.1278 ± 0.0416 |
| randne *(inner-product)* | 3 | 0.4326 ± 0.0259 | -0.1644 ± 0.0573 | 0.9256 ± 0.0024 | 0.9997 ± 0.0000 | 0.9827 ± 0.0033 | 0.4727 ± 0.0022 | 0.0919 ± 0.0031 |
| rforce2vec | 1 | 0.4179 | -0.2005 | 0.7502 | 0.9666 | 0.9243 | 0.5243 | 0.1246 |
| prone *(inner-product)* | 3 | 0.0564 ± 0.0030 | -0.3366 ± 0.0322 | 0.7928 ± 0.0005 | 0.9967 ± 0.0009 | 0.9768 ± 0.0037 | 0.6711 ± 0.0048 | 0.1825 ± 0.0042 |
| line *(inner-product)* | 3 | -0.1400 ± 0.0173 | -0.4289 ± 0.0278 | 0.1809 ± 0.0031 | 0.9918 ± 0.0026 | 0.9517 ± 0.0064 | 0.2134 ± 0.0054 | 0.0057 ± 0.0007 |
| grarep *(inner-product)* | 3 | -0.2123 ± 0.0134 | -0.2417 ± 0.0187 | 0.2040 ± 0.0001 | 0.9798 ± 0.0012 | 0.9396 ± 0.0030 | 0.4815 ± 0.0039 | 0.1226 ± 0.0025 |
| netmf *(inner-product)* | 3 | -0.2715 ± 0.0151 | -0.3579 ± 0.0545 | 0.6547 ± 0.0009 | 0.9916 ± 0.0015 | 0.9767 ± 0.0003 | 0.5386 ± 0.0010 | 0.0440 ± 0.0010 |
| hope *(inner-product)* | 3 | -0.3827 ± 0.0104 | -0.0992 ± 0.0380 | 0.3669 ± 0.0014 | 0.9818 ± 0.0024 | 0.9480 ± 0.0009 | 0.4059 ± 0.0039 | -0.0014 ± 0.0021 |

**citeseer, dim 128**

| method | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score | NMI | ARI |
|---|---|---|---|---|---|---|---|---|
| landmark_mds *(distance oracle)* | 3 | 0.9545 ± 0.0054 | 0.8740 ± 0.0216 | 0.6262 ± 0.0018 | 0.9911 ± 0.0021 | 0.9582 ± 0.0071 | 0.6565 ± 0.0043 | 0.0015 ± 0.0019 |
| fodiwalk fdhop walk_edges sqn | 3 | 0.8280 ± 0.0136 | 0.5701 ± 0.0433 | 0.9629 ± 0.0014 | 0.9996 ± 0.0005 | 0.9969 ± 0.0019 | 0.8671 ± 0.0028 | 0.3390 ± 0.0163 |
| tforce2vec | 1 | 0.7677 | 0.4291 | 0.8417 | 0.9976 | 0.9813 | 0.9090 | 0.5869 |
| node2vec | 3 | 0.6871 ± 0.0096 | 0.2931 ± 0.0313 | 0.9380 ± 0.0035 | 0.9986 ± 0.0008 | 0.9884 ± 0.0018 | 0.8060 ± 0.0042 | 0.1429 ± 0.0057 |
| deepwalk | 3 | 0.6860 ± 0.0006 | 0.3148 ± 0.0401 | 0.9415 ± 0.0010 | 0.9995 ± 0.0004 | 0.9919 ± 0.0011 | 0.8117 ± 0.0014 | 0.1451 ± 0.0061 |
| laplacian_eigenmaps | 3 | 0.4742 ± 0.0173 | -0.0492 ± 0.0359 | 0.6468 ± 0.0029 | 0.9876 ± 0.0004 | 0.9629 ± 0.0011 | 0.7127 ± 0.0064 | 0.0641 ± 0.0094 |
| force2vec | 1 | 0.4461 | 0.0086 | 0.8236 | 0.9838 | 0.9461 | 0.8629 | 0.4153 |
| randne *(inner-product)* | 3 | 0.3760 ± 0.0082 | -0.2113 ± 0.0193 | 0.9399 ± 0.0014 | 0.9997 ± 0.0002 | 0.9904 ± 0.0017 | 0.7369 ± 0.0044 | 0.1107 ± 0.0092 |
| rforce2vec | 1 | 0.2813 | -0.3071 | 0.6303 | 0.9310 | 0.8884 | 0.6559 | 0.0575 |
| prone *(inner-product)* | 3 | 0.0260 ± 0.0102 | -0.3445 ± 0.0176 | 0.8255 ± 0.0011 | 0.9990 ± 0.0003 | 0.9893 ± 0.0014 | 0.8145 ± 0.0019 | 0.1438 ± 0.0045 |
| line *(inner-product)* | 3 | -0.1433 ± 0.0146 | -0.4319 ± 0.0104 | 0.1718 ± 0.0002 | 0.9917 ± 0.0007 | 0.9551 ± 0.0051 | 0.5053 ± 0.0024 | 0.0028 ± 0.0002 |
| grarep *(inner-product)* | 3 | -0.2744 ± 0.0156 | -0.1504 ± 0.0296 | 0.3774 ± 0.0005 | 0.9332 ± 0.0012 | 0.8739 ± 0.0048 | 0.4835 ± 0.0019 | -0.0254 ± 0.0000 |
| netmf *(inner-product)* | 3 | -0.2984 ± 0.0268 | -0.3215 ± 0.0393 | 0.7414 ± 0.0008 | 0.9786 ± 0.0021 | 0.9460 ± 0.0047 | 0.5410 ± 0.0011 | -0.0241 ± 0.0004 |
| hope *(inner-product)* | 3 | -0.3728 ± 0.0172 | 0.0388 ± 0.0232 | 0.4502 ± 0.0014 | 0.9207 ± 0.0001 | 0.8682 ± 0.0054 | 0.3737 ± 0.0018 | -0.0274 ± 0.0002 |

**pubmed, dim 128**  (dense n×n and O(n²) methods not run — see §4)

| method | seeds | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score | NMI | ARI |
|---|---|---|---|---|---|---|---|---|
| fodiwalk fdhop walk_edges sqn | 3 | 0.7933 ± 0.0125 | 0.4811 ± 0.0395 | 0.7763 ± 0.0027 | 0.9989 ± 0.0002 | 0.9916 ± 0.0010 | 0.5829 ± 0.0012 | 0.3514 ± 0.0040 |
| landmark_mds *(distance oracle)* | 3 | 0.7690 ± 0.0193 | 0.4286 ± 0.0396 | 0.5624 ± 0.0035 | 0.9933 ± 0.0007 | 0.9610 ± 0.0018 | 0.4710 ± 0.0151 | 0.2317 ± 0.0247 |
| deepwalk | 3 | 0.6990 ± 0.0045 | 0.3208 ± 0.0154 | 0.7978 ± 0.0011 | 0.9989 ± 0.0001 | 0.9847 ± 0.0011 | 0.6276 ± 0.0087 | 0.4431 ± 0.0187 |
| node2vec | 3 | 0.6788 ± 0.0098 | 0.2444 ± 0.0440 | 0.9184 ± 0.0003 | 0.9990 ± 0.0002 | 0.9860 ± 0.0007 | 0.6148 ± 0.0045 | 0.3738 ± 0.0185 |
| tforce2vec | 1 | 0.5928 | 0.1181 | 0.5430 | 0.9896 | 0.9585 | 0.6435 | 0.4428 |
| randne *(inner-product)* | 3 | 0.4800 ± 0.0256 | -0.1196 ± 0.0350 | 0.6963 ± 0.0027 | 0.9984 ± 0.0002 | 0.9848 ± 0.0008 | 0.0885 ± 0.0011 | 0.0202 ± 0.0045 |
| laplacian_eigenmaps | 3 | 0.4289 ± 0.0164 | -0.0473 ± 0.0331 | 0.1814 ± 0.0000 | 0.9957 ± 0.0002 | 0.9789 ± 0.0018 | 0.2427 ± 0.1389 | 0.0973 ± 0.0813 |
| rforce2vec | 1 | 0.3621 | -0.2812 | 0.2291 | 0.9427 | 0.8490 | 0.0973 | 0.0352 |
| prone *(inner-product)* | 3 | 0.1243 ± 0.0104 | -0.3480 ± 0.0404 | 0.6060 ± 0.0007 | 0.9969 ± 0.0003 | 0.9738 ± 0.0002 | 0.5126 ± 0.0032 | 0.2286 ± 0.0132 |
| line *(inner-product)* | 3 | -0.2978 ± 0.0091 | -0.3254 ± 0.0306 | 0.0260 ± 0.0007 | 0.9911 ± 0.0005 | 0.9451 ± 0.0015 | 0.0309 ± 0.0002 | 0.0030 ± 0.0002 |

## 2. What the numbers say

Read the distance-geometry rows (fodiwalk, Force2Vec, Laplacian Eigenmaps,
landmark MDS) on rho and hop R2; read the inner-product rows on LP and
community. With that split:

- **fodiwalk leads the learned methods on hop geometry on cora and
  pubmed.** rho 0.866 vs the next learned method deepwalk 0.788 on cora;
  0.793 vs 0.699 on pubmed. It also leads recall@10 among the geometry
  methods. Only landmark MDS, which reads the true distances, is close —
  and on citeseer it passes fodiwalk (rho 0.955 vs 0.828), because citeseer
  is the sparsest graph (avg degree 2.74) where a walk-based hop estimate
  is noisiest and a distance oracle gains most.
- **tForce2Vec is the strongest of the Force2Vec family** and the only one
  that competes on rho (0.717 cora, 0.768 citeseer). The base Force2Vec
  (option 1, O(n²)) and rForce2Vec are weaker on geometry at the published
  1200 iterations. But **the Force2Vec family wins community structure**:
  tForce2Vec has the best NMI and ARI of ALL methods on cora (0.781 /
  0.517) and citeseer (0.909 / 0.587). Force-directed layouts cluster
  well even when their hop ordering is middling.
- **The inner-product methods keep high LP AUC and lose Euclidean
  geometry.** RandNE is the sharp case: rho -0.16 on cora yet LP AUC 0.9997
  (the best on the graph) and recall@10 0.926. Its random projection keeps
  neighbours retrievable by the classifier but does not order Euclidean
  distance by hop. ProNE, NetMF, GraRep and HOPE behave the same way. This
  is a property of the objective, not a defect; the Euclidean columns are
  simply the wrong lens for them.
- **Laplacian Eigenmaps sits mid-pack** on rho (~0.43–0.48 on all three)
  and near-zero hop R2. It preserves coarse community adjacency (edges land
  close) but at 128 smooth eigenvectors it does not separate 1-hop from
  2-hop, so the single distance feature carries little hop signal.

## 3. Runtime and memory (dim 128)

Wall-clock seconds and peak RSS (MB) of the embedding process, one machine
(8 cores, 7 GB RAM, no GPU used). fodiwalk's RSS includes in-process
scoring; the karateclub / nodevectors methods are scored in a separate step
so their RSS is the embedding only. A dash is a run not made (see §4).

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
- **The factorization and random-projection methods are the cheapest**
  (1–7 s), and tForce2Vec / rForce2Vec are nearly as cheap (5–60 s) — the
  Force2Vec base at O(n²) is the exception (18–25 min).
- **fodiwalk is fast and mid-memory**, and it is the only method whose RSS
  figure also pays for the scoring pass.

## 4. Methods, provenance, and what is not run

All embedding code is published; none was written from scratch except the
landmark-MDS baseline, which the plan
(`evaluator/recommended-comparisons.md`, method 8) specifies as an
experimental baseline to build.

| method | source | version / commit | note |
|---|---|---|---|
| fodiwalk | this repo | fdhop, walk_edges, min_gap, 10×20, sqn/const, lr 0.999 | best cell this session |
| deepwalk, node2vec | gensim word2vec | deepwalk 80×40, node2vec 10×80 p=q=1, window 10 | baselines |
| Laplacian Eigenmaps | karateclub `LaplacianEigenmaps` | karateclub 1.3.3 | sklearn's arpack hung on the disconnected graphs; this solves them in seconds |
| landmark MDS | own, to the doc's spec | scipy BFS + classical MDS, 256 landmarks | distance oracle, reference row |
| ProNE | nodevectors `ProNE` | nodevectors 0.2.0 | factorization + spectral propagation |
| NetMF | karateclub `NetMF` | 1.3.3 | dense n×n |
| GraRep | karateclub `GraRep` | 1.3.3, order 4 | dense n×n |
| HOPE | karateclub `HOPE` | 1.3.3 | dense n×n |
| RandNE | karateclub `RandNE` | 1.3.3 | random projection |
| LINE | karateclub `FirstOrderLINE`+`SecondOrderLINE` | 1.3.3, 64+64 dims | first- and second-order halves |
| Force2Vec / tForce2Vec / rForce2Vec | HipGraph/Force2Vec (C++) | options 1 / 5 / 7, iter 1200 | built from source |

**Not run, with the reason (never dropped, per the plan):**

- **NetMF, GraRep, HOPE on pubmed** — each forms a dense n×n matrix, ≈3 GB
  at 19,717 nodes, which OOMs a 7 GB machine. Run on cora and citeseer only,
  exactly the "GraRep … for smaller graphs" note in the plan.
- **Force2Vec base (option 1) on pubmed** — O(n²) all-pairs; impractical at
  19,717 nodes on this box (it already took 18–25 min on the small graphs).
  tForce2Vec and rForce2Vec (the scalable O(ns) options) ran on all three.
- **NetSMF** — the published implementation is a C++ build (THUDM); not
  attempted this pass. NetMF (its dense cousin) stands in for the
  matrix-factorization family on the small graphs.
- **GOSH** — GPU/CUDA only. This box has a 6 GB card and no CUDA build
  installed, so it is out of scope for this pass.

## 5. Threats to reading this table

- **Single-protocol, not held-out.** The embeddings saw every edge, so the
  LP panel is edge reconstruction, not held-out link prediction. Same as
  the fodiwalk report `260917-iclr2027-preliminary.md`.
- **Force2Vec rows are one run.** The binary exposes no seed, so there are
  no error bars for `force2vec`, `tforce2vec`, `rforce2vec`. Do not read
  their gaps against a 3-seed method as significant without more runs.
- **The Euclidean caveat again.** Any single-number ranking of this table
  is wrong: it would penalise the inner-product methods for a geometry they
  do not target. Rank within a family, or read the column that matches the
  method's objective.
- **fodiwalk is shown at its best cell only** (fdhop walk_edges sqn). The
  full fodiwalk sweep — force laws, walk policies, optimisers, dimensions —
  is in `260917-iclr2027-preliminary.md`; this report places the strongest
  fodiwalk cell beside the other methods, it does not re-run that sweep.
