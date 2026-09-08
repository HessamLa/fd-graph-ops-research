# Embedding scorecard — 2026-09-05

Every embedding in `data_cache/embeddings/` scored with every metric the
`evaluator` package has. This replaces the 2026-09-04 scorecard: **41
records against 31**, and two graphs over a million nodes that did not exist
in the previous run.

| | |
| --- | --- |
| Embeddings | 41 |
| Graphs | as_skitter (1,696,415), com_youtube (1,134,890), wordnet (82,115), pubmed (19,717), cora (2,708) |
| Dimension | 128 |
| Protocol | `n2v1m`, `max_pairs=50,000`, so every row is `protocol_modified` |
| Scorer wall time | 4,191 s |
| Errors | 0 |
| Raw data | `data_cache/evaluator/store-scores/260905-all-scores.jsonl` |
| Script | `data_cache/evaluator/store-scores/eval_store.py` |

Within a graph, every row was scored on **the same** pair sample and the
same kNN query nodes. Across graphs nothing is comparable, and above a
million nodes two measurement rules change — see section 2.

---

## 1. What changed since 2026-09-04

**Two graphs over a million nodes arrived**, and both overturn something.

**`as_skitter` breaks the geometry entirely.** Both node2vec arms score
Spearman rho at or below zero — **−0.002 and −0.040** — with hop R² of
−0.017 and −0.000. There is no monotone relationship between layout distance
and graph distance, and a small negative one where there is any. Both still
score AUC above 0.99. This is the clearest case yet of link prediction
reporting success on a layout that holds no distance information.

**`com_youtube` is the first graph where a baseline beats fodiwalk.**
deepwalk reaches rho **0.754** against the best fodiwalk arm at **0.692** —
a margin of +0.062. On every other graph fodiwalk leads:

| graph | deepwalk rho | best fodiwalk rho | delta |
| --- | --- | --- | --- |
| **com_youtube** | **0.754** | 0.692 | **+0.062** |
| cora | 0.744 | 0.839 | −0.095 |
| pubmed | 0.662 | 0.776 | −0.114 |
| wordnet | 0.171 | 0.374 | −0.202 |

The reversal is not the search bias. node2vec on the same graph reaches only
0.382 at its best `q`, so deepwalk's advantage there comes from its
objective (hierarchical softmax against negative sampling) or its walk
budget (80×40 against 10×80), not from `p`/`q`. Nothing in the store
separates those two causes.

**The q-sweep stays monotone at million scale.** On com_youtube rho runs
0.382 / 0.343 / 0.284 at q = 0.5 / 1.0 / 2.0, the same DFS-favouring order
found on cora and pubmed. Four graphs now agree on the direction.

---

## 2. Two measurement rules change above a million nodes

Both change what the number **means**, not just what it costs, so both are
recorded per row rather than in a footnote.

**Hop sources are bounded to 200 above 200,000 nodes.** An unrestricted
sample gives `hops.choose_backend` roughly 20,000 distinct sources; its rule
only routes ≤512 sources to the blocked SciPy BFS, so above that it asks for
a PLL index over the whole graph, which does not fit in the memory this
machine has. as_skitter and com_youtube rows are therefore drawn from 200
sources; every other graph is unrestricted. The `sources` figure sits in each
table caption.

**kNN is approximate above 200,000 nodes.** `metrics.knn` switches from
exact brute-force search to `pynndescent` at `RECON_MAX_N`. The `rec@10`
column is marked **`rec@10~`** on those tables. An approximate value must not
be read against an exact one, and the query count also drops from 1,000 to
500 there.

---

## 3. The tables

Rows sorted by Spearman rho. `D` is Somers' D.

### as_skitter — n 1,696,415 · 19,978 hop pairs · sources 200 · τ-b ceiling 0.8490

| arm | budget | acc | auc | rf R² | rho | D | stress | rec@10~ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| node2vec q=0.5 | 10×80/w10 | 0.9579 | 0.9940 | -0.017 | -0.002 | -0.001 | 0.2210 | 0.232 |
| node2vec q=1.0 | 10×80/w10 | 0.9547 | 0.9918 | -0.000 | -0.040 | -0.025 | 0.2250 | 0.225 |

### com_youtube — n 1,134,890 · 20,000 hop pairs · sources 200 · τ-b ceiling 0.8721

| arm | budget | acc | auc | rf R² | rho | D | stress | rec@10~ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deepwalk | 80×40/w10 | 0.9551 | 0.9931 | 0.610 | 0.754 | 0.534 | 0.1796 | 0.455 |
| walk_edges / fdlinear k4=0.01 / plain | 10×20/w5 | 0.9681 | 0.9941 | 0.524 | 0.692 | 0.481 | 0.2180 | 0.331 |
| walk_edges / fdlinear k4=0.01 / plain | 10×80/w10 | 0.9665 | 0.9936 | 0.506 | 0.690 | 0.479 | 0.2192 | 0.328 |
| walk_edges / fdhop k4=1 / plain | 10×80/w10 | 0.9852 | 0.9980 | 0.476 | 0.683 | 0.479 | 0.1789 | 0.498 |
| walk_edges / fdhop k4=1 / plain | 10×20/w5 | 0.9856 | 0.9983 | 0.447 | 0.666 | 0.465 | 0.1855 | 0.547 |
| node2vec q=0.5 | 10×80/w10 | 0.9348 | 0.9870 | 0.130 | 0.382 | 0.250 | 0.2128 | 0.486 |
| node2vec q=1.0 | 10×80/w10 | 0.9292 | 0.9851 | 0.108 | 0.343 | 0.224 | 0.2172 | 0.479 |
| node2vec q=2.0 | 10×80/w10 | 0.9334 | 0.9866 | 0.078 | 0.284 | 0.185 | 0.2246 | 0.467 |

### wordnet — n 82,115 · 20,000 hop pairs · sources all · τ-b ceiling 0.9517

| arm | budget | acc | auc | rf R² | rho | D | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| walk_edges / fdlinear k4=0.01 / plain | 10×20/w5 | 0.9929 | 0.9990 | 0.156 | 0.374 | 0.255 | 0.2104 | 0.814 |
| walk_edges / fdlinear k4=0.01 / plain | 10×80/w10 | 0.9930 | 0.9992 | 0.153 | 0.360 | 0.245 | 0.2119 | 0.704 |
| walk_edges / fdhop k4=1 / plain | 10×20/w5 | 0.9964 | 0.9999 | 0.096 | 0.355 | 0.240 | 0.2142 | 0.989 |
| walk_edges / fdhop k4=1 / plain | 10×80/w10 | 0.9954 | 0.9996 | 0.096 | 0.339 | 0.229 | 0.2167 | 0.940 |
| deepwalk | 80×40/w10 | 0.9892 | 0.9991 | 0.023 | 0.171 | 0.114 | 0.2314 | 0.686 |
| node2vec q=0.5 | 10×80/w10 | 0.9940 | 0.9993 | 0.025 | 0.019 | 0.013 | 0.2420 | 0.465 |
| node2vec q=2.0 | 10×80/w10 | 0.9946 | 0.9994 | 0.022 | 0.012 | 0.008 | 0.2437 | 0.500 |
| node2vec q=1.0 | 10×80/w10 | 0.9933 | 0.9993 | 0.042 | 0.011 | 0.007 | 0.2427 | 0.478 |

### pubmed — n 19,717 · 20,000 hop pairs · sources all · τ-b ceiling 0.8955

| arm | budget | acc | auc | rf R² | rho | D | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| walk_edges / fdhop k4=1 / plain | 10×80/w10 | 0.9912 | 0.9992 | 0.606 | 0.776 | 0.565 | 0.1447 | 0.775 |
| walk_edges / fdhop k4=1 / plain | 10×20/w5 | 0.9915 | 0.9989 | 0.610 | 0.775 | 0.563 | 0.1464 | 0.778 |
| walk_edges / fdlinear k4=0.01 / velocity | 10×20/w5 | 0.9709 | 0.9949 | 0.502 | 0.711 | 0.505 | 0.2092 | 0.468 |
| walk_edges / fdlinear k4=0.01 / plain | 10×20/w5 | 0.9712 | 0.9955 | 0.503 | 0.710 | 0.504 | 0.2100 | 0.469 |
| walk_edges / fdlinear k4=0.01 / plain | 10×80/w10 | 0.9720 | 0.9950 | 0.504 | 0.707 | 0.502 | 0.2111 | 0.470 |
| deepwalk | 80×40/w10 | 0.9736 | 0.9978 | 0.511 | 0.662 | 0.465 | 0.1660 | 0.728 |
| nbr_walk / fdlinear k4=0.01 / velocity | 10×20/w5 | 0.9824 | 0.9986 | 0.418 | 0.655 | 0.458 | 0.1724 | 0.685 |
| nbr_walk / fdlinear k4=0.01 / plain | 10×20/w5 | 0.9851 | 0.9986 | 0.408 | 0.651 | 0.455 | 0.1737 | 0.682 |
| node2vec q=0.5 | 10×80/w10 | 0.9667 | 0.9960 | 0.210 | 0.416 | 0.279 | 0.2023 | 0.644 |
| node2vec q=1.0 | 10×80/w10 | 0.9680 | 0.9961 | 0.173 | 0.362 | 0.240 | 0.2075 | 0.618 |
| node2vec q=2.0 | 10×80/w10 | 0.9667 | 0.9959 | 0.149 | 0.275 | 0.181 | 0.2157 | 0.608 |

### cora — n 2,708 · 16,723 hop pairs · sources all · τ-b ceiling 0.9208

| arm | budget | acc | auc | rf R² | rho | D | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| walk_edges / fdhop k4=1 / plain | 10×80/w10 | 0.9905 | 0.9993 | 0.696 | 0.839 | 0.633 | 0.1708 | 0.964 |
| walk_edges / fdhop / plain | 10×20/w5 | 0.9882 | 0.9990 | 0.705 | 0.835 | 0.629 | 0.1705 | 0.959 |
| walk_edges / fdhop k4=1 / plain | 10×20/w5 | 0.9882 | 0.9990 | 0.705 | 0.835 | 0.629 | 0.1705 | 0.959 |
| walk_edges / fdlinear k4=0.01 / velocity | 10×20/w5 | 0.9697 | 0.9941 | 0.702 | 0.788 | 0.587 | 0.1909 | 0.728 |
| walk_edges / fdlinear k4=0.01 / plain | 10×20/w5 | 0.9692 | 0.9936 | 0.697 | 0.787 | 0.586 | 0.1918 | 0.728 |
| walk_edges / fdlinear k4=0.01 / plain | 10×80/w10 | 0.9664 | 0.9935 | 0.682 | 0.781 | 0.580 | 0.1977 | 0.711 |
| deepwalk | 80×40/w10 | 0.9744 | 0.9980 | 0.601 | 0.744 | 0.540 | 0.2078 | 0.906 |
| node2vec q=0.5 | 10×80/w10 | 0.9697 | 0.9953 | 0.507 | 0.742 | 0.535 | 0.2320 | 0.847 |
| node2vec q=1.0 | 10×80/w10 | 0.9740 | 0.9970 | 0.454 | 0.711 | 0.509 | 0.2377 | 0.864 |
| node2vec q=2.0 | 10×80/w10 | 0.9702 | 0.9980 | 0.387 | 0.638 | 0.450 | 0.2458 | 0.868 |
| nbr_walk / fdlinear k4=0.01 / velocity | 10×20/w5 | 0.9711 | 0.9949 | 0.237 | 0.526 | 0.379 | 0.2603 | 0.854 |
| nbr_walk / fdlinear k4=0.01 / plain | 10×20/w5 | 0.9730 | 0.9954 | 0.223 | 0.516 | 0.371 | 0.2626 | 0.854 |

---

## 4. Verification

Each `config.json` carries a `metrics` block written when the embedding was
made. Those were recomputed independently and compared.

**246 comparisons across 41 records. Maximum absolute difference
4.441e-16.** That is the floating-point noise floor from `n_jobs=-1`
thread-order accumulation in the random forest, not a discrepancy. Every
recorded accuracy, AUC, `f1_score`, rf R², mlp R² and rf MAE in the store is
correct.

---

## 5. What this run does not establish

- **`as_skitter` has only two records**, both node2vec. No fodiwalk arm, no
  deepwalk. Its collapse is a fact about those two embeddings, not a
  statement about the graph — nothing else has been tried on it.
- **`roadnet_ca` and `ncbi_taxonomy` are absent.** The `runner` session
  scheduled them; they have not landed. **A missing directory means the run
  did not finish, never that it scored zero.**
- **com_youtube's deepwalk win has two possible causes** and the store
  cannot separate them: the training objective and the walk budget differ at
  the same time. A deepwalk run at 10×80/w10, or a node2vec run at 80×40
  with hierarchical softmax, would isolate it.
- **One seed per configuration.** No error bars anywhere.
- **`rec@10` is not comparable across graphs**, and above a million nodes it
  is also approximate.
- **Rank and stress metrics have no parity reference.** The link prediction
  and hop-regression columns are verified bit-exact against frozen
  references; these are not.
- **cora still carries the `0ba3b1b3` ambiguity.** That fingerprint is
  `git_dirty: true` and its force law is recorded only in a `notes` field. A
  second record once shared it and ran a different law; that record is gone
  and its removal is unaccounted for.

---

## 6. Method

- One protocol for every row: `n2v1m` with `max_pairs = 50,000`, matching
  the store's own records. Every row carries `protocol_modified: true`.
- Hop pairs and kNN queries drawn once per graph and reused for every
  embedding of that graph. 20,000 pairs requested, minimum hop 2.
- Unreachable pairs filtered with `isfinite` **before** the minimum-hop
  filter. `hops.UNREACHABLE` is `inf`; a minimum-hop filter alone keeps the
  sentinel. cora drops 3,277 of 20,000 (78 components); as_skitter drops 22.
- Hop ground truth from the exact backend chosen automatically — blocked
  SciPy BFS at every size here. No approximation in the hop target.
- Somers' D via the identity `D = τ-b · √((1 − tie_y) / (1 − tie_x))`, not
  `scipy.stats.somersd`, which is O(n²) and costs 160.67 s per embedding on
  a 16,723-pair sample. Every row records `somers_d_via: "identity"`.
- Somers' D and τ-b cannot reach 1.0: the hop target is an integer with
  heavy ties and both divide by a quantity ties shrink. A perfect layout
  scores `1 − tie_frac(y)`. Each table caption carries its ceiling.
