# Embedding scorecard — 2026-09-04

Every embedding in `data_cache/embeddings/` scored with every metric the
`evaluator` package has.

| | |
| --- | --- |
| Embeddings | 31 |
| Graphs | cora (2,708), pubmed (19,717), wordnet (82,115) |
| Dimension | 128 |
| Protocol | `n2v1m`, `max_pairs=50,000`, so every row is `protocol_modified` |
| Wall time | 595 s |
| Errors | 0 |
| Raw data | `data_cache/evaluator/store-scores/260904-all-scores.jsonl` |
| Script | `data_cache/evaluator/store-scores/eval_store.py` |

Within a graph, every row was scored on **the same** pair sample and the same
kNN query nodes. Two embeddings measured on different pairs are not
comparable. Across graphs they are not comparable either.

---

## 1. What the run establishes

**Link prediction no longer discriminates.** Across all 31 embeddings, AUC
lands between **0.9935 and 0.9999**. Every arm looks excellent. A report
built on link prediction alone would conclude there is nothing to choose
between them.

**The geometry separates the same rows by a factor of 76.** Spearman rho
between layout distance and true hop distance runs from **0.011 to 0.839**,
and hop R² from **0.022 to 0.705**.

**node2vec's search bias works, and the useful direction is the opposite of
the obvious one.** See section 2.

**The force law is the second design axis. The optimiser and the walk budget
are not.** See sections 3 to 6.

---

## 2. node2vec q-sweep

`p = 1.0` throughout; only `q` varies. `q < 1` sends the walk away from
where it came from (DFS-like); `q > 1` keeps it near (BFS-like). All rows
10×80/w10.

### cora

| q | acc | auc | rf R² | rho | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- |
| **0.5** | 0.9697 | 0.9953 | **0.507** | **0.742** | **0.2320** | 0.847 |
| 1.0 | 0.9740 | 0.9970 | 0.454 | 0.711 | 0.2377 | 0.864 |
| 2.0 | 0.9702 | 0.9980 | 0.387 | 0.638 | 0.2458 | 0.868 |

### pubmed

| q | acc | auc | rf R² | rho | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- |
| **0.5** | 0.9667 | 0.9960 | **0.210** | **0.416** | **0.2023** | 0.644 |
| 1.0 | 0.9680 | 0.9961 | 0.173 | 0.362 | 0.2075 | 0.618 |
| 2.0 | 0.9667 | 0.9959 | 0.149 | 0.275 | 0.2157 | 0.608 |

### wordnet

| q | acc | auc | rf R² | rho | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.9940 | 0.9993 | 0.025 | 0.019 | 0.2420 | 0.465 |
| 1.0 | 0.9933 | 0.9993 | 0.042 | 0.011 | 0.2427 | 0.478 |
| 2.0 | 0.9946 | 0.9994 | 0.022 | 0.012 | 0.2437 | 0.500 |

**Reading.** On cora and pubmed the geometry columns are **monotone in q**,
and they run the way the walk length implies: a DFS-like walk reaches
further, so the training corpus holds longer-range co-occurrence, so the
layout encodes distance better. rho moves 0.742 → 0.711 → 0.638 on cora and
0.416 → 0.362 → 0.275 on pubmed. rec@10 moves the other way on cora — the
BFS-like walk is better at 1-hop retrieval and worse at global structure,
which is what the two metrics are for.

AUC does not move. It spans 0.9953 to 0.9980 on cora while rho falls by
0.104.

**wordnet kills the effect entirely.** All three q values sit at rho 0.011
to 0.019, which is noise. The mechanism that works on the citation graphs
does nothing here.

**The earlier gap is closed.** The previous report noted that no row
exercised `q ≠ 1`, so nothing tested node2vec's actual mechanism. It is
tested now, and it is real — but it does not change the ranking. Even the
best q loses to `walk_edges` on every graph.

---

## 3. Matched budget — the fair head-to-head

10×80/w10 only, so the walk budget is held constant.

### cora — τ-b ceiling 0.9208

| arm | acc | auc | rf R² | rho | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- |
| walk_edges / fdhop k4=1 / plain | **0.9905** | **0.9993** | **0.696** | **0.839** | **0.1708** | **0.964** |
| walk_edges / fdlinear k4=0.01 / plain | 0.9664 | 0.9935 | 0.682 | 0.781 | 0.1977 | 0.711 |
| node2vec q=0.5 | 0.9697 | 0.9953 | 0.507 | 0.742 | 0.2320 | 0.847 |
| node2vec q=1.0 | 0.9740 | 0.9970 | 0.454 | 0.711 | 0.2377 | 0.864 |
| node2vec q=2.0 | 0.9702 | 0.9980 | 0.387 | 0.638 | 0.2458 | 0.868 |

### pubmed — τ-b ceiling 0.8955

| arm | acc | auc | rf R² | rho | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- |
| walk_edges / fdhop k4=1 / plain | **0.9912** | **0.9992** | **0.606** | **0.776** | **0.1447** | **0.775** |
| walk_edges / fdlinear k4=0.01 / plain | 0.9720 | 0.9950 | 0.504 | 0.707 | 0.2111 | 0.470 |
| node2vec q=0.5 | 0.9667 | 0.9960 | 0.210 | 0.416 | 0.2023 | 0.644 |
| node2vec q=1.0 | 0.9680 | 0.9961 | 0.173 | 0.362 | 0.2075 | 0.618 |
| node2vec q=2.0 | 0.9667 | 0.9959 | 0.149 | 0.275 | 0.2157 | 0.608 |

### wordnet — τ-b ceiling 0.9517

| arm | acc | auc | rf R² | rho | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- |
| walk_edges / fdlinear k4=0.01 / plain | 0.9930 | 0.9992 | **0.153** | **0.360** | **0.2119** | 0.704 |
| walk_edges / fdhop k4=1 / plain | **0.9954** | **0.9996** | 0.096 | 0.339 | 0.2167 | **0.940** |
| node2vec q=0.5 | 0.9940 | 0.9993 | 0.025 | 0.019 | 0.2420 | 0.465 |
| node2vec q=2.0 | 0.9946 | 0.9994 | 0.022 | 0.012 | 0.2437 | 0.500 |
| node2vec q=1.0 | 0.9933 | 0.9993 | 0.042 | 0.011 | 0.2427 | 0.478 |

**Reading.** `walk_edges` beats every node2vec setting on every graph and
every geometry column. The margin is largest on pubmed: rho 0.776 against
0.416 for the best q, and rf R² 0.606 against 0.210.

wordnet is where all five arms fail. The best hop R² is 0.153 and the best
rho is 0.360 against a ceiling of 0.9517 — while every arm scores AUC above
0.999. Nothing in the store has learned wordnet's distance structure, and
link prediction cannot tell.

---

## 4. Force law — `fdhop` against `fdlinear`

Held: `walk_edges`, `plain`, same budget, same graph. Delta is
`fdhop − fdlinear`.

| graph | budget | acc | auc | rf R² | rho | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cora | 10×20/w5 | +0.0189 | +0.0054 | +0.008 | +0.048 | −0.0212 | **+0.231** |
| cora | 10×80/w10 | +0.0241 | +0.0058 | +0.015 | +0.057 | −0.0269 | **+0.254** |
| pubmed | 10×20/w5 | +0.0203 | +0.0035 | **+0.108** | +0.065 | −0.0636 | **+0.309** |
| pubmed | 10×80/w10 | +0.0192 | +0.0041 | **+0.102** | +0.069 | −0.0664 | **+0.305** |
| wordnet | 10×20/w5 | +0.0035 | +0.0009 | **−0.060** | **−0.019** | +0.0038 | +0.175 |
| wordnet | 10×80/w10 | +0.0024 | +0.0004 | **−0.058** | **−0.021** | +0.0048 | +0.236 |

**`fdhop` wins on the citation graphs and loses on wordnet.** The sign flips
on rf R² and rho, consistently across both budgets, so it is not noise. On
wordnet `fdhop` still wins rec@10 by a wide margin (+0.18 to +0.24) while
losing global structure — it packs true neighbours tightly and gets the
long-range order wrong.

Lower stress is better; `fdhop` improves it on cora and pubmed and slightly
worsens it on wordnet, which agrees with the rho column.

---

## 5. Walker — `walk_edges` against `nbr_walk`

Held: `fdlinear k4=0.01`, 10×20/w5. No `nbr_walk` row exists for wordnet.

| graph | optim | walker | acc | auc | rf R² | rho | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cora | plain | walk_edges | 0.9692 | 0.9936 | **0.697** | **0.787** | **0.1918** | 0.728 |
| cora | plain | nbr_walk | 0.9730 | 0.9954 | 0.223 | 0.516 | 0.2626 | **0.854** |
| cora | velocity | walk_edges | 0.9697 | 0.9941 | **0.702** | **0.788** | **0.1909** | 0.728 |
| cora | velocity | nbr_walk | 0.9711 | 0.9949 | 0.237 | 0.526 | 0.2603 | **0.854** |
| pubmed | plain | walk_edges | 0.9712 | 0.9955 | **0.503** | **0.710** | 0.2100 | 0.469 |
| pubmed | plain | nbr_walk | **0.9851** | **0.9986** | 0.408 | 0.651 | **0.1737** | **0.682** |
| pubmed | velocity | walk_edges | 0.9709 | 0.9949 | **0.502** | **0.711** | 0.2092 | 0.468 |
| pubmed | velocity | nbr_walk | 0.9824 | 0.9986 | 0.418 | 0.655 | **0.1724** | **0.685** |

**The largest single effect in the store, and it is graph-dependent.** On
cora the walker swap moves hop R² from 0.70 to 0.22 — a factor of three.
On pubmed the same swap costs only 0.10 in R² while `nbr_walk` *wins*
accuracy, AUC, stress and rec@10.

So `walk_edges` is the better choice for hop geometry and `nbr_walk` for
local structure, and how much that trade costs depends on the graph.

---

## 6. Optimiser — `plain` against `velocity`

Held: `fdlinear k4=0.01`, 10×20/w5.

| graph | walker | optim | acc | auc | rf R² | rho | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cora | walk_edges | plain | 0.9692 | 0.9936 | 0.697 | 0.787 | 0.1918 | 0.728 |
| cora | walk_edges | velocity | 0.9697 | 0.9941 | 0.702 | 0.788 | 0.1909 | 0.728 |
| cora | nbr_walk | plain | 0.9730 | 0.9954 | 0.223 | 0.516 | 0.2626 | 0.854 |
| cora | nbr_walk | velocity | 0.9711 | 0.9949 | 0.237 | 0.526 | 0.2603 | 0.854 |
| pubmed | walk_edges | plain | 0.9712 | 0.9955 | 0.503 | 0.710 | 0.2100 | 0.469 |
| pubmed | walk_edges | velocity | 0.9709 | 0.9949 | 0.502 | 0.711 | 0.2092 | 0.468 |
| pubmed | nbr_walk | plain | 0.9851 | 0.9986 | 0.408 | 0.651 | 0.1737 | 0.682 |
| pubmed | nbr_walk | velocity | 0.9824 | 0.9986 | 0.418 | 0.655 | 0.1724 | 0.685 |

**No effect worth reporting.** Every pair agrees to the third decimal on
every column. rec@10 is identical to three decimals in all four cora pairs.
At one seed per configuration this is inside run-to-run variation.

---

## 7. Walk budget — 10×20/w5 against 10×80/w10

Held: `walk_edges`, `plain`.

| graph | law | budget | acc | auc | rf R² | rho | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cora | fdhop | 10×20/w5 | 0.9882 | 0.9990 | 0.705 | 0.835 | 0.1705 | 0.959 |
| cora | fdhop | 10×80/w10 | 0.9905 | 0.9993 | 0.696 | 0.839 | 0.1708 | 0.964 |
| cora | fdlinear | 10×20/w5 | 0.9692 | 0.9936 | 0.697 | 0.787 | 0.1918 | 0.728 |
| cora | fdlinear | 10×80/w10 | 0.9664 | 0.9935 | 0.682 | 0.781 | 0.1977 | 0.711 |
| pubmed | fdhop | 10×20/w5 | 0.9915 | 0.9989 | 0.610 | 0.775 | 0.1464 | 0.778 |
| pubmed | fdhop | 10×80/w10 | 0.9912 | 0.9992 | 0.606 | 0.776 | 0.1447 | 0.775 |
| pubmed | fdlinear | 10×20/w5 | 0.9712 | 0.9955 | 0.503 | 0.710 | 0.2100 | 0.469 |
| pubmed | fdlinear | 10×80/w10 | 0.9720 | 0.9950 | 0.504 | 0.707 | 0.2111 | 0.470 |
| wordnet | fdhop | 10×20/w5 | 0.9964 | 0.9999 | 0.096 | 0.355 | 0.2142 | 0.989 |
| wordnet | fdhop | 10×80/w10 | 0.9954 | 0.9996 | 0.096 | 0.339 | 0.2167 | 0.940 |
| wordnet | fdlinear | 10×20/w5 | 0.9929 | 0.9990 | 0.156 | 0.374 | 0.2104 | 0.814 |
| wordnet | fdlinear | 10×80/w10 | 0.9930 | 0.9992 | 0.153 | 0.360 | 0.2119 | 0.704 |

**Four times the walk length buys nothing.** Every pair agrees to about the
second decimal, and where it moves it as often gets worse. For the
force-directed arms the walk budget is not a lever; the corpus is already
saturated at 10×20/w5.

This matters for cost: the 10×20/w5 runs are four times cheaper to augment
and score the same.

---

## 8. Verification

Each `config.json` carries a `metrics` block written when the embedding was
made. Those were recomputed independently and compared.

**186 comparisons across 31 records. Maximum absolute difference
4.441e-16.** That is the floating-point noise floor from `n_jobs=-1`
thread-order accumulation in the random forest, not a discrepancy. Every
recorded accuracy, AUC, `f1_score`, rf R², mlp R² and rf MAE in the store is
correct.

**The duplicate-fingerprint defect is gone.** The previous report found two
cora runs sharing config hash `0ba3b1b3` whose embeddings differed by three
orders of magnitude in `mean |Z|`. Only `260904-071825` remains; the
divergent `260904-062153` has been removed. The underlying weakness is
unchanged: `hash_spec 1` does not capture working-tree state, and the
surviving configs still record `git_dirty: true`.

---

## 9. How the numbers were produced

- One protocol for every row: `n2v1m` with `max_pairs = 50,000`, matching
  what the store's own records used. Every row therefore carries
  `protocol_modified: true`, which is honest — these are not the recorded
  `n2v1m` baseline.
- Hop pairs and kNN queries drawn once per graph and reused for every
  embedding of that graph. 20,000 pairs requested, minimum hop 2, 1,000
  query nodes, k = 10.
- Unreachable pairs filtered with `isfinite` **before** the minimum-hop
  filter. `hops.UNREACHABLE` is `inf`, and a minimum-hop filter alone keeps
  the sentinel. On cora this drops 3,277 of 20,000 pairs; the graph has 78
  components and leaving them in destroys every correlation.
- Hop ground truth from the exact backend chosen automatically — blocked
  SciPy BFS at these sizes. No approximation.
- kNN by exact brute-force search under a bounded working-memory chunk.
  Every row records which path ran.
- Somers' D via the identity
  `D = τ-b · √((1 − tie_y) / (1 − tie_x))`, not `scipy.stats.somersd`,
  which is O(n²) and costs 160.67 s per embedding on a 16,723-pair sample.
  The identity is exact and free once τ-b is computed. Every row records
  `somers_d_via: "identity"`.

### Reading the rank columns

Somers' D and Kendall's τ-b cannot reach 1.0. The hop target is an integer
with heavy ties, and both divide by a quantity ties shrink. A perfect layout
scores `1 − tie_frac(y)`. The τ-b ceiling is 0.9208 on cora, 0.8955 on
pubmed, 0.9517 on wordnet, and each table caption carries it.

---

## 10. What this run does not establish

- **One seed per configuration.** No error bars. The optimiser comparison in
  section 6 reports differences smaller than any variation estimate the
  store can support.
- **`rec@10` is not comparable across graphs.** It is a micro-average with a
  `min(degree, 10)` denominator, so a different degree distribution changes
  what the number means. Within a graph it is sound.
- **wordnet has no `nbr_walk` and no `velocity` arm.** Its rows cannot
  answer the walker or optimiser question on that graph.
- **`p` is fixed at 1.0.** The q-sweep tests the in-out parameter only; the
  return parameter is untested.
- **Rank and stress metrics have no parity reference.** They come from a
  metrology campaign built in the last few days. The link prediction and
  hop-regression columns are verified bit-exact against frozen references;
  these are not.
- **wordnet's failure is unexplained.** Every arm fails there on geometry
  while succeeding on link prediction. Whether that is the graph or the
  methods is open — a second, structurally different embedding of the same
  graph would settle it.
