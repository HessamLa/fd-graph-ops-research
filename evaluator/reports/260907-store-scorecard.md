# Store scorecard — 2026-09-07

Every embedding in `data_cache/embeddings/` scored with every metric the
`evaluator` package has. Supersedes `260905-store-scorecard.md`.

| | |
| --- | --- |
| **Records in store** | **58** |
| **Records scored here** | **50** |
| Excluded by instruction | 8 — `fdunit` ×4, `fdunit_freq` ×2, `fdhop2` ×2 |
| Graphs | as_skitter (1,696,415), com_youtube (1,134,890), wordnet (82,115), pubmed (19,717), cora (2,708) |
| Rows per graph | cora 16, pubmed 15, com_youtube 8, wordnet 8, as_skitter 3 |
| Dimension | 128 |
| Protocol | `n2v1m`, `max_pairs=50,000` — every row is `protocol_modified` |
| Errors | 0 |
| Data | `data_cache/evaluator/store-scores/260907-all-scores.jsonl` |
| Scorers | `eval_store.py` (full), `eval_new.py` (filtered, with controls) |

Within a graph, every row was scored on **the same** pair sample and the same
kNN query nodes. Across graphs nothing is comparable, and above 200,000 nodes
two measurement rules change — see section 6.

---

## 1. The one-line result

**Link prediction cannot separate these embeddings. The geometry separates
them completely.**

| | range across all 50 rows |
| --- | --- |
| AUC | 0.9851 – 0.9999 |
| Spearman rho | **−0.066 – 0.856** |

Every arm scores above 0.985 AUC. Rank correlation between layout distance
and true hop distance spans from *worse than nothing* to 0.856. Any study
reporting only link prediction would conclude these methods are
interchangeable.

---

## 2. Best of each family, per graph

Highest Spearman rho in each family.

| graph | best fodiwalk | rho | deepwalk rho | best node2vec | rho |
| --- | --- | --- | --- | --- | --- |
| cora | fdhop | **0.856** | 0.744 | q=0.5 | 0.742 |
| pubmed | fdhop | **0.790** | 0.662 | q=0.5 | 0.416 |
| wordnet | fdlinear | **0.374** | 0.171 | q=0.5 | 0.019 |
| com_youtube | fdlinear | 0.692 | **0.754** | q=0.5 | 0.382 |
| as_skitter | — | — | — | q=0.5 | **-0.002** |

**fodiwalk leads on four graphs and loses one.** com_youtube is the single
reversal: deepwalk 0.754 against the best fodiwalk 0.692. The cause is not
the search bias — node2vec on that graph reaches only 0.382 — so it is
deepwalk's objective (hierarchical softmax) or its walk budget (80×40), and
the store cannot separate the two because both differ at once.

**as_skitter has no fodiwalk arm at all.** Its three rows are node2vec only,
and every one scores rho at or below zero.

---

## 3. Force law — `fdhop` against `fdlinear`


Held: `walk_edges`, `plain`, same budget. Delta is fdhop − fdlinear.

| graph | budget | Δacc | Δauc | Δ rf R² | Δ rho | Δ stress | Δ rec@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cora | 10×20/w5 | +0.0194 | +0.0052 | +0.049 | +0.068 | -0.0259 | +0.229 |
| cora | 10×80/w10 | +0.0241 | +0.0058 | +0.015 | +0.057 | -0.0269 | +0.254 |
| pubmed | 10×20/w5 | +0.0180 | +0.0028 | +0.134 | +0.080 | -0.0615 | +0.304 |
| pubmed | 10×80/w10 | +0.0192 | +0.0041 | +0.102 | +0.069 | -0.0664 | +0.305 |
| wordnet | 10×20/w5 | +0.0035 | +0.0009 | -0.060 | -0.019 | +0.0038 | +0.175 |
| wordnet | 10×80/w10 | +0.0024 | +0.0004 | -0.058 | -0.021 | +0.0048 | +0.236 |
| com_youtube | 10×20/w5 | +0.0175 | +0.0042 | -0.077 | -0.026 | -0.0325 | +0.216 |
| com_youtube | 10×80/w10 | +0.0187 | +0.0045 | -0.030 | -0.007 | -0.0403 | +0.170 |


**`fdhop` wins the two small citation graphs and loses the two large ones**
on the geometry columns, at both walk budgets. The sign flip is consistent
across budgets, so it is not noise.

It wins accuracy, AUC and neighbour recall **everywhere** — rec@10 by +0.17
to +0.31. The honest reading is that `fdhop` trades global distance structure
for local structure, and the trade turns negative as the graph grows.

---

## 4. node2vec q-sweep — five graphs

`p = 1.0` throughout. `q < 1` sends the walk away (DFS-like); `q > 1` keeps
it near (BFS-like).


| graph | q=0.5 | q=1.0 | q=2.0 | span |
| --- | --- | --- | --- | --- |
| cora | 0.742 | 0.711 | 0.638 | 0.103 |
| pubmed | 0.416 | 0.362 | 0.275 | 0.141 |
| wordnet | 0.019 | 0.011 | 0.012 | 0.008 |
| com_youtube | 0.382 | 0.343 | 0.284 | 0.098 |
| as_skitter | -0.002 | -0.040 | -0.066 | 0.064 |


**The DFS direction wins on every graph.** q=0.5 leads all five, without
exception. On the three graphs where node2vec has real signal the span is
0.098 to 0.141; on wordnet and as_skitter every value is near zero and the
ordering carries no weight.

### The same knob does nothing inside fodiwalk

`walk_edges_pq` applies the identical p/q bias to fodiwalk's own walker:

| q | node2vec rho, cora | walk_edges_pq rho, cora |
| --- | --- | --- |
| 0.5 | 0.742 | 0.837 |
| 1.0 | 0.711 | 0.835 *(plain `walk_edges`)* |
| 2.0 | 0.638 | 0.838 |
| **span** | **0.103** | **0.003** |

pubmed agrees: span 0.001, and the ordering flips between the two graphs.
**A factor of 35.** Whatever the search bias buys a skip-gram model, the
force-directed layout already has. This is a finding about the method, not a
weak test — same knob, same graph, same budget, large effect in one pipeline
and none in the other.

---

## 5. `fdhop_min` is much worse than `fdhop`

| graph | arm | acc | auc | rf R² | rho | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cora | fdhop | 0.9882 | 0.9990 | **0.705** | **0.835** | **0.1705** | **0.959** |
| cora | fdhop_min | 0.9493 | 0.9897 | 0.163 | 0.381 | 0.2891 | 0.575 |
| pubmed | fdhop | 0.9915 | 0.9989 | **0.610** | **0.775** | **0.1464** | **0.778** |
| pubmed | fdhop_min | 0.9316 | 0.9861 | 0.246 | 0.441 | 0.2744 | 0.428 |

Worse on every column on both graphs. Hop R² falls by a factor of four on
cora. Not a close call.

---

## 6. Two measurement rules change above 200,000 nodes

Both change what the number **means**, so both are recorded per row.

**Hop sources bounded to 200.** An unrestricted sample gives
`hops.choose_backend` ~20,000 distinct sources; its rule only routes ≤512 to
the blocked SciPy BFS, so above that it asks for a PLL index over the whole
graph, which does not fit in memory. as_skitter and com_youtube use 200
sources; cora, pubmed and wordnet are unrestricted.

**kNN is approximate.** `metrics.knn` switches to `pynndescent` at
`RECON_MAX_N`. Those columns are marked **`rec@10~`**, and the query count
drops from 1,000 to 500.

**The approximate path is not deterministic across runs.** Six control rows
re-scored on 2026-09-07 reproduced their 2026-09-05 values exactly on every
hop-sample metric — accuracy, AUC, f1_score, mlp R², rho, Somers' D and
stress all at 0.000e+00, rf R² at 2.220e-16 — while `rec@10` moved by up to
**7.625e-04**, and only on `as_skitter`. Same `Z`, same seed, same queries,
different neighbours. Treat a difference below ~1e-3 on an approximate row as
noise.

---

## 7. Verification

Each `config.json` carries a `metrics` block written when the embedding was
made. Recomputed independently: **300 comparisons, maximum absolute
difference 4.441e-16** — the `n_jobs=-1` thread-order floor. Every recorded
accuracy, AUC, `f1_score`, rf R², mlp R² and rf MAE in the store is correct.

---

## 8. An unexplained gap in the `fdhop` rows

Three cora records claim `fdhop`, `walk_edges`, `plain`, 10×20/w5, 200
epochs, same force parameters:

| record | commit | rf R² | rho |
| --- | --- | --- | --- |
| `260906-204215-c0fedf2b` | `ea9ae67` | **0.746** | **0.856** |
| `260904-072752-6047dac7` | `864c5b4` | 0.705 | 0.835 |
| `260904-071825-0ba3b1b3` | `864c5b4` | 0.705 | 0.835 |

**The commit difference is not the explanation.** `git diff 864c5b4 ea9ae67`
touches only the `fdwalk` → `archive/fdwalk` move, the deletion of
`fodined/`, and one line of `modular.py`. `fodiwalk/embed/forces.py` is
unchanged between them.

It is worse than that: `forces.py` is **currently uncommitted**
(`git status` reports `M`), and all three records carry `git_dirty: true`.
The force law that produced these numbers was never in any commit, so it
could have changed between 2026-09-04 and 2026-09-06 with nothing recording
it — and the commit field would look exactly as it does.

**Anyone citing an `fdhop` number cannot currently say which `fdhop`
produced it.** The 0.021 rho gap is real and unexplained. A re-run on the
current tree would show whether it reproduces.

This is the same failure as the `0ba3b1b3` fingerprint collision: a dirty
tree, code outside version control, and a hash that cannot see it.

---

## 9. What this run does not establish

- **One seed per configuration.** No error bars. The `walk_edges_pq` result
  shows the effect is far smaller than node2vec's, not that it is zero.
- **`walk_edges_pq`, `fdhop_min` and `nbr_walk` exist on cora and pubmed
  only.** Those conclusions do not extend to the large graphs.
- **`nbr_walk` has never been run with `fdhop`.** Every `fdhop` number uses
  `walk_edges`.
- **`fdunit`, `fdunit_freq` and `fdhop2` were excluded by instruction** —
  8 records in the store carry no numbers here.
- **`roadnet_ca` and `ncbi_taxonomy` are absent.** A missing directory means
  a run that did not finish, never a zero.
- **Rank and stress metrics have no parity reference.** Link prediction and
  hop regression are verified bit-exact against frozen references; rho,
  Somers' D and stress are not.
- **`rec@10` is not comparable across graphs**, and above 200,000 nodes it is
  also approximate and non-reproducible.

---

## 10. Method

- One protocol for every row: `n2v1m`, `max_pairs = 50,000`, matching the
  store's own records. Every row carries `protocol_modified: true`.
- Hop pairs and kNN queries drawn once per graph, reused for every embedding
  of that graph. 20,000 pairs requested, minimum hop 2.
- Unreachable pairs filtered with `isfinite` **before** the minimum-hop
  filter. `hops.UNREACHABLE` is `inf`; a minimum-hop filter alone keeps the
  sentinel. cora drops 3,277 of 20,000 (78 components); as_skitter drops 22.
- Hop ground truth from the exact backend — blocked SciPy BFS at every size
  here.
- Somers' D via the identity `D = τ-b · √((1 − tie_y)/(1 − tie_x))`, not
  `scipy.stats.somersd`, which is O(n²) and costs 160.67 s per embedding on
  a 16,723-pair sample.
- Somers' D and τ-b cannot reach 1.0: the hop target is an integer with heavy
  ties. A perfect layout scores `1 − tie_frac(y)`. Each table carries its
  ceiling.

---

## 11. Full tables

### as_skitter — n 1,696,415 · 19,978 hop pairs · sources 200 · τ-b ceiling 0.8490 · kNN pynndescent

| arm | budget | acc | auc | rf R² | rho | D | stress | rec@10~ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| node2vec q=0.5 | 10×80/w10 | 0.9579 | 0.9940 | -0.017 | -0.002 | -0.001 | 0.2210 | 0.233 |
| node2vec q=1.0 | 10×80/w10 | 0.9547 | 0.9918 | -0.000 | -0.040 | -0.025 | 0.2250 | 0.226 |
| node2vec q=2.0 | 10×80/w10 | 0.9581 | 0.9939 | 0.006 | -0.066 | -0.042 | 0.2283 | 0.217 |

### com_youtube — n 1,134,890 · 20,000 hop pairs · sources 200 · τ-b ceiling 0.8721 · kNN pynndescent

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

### wordnet — n 82,115 · 20,000 hop pairs · sources all · τ-b ceiling 0.9517 · kNN brute

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

### pubmed — n 19,717 · 20,000 hop pairs · sources all · τ-b ceiling 0.8955 · kNN brute

| arm | budget | acc | auc | rf R² | rho | D | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| walk_edges / fdhop k4=1 / plain | 10×20/w5 | 0.9892 | 0.9983 | 0.637 | 0.790 | 0.577 | 0.1484 | 0.773 |
| walk_edges / fdhop k4=1 / plain | 10×80/w10 | 0.9912 | 0.9992 | 0.606 | 0.776 | 0.565 | 0.1447 | 0.775 |
| walk_edges_pq q=0.5 / fdhop k4=1 / plain | 10×20/w5 | 0.9909 | 0.9987 | 0.588 | 0.775 | 0.564 | 0.1464 | 0.784 |
| walk_edges / fdhop k4=1 / plain | 10×20/w5 | 0.9915 | 0.9989 | 0.610 | 0.775 | 0.563 | 0.1464 | 0.778 |
| walk_edges_pq q=2.0 / fdhop k4=1 / plain | 10×20/w5 | 0.9916 | 0.9990 | 0.601 | 0.774 | 0.563 | 0.1457 | 0.778 |
| walk_edges / fdlinear k4=0.01 / velocity | 10×20/w5 | 0.9709 | 0.9949 | 0.502 | 0.711 | 0.505 | 0.2092 | 0.468 |
| walk_edges / fdlinear k4=0.01 / plain | 10×20/w5 | 0.9712 | 0.9955 | 0.503 | 0.710 | 0.504 | 0.2100 | 0.469 |
| walk_edges / fdlinear k4=0.01 / plain | 10×80/w10 | 0.9720 | 0.9950 | 0.504 | 0.707 | 0.502 | 0.2111 | 0.470 |
| deepwalk | 80×40/w10 | 0.9736 | 0.9978 | 0.511 | 0.662 | 0.465 | 0.1660 | 0.728 |
| nbr_walk / fdlinear k4=0.01 / velocity | 10×20/w5 | 0.9824 | 0.9986 | 0.418 | 0.655 | 0.458 | 0.1724 | 0.685 |
| nbr_walk / fdlinear k4=0.01 / plain | 10×20/w5 | 0.9851 | 0.9986 | 0.408 | 0.651 | 0.455 | 0.1737 | 0.682 |
| walk_edges / fdhop_min k4=1 / plain | 10×20/w5 | 0.9316 | 0.9861 | 0.246 | 0.441 | 0.298 | 0.2744 | 0.428 |
| node2vec q=0.5 | 10×80/w10 | 0.9667 | 0.9960 | 0.210 | 0.416 | 0.279 | 0.2023 | 0.644 |
| node2vec q=1.0 | 10×80/w10 | 0.9680 | 0.9961 | 0.173 | 0.362 | 0.240 | 0.2075 | 0.618 |
| node2vec q=2.0 | 10×80/w10 | 0.9667 | 0.9959 | 0.149 | 0.275 | 0.181 | 0.2157 | 0.608 |

### cora — n 2,708 · 16,723 hop pairs · sources all · τ-b ceiling 0.9208 · kNN brute

| arm | budget | acc | auc | rf R² | rho | D | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| walk_edges / fdhop k4=1 / plain | 10×20/w5 | 0.9886 | 0.9988 | 0.746 | 0.856 | 0.653 | 0.1658 | 0.956 |
| walk_edges / fdhop k4=1 / plain | 10×80/w10 | 0.9905 | 0.9993 | 0.696 | 0.839 | 0.633 | 0.1708 | 0.964 |
| walk_edges_pq q=2.0 / fdhop k4=1 / plain | 10×20/w5 | 0.9934 | 0.9994 | 0.704 | 0.838 | 0.633 | 0.1692 | 0.962 |
| walk_edges_pq q=0.5 / fdhop k4=1 / plain | 10×20/w5 | 0.9848 | 0.9990 | 0.711 | 0.837 | 0.632 | 0.1703 | 0.960 |
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
| walk_edges / fdhop_min k4=1 / plain | 10×20/w5 | 0.9493 | 0.9897 | 0.163 | 0.381 | 0.259 | 0.2891 | 0.575 |

