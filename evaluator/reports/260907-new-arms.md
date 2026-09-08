# New arms scored — 2026-09-07

Nine previously unscored records, plus six control rows. `fdunit`,
`fdunit_freq` and `fdhop2` were excluded by instruction.

| | |
| --- | --- |
| Records in store | 58 |
| Records scored (merged file) | **50** |
| Excluded by instruction | 8 (`fdunit` ×4, `fdunit_freq` ×2, `fdhop2` ×2) |
| Errors | 0 |
| Merged data | `data_cache/evaluator/store-scores/260907-all-scores.jsonl` |
| This run only | `data_cache/evaluator/store-scores/260907-new-scores.jsonl` |
| Previous set | `260905-all-scores.jsonl` (41 rows) |

---

## 1. The merge is verified, not assumed

Six already-scored rows were re-run alongside the new ones. Maximum absolute
difference against their 2026-09-05 values:

| metric | max difference |
| --- | --- |
| accuracy, auc, f1_score | **0.000e+00** |
| mlp R², rho, Somers' D, stress | **0.000e+00** |
| rf R² | 2.220e-16 |
| `rec@10` | **7.625e-04** |

Every metric that depends on the hop pair sample reproduces **exactly**, so
the fixture rebuilt identically and merging old with new rows is sound. rf R²
moves at the `n_jobs=-1` thread-order floor.

**`rec@10` is the exception, and it is a property of the metric.** Only
`as_skitter` moved; cora and pubmed reproduced to zero. Those two use exact
brute-force kNN, `as_skitter` uses `pynndescent` above 200,000 nodes.

**New finding: the approximate kNN path is not deterministic across runs.**
Same `Z`, same seed, same query nodes, different neighbours at the 1e-4
level. The `rec@10~` column was already marked as not comparable across
graphs; it is now also known not to be reproducible run to run. Treat a
difference below about 1e-3 on an approximate row as noise.

---

## 2. `walk_edges_pq` — the p/q bias does not transfer to fodiwalk

The clearest result of this run, and it is negative.

### cora

| arm | acc | auc | rf R² | rho | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- |
| walk_edges_pq q=2.0 / fdhop | 0.9934 | 0.9994 | 0.704 | 0.838 | 0.1692 | 0.962 |
| walk_edges_pq q=0.5 / fdhop | 0.9848 | 0.9990 | 0.711 | 0.837 | 0.1703 | 0.960 |
| walk_edges / fdhop | 0.9882 | 0.9990 | 0.705 | 0.835 | 0.1705 | 0.959 |

### pubmed

| arm | acc | auc | rf R² | rho | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- |
| walk_edges_pq q=0.5 / fdhop | 0.9909 | 0.9987 | 0.588 | 0.775 | 0.1464 | 0.784 |
| walk_edges / fdhop | 0.9915 | 0.9989 | 0.610 | 0.775 | 0.1464 | 0.778 |
| walk_edges_pq q=2.0 / fdhop | 0.9916 | 0.9990 | 0.601 | 0.774 | 0.1457 | 0.778 |

**rho spans 0.003 on cora and 0.001 on pubmed.** Both are inside the noise a
single seed can support, and the ordering flips between the two graphs.

Compare node2vec, where the same parameter is a strong axis:

| q | node2vec rho, cora | walk_edges_pq rho, cora |
| --- | --- | --- |
| 0.5 | 0.742 | 0.837 |
| 1.0 | 0.711 | 0.835 (plain `walk_edges`) |
| 2.0 | 0.638 | 0.838 |
| **span** | **0.104** | **0.003** |

The search bias moves node2vec by 0.104 and fodiwalk by 0.003 — a factor of
35. **Whatever the p/q bias buys a skip-gram model, the force-directed
layout already has.** That is a real finding about the method, not a null
result from a weak test: the same knob on the same graph with the same
budget produces a large effect in one pipeline and none in the other.

---

## 3. `fdhop_min` is much worse than `fdhop`

| graph | arm | acc | auc | rf R² | rho | stress | rec@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cora | fdhop | 0.9882 | 0.9990 | **0.705** | **0.835** | **0.1705** | **0.959** |
| cora | fdhop_min | 0.9493 | 0.9897 | 0.163 | 0.381 | 0.2891 | 0.575 |
| pubmed | fdhop | 0.9915 | 0.9989 | **0.610** | **0.775** | **0.1464** | **0.778** |
| pubmed | fdhop_min | 0.9316 | 0.9861 | 0.246 | 0.441 | 0.2744 | 0.428 |

Worse on every column on both graphs, by a wide margin — hop R² falls by a
factor of four on cora. This is not a close call and does not need a second
seed to read.

---

## 4. A newer `fdhop` run scores better than the older ones

Three cora records claim `force.law = fdhop` with `walk_edges`, `plain`,
10×20/w5, 200 epochs:

| record | commit | `force.params` | rf R² | rho |
| --- | --- | --- | --- | --- |
| `260906-204215-c0fedf2b` | `ea9ae67` | k1, k4=1.0, kr | **0.746** | **0.856** |
| `260904-072752-6047dac7` | `864c5b4` | k1, k4=1.0, kr | 0.705 | 0.835 |
| `260904-071825-0ba3b1b3` | `864c5b4` | k1, kr (no k4) | 0.705 | 0.835 |

**The newer record is on a different commit** — `ea9ae67` against `864c5b4`
— and scores higher on both geometry columns. All three record
`git_dirty: true`.

Two readings and the store cannot separate them: `fdhop` changed between the
two commits, or this is seed-level variation. The two `864c5b4` records agree
to four decimals with each other despite differing in whether `k4` was
recorded, which argues the spread is not seed noise — but that is inference,
not measurement. **A second run at `ea9ae67` would settle it.**

Anyone citing an `fdhop` number should say which commit produced it.

---

## 5. `as_skitter` q-sweep, now complete

| q | rho | rf R² | acc | auc |
| --- | --- | --- | --- | --- |
| 0.5 | −0.0023 | −0.0175 | 0.9579 | 0.9940 |
| 1.0 | −0.0400 | −0.0003 | 0.9547 | 0.9918 |
| 2.0 | −0.0658 | +0.0059 | 0.9581 | 0.9939 |

All three are at or below zero. The ordering follows the DFS-favouring
direction seen on cora, pubmed and com_youtube, but every value is
indistinguishable from no relationship at all, so the ordering carries no
weight here.

AUC stays above 0.99 throughout. `as_skitter` remains the sharpest case of
link prediction reporting success on a layout that holds no distance
information — and it still has no fodiwalk arm to compare against.

---

## 6. What this run does not establish

- **`walk_edges_pq` and `fdhop_min` exist on cora and pubmed only.** No
  wordnet, no com_youtube, no as_skitter. Both conclusions are limited to
  the two small citation graphs.
- **One seed per configuration.** The `walk_edges_pq` result is a
  no-difference finding at single seed; it shows the effect is far smaller
  than node2vec's, not that it is exactly zero.
- **`fdunit`, `fdunit_freq` and `fdhop2` were excluded by instruction** and
  are unscored. Eight records in the store carry no numbers here.
- **The `ea9ae67` versus `864c5b4` question is open**, and it affects every
  `fdhop` number in the file.
