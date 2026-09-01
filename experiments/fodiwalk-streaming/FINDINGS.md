# Findings: streaming fodiwalk, memory, and the optimiser

Recorded 2026-08-28 and 2026-08-29. Every number here was measured on this
machine (15 GB RAM, GeForce GTX 950 with 2048 MiB) and re-run by the
coordinator, not taken from an agent's report.

Raw data: `results.tsv`, `optim_pubmed.tsv`, `final_*.tsv`, `logs/`.
Related: `../fodiwalk/REPORT_1M.md`, `REPORT.md`.

---

## 1. `D` is dropped. Three flat arrays for each node

The augmented graph is no longer a matrix. For each node it is **partner
id, min hop `h`, frequency**, each with its own best data type.

**Nothing in the tree does matrix algebra on `D`.** Checked: no `.dot`, no
`@`, no transpose, no axis sum, no `getrow`, no `toarray`. The only
`D[i, j]` references anywhere are DOCSTRINGS describing an "indexability
contract" that no live code exercises (`fodiwalk/base.py:52`,
`forcedirected/csr.py:16`). Every real reader touches `.indptr`,
`.indices`, `.data`, `.shape`, `.nnz` and nothing else, so a plain object
substitutes for `scipy.sparse.csr_matrix` with NO change to
`forcedirected/`. The one exception is `merge.add_far_pairs`, which needs
`.tocoo()` because it depends on scipy SUMMING duplicate COO coordinates.

Two strategies, both wanted:

* **precomputed** -- walk once, aggregate once, reuse for every epoch.
* **streaming** -- aggregate inside each epoch, use, discard. Nothing of
  size `nnz` is ever live.

**Bit-exactness is not a goal at this stage** (project owner, 2026-08-28).
`golden.py` and `ab.py` bound the earlier code-move work; they do not bind
this line, and a changed digest is not a defect here.

## 2. The memory rewrite (commit a837f39)

`nbr_walk` augmentation, com_youtube cut to 150,000 nodes, measured on the
same script against both trees:

| | before | after |
| --- | --- | --- |
| peak RSS | 3055 MB | **1195 MB** |
| wall time | 23.9 s | **12.2 s** |
| peak / resting | 6.17x | 2.41x |

Full com_youtube then completed for the first time: 1,134,890 nodes,
D.nnz 145,222,742, **4630 MB peak, 91 s**. The 2026-08-14 report
("node2vec finishes, fodined does not") is answered.

What was wrong: a global `np.argsort` over 26M keys THAT WERE ALREADY
SORTED (+203 MB permutation, +609 MB four-array gather); a `row * n + col`
int64 key array (+198 MB) carrying a row number CSR offsets give free;
`D` and `freq` storing the same structure twice; a COO round trip that
sorts and de-duplicates input already sorted and unique; and a
`counts.astype(np.float64)` on an array that was already float64.

**Verified byte-exact past where the gates reach.** The merge block is
20,000 rows and cora is 2,708, so `golden.py` only ever exercised ONE
block. Checked against the pre-change tree at n=60,000 (three blocks,
10.5M pairs): `indptr`, `indices`, `data`, `freq` all identical. Then
eight configs at n=50,000 -- default, low_deg, rowcap16, rowcap_lowdeg,
buckets, freq_node, node2vec, far2000 -- every field identical.

## 3. Why fodiwalk used more memory than node2vec

Same walks; opposite fate.

**node2vec stores no pair.** Its walks go to DISK -- 113,489,000 tokens,
0.74 GB, **6.52 bytes per walk index as text** -- and
`Word2Vec(corpus_file=...)` streams them. RAM holds only the model:

| component | MB |
| --- | --- |
| `wv.vectors` 1,134,890 x 64 float32 | 277.1 |
| `syn1neg` (negative=5) | 277.1 |
| vocab `key_to_index` + `index_to_key` (measured 115 B/key) | 124.3 |
| `cum_table` | 8.7 |
| accounted | 687.1 |
| whole-run peak (logged) | **1382** |

**fodiwalk stores every pair**, because the force law reads a whole ROW
each epoch for hundreds of epochs. At 150,000 nodes the live buffers are:

| buffer | MB | note |
| --- | --- | --- |
| plan tiles | 187.5 | padded float32 -- a SECOND copy of the pairs |
| `D.data` float64 | 108.0 | `h`, an integer 1..19 |
| `freq` float64 | 108.0 | a count |
| `D.indices` int32 | 54.0 | needed |
| sum live | 482.7 | peak during plan build 1467 |

Both planes ALIAS their sources (`plane[0] is D.data`, `plane[1] is
freq`) -- there is no redundant plane copy. The avoidable costs are the
float64 storage of integer-valued data (the kernel casts to float32 at
`sell_c_sigma.py:308` anyway) and the plan being a second full copy.

## 4. The GPU wall was the allocator, not the card

Three runs died at `RESOURCE_EXHAUSTED: allocating 290,531,840 bytes`
under `--chunks` 1, 4 and 8 alike. That is exactly `1,134,890 x 64 x 4`:
one full `(n, d)` float32 array.

Two structural facts:

* **`--chunks` cannot bound device memory.** `embed/planner.py:117` is
  `plans.append(device_put(plan) if resident else plan)` INSIDE the chunk
  loop, with `resident = not chunk_host` at line 87. Without
  `--chunk-host` every chunk stays on the card.
* **The failure is not in the plan.** `optim.step_plain` is
  `Z + lr * dZ` on the FULL, unchunked `Z` and `dZ`, so the floor is
  `Z + dZ + one temporary` = **831.2 MB** however the plan is split.

But 831 MB is 41% of 2048 MiB. `XLA_PYTHON_CLIENT_PREALLOCATE=false` did
NOT help (identical failure). **`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`
DID**: the run completes at peak GPU 1984-1993 MB. JAX's default 75%
preallocation caps the pool near 1536 MB, and that was the whole obstacle.
An earlier conclusion that "the card is too small" was WRONG.

GPU is **7.9x** faster than CPU for this: 1.13 s/epoch against 8.94.

## 5. Stored `D` against node2vec at a million nodes

| | node2vec | fodiwalk @200 | fodiwalk @600 |
| --- | --- | --- | --- |
| wall time | 888.8 s | **461.6 s** | 875.1 s |
| peak RSS | **1382 MB** | 5897 MB | 5897 MB |
| accuracy | 0.9750 | **0.9754** | 0.9718 |
| f1 | 0.9753 | 0.9752 | 0.9714 |
| auc | 0.9964 | **0.9973** | 0.9964 |
| hop R2 | 0.042 | **0.4283** | 0.4332 |

**Link prediction PEAKS at epoch 200 and then FALLS** -- 0.9754, 0.9718,
0.9628 at 2000 -- while hop R2 rises the whole way, 0.4283, 0.4332,
0.4625. The two tasks want different amounts of relaxation, and the
package default of 2000 epochs is past the best point for link
prediction. A report that gives one metric picks the wrong epoch count.

## 6. Streaming: the campaign

14 runs, 7 graphs, 2,708 to 2,937,016 nodes, up to 11,095,298 edges. All
finished. Full table in `REPORT.md`.

**Memory does not track graph size**: 922 MB at 2,708 nodes, 2749 MB at
1,696,415 nodes and 11M edges -- a 626x larger graph for 3.0x the memory,
and most of the 922 MB floor is the JAX runtime. The peak is one BATCH
plus `Z` and it stops moving after epoch 1: com_youtube reports the same
peak at 5 epochs (1690 MB) and at 50 (1674 MB) while the pairs consumed
rise from 386 million to 3.86 billion.

Batch is the dial (150,000 nodes): 4096 -> 1523 MB at 5.4 s/epoch;
1024 -> 1140 MB at 7.6 s; 256 -> 943 MB at 17.6 s.

**Streaming does NOT win on small graphs**: 1523 MB against the stored
path's 1467 MB at 150,000 nodes. It pays only once `D` would dominate.

Against node2vec on com_youtube at 50 epochs: **699.5 s against 888.8 s,
1674 MB against 1382 MB (1.21x), accuracy 0.9756 against 0.9750, f1
level, auc 0.9967 against 0.9964, hop R2 0.4084 against 0.042.** The
memory gap that was 4515 MB with stored `D` is now 292 MB.

## 7. Hop distance splits at average degree ~3

| graph | avg degree | 5 epochs | 50 epochs | |
| --- | --- | --- | --- | --- |
| `as_skitter` | 13.08 | +0.3900 | +0.4700 | improves |
| `com_youtube` | 5.27 | +0.1062 | +0.4084 | improves |
| `pubmed` | 4.50 | +0.1433 | +0.4037 | improves |
| `cora` | 3.90 | +0.0606 | +0.2179 | improves |
| `roadnet_ca` | 2.82 | -0.0008 | -0.0043 | flat |
| `wordnet` | 2.06 | +0.0044 | +0.0104 | flat |
| `ncbi_taxonomy` | 2.00 | +0.0109 | +0.0230 | flat |

Clean separation with nothing in between. **Structural, not
under-training**: ten times the epochs moved `roadnet_ca` from -0.0008 to
-0.0043 with hop MAE stuck at 117.4 -> 117.6. Two trees and a road
network fail; 64 Euclidean dimensions cannot hold their distances.

**Link prediction does not care.** `roadnet_ca` posts the campaign's best
AUC -- 0.9994 at 5 epochs and **1.0000 at 50** -- with a NEGATIVE hop R2.

## 8. The effective-learning-rate law

The force-directed method converges when `lim(x->inf) f(x)/x < 1.0`
(project owner). The operational form:

    effective lr = lr x the DC gain of the update rule,  edge at 1.0

| rule | DC gain |
| --- | --- |
| `plain`, `sgd`, `velocity` | 1 |
| `momentum` `m = b*m + dZ` | `1/(1-b)` = **10** at b=0.9 |
| `nesterov` | `1 + b/(1-b)` = `1/(1-b)` = **10** at b=0.9 |
| `fa2` | speed capped at `k_max` = **10** |
| `adam`, `sqn` | adaptive, no fixed gain |

**This predicted all 16 recorded outcomes with no misses**, including
every divergence. `momentum` at lr 0.1 works because 0.1 x 10 = 1.0
exactly; at 0.9, 0.999 and 1.0 the effective lr is 9-10 and every run
produced `nan`. It then predicted `fa2`'s divergence from a mechanism it
had never seen: `fa2` at lr 0.999 died at epoch 15 on pubmed with 891,520
non-finite values, and runs at lr 0.0999.

**Generalised momentum** `m = b*m + a*dZ` has gain `a/(1-b)`, so
`a + b = 1` gives gain 1 and IS `velocity`; `a + b < 1` gives gain < 1.
Measured on cora, 50 epochs, lr 1.0, streaming:

| config | a+b | gain | eff lr | dz | max\|Z\| | acc | auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `plain` | -- | 1.00 | 1.00 | 10.9 | 119 | 0.9735 | 0.9952 |
| `momentum` a=1.0 b=0.9 | 1.90 | 10.0 | 10.0 | **1.11e9** | **6.0e10** | 0.8916 | 0.9529 |
| `momentum` a=0.5 b=0.4 | 0.90 | 0.833 | 0.833 | 10.3 | 114 | **0.9740** | 0.9952 |

The classic rule blows up by eight orders of magnitude; `a=0.5, b=0.4`
lands it back beside `plain` and slightly ahead on accuracy. NOTE: it did
not reach `nan` here, so the finite-check did not fire -- streaming's
per-batch updates kept it just short of overflow where the stored path
went to `nan`. `dz` is the honest signal, not the finite flag.

**GLOBAL RULE (project owner, 2026-08-29): no learning rate is ever 1.0.**
0.999 or less, and a decay schedule starts at 0.999 or less. This binds
the effective lr too, so a gain-10 rule needs `lr < 0.1`.

## 9. Decay REVERSES between the two strategies

Stored `D`, pubmed, lr 0.999: decay was the single biggest lever, and it
rescued exactly the runs whose step was too large. `adam` 0.219 -> 0.517,
`sgd` 0.275 -> 0.496, `plain` 0.288 -> 0.494 -- every one of those had
`dz > 1` without it. The already-stable rules (`velocity` 0.505,
`sqn` 0.508) gained nothing.

Streaming, pubmed, 200 epochs, balanced grid of 6 rules x 2 schedules at
effective lr 0.999 for every row:

| rule | const | linear | delta |
| --- | --- | --- | --- |
| `nesterov` | 0.5820 | 0.4626 | **-0.119** |
| `adam` | 0.4395 | 0.3393 | **-0.100** |
| `sgd` | 0.4486 | 0.3571 | -0.092 |
| `plain` | 0.5808 | 0.4921 | -0.089 |
| `velocity` | **0.5871** | 0.5062 | -0.081 |
| `fa2` | 0.2041 | 0.1765 | -0.028 |

**Decay hurts all six, and the effect is the same size at 50 and at 200
epochs** (-0.08 to -0.12 both times), so it is NOT an under-training
artefact -- which was the confound the 200-epoch arm was run to settle.
Streaming already injects noise through fresh walks each epoch and does
not need annealing.

## 10. Optimiser verdict

**`plain`, lr 0.999, constant.**

`velocity` scored 0.5871 against `plain`'s 0.5808 -- a gap of 0.0063. That
is NOT a real difference: the recorded three-seed runs on pubmed show hop
R2 spreads of **0.039-0.046 across seeds alone**
(`D_p1.0_q1.0`: 0.427 / 0.413 / 0.459), and this grid ran ONE seed. The
top three -- `velocity` 0.5871, `nesterov` 0.5820, `plain` 0.5808 -- are
indistinguishable.

The tiebreak is cost. State arrays of the shape of `Z`, at 1,134,890
nodes and dim 64 (one array = 277 MB):

| rule | arrays | MB @1M | fits 2048 MiB |
| --- | --- | --- | --- |
| `plain`, `sgd` | 0 | 0 | yes |
| `momentum`, `nesterov`, `fa2`, `velocity` | 1 | 277 | yes |
| `adam` | 2 | 554 | yes |
| `sqn` | 8 | 2217 | **NO** |

`plain` is free, tied on quality, and has **zero divergences in 230+
recorded runs**. `sqn` cannot run at a million nodes at all -- a real
loss, since its quality is good (cora 0.597), not a dismissal.

**Nothing converged at 200 epochs.** Every config gained +0.12 to +0.22
hop R2 from 50 to 200. Epochs are currently a bigger lever than the
optimiser, and they cost no memory.

## 11. Corrections made during this work

Recorded because each one was stated confidently and was wrong.

1. **"The card is too small."** It was JAX's 75% preallocation.
   `MEM_FRACTION=0.95` runs it.
2. **"Adam collapses under streaming" (0.1295, 4x worse).** That was
   linear decay plus a 50-epoch budget. At const and 200 epochs Adam
   reaches 0.4395. Still mid-table; the claim was overstated.
3. **"Adam scores 0.6188 on cora."** True for ONE A/B cell
   (`nbr_walk` + adam + lr 0.1). Across 7 recorded cora runs Adam
   averages 0.9201. The figure was a worst case quoted as typical.
4. **"plane[1] is a separate copy."** It aliases `freq`. The measuring
   script compared it against `D.data` only.
5. **A benchmark was run on `pairs=walk`** -- the symmetric policy, which
   the memory work never touched -- instead of `nbr_walk`. It burned 20
   minutes and pushed the machine to 80% swap. It did produce a real
   finding (section 12).
6. **Streaming resting memory.** The predicted 5.1 GB -> 2.1 GB drop did
   NOT land: `D.data` stays float64 because `golden.py` hashes it.

## 12. Footnote: the `walk` policy does not reach a million nodes

`walk_pair_stats` prunes when its accumulator passes a FIXED
`prune_max = 4,000,000`, while the natural per-node ceiling grows with
`n`. Past ~70,000 nodes the prune fires on nearly every block without
shrinking much, so the cost becomes a repeated full sort of a growing
array. Measured: n=50,000 -> 15.60 s (4 prunes); n=100,000 -> 66.15 s
(9 prunes). **4.2x the time for 2x the nodes.** Not fixed --
`fodiwalk/` was off limits during a benchmark.
