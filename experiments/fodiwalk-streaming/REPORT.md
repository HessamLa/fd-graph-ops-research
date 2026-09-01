# Streaming fodiwalk across every graph the loader supports

Streaming means: for each batch of nodes, walk from those nodes only, fuse
the walks into three flat arrays (partner id, min hop `h`, frequency), take
the gradient for those rows, update them, and DISCARD everything. `D` is
never built and nothing of size `nnz` is ever live.

**Fourteen runs: seven graphs at 5 and at 50 epochs, 2,708 to 2,937,016
nodes and up to 11,095,298 edges. Every run finished.**

Driver: `bench_stream.py`. Runner: `run_campaign.sh`. Raw rows:
`results.tsv`. One log per run under `logs/`.

## Input parameters

Identical for every run except where the table says otherwise:

| parameter | value |
| --- | --- |
| walks per node | 5 |
| walk length | 20 steps |
| walk generator | uniform, first order (`p = q = 1`) -- the walk of DeepWalk and of node2vec at p=q=1 |
| pair policy | `nbr_walk`: every neighbour at `h = 1`, plus every node a walk from the row reached, at `h` = the first step |
| embedding dimension | 64 |
| force law | `fdlinear`, `k1 = 0.999`, `k4 = 0.01`, `kr = 1.0`, sign `-1.0` |
| weight | `min_gap` |
| learning rate | 1.0, constant |
| optimiser | plain (`Z + lr * dZ`) |
| batch (nodes per step) | 512 for cora, 1024 for the rest |
| epochs | 5 and 50 (see table) |
| seed | 42 |
| device | GPU, GeForce GTX 950, 2048 MiB |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 0.95 -- REQUIRED; the 75% default is not enough |
| scoring | `evaluator`, protocol `n2v1m`, LP `max_pairs = 50,000` |

Machine: 15 GB RAM, GTX 950 with 2048 MiB.

`max_pairs = 50,000` matches the archived node2vec run of
`experiments/large-graph-node2vec/`, which passed `--lp-pairs 25000` and
doubled it. It is an override of the `n2v1m` default of 80,000, so every LP
row here reports `protocol_modified=True`, exactly as that node2vec run
does. The two are comparable TO EACH OTHER; neither is the bare protocol.
Hop distance is unmodified on both sides.

## Results

| graph | nodes | edges | avg deg | epochs | wall s | peak RSS MB | accuracy | precision | recall | f1 | auc | hop R2 | hop MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cora` | 2,708 | 5,278 | 3.90 | 5 | 0.8 | 922 | 0.9522 | 0.9732 | 0.9299 | 0.9511 | 0.9921 | 0.0606 | 1.486 |
| `cora` | 2,708 | 5,278 | 3.90 | 50 | 2.6 | 922 | 0.9735 | 0.9902 | 0.9564 | 0.9730 | 0.9952 | 0.2179 | 1.354 |
| `pubmed` | 19,717 | 44,324 | 4.50 | 5 | 2.0 | 1,004 | 0.9566 | 0.9657 | 0.9468 | 0.9562 | 0.9919 | 0.1433 | 1.102 |
| `pubmed` | 19,717 | 44,324 | 4.50 | 50 | 13.0 | 994 | 0.9827 | 0.9914 | 0.9738 | 0.9825 | 0.9984 | 0.4037 | 0.924 |
| `wordnet` | 82,115 | 84,427 | 2.06 | 5 | 5.6 | 1,115 | 0.9557 | 0.9851 | 0.9254 | 0.9543 | 0.9898 | 0.0044 | 2.432 |
| `wordnet` | 82,115 | 84,427 | 2.06 | 50 | 49.5 | 1,113 | 0.9669 | 0.9945 | 0.9390 | 0.9660 | 0.9895 | 0.0104 | 2.432 |
| `com_youtube` | 1,134,890 | 2,987,624 | 5.27 | 5 | 74.3 | 1,690 | 0.9697 | 0.9728 | 0.9664 | 0.9696 | 0.9947 | 0.1062 | 0.942 |
| `com_youtube` | 1,134,890 | 2,987,624 | 5.27 | 50 | 699.5 | 1,674 | 0.9756 | 0.9855 | 0.9654 | 0.9753 | 0.9967 | 0.4084 | 0.739 |
| `as_skitter` | 1,696,415 | 11,095,298 | 13.08 | 5 | 137.9 | 2,749 | 0.9491 | 0.9602 | 0.9370 | 0.9485 | 0.9892 | 0.3900 | 0.613 |
| `as_skitter` | 1,696,415 | 11,095,298 | 13.08 | 50 | 1243.1 | 2,738 | 0.9811 | 0.9924 | 0.9696 | 0.9809 | 0.9987 | 0.4700 | 0.566 |
| `roadnet_ca` | 1,965,206 | 2,766,607 | 2.82 | 5 | 141.6 | 1,954 | 0.9825 | 0.9959 | 0.9690 | 0.9823 | 0.9994 | -0.0008 | 117.433 |
| `roadnet_ca` | 1,965,206 | 2,766,607 | 2.82 | 50 | 1360.4 | 1,935 | 0.9951 | 0.9980 | 0.9922 | 0.9951 | 1.0000 | -0.0043 | 117.595 |
| `ncbi_taxonomy` | 2,937,016 | 2,937,015 | 2.00 | 5 | 278.5 | 2,207 | 0.9836 | 0.9913 | 0.9758 | 0.9835 | 0.9974 | 0.0109 | 9.500 |
| `ncbi_taxonomy` | 2,937,016 | 2,937,015 | 2.00 | 50 | 2747.9 | 2,214 | 0.9843 | 0.9975 | 0.9710 | 0.9841 | 0.9946 | 0.0230 | 9.461 |


### Cost by stage, and the pairs that were thrown away

| run | pairs per epoch | pairs used and discarded | t_load | t_embed | t_lp | t_da |
| --- | --- | --- | --- | --- | --- | --- |
| `cora` e5 | 121,588 | 607,944 | 0.0 | 0.8 | 1.2 | 1.5 |
| `cora` e50 | 121,595 | 6,079,787 | 0.0 | 2.6 | 1.3 | 4.2 |
| `pubmed` e5 | 1,228,837 | 6,144,189 | 0.0 | 2.0 | 9.3 | 9.1 |
| `pubmed` e50 | 1,228,299 | 61,414,982 | 0.1 | 12.9 | 12.8 | 2.9 |
| `wordnet` e5 | 2,410,702 | 12,053,513 | 0.2 | 5.4 | 11.3 | 5.8 |
| `wordnet` e50 | 2,411,279 | 120,563,985 | 0.2 | 49.3 | 9.7 | 5.0 |
| `com_youtube` e5 | 77,252,299 | 386,261,499 | 2.4 | 71.9 | 15.8 | 90.9 |
| `com_youtube` e50 | 77,257,076 | 3,862,853,800 | 2.3 | 697.2 | 16.5 | 83.0 |
| `as_skitter` e5 | 139,627,470 | 698,137,352 | 14.3 | 123.5 | 30.3 | 178.8 |
| `as_skitter` e50 | 139,625,105 | 6,981,255,285 | 13.6 | 1229.6 | 23.6 | 153.7 |
| `roadnet_ca` e5 | 56,683,080 | 283,415,404 | 5.8 | 135.9 | 17.7 | 131.2 |
| `roadnet_ca` e50 | 56,684,851 | 2,834,242,550 | 6.4 | 1354.1 | 16.0 | 116.1 |
| `ncbi_taxonomy` e5 | 109,721,502 | 548,607,514 | 3.7 | 274.8 | 22.2 | 149.7 |
| `ncbi_taxonomy` e50 | 109,720,831 | 5,486,041,584 | 4.3 | 2743.6 | 18.3 | 134.5 |

## 1. Memory does not track graph size

922 MB at 2,708 nodes. 2,749 MB at 1,696,415 nodes and 11,095,298 edges.
**A 626x larger graph costs 3.0x the memory**, and most of the 922 MB floor
is the JAX runtime, not data: `Z` for cora is 1 MB.

The peak is one BATCH plus `Z`, and the batch is a parameter. It stops
moving after the first epoch: `com_youtube` reports the same peak at 5
epochs (1,690 MB) and at 50 (1,674 MB), while the pairs it consumed rose
from 386 million to 3.86 BILLION.

Batch size is the dial. Measured at 150,000 nodes of com_youtube:

| batch | peak RSS | per epoch |
| --- | --- | --- |
| 4096 | 1,523 MB | 5.4 s |
| 1024 | 1,140 MB | 7.6 s |
| 256 | 943 MB | 17.6 s |

**Streaming does NOT win on small graphs.** At 150,000 nodes it needs
1,523 MB against the stored-`D` path's 1,467 MB, because the runtime floor
swamps the data. It pays only once `D` would dominate.

## 2. Against node2vec on com_youtube

The one graph with a recorded baseline
(`experiments/large-graph-node2vec/node2vec_com_youtube_1M_dim64.log`).
Same walks: 5 x 20, seed 42, dim 64.

| | node2vec | streaming fodiwalk @50 |
| --- | --- | --- |
| wall time | 888.8 s | **699.5 s** |
| peak RSS | **1,382 MB** | 1,674 MB |
| accuracy | 0.9750 | **0.9756** |
| f1 | 0.9753 | 0.9753 |
| auc | 0.9964 | **0.9967** |
| **hop R2** | 0.042 | **0.4084** |
| hop MAE | 0.910 | **0.739** |

**Faster, better or equal on every quality metric, 9.7x better on hop
distance, at 1.21x the memory.** Link prediction is a tie on f1 and a hair
ahead on accuracy and AUC. Memory is the only axis node2vec still wins, and
the gap is now 292 MB rather than the 4,515 MB it was with stored `D`.

Against the stored-`D` path on the same graph (5,897 MB peak, 461.6 s at
200 epochs, R2 0.4283): streaming reaches the same quality band at **3.5x
less memory** for about 1.5x the time.

## 3. Epochs matter more than expected, and cost nothing in memory

| graph | 5 epochs | 50 epochs |
| --- | --- | --- |
| cora accuracy | 0.9522 | 0.9735 |
| pubmed accuracy | 0.9566 | 0.9827 |
| com_youtube accuracy | 0.9697 | 0.9756 |
| cora hop R2 | 0.0606 | 0.2179 |
| pubmed hop R2 | 0.1433 | 0.4037 |
| com_youtube hop R2 | 0.1062 | 0.4084 |

pubmed gains 2.6 accuracy points and triples its hop R2 for 11 extra
seconds. Peak memory is unchanged in every case. **Five epochs undersells
this method**; it was the streaming default, not a converged setting.

## 4. Hop distance splits by AVERAGE DEGREE, and the split is sharp

Hop R2, 5 epochs -> 50 epochs, ordered by average degree:

| graph | avg degree | 5 epochs | 50 epochs | |
| --- | --- | --- | --- | --- |
| `as_skitter` | 13.08 | +0.3900 | **+0.4700** | improves |
| `com_youtube` | 5.27 | +0.1062 | **+0.4084** | improves |
| `pubmed` | 4.50 | +0.1433 | **+0.4037** | improves |
| `cora` | 3.90 | +0.0606 | +0.2179 | improves |
| `roadnet_ca` | 2.82 | -0.0008 | -0.0043 | flat |
| `wordnet` | 2.06 | +0.0044 | +0.0104 | flat |
| `ncbi_taxonomy` | 2.00 | +0.0109 | +0.0230 | flat |

**The split is clean at an average degree near 3, with nothing in between.**
Every graph above it gains substantially from more epochs; every graph below
it stays put. The three that fail are two trees (`ncbi_taxonomy` at exactly
2.00, `wordnet` at 2.06) and a road network (`roadnet_ca`, MAXIMUM degree
12, high diameter).

**This is structural, not under-training, and the 50-epoch runs prove it.**
Ten times the epochs moved `roadnet_ca` from -0.0008 to -0.0043 and left its
hop MAE at 117.4 -> 117.6. `ncbi_taxonomy` went 0.0109 -> 0.0230, still an
order below the dense graphs. A road network's hop distances run to hundreds
of steps and a sparse tree's are dominated by depth; 64 Euclidean dimensions
hold neither. More epochs will not fix this: it is the wrong metric for these
graphs, or the wrong dimension, or both.

**Link prediction does not care.** `roadnet_ca` posts the best AUC of the
campaign -- 0.9994 at 5 epochs and **1.0000 at 50** -- while its hop R2 is
NEGATIVE. `ncbi_taxonomy` reaches 0.9836 accuracy with hop R2 0.023. The two
tasks measure different things, and these graphs are the sharpest
demonstration of it in the campaign. A report that gives only one of them is
misleading.

## 5. What is NOT established here

- **Convergence.** Nothing here shows streaming reaches the same optimum
  as the stored-`D` path. It reaches a comparable quality band at 50
  epochs on com_youtube. That is a measurement, not a proof.
- **The optimiser is plain, and only plain**: `Z + lr * dZ` with `lr = 1.0`
  fixed, no momentum, no Adam, no state between steps, no learning-rate
  schedule. The degree division (`F * inv_deg`) is the only per-row scaling.
  `optim.py`'s other seven rules were not tried under streaming. On the
  stored path Adam scored far WORSE than plain on this force law (cora
  accuracy 0.6188 against 0.9621), so plain is the rule this physics works
  with -- but that was not re-checked here.
- **Bit-exactness is not a goal** and was not tested. The user set that
  aside on 2026-08-28: the objective at this stage is memory.
- Streaming changes the algorithm twice over, and both are inherent:
  walks are **fresh in every epoch** (a stochastic sample, not one fixed
  `D`), and the update is **immediate per batch** (Gauss-Seidel, not the
  stored path's Jacobi).
- Only `nbr_walk` was used. The symmetric `walk` policy does not reach
  these sizes (`experiments/fodiwalk/REPORT_1M.md` section 8).

## Reproduce

```bash
./experiments/fodiwalk-streaming/run_campaign.sh \
    cora:5:512:gpu pubmed:50:1024:gpu com_youtube:50:1024:gpu
```

The argument is `graph:epochs:batch:device`. Each run appends one row to
`results.tsv` and writes its own log, so a failure leaves earlier rows
intact and names itself in the gap.
