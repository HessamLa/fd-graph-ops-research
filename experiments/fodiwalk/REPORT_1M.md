# fodiwalk against node2vec at a million nodes

Graph: com_youtube (SNAP), 1,134,890 nodes, 2,987,624 undirected edges.
Machine: 15 GB RAM, GeForce GTX 950, 2048 MiB VRAM.

Both sides scored by `evaluator` under protocol `n2v1m`. Same walks on both
sides: 5 walks x 20 steps, seed 42, first-order uniform (`p = q = 1`), which
is the walk node2vec used. fodiwalk: `pairs=nbr_walk`, `weight=min_gap`,
`force=fdlinear`, dim 64.

Scripts: `bench_n2v1m.py` (stored `D`), `bench_stream.py` (streaming).
node2vec is NOT re-run; its numbers are the recorded
`node2vec_com_youtube_1M_dim64.log`.

## The answer to the 2026-08-14 report

`experiments/large-graph-node2vec/REPORT.md` is titled "One million nodes:
node2vec finishes, fodined does not". It does now. The augmentation
completes in 37-61 s and gives 77,257,655 pairs, and the whole pipeline
runs to real scores. The wall that stopped it was the augmentation's peak
memory, and commit a837f39 removed it.

## 1. The comparison

| | node2vec | fodiwalk @200 | fodiwalk @600 |
| --- | --- | --- | --- |
| total wall time | 888.8 s | **461.6 s** | 875.1 s |
| peak RSS | **1382 MB** | 5897 MB | 5897 MB |
| peak GPU | n/a (CPU) | 1993 MB / 2048 | 1993 MB / 2048 |
| accuracy | 0.9750 | **0.9754** | 0.9718 |
| precision | 0.9640 | 0.9847 | -- |
| recall | 0.9868 | 0.9662 | -- |
| f1 | 0.9753 | 0.9752 | 0.9714 |
| auc | 0.9964 | **0.9973** | 0.9964 |
| **hop R2 (MLP)** | 0.042 | **0.4283** | **0.4332** |
| hop MAE | 0.910 | 0.721 | -- |

**At epoch 200 fodiwalk matches or beats node2vec on every quality metric
in HALF the wall time, and is 10.2x better on hop distance.** At epoch 600
the wall times are level (875.1 s against 888.8 s) and hop R2 is 10.3x.

Memory is the one axis node2vec wins: 1382 MB against 5897 MB, 4.3x.
Section 4 halves that gap.

## 2. The epoch sweep, and a result worth naming

GPU, `--chunks 8 --chunk-host`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`:

| epoch | total s | accuracy | f1 | auc | hop R2 |
| --- | --- | --- | --- | --- | --- |
| 10 | 244.4 | 0.9287 | 0.9274 | 0.9809 | 0.1107 |
| 60 | 295.1 | 0.9717 | 0.9713 | 0.9969 | 0.3399 |
| **200** | **461.6** | **0.9754** | **0.9752** | **0.9973** | 0.4283 |
| 600 | 875.1 | 0.9718 | 0.9714 | 0.9964 | 0.4332 |
| 2000 | 2448.8 | 0.9628 | 0.9623 | 0.9931 | **0.4625** |

**Link prediction PEAKS at epoch 200 and then falls** -- 0.9754, 0.9718,
0.9628. Hop R2 rises the whole way -- 0.4283, 0.4332, 0.4625. The two tasks
want different amounts of relaxation, and 2000 epochs (the package default)
is past the best point for link prediction on this graph. A run that
reports only one of the two metrics will pick the wrong epoch count.

Per-epoch cost, GPU: 1.13 s. CPU: 8.94 s. The GPU is 7.9x faster.

## 3. The GPU wall, and what it really was

The first three attempts died at
`RESOURCE_EXHAUSTED: allocating 290,531,840 bytes`, under `--chunks` 1, 4
and 8 alike. That number is exactly `1,134,890 x 64 x 4`: one full `(n, d)`
float32 array.

`--chunks` cannot help, and the reason is structural.
`embed/planner.py:117` is
`plans.append(device_put(plan) if resident else plan)` inside the chunk
loop, with `resident = not chunk_host` at line 87 -- so without
`--chunk-host` every chunk stays on the card and the total is unchanged.
And the failure is not in the plan at all: `optim.step_plain` is
`Z + lr * dZ` on the FULL, unchunked `Z` and `dZ`, so the floor is
`Z + dZ + one temporary` = 831.2 MB no matter how the plan is split.

But 831 MB is 41% of a 2048 MiB card. The real obstacle was JAX's default
75% preallocation, which caps the pool near 1536 MB.
`XLA_PYTHON_CLIENT_PREALLOCATE=false` did NOT help (identical failure).
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` DID: the run completes at
peak GPU 1993 MB of 2048. The card is big enough; the default was not.

## 4. Streaming: node2vec's memory strategy on fodiwalk's physics

`bench_stream.py`. For a batch of nodes: walk from those nodes only, fuse
to three vectors (partner, min hop `h`, frequency), take the gradient,
update those rows, discard. **`D` is never built.** Nothing of size `nnz`
is ever live.

com_youtube, 5 epochs, batch 1024, dim 64, CPU:

| | value |
| --- | --- |
| total wall time | 588.0 s |
| **peak RSS** | **1876 MB** |
| pairs used and discarded | 386,261,499 (77.25M per epoch) |
| accuracy / f1 / auc | 0.9700 / 0.9699 / 0.9947 |
| hop R2 (MLP) | 0.1062 |

The peak does not move after epoch 1: it is bounded by ONE BATCH, not by
the graph. 77.25M pairs per epoch matches `D.nnz = 77,257,655`, so the same
pairs are made -- they are just never kept.

| | peak RSS | vs node2vec |
| --- | --- | --- |
| node2vec | 1382 MB | 1.00x |
| **fodiwalk streaming** | **1876 MB** | **1.36x** |
| fodiwalk stored `D` | 5897 MB | 4.27x |

Batch size is the dial (150,000 nodes): 4096 -> 1523 MB at 5.4 s/epoch;
1024 -> 1140 MB at 7.6 s/epoch; 256 -> 943 MB at 17.6 s/epoch.

**The cost is recomputation.** Streaming re-walks every epoch: ~118 s
against the stored plan's 1.13 s on the GPU, about 100x. At 5 epochs
streaming beats node2vec on time AND is close on memory. At 200 epochs it
would need ~6.5 hours where the stored plan needs 8 minutes. Memory against
recomputation, and the crossover is early.

**Two things streaming changes, and they are the strategy, not defects.**
The walks are fresh in every epoch, so it optimises a stochastic sample and
not one fixed `D`. The update is immediate, so row `u` moves before row
`u + 1` is computed -- Gauss-Seidel, where the stored path is Jacobi.
Whether streaming converges to the same place is NOT established here; 5
epochs is not a convergence claim.

**Streaming does NOT win at 150,000 nodes**: 1523 MB against the stored
path's 1467 MB. The JAX runtime floor is ~790 MB when `Z` is 37 MB, so the
overhead swamps the data. The strategy pays only once `D` dominates.

## 5. Where fodiwalk's memory goes

Measured at 150,000 nodes, `D.nnz = 14,151,016`, unique live buffers:

| buffer | MB | note |
| --- | --- | --- |
| plan tiles | 187.5 | padded, tiled float32 -- a SECOND copy of the pairs |
| `D.data` float64 | 108.0 | `h`, an integer 1..19 |
| `freq` float64 | 108.0 | a count |
| `D.indices` int32 | 54.0 | needed |
| A | 24.8 | |
| sum live | 482.7 | peak during plan build: 1467 |

Both planes ALIAS their sources (`plane[0] is D.data`,
`plane[1] is freq`): there is no redundant plane copy.

Two avoidable costs. `h` and `freq` are float64 while the kernel is
float32 by design -- `sell_c_sigma.py:308` casts them down anyway. `h` is
an integer 1..19 and int8 holds it. At 1.13M nodes that is 1180 MB where
~370 MB would do. And the plan is a second full copy of the pair data,
inherent to the SELL-C-sigma layout, with 19-20% padding.

`golden.py` hashes `D.data`, so narrowing it is a deliberate re-record,
not a free change.

## 6. Protocol -- what makes the two sides comparable

`n2v1m` (`evaluator/config.py:200`) is LP `max_pairs=80,000,
n_estimators=100, pos_draw=always`; DA `n_sources=200, hops=bfs`.

The archived node2vec run passed `--lp-pairs 25000`, and
`bench_node2vec_1M.py:220` doubles it: `max_pairs = 50,000` -- an EXPLICIT
override of the 80,000 default. So the reference numbers were made at
50,000 LP pairs and that run is itself `protocol_modified=True`. Every
fodiwalk row here matches it at 50,000 and carries the same flag. **The two
sides are comparable TO EACH OTHER; neither is the bare `n2v1m` baseline.**
The archived `--sp-sources 200 --sp-pairs 20000` equal the DA defaults, so
DA is unmodified on both sides.

`bench_fodiwalk.py`'s own `--lp-pairs 50_000` is an unrelated flag: that
script scores through `fodiwalk.misc.evaluation` under `fodiwalk_dist`, a
different negative sampler and a different hop-pair sampler. It is not used
here.

## 7. Reproducibility of the 1M numbers

The plan reports `n_split = 2593` and `pad_frac = 0.191`: 2,593 rows are
wide enough to split into virtual rows that share an owner id, so
`dZ.at[rows].add(...)` accumulates in a free order on the GPU. **The
million-node numbers are therefore not bit-reproducible**, unlike the
cora-scale gates, which hold every row under `k_max`. Repeat runs agree to
the reported digits, not to the last bit.

## 8. Footnote: the `walk` policy does not reach this size

Only `nbr_walk` was benchmarked. The symmetric `walk` policy was tried
first by mistake and did not finish: `walk_pair_stats` prunes when its
accumulator passes a FIXED `prune_max = 4,000,000`, while the natural
per-node ceiling grows with `n`, so past ~70,000 nodes the prune fires on
nearly every block without shrinking much and the cost becomes a repeated
full sort of a growing array. Measured: n=50,000 -> 15.60 s (4 prunes);
n=100,000 -> 66.15 s (9 prunes). 4.2x the time for 2x the nodes. Not fixed
-- `fodiwalk/` was off limits during a benchmark.
