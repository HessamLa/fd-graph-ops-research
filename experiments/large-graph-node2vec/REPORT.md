# One million nodes: node2vec finishes, fodined does not

Date: 2026-08-14
Graph: com_youtube (SNAP), 1,134,890 nodes and 2,987,624 edges
Machine: about 3 GB of free RAM, GeForce GTX 950 with 2 GB

Logs:
* node2vec: `node2vec_com_youtube_1M.log` (this directory)
* fodined: `../modular-graphs/modular_com_youtube_1M.log`

## The result

| | node2vec | fodined (`modular.py`) |
| --- | --- | --- |
| Finished | **yes** | **no** |
| Time | 764.6 s (12.7 min) | died after 20 s |
| Peak RAM | 2212 MB | 2465 MB, and it was still growing |
| Stage reached | all | the augmentation, before any distance |

## node2vec

| Stage | Time | Note |
| --- | --- | --- |
| Load | 1.5 s | pandas reads the edge list |
| Walks | 27.9 s | 5,674,450 walks x 20 steps = 113M tokens, 0.74 GB on disk |
| Word2Vec | 732.5 s | 3 epochs, 8 workers, vocabulary 1,134,890 |
| Gather | 2.7 s | the vectors go into one (n, 128) array |
| **Total** | **764.6 s** | peak RSS 2212 MB |

Link prediction on 80,000 pairs:

| accuracy | precision | recall | F1 | AUC |
| --- | --- | --- | --- | --- |
| 0.9776 | 0.9602 | 0.9964 | 0.9780 | 0.9984 |

Hop distance approximation, 20,000 pairs from 200 source nodes:

| model | MAE | MRE | RMSE | R2 | exact |
| --- | --- | --- | --- | --- | --- |
| mean baseline | 1.069 | 0.211 | 1.345 | -0.000 | 32.3% |
| random forest | 1.042 | 0.205 | 1.341 | 0.005 | 31.1% |
| MLP | 1.032 | 0.204 | 1.315 | **0.043** | 30.6% |

## Why fodined stopped

`modular.py` stopped inside the augmentation, and not in the embedding. It
never computed one distance, and it never used the GPU.

The augmentation asks for `N_NEW_EDGES = n * log10(n)` = 6,871,706 new pairs.
The rejection loop keeps Python sets for those pairs:

| Structure | Content | Approximate memory |
| --- | --- | --- |
| `taken` | 13.7M tuples (both directions) | 1.8 GB |
| `edge_set` | 6.0M tuples | 0.8 GB |
| `new_pairs` | 6.9M tuples | 0.6 GB |

That is more than 3 GB before any work starts. The process passed 2.4 GB
after 20 s.

Two more walls stand behind that one, and the run never reached them:

1. **The hop distances.** The sample touches nearly all the 1.13M nodes as
   sources. One BFS on this graph needs 0.36 s (measured in
   `experiments/bench_large_graphs.py`), thus 1.13M sources need about
   **114 hours**.
2. **The GPU.** `Z` and `dZ` at 1,134,890 x 128 float32 are 1.16 GB
   together, and the plan tiles add about 300 MB. The card has 2 GB.

## What fodined needs for this size

1. **Vectorized pair sampling.** The sets must go. A sorted array of the
   composite keys `u * n + v` and `np.searchsorted` do the same test with a
   small fraction of the memory. `experiments/bench_large_graphs.py` already
   holds that code.
2. **A different source of the hop distances.** Pruned Landmark Labeling, or
   a limited set of sources, or the landmark approximation. The
   `experiments/` directory measured all three.
3. **A smaller `n_dim`, or the CPU.** 128 dimensions do not fit on a 2 GB
   card at this size.

Only the first of the three is a small change.

## The honest comparison

node2vec wins this comparison, but the two methods do not solve the same
problem. node2vec needs no distance at all: a random walk is local, thus it
never asks a global question. fodined asks for the exact hop distance of
millions of pairs, which is a global question, and that is the whole cost.

The quality supports this. node2vec reaches an AUC of 0.9984 on link
prediction, which is a local property. It reaches an R2 of only 0.043 on the
hop distance, which is a global property. On the smaller graphs, where both
methods run, fodined gives a better R2 on the hop distance than node2vec on
2 of the 4 graphs (pubmed 0.340 against 0.080, and wordnet 0.173 against
0.106).

Thus the correct summary is not "node2vec is better". It is: **at a million
nodes on this machine, only node2vec runs at all**, and the reason is the
augmentation of fodined, and not its embedding engine.

## A warning about the guard script

`../modular-graphs/run_guarded.sh` watches the memory of the child and kills
it before the machine suffers. The first version watched the WRONG process:
`$!` after `timeout ... &` gives the PID of `timeout`, thus the guard read
2 MB while the python process reached 4.8 GB with 2 GB of free RAM. The
script now starts python directly. Read the comment before an edit.
