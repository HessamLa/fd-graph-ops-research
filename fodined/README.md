# FoDiNed

FoDiNed is the name of this package. The name is short for **Fo**rce-**Di**rected
**N**ode **E**mbe**d**ding. The code does not use the name. The directory name
is `fodined`.

FoDiNed puts the nodes of a graph in a space of `d` dimensions. Two forces move
each node. The attractive force pulls a node to its neighbors. The repulsive
force pushes a node away from the other nodes. The two forces become equal after
sufficient epochs. The positions at that time are the embedding.

## What is in this directory

| Path | Function |
| --- | --- |
| `modular.py` | The script. It does the full task from the edgelist to the link prediction scores. |
| `core/force_directed.py` | The `ForceDirected` base class. It holds the epoch loop, the batch loop, and the callbacks. |
| `core/csr.py` | Two helpers for a CSR matrix: `row_of` and `n_rows`. |
| `embedding/shell_force.py` | The force law, and the quantities of the force law. |
| `embedding/sell_c_sigma.py` | The SELL-C-sigma layout machinery, and the JAX kernel. |
| `embedding/drop.py` | The random drop regularizer. |

These files are a copy of the related files in `fdge_jax_sell_c_sigma/`. Only
the import lines are different: they are relative now. Thus this directory has
no dependency on the origin package. Each file has a note about its origin at
the top.

This directory holds only the files that `modular.py` uses. The origin package
has more modules, for example `models.py`, `graph_building/`, and
`graph_augmenting/`. `modular.py` does not use them.

## How to start the script

Start the script from the repo root (`fdmap/`). Both of these commands are
correct:

```bash
.venv/bin/python fodined/modular.py
.venv/bin/python -m fodined.modular
```

The script puts the parent directory of `fodined` on `sys.path` at line 40.
Python puts only the directory of the script on `sys.path`. Thus the import of
`fodined` fails without this step.

## What the script does

The script runs from the first line to the last line. It has no main function.
There are four steps:

1. **It loads the graph.** The source is the Cora edgelist at
   `data_cache/cora/cora.cites`. The script changes the paper IDs to row
   indices `0` to `n-1`. Then it makes a symmetric CSR matrix. Each edge has a
   weight of 1.0.
2. **It augments the graph.** The script takes 1000 random pairs of nodes that
   have no edge. It adds an edge for each pair. The weight of the new edge is
   the hop distance between the two nodes. If there is no path between the two
   nodes, the weight is 100.0.
3. **It embeds the nodes.** The class `SellCSigmaFD` is a subclass of
   `ForceDirected`. It supplies the two stage hooks: `augment_graph` makes the
   batch plan one time, and `forces` calls the JAX kernel one time for each
   epoch. The result is an array of 128 dimensions for each node.
4. **It does link prediction.** A random forest classifier learns to find a
   pair of nodes that has an edge. The feature of a pair is the Hadamard
   product of the two embeddings. The script prints the accuracy, the
   precision, the recall, the F1-score, and the AUC.

## Results

These are the results of one run on one GPU. The seed is 42.

```
n = 2708 nodes, 5278 undirected edges, 12556 stored entries after the augmentation
2000 epochs in 8 s to 19 s (the load and the clock of the GPU change this time)
accuracy 0.9427 | precision 0.9448 | recall 0.9403 | f1 0.9426 | auc 0.9833
```

The scores and the final `||dZ||` are the same as the scores of
`fdge_jax_sell_c_sigma/modular.py`. The two scripts do the same calculation.

**Be careful with these scores.** The embedding uses all the edges. The test
edges are thus not new to the embedding. The scores show how much graph
structure the embedding keeps. They do not show the performance on unknown
edges. To measure that performance, remove the test edges from the graph before
step 3.

## Requirements

The script needs `jax` with CUDA, `numpy`, `scipy`, and `scikit-learn`. The
virtual environment at `.venv/` has all of them.
