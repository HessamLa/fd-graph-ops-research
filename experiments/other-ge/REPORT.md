# node2vec and Poincare: the baselines for fodined

Date: 2026-08-14
Script: `bench_other_ge.py`
Raw output: `results_2026-08-14.log`

## The question

`fodined` is a force-directed embedding. Two other methods embed the same
graphs, and they do the same tasks. The numbers of fodined then have
something to stand against.

## The answer

**node2vec wins on every graph, and it is 4x to 9x faster.** Poincare never
wins the link prediction, but the gap becomes very small on the two trees,
which is where the theory says hyperbolic space must help.

Both methods approximate the hop distance badly. The best R2 is 0.32, and
most values are below 0.15.

---

## Setup

| Item | Value |
| --- | --- |
| Dimensions | 128 for both methods |
| Seed | 42 |
| node2vec | 10 walks for each node, length 40, window 5, 5 epochs, p = q = 1 |
| Poincare | gensim `PoincareModel`, 50 epochs, 10 negative samples |
| Graphs | cora, pubmed, wordnet, ncbi_taxonomy |
| Truncation | one subtree of a maximum of 20,000 nodes for the two trees |

`p = q = 1` makes the second-order walk of node2vec a uniform walk, thus
this is the DeepWalk configuration of node2vec. A uniform walk is one
vectorized NumPy operation. Use `--p` and `--q` for the true walk, which is
slow.

The hop regression uses ONE feature: the distance of the pair in the space
of the method. node2vec gets the Euclidean distance, and Poincare gets the
Poincare distance. A Euclidean distance in the Poincare ball has no meaning.
Pairs at hop 1 are removed, because a neighbour is the easy case.

---

## Results

### Link prediction

| Graph | Method | accuracy | precision | recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| cora | node2vec | **0.9777** | 0.9865 | **0.9688** | **0.9775** | **0.9974** |
| cora | poincare | 0.9437 | 0.9709 | 0.9148 | 0.9420 | 0.9871 |
| pubmed | node2vec | **0.9763** | 0.9856 | **0.9667** | **0.9761** | **0.9972** |
| pubmed | poincare | 0.9547 | 0.9685 | 0.9401 | 0.9541 | 0.9894 |
| wordnet | node2vec | **0.9910** | 0.9969 | **0.9850** | **0.9909** | **0.9994** |
| wordnet | poincare | 0.9884 | 0.9967 | 0.9800 | 0.9883 | 0.9987 |
| ncbi_taxonomy | node2vec | **0.9863** | 0.9928 | **0.9796** | **0.9862** | **0.9985** |
| ncbi_taxonomy | poincare | 0.9837 | 0.9939 | 0.9734 | 0.9836 | 0.9972 |

### Hop distance approximation

The baseline gives the mean of the train set to every pair.

| Graph | Method | base MAE | RF MAE | MLP MAE | MLP R2 | MLP exact |
| --- | --- | --- | --- | --- | --- | --- |
| cora | node2vec | 1.527 | 1.232 | **1.216** | **0.324** | 26.9% |
| cora | poincare | 1.527 | 1.445 | 1.426 | 0.123 | 23.2% |
| pubmed | node2vec | 1.179 | 1.140 | 1.129 | 0.080 | 29.2% |
| pubmed | poincare | 1.179 | 1.133 | **1.118** | **0.094** | 29.7% |
| wordnet | node2vec | 1.583 | 1.505 | **1.507** | **0.106** | 21.1% |
| wordnet | poincare | 1.583 | 1.526 | 1.517 | 0.069 | 20.0% |
| ncbi_taxonomy | node2vec | 1.220 | 1.147 | 1.120 | **0.151** | 29.9% |
| ncbi_taxonomy | poincare | 1.220 | 1.158 | 1.120 | 0.124 | 30.7% |

### Speed

| Graph | n | node2vec | Poincare | node2vec is faster by |
| --- | --- | --- | --- | --- |
| cora | 2,708 | 5.0 s | 31.4 s | 6.3x |
| pubmed | 19,717 | 38.7 s | 174.8 s | 4.5x |
| wordnet | 20,000 | 35.3 s | 238.2 s | 6.7x |
| ncbi_taxonomy | 17,672 | 29.7 s | 261.4 s | 8.8x |

node2vec holds 510 to 596 nodes each second. Poincare holds 68 to 113. The
walks of node2vec are only 6% of its time; Word2Vec takes the rest.

---

## Findings

### 1. node2vec wins the link prediction on every graph

The AUC of node2vec is above the AUC of Poincare on all four graphs. The
size of the win is not the same everywhere. See finding 2.

### 2. The trees show the effect that the theory predicts, but it is not
       sufficient

The difference of the AUC:

| Graph | Type | AUC difference |
| --- | --- | --- |
| cora | citation graph | 0.0103 |
| pubmed | citation graph | 0.0078 |
| wordnet | tree | 0.0007 |
| ncbi_taxonomy | tree | 0.0013 |

The gap on a tree is 6 to 15 times smaller than the gap on a citation
graph. Thus hyperbolic space really helps a hierarchy, exactly as the theory
of the Poincare paper says. But it helps only enough to make the result
equal, and not enough to win.

### 3. Both methods approximate the hop distance badly

The best R2 is 0.324, and it belongs to node2vec on cora. Six of the eight
values are below 0.15. The MAE improves the baseline by only 4% to 20%.

A single number, which is the distance of the pair, does not hold the hop
distance. This agrees with the same measurement on fodined.

### 4. Poincare is 4x to 9x slower

The reason is the implementation, and not the mathematics. gensim trains
Poincare on the CPU with one thread for the gradient, and Word2Vec uses all
the cores. A GPU implementation of Poincare would change this row of the
table, and it would not change the quality rows.

### 5. Precision is high and recall is lower, for every method and graph

Every one of the 8 runs has a precision above its recall. Both methods
therefore miss true edges more often than they invent false ones. The
threshold of the classifier is at 0.5. A lower threshold moves the balance.

---

## Comparison with fodined (indicative)

The numbers of fodined come from `fodined/modular.py` on cora, and not from
this harness. The protocols are near, but not equal: `modular.py` uses 200
trees in the random forest, and this script uses 100. Read this table as an
indication, and not as a controlled comparison.

| Method | cora AUC | cora hop R2 | cora hop MAE |
| --- | --- | --- | --- |
| node2vec | 0.9974 | 0.324 | 1.216 |
| **fodined** | 0.9945 | 0.231 | 1.336 |
| poincare | 0.9871 | 0.123 | 1.426 |

fodined is between the two methods on both tasks on cora. node2vec is
better than fodined on this graph.

A comparison of the speed needs one harness. fodined runs on the GPU, and
both methods here run on the CPU.

---

## A warning about the earlier result

`results_2026-08-13_bfs-truncation-INVALID.log` holds the first run. Do not
use it. The truncation of that run took a BFS from the node of the highest
degree. NCBI has nodes with tens of thousands of children, thus the result
was a STAR: one node of degree 19,999, and 19,999 leaves.

The signs of the fault were clear:

* node2vec reached exactly 1.0000 on accuracy, precision, recall, F1, and
  AUC.
* Every pair with no edge had a hop distance of exactly 2.
* Poincare stopped with `Cannot sample 10 negative nodes from a set of 2`,
  because one node is a neighbour of all the others.

`_pick_subtree` now takes a subtree, which keeps the depth. The eccentricity
of NCBI changed from 2 to 12. Use `--truncate bfs` to get the old behaviour,
but read the docstring first.

---

## Limits

* **One seed.** The `drop_strategies` experiment used three seeds and it
  found a variance of ±0.001 on the AUC. The differences of finding 2
  (0.0007 and 0.0013) are of that size, thus the equality of the two methods
  on the trees is safe, but the ORDER of the two is not.
* **128 dimensions for Poincare.** The Poincare paper shows its advantage at
  5 to 50 dimensions, because a small number of hyperbolic dimensions is
  the point of the method. 128 dimensions give Euclidean space enough room,
  thus this experiment does not test Poincare where it is strongest.
* **50 epochs for Poincare.** The paper uses more.
* **The trees are truncated** to 20,000 nodes of 82,144 and 2,937,016.
* **p = q = 1** for node2vec, thus this is DeepWalk and not the full method.

## How to repeat

```bash
.venv/bin/python experiments/other-ge/bench_other_ge.py --max-nodes 20000

# Poincare in the dimension where it is strong:
.venv/bin/python experiments/other-ge/bench_other_ge.py \
    --graphs wordnet,ncbi_taxonomy --methods poincare --dim 10
```
