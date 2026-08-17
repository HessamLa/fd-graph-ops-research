# References

Created: 2026-08-15. Last updated: 2026-08-16 23:25 PDT.

What each work gives to the fdwalk design. The numbers are the citation keys
of PLAN.md and FINDINGS.md.

## The walk as a source of pairs

[1] Fruchterman, T. M. J., and Reingold, E. M. "Graph Drawing by
Force-Directed Placement." *Software: Practice and Experience* 21(11), 1991,
1129-1164.
The origin of the force model: attraction between the pairs that have an
edge, repulsion between every pair. fodined keeps that shape, and it changes
which pairs attract.

[2] Kamada, T., and Kawai, S. "An algorithm for drawing general undirected
graphs." *Information Processing Letters* 31(1), 1989, 7-15.
The layout of a spring whose rest length is the GRAPH DISTANCE of the two
nodes. This is the reason that `D` holds a hop distance and not a 0 or a 1,
and it is the reason that the weight of a pair matters at all.

[3] Perozzi, B., Al-Rfou, R., and Skiena, S. "DeepWalk: Online Learning of
Social Representations." *KDD* 2014. arXiv:1403.6652.
The first work that reads a graph as a corpus of random walks. It gives the
budget form: `r` walks of length `l` from each node.

[4] Grover, A., and Leskovec, J. "node2vec: Scalable Feature Learning for
Networks." *KDD* 2016. arXiv:1607.00653.
The biased walk, with the return parameter `p` and the in-out parameter `q`.
At `p = q = 1` it is the walk of DeepWalk, which is the walk that our
baseline uses. The bias is an axis that fdwalk can take later.

[5] Levy, O., and Goldberg, Y. "Neural Word Embedding as Implicit Matrix
Factorization." *NIPS* 2014.
Skip-gram with negative sampling factors a shifted PMI matrix. Thus the
co-occurrence counts of a walk hold the information, and the neural network
is one way to read it. Weight rule A3 comes from this.

[6] Qiu, J., Dong, Y., Ma, H., Li, J., Wang, K., and Tang, J. "Network
Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and
node2vec." *WSDM* 2018. arXiv:1710.02971.
The same statement for graphs: DeepWalk and node2vec factor a matrix whose
entries are walk co-occurrence probabilities. This is the direct support of
the fdwalk idea. fdwalk gives that matrix to a force law instead of to a
factorization.

[7] Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J., and Mei, Q. "LINE:
Large-scale Information Network Embedding." *WWW* 2015. arXiv:1503.03578.
First-order and second-order proximity, and edge sampling with an alias
table. The reason that variant B2 always keeps the original edges: the
first-order proximity is the part that a walk can miss.

## The pairs that repel

[8] Tang, J., Liu, J., Zhang, M., and Mei, Q. "Visualizing Large-scale and
High-dimensional Data." *WWW* 2016 (LargeVis). arXiv:1602.00370.
A sparse neighbour graph attracts, and uniformly random pairs repel. This is
the same structure as the far pairs of fodined, and it is the evidence that
a small number of random negative pairs is enough.

[9] McInnes, L., Healy, J., and Melville, J. "UMAP: Uniform Manifold
Approximation and Projection for Dimension Reduction." arXiv:1802.03426,
2018.
The same attraction and repulsion on a sparse kNN graph, with a different
weight rule. The cap `M` of fdwalk is the same construction as the `k` of a
kNN graph.

[10] Jacomy, M., Venturini, T., Heymann, S., and Bastian, M. "ForceAtlas2, a
Continuous Graph Layout Algorithm for Handy Network Visualization Designed
for the Gephi Software." *PLoS ONE* 9(6), 2014, e98679.
The "local speed" rule: each node gets its own step size, from the swinging
of that node between two iterations. Variant C4. It is the force-directed
answer to the problem that Adam solves in deep learning.

[12] Böhm, J. N., Berens, P., and Kobak, D. "Attraction-Repulsion Spectrum
in Neighbor Embeddings." *Journal of Machine Learning Research* 23(95),
2022, 1-32. arXiv:2007.08902.
The balance of attraction against repulsion is one axis, and it moves the
result from a local structure to a global structure. Our `K1..K4` sit
somewhere on that axis. The work explains why a change of the far weight
(H7) can change the character of the embedding, and what to look for.

## The update rule

[11] Kingma, D. P., and Ba, J. "Adam: A Method for Stochastic Optimization."
*ICLR* 2015. arXiv:1412.6980.
Variant C3. It holds two state arrays, which is the arithmetic of H6.

[13] Sutskever, I., Martens, J., Dahl, G., and Hinton, G. "On the importance
of initialization and momentum in deep learning." *ICML* 2013.
Heavy-ball and Nesterov momentum, variants C1 and C2, and the reason to
expect fewer epochs.

[14] Loshchilov, I., and Hutter, F. "Decoupled Weight Decay Regularization."
*ICLR* 2019. arXiv:1711.05101.
Only if a variant needs a decay on `Z`. It is not in the plan today.

## The measurement

[15] Akiba, T., Iwata, Y., and Yoshida, Y. "Fast Exact Shortest-Path
Distance Queries on Large Networks by Pruned Landmark Labeling." *SIGMOD*
2013.
The exact distances for the evaluation. Our own benchmark
(`../bench_shortest_path_gemsec.py`) measured that PLL beats a chunked SciPy
BFS by 5 to 15 times up to 50,000 nodes, and that it needs more RAM than
this machine has at 1M nodes. Thus the evaluation of a 1M graph uses a
sample of sources.

[16] Tsitsulin, A., Mottin, D., Karras, P., and Müller, E. "VERSE:
Versatile Graph Embeddings from Similarity Measures." *WWW* 2018.
arXiv:1803.04742.
An embedding that keeps a chosen similarity, and personalized PageRank is
one choice. A later axis for the weight, if the walk gap and the PMI both
fail.

## The data

[17] Sen, P., Namata, G., Bilgic, M., Getoor, L., Gallagher, B., and
Eliassi-Rad, T. "Collective Classification in Network Data." *AI Magazine*
29(3), 2008, 93-106.
Cora and PubMed.

[18] Leskovec, J., and Krevl, A. "SNAP Datasets: Stanford Large Network
Dataset Collection." 2014. http://snap.stanford.edu/data
com-youtube, as-skitter, and roadNet-CA.

[19] Yang, J., and Leskovec, J. "Defining and Evaluating Network Communities
Based on Ground-Truth." *ICDM* 2012.
The com-youtube graph: 1,134,890 nodes, 2,987,624 edges, maximum degree
28,754 in our load.

[20] Leskovec, J., Lang, K., Dasgupta, A., and Mahoney, M. "Community
Structure in Large Networks: Natural Cluster Sizes and the Absence of Large
Well-Defined Clusters." *Internet Mathematics* 6(1), 2009, 29-123.
roadNet-CA: 1,965,206 nodes and a maximum degree of 12. The contrast graph
of the whole study.

## The style

[21] ASD-STE100, *Simplified Technical English*. Aerospace and Defence
Industries Association of Europe.
Every document of this directory follows it.

## Inside this repository

* `../large-graph-node2vec/REPORT.md` -- the node2vec baseline at 1.13M
  nodes, and the measured reason that fodined died there.
* `../other-ge/REPORT.md` -- node2vec and Poincaré on the small graphs, and
  the evaluation harness that fdwalk reuses.
* `../drop_strategies/REPORT.md` -- the drop rate is also a stabiliser, and
  the row-skipping measurements.
* `../modular-graphs/` -- the logs of `fodined/modular.py` on each graph, and
  the memory guard.
* `../bench_shortest_path_gemsec.py` -- the cost of the exact distances.
