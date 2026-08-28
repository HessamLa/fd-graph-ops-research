This document is maintained by the user.

[Thu Aug 27 10:57:41 PM PDT 2026]

In fodiwalk we are storing walks and its corresponding data in CSR format. This is overkill. The walks generated during augmentation must be stored as flat array, and then passed to the embedding stage. We can store index of the nodes in a walk, as well as their min-hop and frequency in flat array format. This will save huge amount of space and make memory access much faster. The byproduct is that we will not need the ``D`` or hops matrix anymore.

The smart strategy would be to use the same walk generator used with node2vec. This way, we can compare out method vs node2vec and DeepWalk. How to use this? Just have the walk generator produce the walks. Then, for each node, reduce its corresponding walks to three same sized vector arrays: node indices, min hops, and frequencies. In worst case scenario the space will be $3 · r · |V| · l$, and the best case would be $3 · l$. 

I am expecting this to be much faster.

[Mon Aug 17 11:12:54 PM PDT 2026]

We have been working on the experiments/fdwalk so far. In fdwalk, we decided to use random walk for augmentation. 

**For graph augmentation:**

In previous methods, we included all nodes in k-ball vicinity of a node. Exploring the graph in this way was very time consuming for dense graphs or graphs with hub nodes. It also used more memory.

In fdwalk, we perform random walks to discover reachable nodes. For calculating embedding gradient of each node u, we use the other nodes which show up in the walk of node u. We retain min hop and frequency that the node showed up in the walk.

There are also a bunch of other random walk and augmentation details that are documented.

We also adopted a non-symmetric D matrix. 

    Now, while we want to keep all the edges, but we want to consider some of the edges when calculating embedding of one end. In other words, considering that u,v are immediate neighbors in the original graph. Then calculating embedding gradient for u, we can avoid v (omitting the edge from u's perspecive), and when calculating v's, we consider u (including the edge from v's perspective).

    With this regime, we can reduce the memory usage dramatically.

    Here is the regime:
    1. For immediate neighbors u,v, only include v on list of u, if deg v ≥ deg u.
    2. If v is on the walk list of u.


**For force function:**

We are also exploring a new force function in parallel to the previous ones.

First of all, we decided to only calculate the attractive force for immediate neighbors: ``Fa = 0`` for ``h ≥ 2``

Also, 

in the previous force function, we defined Fr as the following

``Fr = -k3 · h · exp(-k4 · x)``

*fdlinear force function:*
```
Fa = k1 · x                    for h = 1 only   (no 1/deg)
Fr = -kr · exp(-k4·x)          for h = 1
Fr = -(h/freq) · exp(-k4·x)    for h ≥ 2        (freq = pair co-occurrence count in the walk)
```