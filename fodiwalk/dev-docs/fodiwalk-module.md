Fodiwalk is a force-directed graph embedding method that uses graph walk methods to generate the set of nodes for calculating the force and embedding gradient.

$$
F_u = \sum_{v \in \text{Walk}(u)} F_{u,v}^{(a)} + F_{u,v}^{(r)} 
$$

The main contribution in Fodiwalk compared to ForceDirected Graph Embedding is that here, the graph augmentation stages is performed using graph walk.

---

Either the module is developed using object oriented and classes, or using functional approach, there must be three diferent categories of methods based on the following three stages.

1. Graph construction
2. Graph augmentation
3. Embedding

There are other helper functions and methods as well, such as data loader, file operations, analysis and results, profilers, etc. But they are not crucial to Fodiwalk.

Everything must be captured under these three categories. These categories share limited configuration parameters, and each one will generate and pass a limited number of data.

In graph construction data is loaded, prepared into graph and its accompanying data if applicable, and passed to graph augmentation.

In Graph augmentation, graph is recieved (with its accompanying data if exists). Depending on the donwstream embedding configuration, the graph is augmented and updated by adding more edges or selecting node pairs or taging edges etc.

In Graph embedding, an optimized sparse graph operations engine, the force functions are applied on graph elements and the embeddings are generated.

These categories must be respected to make the code readable and traceable, and not turn into a slop.

---

Baseline Fodiwalk has a simple walk which generates the $\text{Walk}(u)$ set such that $\deg v ≥ \deg u,   \forall v \in \text{Walk}(u)$, i.e. all nodes in the walk set of $u$ have degrees larger than $u$.

Baseline Fodiwalk uses the following force function.

```
let z_uv = z_v - z_u

h == 1:  Fa = k * z_uv
         Fr = - exp(||z_uv||) * (z_uv) / ||z_uv||

h >= 2:  Fa = 0
         Fr = -(h_uv / freq_uv) * exp(||z_uv||) * (z_uv) / ||z_uv||

# h_uv: the minimum hop distance between u and v found during walks
# freq_uv: how often v appeared in the walks with respect to u
```

In this function, ``z_u`` is the embedding of node $u$.

In order to save memory, ``h_uv`` and ``freq_uv`` could be fused and saved as ``h_uv / freq_uv``.

Using walks, essentially the ``D`` matrix becomes assymetric, which is fine.

