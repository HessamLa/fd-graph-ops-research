Mathemtically is similar to a ForceDirected Graph embedding, as found in `forcedirected_numba`.

But here we have a sparse weighted graph. Only the forces between connected edges will be calculated. All weights are positive and larger than 0.

Here is the force calculation, similar to what existed before

- d_uv \in R : weight of the edge <u,v>
- z[u] \in R^d: embedding of node u in a d-dimensional space
- dz[u] \in R^d: gradient
- z_uv \in R^d: the vector from embedding of u to v
- N(u) : set of neighbors of u
- ||z_uv|| : magtinude of the vector (L2 norm)

- F_u_attr : net attractive force on node u
- F_u_repl : net repulsive force on node u
- F_u = F_u_attr - F_u_repl : net force on node u

```
  LOOP:
    # calculate gradients
    Foreach u in V: 
      F_u_attr = ...
      F_u_repl = ...

      F_u = F_u_attr + F_u_repl
      # update the gradient;
      dz[u] = ...
    # update embeddings
    Foreach u in V:
      z[u] += dz[u]
```

## New Design   

But we are going to pursue a different design here.
 
1. We will prepare a weighted matrix (sparse or dense) to represent the graph
2. Then we feed the matrix to the force directed graph embedding algorithm

This way, we can break the logic into two separate compartments. 

- **Graph Building**. In this compartment two functions may be used depending on the scenario
  - `make_graph`: a graph is generated from an existing data (using algorithms such as MST and knn)
  - `augment_graph`: an existing graph is augmented by adding weighted edges or modifying them
  
- **Force-Directed Graph Embedding**. In this compartment, the force functions are determined. The graph weights are constant inputs to this compartment.
  - `embed`: an augmented weighted graph is used to embed the graph nodes in the d-dimensional space using force functions.

Consider these flows:
- Embedding: `G (weighted/unweighted) -> [Augmenting Graph] -> D -> [Embed] -> Z`
- Fitting (or Mapping): `data -> [Making Graph] -> G -> [Augmenting Graph] -> D -> [Embed] -> Z`

**Object-oriented style:**
```
class FDModel(ForceDirected):
  
  def make_graph (self, data, ...):
    NotImplemented
  
  def augment_graph (self, G, ...):
    NotImplemented
  
  def forces (self, ...):
    NotImplemented

  def embed (self, G, epochs:int=1000, dZ_threshold ...):
    D = self.augment_graph (G)
    # the force-dircted embedding algorithm
    LOOP over epochs :
      # Calculate gradients using self.forces
      dZ = ...

      # Early stop check
      if (np.linalg.norm(dZ, axis=1).mean() < dZ_threshold):
        break
      
      # Update embeddings
      Z += dZ

  def fit(data, epochs:int=1000, ...):
    G = self.make_graph (data, ...)
    Z = self.embed (G)



# Embedding an existing graph
model = FDModel(...)
Z = model.embed(G, ...)

# Fitting data
model = FDModel(...)
Z = model.fit (data, ...)
```

**Functional style:**

```
G = make_graph(data, ...) # Making graph
D = augment_graph(G, ...) # Augmenting graph
Z = embed(D, ...) # Embedding the augmented graph
```


## What has changed?

The main flow of the algorithm is the same. In this model, we are separating the logic into three stages as the following:
1. Making graph
2. Augmenting graph
3. Embedding graph

## Functions and Stages

To better illustrate the new design, we use the current implementation as a reference. In the current implementation, some functions concretely belong to specific stages. It is fair to say that the stage is defined by those functionalities:

1. Making graph
   - `make_graph` in `fdge_numba/graphmaking/registry.py`
2. Augmenting graph
   - `get_hops_csr` in `fdge_numba/forcedirected_numba/model_204_shell.py`. Connects each node pair `u` and `v` using an edge with weight set to the shortest path distance, d(u,v) in the original graph.
3. Embedding graph
   - `_forces_204_hidx`
   - `_scalar_force`

Other functions are for preparing other variables and coefficient matrices. During actual implementation, the designer would use the correct order of calling them.


## Other Concerns

In the original code, there are some variables and matrices that can be derived in between the stages.

For example, the set `S_h(u)` is the set of nodes in the shell of `u`. This can be derived from G.

Sparcity or density of the matrices is determined during implementation. Nevertheless, all methods and functions must return indexable data structures, such as `G[i,j]` and `D[i,j]`

The existing code has some assumptions hard-coded into it. For example, with `get_hops_csr`, the augmented graph becomes a dense graph, making the complexity O(n^2). There are other methods that the augmented graph can remain sparse and reduce the complexity to O(n log n), or even O(n).


