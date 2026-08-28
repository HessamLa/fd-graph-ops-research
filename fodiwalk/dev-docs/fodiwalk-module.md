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

In Graph embedding, an optimized sparse graph operations engine, the force functions are applied on graph elements and the embeddings are generated. This stage shall not do any graph analysis or data perapration. It must only consume the data. Its main goal is to apply the force function on the input data using the best implementation to optimize resource utilization.

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

---

## The stage contract (2026-08-27)

The three categories are STAGES. A stage produces material for the next
one and consumes material from the one before. **That is the only thing
that happens between stages.**

**There is no interaction of any other sort.** No stage imports another
stage. No stage calls a function of another stage. No stage asks another
stage a question. A stage does not read another stage's registry, table or
constant.

What crosses between two stages is DATA, and the contract is the TYPE and
the SHAPE of that data. Nothing else.

Each stage owns its own functions and its own data structures.

```
make_graph  --[ A: symmetric CSR, zero diagonal, sorted indices ]-->
augment_graph  --[ Augmentation: D, freq, stats, info ]-->
embed  --[ Z: (n, n_dim) ]-->
```

### What this rules out, with the case that made it explicit

A force law belongs to the embedding stage. The values a law reads, and
the divisor it needs, are therefore built in the embedding stage TOO --
even though they are computed from what the augmentation produced.

It is tempting to build them in the augmentation stage, on the reasoning
that they are "preparation". That reasoning is wrong, and it shows up
immediately as an import: the augmentation stage has to ask the law which
values it needs. The moment stage 2 imports stage 3, the stages are no
longer separable, and the boundary that the three categories exist to
create is gone.

The test: if a stage needs to know the name of a force law, a policy of
another stage, or the shape of another stage's internals, it is doing work
that belongs elsewhere.

### Where the stages meet

`model.py` composes the three. It takes what stage 2 produced and hands it
to stage 3. It belongs to no stage, and it is the ONLY place they meet.

