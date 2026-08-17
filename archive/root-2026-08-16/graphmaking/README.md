# graphmaking

Graph construction strategies for force-directed embedding pipelines.

## Usage

```python
from graphmaking import make_graph, available_strategies

available_strategies()
# ['mst', 'mst_k_minimum', 'union_mst_nndescent']

G = make_graph(data, "mst")
G = make_graph(data, "union_mst_nndescent", k_neighbors=10)
G = make_graph(data, "mst_k_minimum", k_neighbors=10)
```

`data` is an `(n, d)` array-like. Every strategy returns a simple,
undirected, unweighted `networkx.Graph` on nodes `0..n-1`, guaranteed
connected (an MST is always included). Each edge carries a `source`
attribute in `{'mst', 'knn', 'fill'}` for provenance.

## Strategies

| name | description |
|---|---|
| `mst` | Exact Euclidean minimum spanning tree. `n-1` edges. |
| `union_mst_nndescent` | Union of MST edges and PyNNDescent kNN edges. Requires `k_neighbors`. |
| `mst_k_minimum` | MST plus nearest-non-neighbor fill edges so every node has degree `>= min(k, n-1)`. Requires `k_neighbors`. |

## Layout

```
graphmaking/
    __init__.py           # public API: make_graph, register_strategy, available_strategies
    registry.py            # strategy registry + make_graph dispatcher
    building_blocks.py     # shared helpers: mst_edges, knn_edges, base_graph, add_edges
    strategies/
        __init__.py        # imports every strategy module to register it
        mst.py
        union_mst_nndescent.py
        mst_k_minimum.py
```

## Adding a strategy

1. Create `strategies/my_strategy.py`:
   ```python
   from ..registry import register_strategy
   from ..building_blocks import base_graph, add_edges

   @register_strategy("my_strategy")
   def _strategy_my_strategy(data, **kwargs):
       G = base_graph(data.shape[0])
       # ... add edges ...
       return G
   ```
2. Import it in `strategies/__init__.py`:
   ```python
   from . import mst, union_mst_nndescent, mst_k_minimum, my_strategy
   ```

Registration happens on import, so a strategy that isn't imported in
`strategies/__init__.py` won't be reachable through `make_graph`.

## Notes

- MST is computed exactly from a dense pairwise-distance matrix
  (`scipy.sparse.csgraph.minimum_spanning_tree`). Fine up to roughly
  10-20k points; swap `mst_edges` in `building_blocks.py` for an
  approximate MST beyond that.
- `make_graph(..., weighted=True)` is reserved for future weighted
  variants and currently raises `NotImplementedError`.

## Dependencies

`numpy`, `scipy`, `networkx`, `pynndescent`
