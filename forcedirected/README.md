# forcedirected

One force-directed embedding engine. Give it a weighted matrix `D`; it
relaxes an embedding `Z` against it.

## Goal

- ONE implementation, reused. The engine and the kernel were rewritten in
  each package that needed them. They are here now, and each caller
  forwards.
- Every dependency points AT this package. It reads numpy, scipy, jax and
  its own modules, and NOTHING of this repository. Thus no cycle is
  possible, and `fodined` does not depend on `fodiwalk` to use the kernel.
- The kernel is used only inside the engine. That is why it lives here and
  not in a package of its own.

## What is here

| file | what it holds |
| --- | --- |
| `force_directed.py` | `ForceDirected`: the epoch loop, the row batching, the callbacks, the `Z` update. NO force law -- `forces` dispatches to one the subclass supplies |
| `sell_c_sigma.py` | the SELL-C-sigma layout and the jitted kernel. Hub split, width sort, ladder quantize, pack, pad. NO physics |
| `optim.py` | eight update rules: `plain`, `momentum`, `nesterov`, `adam`, `fa2`, `velocity`, `sgd`, `sqn`. A rule is pure |
| `csr.py` | `row_of`, `n_rows`. numpy only |
| `PARITY.md` | what the moves promised, and how each was proved |

## Use

```python
from forcedirected import ForceDirected, Callback_Base, make_plan, step
```

Subclass and supply the two stages the engine does not own:

```python
class MyModel(ForceDirected):
    def augment_graph(self, G, **kw): ...                    # -> D
    def forces(self, Z, D, row_start, row_end, **kw): ...    # -> (rows, d)
```

`forces` gets a ROW RANGE and never a set of pairs: the kernel writes
`dZ.at[rows].add(...)`, thus only disjoint rows make the parts additive.

## Callers

`fodiwalk/core/{force_directed,sell_c_sigma,csr}.py` and
`fodiwalk/misc/optim.py` and `fodined/embedding/sell_c_sigma.py` are
FORWARDERS. They hold no algorithm. Edit the code here.

## Tests

```bash
.venv/bin/pytest forcedirected/tests -q
.venv/bin/python forcedirected/tests/reconstruct_pre_unification.py
```

The second one recovers the pre-unification kernel byte-exact, by
reversing enumerated edits. It keys on the sha256 of `sell_c_sigma.py`.
An edit to that file breaks the route; `PARITY.md` section 9 says what to
do instead.

## One property to know

`step` is NOT bit-reproducible in general. A row wider than `k_max` splits
into virtual rows that share an owner id, and `dZ.at[rows].add(...)` then
accumulates in a free order on a GPU. The threshold is three addends in
one batch. Cora stops at two and is exact; PubMed reaches four and is not.
Do not write a bit-equality test on PubMed.
