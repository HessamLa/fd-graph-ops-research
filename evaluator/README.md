# evaluator

One measurement of the quality of a graph embedding.

Four files of this repository each scored an embedding a different way:
another pair budget, another classifier size, another feature width,
another source of the hop ground truth. A number of one was not comparable
to a number of another, and nothing said so. This package holds one
implementation, and it makes the disagreement explicit: each recorded
baseline is a **named protocol**, and each result record carries the name.

Two tasks:

| task | question | model |
|---|---|---|
| `link_prediction` | does the geometry hold the adjacency? | classifier on a pair feature |
| `dist_approx` | does the geometry hold the hop distance? | regressor on a pair feature |

---

## 1. Run it

The package is **not installed**. Run it from the repository root, or put
the root on `PYTHONPATH`:

```bash
cd /home/h/gnn/fd-graph-embedding/fdmap
.venv/bin/python -m evaluator cora emb/cora.npy --all
```

```python
import sys; sys.path.insert(0, ROOT)      # the pattern the bench scripts use
import evaluator as ev
```

Required: `numpy`, `scipy`, `scikit-learn`. Optional: `networkit` (the
exact PLL hop backend), `fodiwalk` (the dataset registry, for a bare graph
name such as `cora`).

---

## 2. Quick start

### CLI

```
python -m evaluator GRAPH EMBEDDING [tasks] [options]
```

```bash
.venv/bin/python -m evaluator cora emb/cora.npy --all \
    --protocol otherge --tag method=spectral --results runs/otherge.jsonl
```

```
  protocol=otherge seed=42
   link_prediction     0.6s  accuracy=0.9313 precision=0.9160 recall=0.9498 f1=0.9326 auc=0.9775
       dist_approx     5.0s  baseline.mae=1.5269 ... rf.mae=1.3231 ... mlp.mae=1.3187 ...
```

### API

```python
import evaluator as ev

rep = ev.evaluate("cora", "emb/cora.npy",
                  tasks=("link_prediction", "dist_approx"),
                  protocol="otherge", seed=42,
                  tags={"method": "spectral"})
print(rep.table())
rep.write("runs/otherge.jsonl")
```

One task alone, and the `TaskResult` directly:

```python
res = ev.link_prediction("cora", Z, protocol="otherge")
res.scores        # {'accuracy': 0.9313, ..., 'auc': 0.9775}
res.sizes         # {'positives': 5278, 'negatives': 5278, ...}
res.cfg           # the resolved settings, plus protocol and protocol_modified
```

`evaluate()` and the CLI reach the **same** task functions. Neither adds a
measurement step the other misses.

---

## 3. The protocol rule

A keyword left at `None` takes the value of the protocol. A keyword given
**explicitly** overrides it and sets `protocol_modified` on the record.
`table()` marks such a run with a leading `*`:

```
* protocol=otherge seed=42
*      dist_approx     2.2s  baseline.mae=1.5116 ...
```

The `*` means: **do not compare these numbers to the recorded baseline.**
The flag reports the intent of the caller, not the difference of two
objects — an explicit keyword sets it even when the value equals the one
the protocol already held.

### The protocols

| name | source | LP | DA |
|---|---|---|---|
| `otherge` | `experiments/other-ge/bench_other_ge.py` | 80,000 pairs, 100 trees | distance feature, PLL hops |
| `n2v1m` | `experiments/large-graph-node2vec/bench_node2vec_1M.py` | 80,000 pairs, 100 trees, `pos_draw=always` | 200 sources, BFS hops |
| `fodined` | `fodined/modular.py` | 50,000 pairs, 200 trees, `neg_draw=far_pairs` | 2,000 pairs, 200 trees, leaf 1, MLP (128, 64) |
| `fodiwalk_dist` | `fodiwalk/misc/evaluation.py` | 50,000 pairs, 200 trees, `neg_draw=far_pairs` | 200 sources, `pair_draw=bfs_rows`, **distance** feature |
| `fodiwalk_vec` | same | same | same, **vector** feature |
| `default` | no baseline | the dataclass defaults | the dataclass defaults |

`fodiwalk` is one baseline under two names because every caller sweeps
both feature widths. One name for two rows would make `protocol_modified`
lie.

Inspect a protocol from Python:

```python
lp, da = ev.PROTOCOLS["otherge"]
lp.max_pairs        # 80000
da.hops             # 'pll'
```

Full provenance, cell by cell, is in the docstring of
[config.py](config.py) — including the choices a baseline made badly, kept
because a repair here makes a recorded number unreproducible.

---

## 4. Tasks

### `link_prediction`

Positives are real edges. Negatives are drawn pairs with no edge. The
classifier reads one feature of the two vectors.

```python
ev.link_prediction(graph, Z, *, protocol="default", seed=42, rng=None,
                   max_pairs=None, test_size=None, neg_ratio=None,
                   feature=None, model=None, n_estimators=None,
                   pos_draw=None, neg_draw=None, strict=True)
```

| setting | values | default |
|---|---|---|
| `max_pairs` | positives **and** negatives together | 50,000 |
| `test_size` | | 0.2 |
| `neg_ratio` | negatives for each positive | 1.0 |
| `feature` | `hadamard`, `l1`, `concat`, `avg` | `hadamard` |
| `model` | `rf`, `logreg` | `rf` |
| `n_estimators` | | 200 |
| `pos_draw` | `over_cap`, `always` | `over_cap` |
| `neg_draw` | `reject`, `far_pairs` | `reject` |

Scores: `accuracy`, `precision`, `recall`, `f1`, `auc`.

`pos_draw` and `neg_draw` exist only to reproduce old runs. Each names a
different way of drawing pairs out of the same generator. The two ways
give different pairs **and** leave the generator in a different state, so
every draw after them differs too. Do not change either to tune a score.

### `dist_approx`

The regressor reads the two vectors of a pair and must say how many hops
apart the nodes are. The target is the graph's **true** distance, read by
`evaluator.hops`, never a stored augmentation weight.

```python
ev.dist_approx(graph, Z, *, protocol="default", seed=42, rng=None,
               n_pairs=None, n_sources=None, min_hop=None, feature=None,
               metric=None, hops=None, models=None, test_size=None,
               strict=True)
```

| setting | values | default |
|---|---|---|
| `n_pairs` | | 20,000 |
| `n_sources` | draw the first node from N sources; 0 = every node | 0 |
| `min_hop` | drop the pairs closer than this | 2 |
| `feature` | `distance` (1 column), `vector` (`n_dim` columns) | `distance` |
| `metric` | `euclidean`, `poincare`, `cosine`, `dot` | `euclidean` |
| `hops` | `auto`, `pll`, `bfs`, `landmark` | `auto` |
| `models` | any of `baseline`, `rf`, `mlp` | all three |

Scores, for each model: `mae`, `mre`, `rmse`, `r2`, `exact`. `baseline`
predicts the train mean; a model that does not beat it read nothing out of
the geometry.

**The feature width is part of the protocol.** `distance` gives one column
and `vector` gives `n_dim`. Neither ever falls back to the other: a model
with 128 features can win only because it has more of them. An unknown
name raises.

---

## 5. CLI reference

Task selection — `--all`, or one flag for each task, or the paren sugar:

```bash
python -m evaluator cora emb.npy -lp -da
python -m evaluator cora emb.npy --task "dist_approx(n_pairs=5000, feature=distance)"
```

`--task NAME(key=value, ...)` is repeatable and equal to the flag form. An
unknown key raises and names the valid keys, read from the task signature
itself. An explicit `--lp-*` / `--da-*` flag wins over the sugar.

| group | flags |
|---|---|
| link prediction | `--lp-max-pairs` `--lp-test-size` `--lp-neg-ratio` `--lp-feature` `--lp-estimators` |
| dist approx | `--da-pairs` `--da-sources` `--da-min-hop` `--da-feature` `--da-hops` `--da-models` |
| global | `--protocol` `--metric` `--seed` `--max-nodes` `--results` `--tag K=V` `--no-strict` `--quiet` `--json` |

Every option carries its task prefix, thus `--lp-feature` can never touch
`dist_approx`, although both tasks hold a field named `feature`.

`--results PATH` writes the record: `.jsonl` appends one line, `.json`
writes one object. `--json` prints the whole record to stdout and nothing
else.

An error prints as `error: message` to stderr with exit code 1, never a
traceback.

---

## 6. Inputs

`load_graph` dispatches on the **type** of the source:

| source | read as |
|---|---|
| `scipy.sparse` matrix | used directly |
| `(m, 2)` int array | an edge list |
| a path `.npz` | scipy sparse, else a key `edges` or `A` |
| a path `.mtx` | Matrix Market |
| any other existing path | a text edge list; `#` is a comment |
| a name that is not a path | the registry (`cora`, `pubmed`, `wordnet`, ...) via `fodiwalk` |

`A` always comes back as a symmetric CSR of 1.0, sorted indices, zero
diagonal, ids contiguous on `0..n-1`.

`load_embedding`:

| source | read as |
|---|---|
| numpy array | used directly |
| `.npy` | the array |
| `.npz` | key `Z`, else the only key |
| `.csv` / `.tsv` | a matrix, with an optional leading node-id column |
| anything else | word2vec text (`count dim` header) |

A node the word2vec vocabulary missed keeps an all-zero row. The count
reaches the record as `embedding.zero_rows`, and guard C3 reads it.

Both loaders are re-exported: `ev.load_graph`, `ev.load_embedding`. Pass
the loaded `(A, n, info)` triple or `(Z, info)` pair straight into a task
to read a large graph one time only.

---

## 7. Hop backends

| backend | exact | cost |
|---|---|---|
| `pll` | yes | NetworKit PrunedLandmarkLabeling. One index for each graph; a query intersects two label sets. |
| `bfs` | yes | SciPy `shortest_path`, in blocks of sources. One sweep for each distinct source. |
| `landmark` | **no**, an upper bound | a few full sweeps. Ask for it by name; `auto` never picks it. |

`auto` counts the distinct sources and applies the rule measured in
`experiments/bench_shortest_path_gemsec.py`: few sources on a big graph
(`n > 200,000`, `n_sources <= 512`) go to `bfs`, because the PLL index for
that graph would cost more than the whole query set. Everything else goes
to `pll` when NetworKit is importable.

No function here ever builds an `(n, n)` array.

---

## 8. Guards

A guard stops a run that would print a valid-looking number that means
nothing.

The incident: a BFS truncation in `bench_other_ge._pick_subtree` turned
NCBI into a **star** — one hub of degree 19,999 and 19,999 leaves. Every
non-edge pair sat exactly 2 hops apart, thus every method reached a
perfect score. Nothing crashed.

| id | check | fires when |
|---|---|---|
| C1 | `check_shape` | `Z.shape[0] != n` |
| C2 | `check_finite` | `Z` holds a NaN or an inf |
| C3 | `check_zero_rows` | over 50% of the rows of `Z` are all-zero |
| C4 | `check_metric` | `metric="poincare"` and a row has norm >= 1 |
| C5 | `check_class_balance` | the positive fraction leaves `[20%, 80%]` |
| C6 | `check_target_spread` | the hop target holds under 2 distinct values |
| C7 | `check_target_degeneracy` | one hop value holds over 90% of the sample — the star |
| C8 | `check_sample_shortfall` | the sampler returned under 50% of the request |
| C9 | `check_reachability` | under 10% of the sampled pairs are reachable |

Each message names three things — the id, the measured value, the
threshold:

```
error: C1: Z has 100 rows and the graph has n=2708 nodes (threshold: equal).
A mismatched embedding scores the wrong nodes.
```

`strict=True` (the default) raises `evaluator.DegenerateEvaluation`.
`strict=False` (`--no-strict`) records the same message in
`TaskResult.warnings` and continues.

---

## 9. The record

`Report.to_dict()` is the JSON schema of the project, `evaluator/1`:

```json
{
  "created": "2026-08-24T07:36:45+00:00",
  "protocol": "otherge",
  "protocol_modified": false,
  "seed": 42,
  "graph":     {"source": "cora", "n": 2708, "n_edges": 5278, "avg_degree": 3.898},
  "embedding": {"source": "emb.npy", "shape": [2708, 32], "dtype": "float64", "zero_rows": 0},
  "env":       {"python": "3.10.12", "numpy": "1.26.4", "scipy": "1.15.3",
                "sklearn": "1.7.2", "networkit": "11.2.1"},
  "tags":      {"method": "spectral"},
  "tasks": {
    "link_prediction": {
      "task": "link_prediction",
      "cfg":      {"max_pairs": 80000, "...": "...",
                   "protocol": "otherge", "protocol_modified": false},
      "scores":   {"accuracy": 0.9313, "precision": 0.9160, "recall": 0.9498,
                   "f1": 0.9326, "auc": 0.9775},
      "sizes":    {"positives": 5278, "negatives": 5278, "edges": 5278,
                   "pairs": 10556, "train": 8444, "test": 2112},
      "seconds":  0.5477,
      "warnings": []
    }
  },
  "schema": "evaluator/1"
}
```

Every value is a plain Python type. The `cfg` of each task repeats
`protocol` and `protocol_modified`, thus one `.jsonl` line read alone
still says which settings produced its scores.

The harmonic mean is spelled `f1`. `fodined/link_prediction.py` and
`fodiwalk/misc/evaluation.py` spell it `f1-score`; map one to the other
when you compare an old record to a new one.

---

## 10. Reproducibility

- Each task makes its own generator, `np.random.default_rng(seed)`. That
  reproduces the frozen references, each of which makes its generator at
  its first line. A two-task report therefore holds the same numbers as
  two separate runs.
- **The draw order is the contract.** The generator is a stream: a moved
  or added draw changes every later number and nothing crashes. The step
  order of each task is written at the top of its module. Do not reorder
  it.
- `seed` and `rng` stay separate: `seed` goes to sklearn (`random_state`),
  `rng` goes to the sampling.
- `tests/reference/` holds verbatim frozen copies of the four original
  implementations, so a parity reference cannot drift when the live file
  is edited.
- **A reference number is not bit-stable against itself.** A
  RandomForest with `n_jobs=-1` accumulates over threads in a
  nondeterministic order; three runs of the same seed spread by about
  4e-16. Compare with a tolerance. Never assert bit-equality on the output
  of an estimator built with `n_jobs=-1`.
- `import evaluator` pulls in no `sklearn`, no `networkit` and no
  `gensim`, and it costs under 1.0 s. The task functions import sklearn
  inside their bodies. Do not lift any of those to a module top.

### Which scores survive an embedding that is not bit-reproducible

This package is deterministic for a given embedding. The embedding itself
may not be. If yours comes off a GPU, some scores here move between runs
and some do not, and the split is not random.

| stable | moves |
|---|---|
| `auc` | `accuracy`, `f1` |
| `mae`, `rmse`, `r2` on the `distance` feature | the same three on the `vector` feature |

Two reasons. `accuracy` and `f1` read a hard cut at probability 0.5, so a
tiny shift flips the pairs sitting on the line; `auc` reads the ranking and
does not care. And the `vector` feature trains an MLP on `n_dim` columns,
which amplifies a small input change, while the `distance` feature gives
the model one column.

Measured by another session on pubmed, three runs, same seed, same graph,
identical augmentation:

```
       accuracy    f1      r2 (vector)   mae (vector)
run A   0.9709   0.9706      0.509          0.812
run B   0.9704   0.9701      0.487          0.828
run C   0.9703   0.9700      0.472          0.843

stable across the same three: auc 0.9951, r2 (distance) 0.522,
                              mae (distance) 0.804
```

The cause is outside this package: a scatter-add in
`forcedirected/sell_c_sigma.py`, where three or more float32 values landing
on one address add in whatever order the GPU chooses. Cora never exceeds
two, so cora reproduces exactly; pubmed reaches four, so it does not.

**What to do. Hash `Z` first.** That is the one check that gives a
definite answer, and no tolerance can replace it:

- Two runs, bit-identical `Z`, different scores → **a bug in this
  package.** Report it.
- Two runs, different `Z` → the scores are expected to differ. No band
  needed.

Rest a claim on `auc` and on the `distance`-feature regression.

For a rough sense of scale when you cannot hash, movement of at least this
size **has been observed and is not a bug**:

| field | observed range |
|---|---|
| `accuracy`, `f1` | ± 0.001 |
| `mae`, `vector` feature | ± 0.035 |
| `r2`, `vector` feature | ± 0.040 |

These are absolute, not percentages — a percentage band is meaningless for
`r2`, whose zero point is arbitrary. They come from four runs at one seed
on one graph, so read them as a floor, not a guarantee: four samples cannot
bound a tail.

The ordering, though, is structural and holds on any graph. `accuracy` and
`f1` move by whatever sits within 3.1e-06 of the 0.5 cut, which is a thin
slice. A `vector`-feature score moves by however much a 128-column MLP
amplifies that same 3.1e-06, which has no small bound. So the `vector`
scores are always the noisiest of the four and `accuracy`/`f1` always the
quietest.

---

## 11. Adding a task

One file, plus one registry line:

```python
# evaluator/tasks/my_task.py
from ..report import TaskResult
from . import register, resolve_rng, load_graph, load_embedding, cfg_record

@register("my_task")
def my_task(graph, Z, *, protocol="default", seed=42, rng=None,
            strict=True, my_setting=None) -> TaskResult:
    ...
    return TaskResult(task="my_task", cfg={}, scores={}, sizes={},
                      seconds=0.0, warnings=[])
```

Import it at the bottom of [tasks/\_\_init\_\_.py](tasks/__init__.py).
`evaluate()` and the CLI read the rest of the signature with
`keywords()`, thus the new settings need no change in `cli.py`.

Import sklearn **inside** the function body, never at the module top.

---

## 12. Status and limits

- **No test suite yet.** Parity against the frozen references has been run
  by hand, not by a committed gate. `tests/` holds the references only.
- **No packaging.** Run from the repository root or set `PYTHONPATH`.
- `PROTOCOLS` is a plain mutable dict. Do not write to it.
- The `dist_approx` row of protocol `fodined` **reproduces no published
  number**. `modular.py` computes no hop distance; it reads a stored
  augmentation weight. This package reads the true distance from the
  graph. The two are different measurements, and the difference is
  intentional. The parity gate covers the link prediction of `fodined`
  only.
- Under `--no-strict` a guard warning reaches the record but not the
  printed table.
- A high link-prediction AUC on a graph with a skewed degree distribution
  can come from degree alone. Nothing here flags that yet.

---

## 13. Map of the files

| file | holds |
|---|---|
| [\_\_init\_\_.py](__init__.py) | `evaluate()`, the public names |
| [cli.py](cli.py) | the argument parser, the paren sugar. No measurement. |
| [config.py](config.py) | `LPCfg`, `DACfg`, `PROTOCOLS`, and the provenance of each cell |
| [tasks/link_prediction.py](tasks/link_prediction.py) | the six steps of the LP task |
| [tasks/dist_approx.py](tasks/dist_approx.py) | the eight steps of the DA task |
| [pairs.py](pairs.py) | the samplers, and their draw-order contracts |
| [hops.py](hops.py) | the three hop backends |
| [metrics.py](metrics.py) | the distances and the pair features |
| [scoring.py](scoring.py) | the two frozen score sets |
| [guards.py](guards.py) | C1 to C9 |
| [report.py](report.py) | `TaskResult`, `Report`, the JSON cast |
| [io.py](io.py) | the graph reader, the embedding reader, the record writer |
| [dev-docs/PRD.md](dev-docs/PRD.md) | the block briefs, the invariants, the parity gate |
