# evaluator -- Product Requirements Document

Created 2026-08-22. Authority on WHAT the `evaluator` package must contain,
what each block reads and writes, and what each block must satisfy.
`ORCHESTRATION.md` is the companion: it says WHO builds each block and in
what order.

Source of the idea: `blueprint.md` in this directory. This document keeps
every requirement of the blueprint and adds the constraints that the three
existing call sites make necessary.

---

## 1. Purpose

One module measures the quality of a graph embedding. Today four files
measure it, and each one measures it a different way:

| File | Link prediction | Hop distance |
|---|---|---|
| `experiments/other-ge/bench_other_ge.py` | RF(100), 40k positives | 1 feature (distance), RF(100, leaf 25), MLP(64,32) |
| `experiments/large-graph-node2vec/bench_node2vec_1M.py` | RF(100), 40k positives | 1 feature, 200 BFS sources |
| `fodined/modular.py` | RF(200), 50k pairs | reads `D.data`, NOT the true hops |
| `fodiwalk/misc/evaluation.py` | RF(200), `max_pairs//2` | `n_dim` features, RF(100), MLP(256,128) |

The four disagree on the classifier size, on the pair budget, on the
feature, and on the source of the ground truth. A number of one file is
therefore not comparable to a number of another file. `evaluator` ends the
duplication and it makes the disagreement EXPLICIT: every setting that
differs becomes a named protocol, and every result record carries the
protocol name.

A later consumer is an AGENT that calls this module as a tool. That is
NOT a requirement of this project, and section 13 says why the design
keeps it cheap: one file over `evaluate()`, with no change to any block
below.

`fodiwalk/misc/evaluation.py` states the rule that this package enforces:

> A model with 128 features can win only because it has more of them, thus
> the two rows must not be mixed. ... A number is comparable to the
> baseline that used the SAME settings, and to no other.

## 2. Goals

1. **G1 Two surfaces.** A CLI (`python -m evaluator ...`) and an importable
   API (`evaluator.link_prediction(...)`). The two share one code path; the
   CLI is a thin argument parser plus a printer.
2. **G2 Extensible tasks.** A new task is one file plus one registry line.
   Nothing else changes.
3. **G3 Named protocols.** Every recorded baseline of this repository has a
   protocol name. A run states its protocol; a result record carries it.
4. **G4 Exact parity.** With the matching protocol and seed, `evaluator`
   reproduces the numbers of the two benchmark scripts EXACTLY.
5. **G5 Scale.** A graph of 1.13M nodes runs inside the peak memory that
   `experiments/large-graph-node2vec/REPORT.md` records.
6. **G6 Guards.** The run STOPS before it prints a meaningless number. A
   degenerate sample raises; it does not score 1.0000 in silence.
7. **G7 Machine-readable output.** One JSON record for each run, with the
   configuration, the sizes, the scores, the timings, and the provenance.
8. **G8 Two surfaces, one code path.** A human uses the CLI. A script
   imports the API. The two reach the same task functions; a surface holds
   argument parsing, printing and error shaping, and no measurement.
   `--json` makes the CLI machine-readable, thus a caller that is not a
   Python process still gets the numbers.

## 3. Non-goals

* `evaluator` does not build embeddings. It reads them.
* `evaluator` does not own the dataset registry.
  `fodiwalk.make_graph.datasets` owns it. `evaluator` calls it when the
  graph argument is a name and not a path.
* `evaluator` does not plot, and it does not write a report in Markdown.
* `evaluator` adds NO new dependency. In particular it does NOT use the
  `mcp` package, which is absent from `.venv/`. An agent tool surface is
  deferred; see section 13.
* `evaluator` does not delete or change `fodiwalk/misc/evaluation.py`.
  That file stays; `fodiwalk` keeps its own measurement until its own
  campaign chooses to move.

---

## 4. Target tree

```
evaluator/
    __init__.py            the public API. Re-exports and the TASKS registry.
    __main__.py            `python -m evaluator` -> cli.main()
    cli.py                 argparse, the paren sugar, and the printer
    config.py              TaskCfg dataclasses + the PROTOCOLS table
    io.py                  load_graph, load_embedding, write_report
    pairs.py               positive/negative pair sampling
    hops.py                ground-truth hop distance, three backends
    metrics.py             distance registry, pair features
    scoring.py             classification and regression score sets
    guards.py              the checks that stop a meaningless run
    report.py              TaskResult, Report, the JSON schema
    tasks/
        __init__.py        TASKS = {...}; register()
        link_prediction.py
        dist_approx.py
    tests/
        test_parity.py     the gate of section 9
        test_api.py        the frozen public surface
        test_guards.py
        test_cli.py
        test_structure.py  the import graph of section 4
    dev-docs/
        blueprint.md  PRD.md  ORCHESTRATION.md  CATALOG.md
```

Import discipline, one way only:

```
    config -> (nothing)
    metrics, scoring, guards -> numpy / scipy / sklearn only
    pairs, hops -> numpy / scipy (+ optional networkit)
    io -> numpy / scipy (+ optional fodiwalk.make_graph.datasets)
    report -> config, io          (write() delegates; io imports no report)
    tasks/* -> config, pairs, hops, metrics, scoring, guards, report
    __init__ -> tasks, report, io, config
    cli -> __init__, io
```

`tasks/` never imports `cli`. A cycle is a defect.

---

## 5. The public API

### 5.1 The task functions

```python
import evaluator as ev

lp = ev.link_prediction(A, Z, protocol="otherge", seed=42)
da = ev.dist_approx(A, Z, protocol="otherge", metric="poincare", seed=42)
```

Both return a `TaskResult`. Both accept a `scipy.sparse` matrix, an
`(m, 2)` edge array, or a path. Both accept an `(n, d)` array or a path.

### 5.2 The batch surface

```python
rep = ev.evaluate(A, Z, tasks=("link_prediction", "dist_approx"),
                  protocol="otherge", seed=42, tags={"method": "node2vec"})
rep.write("results.jsonl")     # append one line
print(rep.table())             # the one-line summary a log prints
```

### 5.3 Signatures, frozen

```python
def link_prediction(graph, Z, *, protocol="default", seed=42, rng=None,
                    max_pairs=None, test_size=None, neg_ratio=None,
                    feature=None, model=None, n_estimators=None,
                    pos_draw=None, strict=True) -> TaskResult

def dist_approx(graph, Z, *, protocol="default", seed=42, rng=None,
                n_pairs=None, n_sources=None, min_hop=None, feature=None,
                metric=None, hops=None, models=None, test_size=None,
                strict=True) -> TaskResult

def evaluate(graph, Z, *, tasks=("link_prediction",), protocol="default",
             seed=42, tags=None, strict=True, **task_kw) -> Report
```

A keyword left at `None` takes the value of the protocol. A keyword given
explicitly overrides the protocol AND marks the record
`"protocol_modified": true`, thus a changed number cannot claim the name of
a recorded baseline.

---

## 6. The CLI

```
.venv/bin/python -m evaluator GRAPH EMBEDDING [tasks] [options]
```

`GRAPH` is a path (`.txt`, `.edges`, `.npz`, `.mtx`) or a registry name
(`cora`, `pubmed`, `wordnet`, `ncbi_taxonomy`, `com_youtube`, ...).
`EMBEDDING` is a path (`.npy`, `.npz`, `.csv`, `.tsv`).

### 6.1 The task flags

```
--link-prediction, -lp        run the link prediction task
--dist-approx, -da            run the hop distance approximation task
--all                         run every registered task
```

### 6.2 The per-task options

Each option carries the task prefix, thus two tasks cannot collide.

```
--lp-max-pairs N        positives and negatives together
--lp-test-size F        the test fraction (0.2)
--lp-neg-ratio F        negatives for each positive (1.0)
--lp-feature NAME       hadamard | l1 | concat | avg
--lp-estimators N       the trees of the forest

--da-pairs N            the pairs to sample
--da-sources N          draw the first node from N sources only. 0 = all
--da-min-hop N          drop the pairs closer than this (2)
--da-feature NAME       distance (1 column) | vector (n_dim columns)
--da-hops NAME          auto | pll | bfs | landmark
--da-models LIST        baseline,rf,mlp
```

### 6.3 The global options

```
--protocol NAME     otherge | n2v1m | fodined | fodiwalk | default
--metric NAME       euclidean | poincare | cosine | dot
--seed N            42
--max-nodes N       truncate a registry graph (0 = no truncation)
--results PATH      .json writes one record; .jsonl appends one line
--tag K=V           repeatable. Goes into the record as provenance
--no-strict         a guard writes a warning and does not raise
--quiet             the record only, no table
```

### 6.4 The machine-readable output

```
--json              write the whole record as JSON to stdout, and NOTHING
                    else. No table, no progress line. A caller that is not
                    a Python process reads stdout and parses it.
```

`--json` implies `--quiet`. It works with `--results`: the record goes to
both. The exit code stays the contract of B12: 0 on success, 1 when a
guard raises.

### 6.5 The paren sugar

The blueprint writes `--link-prediction(split=20, negposratio=1)`. A shell
does not pass that reliably, thus the same thing has a quoted form:

```
--task "link_prediction(test_size=0.2, neg_ratio=1)"
--task "dist_approx(n_pairs=20000, feature=distance)"
```

`cli.parse_call` reads it: a name, then `key=value` pairs separated by
commas. A value is read as int, then float, then bool (`true`/`false`),
then string. An unknown key RAISES and names the keys that the task has.
`--task` may repeat. It is equal to the flag form; a run may mix the two,
and an explicit flag wins over the sugar.

### 6.6 An example

```
.venv/bin/python -m evaluator cora emb/cora_n2v.npy \
    --link-prediction --dist-approx --protocol otherge \
    --tag method=node2vec --tag dim=128 --results runs/otherge.jsonl
```

---

## 7. Blocks

Each block below states its inputs, its outputs, its internal functions,
and what it must satisfy. A block is one work unit of
`ORCHESTRATION.md`.

### B1. `config.py` -- the settings and the protocols

**In.** Nothing. This module is the bottom of the tree and it imports
`dataclasses` only.

**Out.** Two frozen dataclasses and one table.

```python
@dataclasses.dataclass(frozen=True)
class LPCfg:
    max_pairs: int = 50_000      # positives AND negatives together
    test_size: float = 0.2
    neg_ratio: float = 1.0
    feature: str = "hadamard"    # hadamard | l1 | concat | avg
    model: str = "rf"            # rf | logreg
    n_estimators: int = 200
    pos_draw: str = "over_cap"   # over_cap | always -- see below

@dataclasses.dataclass(frozen=True)
class DACfg:
    n_pairs: int = 20_000
    n_sources: int = 0           # 0 = every node is a candidate source
    min_hop: int = 2             # a neighbour is the easy case; drop it
    feature: str = "distance"    # distance (1 col) | vector (n_dim cols)
    metric: str = "euclidean"
    hops: str = "auto"           # auto | pll | bfs | landmark
    pair_draw: str = "reject"    # reject | bfs_rows -- see below
    models: tuple = ("baseline", "rf", "mlp")
    test_size: float = 0.2
    rf_estimators: int = 100
    rf_min_leaf: int = 25
    mlp_hidden: tuple = (64, 32)
    mlp_early_stop: bool = True
```

**`fodiwalk` is TWO protocols, and that is the finding of 2026-08-22.**
The first draft had one `fodiwalk` row with `feature=vector`, read from
the default of `task_hop`. That default is not the protocol. All three
callers of that function --- `fodiwalk/tests/harness.py:63`,
`experiments/fodiwalk/bench_fodiwalk.py:245` and
`experiments/fdwalk/bench_fdwalk.py:856` --- run

```python
for feature in ("distance", "vector"):
    ... task_hop(Z, u, v, d, seed, feature) ...
```

thus the baseline records BOTH widths and reports both. One name for two
recorded rows would make `protocol_modified` lie: a user who asked for the
other width would get the flag that says "this cannot be compared to a
recorded baseline", when it reproduces one exactly. Two names keep the
flag honest. The peer session that owns that file changed the DEFAULT to
`distance` on 2026-08-22; the default is behaviour-neutral, because every
caller passes the value explicitly.

`pair_draw` is the second compatibility knob, and it is the finding of
2026-08-22 (agent A10). Three baselines draw their hop pairs with the
non-edge REJECTION sampler. `fodiwalk` does not: `hop_sample`
(`fodiwalk/misc/evaluation.py:164`) holds ONE BFS row in the memory and it
draws random targets OF THAT ROW, thus it consumes the generator for each
source and not for each batch, and it never rejects an edge. Both
baselines carry `n_sources=200` and `n_pairs=20_000`, thus no other field
tells them apart, and parity test P6 is unreachable without this switch.

`pos_draw` is a legacy compatibility knob and it exists for one reason:
`bench_other_ge.py` calls `rng.choice` only when the edge count is over the
cap, and `bench_node2vec_1M.py` calls it always. The two consume the
generator differently, thus the negatives that follow are different pairs.
Exact parity needs both behaviours.

**The protocol table.** One entry for each recorded baseline.

| name | LP | DA |
|---|---|---|
| `otherge` | `max_pairs=80_000`, `n_estimators=100`, `pos_draw=over_cap` | `feature=distance`, `hops=pll`, `rf_estimators=100`, `rf_min_leaf=25`, `mlp_hidden=(64,32)`, `mlp_early_stop=True` |
| `n2v1m` | `max_pairs=80_000`, `n_estimators=100`, `pos_draw=always` | `n_sources=200`, `hops=bfs`, everything else as `otherge` |
| `fodined` | `max_pairs=50_000`, `n_estimators=200`, `pos_draw=over_cap` | `n_pairs=2_000`, `feature=distance`, `rf_estimators=200`, `rf_min_leaf=1`, `mlp_hidden=(128,64)`, `mlp_early_stop=True` |
| `fodiwalk_dist` | `max_pairs=50_000`, `n_estimators=200` | `n_sources=200`, `hops=bfs`, `pair_draw=bfs_rows`, `feature=distance`, `rf_estimators=100`, `rf_min_leaf=1`, `mlp_hidden=(256,128)`, `mlp_early_stop=False` |
| `fodiwalk_vec` | the same | the same, with `feature=vector` |
| `default` | the dataclass defaults | the dataclass defaults |

`otherge` uses `max_pairs=80_000` because the source script caps the
POSITIVES at 40,000 and it then draws an equal number of negatives.
`max_pairs` counts both classes.

**A CONFLICT to resolve before B10 is built.** The `fodined` row says
`feature=distance`. Two sources disagree:

* the LIVE code, `fodined/modular.py`, builds
  `sp_X = np.linalg.norm(Z[u] - Z[v], axis=1)[:, None]` -- ONE column. The
  two `vector` forms above that line are commented out.
* the docstring of `fodiwalk/misc/evaluation.task_hop` says that `vector`
  "is the protocol of `fodined/modular.py`".

The table follows the CODE, because the code produced the recorded
numbers. The agent that builds B10 must confirm this against the recorded
log before it writes the row, and it must record the answer in
`CATALOG.md`. A wrong cell here makes P5 fail, which is what P5 is for.

**A second difference of the `fodined` row.** `fodined/modular.py` reads
the hop target from `D.data` -- the weights that the AUGMENTATION stored.
`evaluator` reads the TRUE hop distance from the graph. The two are not
the same measurement: `D` holds only the pairs that the policy sampled,
and its unreachable sentinel is `n`. This is an intentional change and not
a parity break. P5 therefore tests the LINK PREDICTION of `fodined` only.
The hop row of `fodined` has no parity reference, and `CATALOG.md` records
that.

**Internal functions.**
`get(name) -> (LPCfg, DACfg)`; `override(cfg, **kw) -> (cfg2, modified)`.

**Must satisfy.**
1. `config.py` imports `dataclasses` and nothing else.
2. Every field of both dataclasses appears in `tests/test_api.py` with its
   default value written literally. A renamed field breaks that test.
3. `override(cfg)` with no keyword returns `(cfg, False)`; with one
   keyword whose value equals the current value it still returns `True`.

### B2. `io.py` -- reading a graph, reading an embedding, writing a record

**In.** A path or a name. A path or an array.

**Out.** `(A, n)` where `A` is a symmetric CSR of 1.0 with sorted indices
and a zero diagonal. `Z` as a 2-D array with `Z.shape[0] == n`.

**Internal functions.**

```python
load_graph(src, max_nodes=0, seed=42) -> (A, n, info)
```
* `src` is a `scipy.sparse` matrix -> normalize it and return it.
* `src` is an `(m, 2)` integer array -> `to_csr`.
* `src` is a path -> the suffix chooses the reader: `.npz` (scipy, or a
  file with an `edges`/`A` key), `.mtx`, otherwise a whitespace or comma
  edge list that skips `#` comment lines.
* `src` is a name -> `fodiwalk.make_graph.datasets.load(name, max_nodes,
  seed)`. An `ImportError` becomes a clear message that names the package,
  and not a traceback.

`info` records the source, `n`, the undirected edge count, and the average
degree.

```python
load_embedding(src, n=None) -> (Z, info)
```
`.npy`; `.npz` (the key `Z`, else the only key); `.csv`/`.tsv` (an optional
first column of node ids); a word2vec text file (a header line of
`count dim`). `info` records the source, the shape, the dtype, and
`zero_rows` -- the count of rows that are all zero, because the benchmark
scripts leave a zero row for a node that the vocabulary misses.

```python
write_report(report, path) -> None
```
`.jsonl` appends one line and it creates the parent directory. `.json`
writes one indented object. Any other suffix raises.

**Must satisfy.**
1. `load_graph` on a 4-node path graph edge list gives `A.nnz == 6`,
   `A.diagonal().sum() == 0`, `A.has_sorted_indices`.
2. The node ids are made contiguous. A file with the ids `{5, 9, 40}`
   gives `n == 3`.
3. `load_graph("cora")` gives `n == 2708`.
4. A `.jsonl` write of two reports gives a file of two lines, and each
   line parses as JSON.
5. `io.py` imports `fodiwalk` INSIDE the function, and not at the top.
   `import evaluator` must work when `fodiwalk` is absent.

### B3. `pairs.py` -- the samples

**In.** `A`, `n`, a count, an `rng`.

**Out.** An `(k, 2)` int64 array.

**Internal functions.**

```python
positives(A, max_count, rng, draw="over_cap") -> (k, 2)
negatives(A, n, count, rng, sources=None) -> (k, 2)
pairs_for_hops(A, n, count, rng, sources=None, draw="reject") -> (k, 2)
```

`negatives` is the rejection sampler of the two benchmark scripts: a sorted
array of the composite keys `u * n + v`, then one `np.searchsorted` for a
whole batch. It must NOT use a Python set. The set form needed 1.8 GB at a
million nodes (`experiments/large-graph-node2vec/REPORT.md`).

**THE DRAW ORDER IS PART OF THE CONTRACT.** `fodiwalk/tests/harness.py`
states it: "THE ORDER OF THE GENERATOR IS PART OF THE PARITY." Each
function must consume the generator in the same order as the source it
replaces, and the docstring must write that order down:

```
negatives(): while short of `count`:
    draw = (count - have) * 2 + 1024
    u = rng.choice(sources, draw)  if sources else  rng.integers(0, n, draw)
    v = rng.integers(0, n, draw)
```

`u` before `v`. A swap changes every pair.

`draw="bfs_rows"` is the OTHER order, and it is not a variant of the same
loop. It is the loop of `fodiwalk/misc/evaluation.hop_sample`: one BFS row
at a time, `per = n_pairs // n_sources` targets drawn from that row with
`rng.choice(n, size=min(per * 2, n), replace=False)`, then the first `per`
reachable ones kept. It returns the hop distances TOO, because the row is
already in the memory and a second pass would cost another BFS. Thus
`pairs_for_hops(draw="bfs_rows")` returns `(pairs, hops)` and the caller
skips `hops.hop_distance`. B10 step 3 branches on this.

**Must satisfy.**
1. No pair of `negatives` has an edge in `A`, and no pair has `u == v`.
2. `positives` gives each undirected edge one time, and never both
   directions.
3. With `draw="over_cap"` and an edge count under the cap, the generator
   state after the call is UNCHANGED.
4. On `com_youtube` (n = 1.13M), `negatives(..., 40_000)` uses less than
   400 MB above the size of `A`.
5. Byte-for-byte equality with the `sample_non_edges` of
   `bench_other_ge.py` for the same seed and the same count.

### B4. `hops.py` -- the ground truth

**In.** `A`, `n`, an `(k, 2)` pair array, a backend name.

**Out.** A float64 array of `k` hop distances. An unreachable pair gets
`np.inf`.

**Internal functions.**

```python
hop_distance(A, n, pairs, backend="auto", rng=None, budget=200e6) -> (k,)
_pll(A, n, pairs)          exact. networkit PrunedLandmarkLabeling
_bfs(A, n, pairs, budget)  exact. scipy shortest_path, in source blocks
_landmark(A, n, pairs, rng, count=64)   approximate. An UPPER bound
choose_backend(n, n_sources) -> str
```

The three facts that cost time to find, and that the docstrings must hold:

* PLL returns `2**64 - 1` for an unreachable pair. This is not documented
  in NetworKit. Test for it.
* `nk.GraphFromCoo` gave a segmentation fault on a symmetric matrix with a
  `data` array. Pass the UPPER TRIANGLE as two `uint64` `(i, j)` arrays.
* `scipy.sparse.csgraph.shortest_path` always returns a DENSE
  `(len(indices), n)` array. `indices=` limits the rows and not the
  density. One block of `budget / (n * 8)` sources holds the peak near
  200 MB.

`choose_backend` states the crossover that
`experiments/bench_shortest_path_gemsec.py` measured: the cost of PLL is
fixed for each graph, and the cost of SciPy grows with the count of
DISTINCT sources. Thus:

```
n_sources > 0 and n_sources <= 512 and n > 200_000   ->  "bfs"
networkit importable                                 ->  "pll"
otherwise                                            ->  "bfs"
```

`landmark` is never automatic. It is approximate, and a user asks for it.
It reuses `fodiwalk.augment_graph.landmarks` when that package is present,
because that code is already validated at 1.13M nodes.

**Must satisfy.**
1. On a 200-node random graph, `_pll`, `_bfs` and
   `scipy.sparse.csgraph.shortest_path` agree on 5,000 pairs, exactly.
2. `_landmark` is an upper bound: `_landmark >= _bfs` for every pair, and
   the report gives the mean error.
3. An unreachable pair gives `inf` from every backend. A two-component
   graph proves it.
4. `_bfs` on `com_youtube` with 200 sources stays under 600 MB above `A`.
5. `hop_distance` never builds an `(n, n)` array.

### B5. `metrics.py` -- the distance, and the pair feature

**In.** `Z`, and two index arrays.

**Out.** A distance vector, or a feature matrix.

**Internal functions.**

```python
DISTANCES = {"euclidean": ..., "poincare": ..., "cosine": ..., "dot": ...}
distance(Z, u, v, metric="euclidean") -> (k,)
FEATURES = {"hadamard": ..., "l1": ..., "concat": ..., "avg": ...}
feature(Z, u, v, kind="hadamard") -> (k, ?)
```

The Poincare distance is
`arccosh(1 + 2*|u-v|^2 / ((1-|u|^2)(1-|v|^2)))`. A Euclidean distance in
the Poincare ball has no meaning, thus a Poincare embedding must not take
the default. `guards.check_metric` enforces it.

`hadamard`, `l1` and `avg` are symmetric in `(u, v)`, as an undirected pair
feature must be. `concat` is NOT symmetric; it is available and the
docstring says why a user should not choose it for an undirected graph.

**Must satisfy.**
1. `distance(Z, u, v, m) == distance(Z, v, u, m)` for every metric.
2. `poincare` on two points at the origin gives `1.41e-6`, and NOT 0.0.
   That value is `arccosh(1 + 1e-12)`, a direct consequence of the second
   clamp of the reference implementation. Exact parity and a literal zero
   cannot both hold, and parity wins. A test that asserts `== 0.0` here
   fails against a correct implementation; assert `< 1e-5`.
   The distance grows without bound as a point goes to the unit sphere:
   0.5 -> 1.099, 0.9 -> 2.944, 0.99 -> 5.293, 0.999 -> 7.600.
3. `feature` with `hadamard` equals `Z[u] * Z[v]` exactly, and it never
   normalizes. A10 checked all four baselines on 2026-08-22 and NONE of
   them normalizes `Z`; the `LPCfg.normalize` field of the first draft was
   removed for that reason.
4. Every function is vectorized. No Python loop over the pairs.

### B6. `scoring.py` -- the score sets

**In.** `y_true`, `y_pred`, and for a classifier `y_prob`.

**Out.** A `dict[str, float]`. The keys are frozen.

```python
classify_scores(y, pred, prob) -> {accuracy, precision, recall, f1, auc}
regress_scores(y, pred)        -> {mae, mre, rmse, r2, exact}
```

`exact` is `mean(rint(pred) == y)`. A hop distance is an integer, thus this
reads easier than the MAE. `precision`, `recall` and `f1` use
`zero_division=0`.

**Must satisfy.**
1. The five classification keys are spelled `f1` and not `f1-score`.
   `fodined/link_prediction.py` writes `f1-score`; this package writes
   `f1`, and `report.py` records the change.
2. A perfect prediction gives 1.0 for every classification key.
3. `regress_scores` on a constant prediction gives `r2 <= 0`.

### B7. `guards.py` -- stop before a meaningless number

**In.** The sample that a task built.

**Out.** Nothing, or a raised `DegenerateEvaluation`. With
`strict=False`, a warning string that the record keeps.

The reason for this block: `bench_other_ge.py` has a whole docstring about
a BFS truncation that turned NCBI into a star. Every pair without an edge
was then exactly 2 hops apart, and EVERY method reached a perfect score.
The run gave a valid-looking number that meant nothing. `fodiwalk` answers
the same class of problem with `core/plan_contract.py`, which raises and
never warns. This block is that pattern, for a measurement.

**The checks.**

| id | check | raises when |
|---|---|---|
| C1 | shape | `Z.shape[0] != n` |
| C2 | finite | `Z` holds a NaN or an inf |
| C3 | zero rows | more than 50% of the rows of `Z` are all zero |
| C4 | metric domain | `metric="poincare"` and any `\|z\| >= 1` |
| C5 | class balance | the positive fraction is outside `[0.2, 0.8]` |
| C6 | target spread | the hop target has fewer than 2 distinct values |
| C7 | target degeneracy | one hop value holds more than 90% of the sample |
| C8 | sample shortfall | the sampler returned less than 50% of the request |
| C9 | reachability | less than 10% of the sampled pairs are reachable |

C7 is the NCBI star. C3 is the out-of-vocabulary zero row. C4 is the
Poincare defect.

**Must satisfy.**
1. A star graph of 500 leaves raises `DegenerateEvaluation` with the id
   `C7` in the message.
2. `strict=False` returns instead of raising, and the returned warning
   holds the same id.
3. Every check names the id, the measured value, and the threshold.

### B8. `report.py` -- the record

**In.** The results of the tasks, and the provenance.

**Out.** One JSON object.

```python
@dataclasses.dataclass
class TaskResult:
    task: str          # "link_prediction"
    cfg: dict          # the resolved configuration, every field
    scores: dict       # {model: {metric: value}} or {metric: value}
    sizes: dict        # the counts a log prints
    seconds: float
    warnings: list     # the guard ids that `strict=False` downgraded

@dataclasses.dataclass(kw_only=True)      # kw_only: a default may precede
class Report:                             # a required field. Python 3.10+.
    schema: str = "evaluator/1"
    created: str
    protocol: str
    protocol_modified: bool
    seed: int
    graph: dict        # from io.load_graph
    embedding: dict    # from io.load_embedding
    env: dict          # python, numpy, scipy, sklearn, networkit versions
    tags: dict
    tasks: dict        # {name: TaskResult}
```

**Internal functions.** `Report.to_dict()`, `Report.write(path)`,
`Report.table()` -- one line for each task, the format of the `summary`
block of `bench_other_ge.py`.

**Must satisfy.**
1. `json.dumps(report.to_dict())` succeeds. Every value is a JSON scalar,
   a list, or a dict.
2. NO numpy object survives `to_dict()`, and `np.float64` is the trap.
   It SUBCLASSES Python `float`, thus an `isinstance(o, float)` test
   passes it through untouched and `json.dumps` still succeeds. The bug is
   then invisible until a stricter encoder, or a `type(v) is float` test,
   meets it. Test the cast by walking the dict and asserting that no value
   has `type(v).__module__ == "numpy"`. Order the checks so the numpy test
   comes FIRST, before any `isinstance` against a builtin.
3. `table()` never raises on a record that `to_dict()` can serialize. A
   score that is not a real number is printed with `repr`, and not with a
   float format.
4. The record holds `protocol` and `protocol_modified`. A record with
   `protocol_modified == true` must not be compared to a recorded
   baseline, and `table()` marks it with a `*`.
5. `env` holds the version of every library that took part.

### B9. `tasks/link_prediction.py`

**In.** `A`, `n`, `Z`, `LPCfg`, `rng`, `seed`, `strict`.

**Out.** A `TaskResult`.

**The steps, in this order.** The order is the parity contract.

```
0. guard C1 (shape) FIRST, before any indexing of `Z`.
1. pos = pairs.positives(A, cfg.max_pairs // 2, rng, cfg.pos_draw)
2. neg = pairs.negatives(A, n, round(pos.shape[0] * cfg.neg_ratio), rng)
3. X   = metrics.feature(Z, ..., cfg.feature);  y = [1...1, 0...0]
4. guards C2, C3, C5   (C1 already ran at step 0)
5. train_test_split(test_size=cfg.test_size, random_state=seed, stratify=y)
6. fit the model; scoring.classify_scores
```

**C1 runs at step 0 and not at step 4.** The first draft put it at step 4.
A `Z` with FEWER rows than `n` then raises a bare `IndexError` inside the
fancy indexing of step 3, before C1 can produce its message. The check
consumes no generator draw, thus moving it costs no parity. Agent A15
found this on 2026-08-22.

**Must satisfy.**
1. On cora with the `otherge` protocol and seed 42, the five scores equal
   the scores of `bench_other_ge.py` to 1e-12.
2. On cora with the `fodined` protocol and seed 42, the five scores equal
   the scores of `fodiwalk/misc/evaluation.link_prediction`.
3. `sizes` holds `positives`, `negatives`, `edges`, `pairs`, `train`,
   `test`.

### B10. `tasks/dist_approx.py`

**In.** `A`, `n`, `Z`, `DACfg`, `rng`, `seed`, `strict`.

**Out.** A `TaskResult` whose `scores` is `{model: {metric: value}}`.

**The steps, in this order.**

```
1. sources = rng.choice(n, cfg.n_sources, replace=False)  if n_sources else None
2. pr  = pairs.pairs_for_hops(A, n, cfg.n_pairs, rng, sources, cfg.pair_draw)
3. h   = hops.hop_distance(A, n, pr, cfg.hops, rng)
       -- SKIPPED when cfg.pair_draw == "bfs_rows": step 2 already gave `h`
4. keep = isfinite(h) & (h >= cfg.min_hop)
       -- the filter is applied HERE, after the sample, and never inside
          the sampler. `hop_sample` keeps `d > 0` and its CALLER applies
          `d >= 2` (`fodiwalk/tests/harness.py:29`). A filter moved inside
          the sampler changes the pair count and breaks P6.
5. guards C6, C7, C8 on the KEPT target; C9 on the RAW `h`
6. X = metrics.distance(...) [:, None]   if cfg.feature == "distance"
       metrics.feature(..., "l1")        if cfg.feature == "vector"
7. train_test_split(test_size, random_state=seed)
8. baseline = the train mean; rf; mlp on a StandardScaler
```

Step 6 is the rule of `fodiwalk/misc/evaluation.task_hop`: `distance` gives
ONE column and `vector` gives `n_dim` columns. The two are different
protocols and their numbers must never share a table.

**Must satisfy.**
1. On cora with `otherge` and seed 42, `rf` and `mlp` equal the numbers of
   `bench_other_ge.task_sp_regression` to 1e-12.
2. `feature="vector"` and `feature="distance"` write different `cfg` in
   the record, thus a reader can see which protocol produced a row.
3. `sizes` holds `n_pairs`, `hop_min`, `hop_max`, `reachable`, `backend`.
   `backend` names the RESOLVED backend and never the string `auto`.
   `reachable` is the count BEFORE the filter of step 4.
4. C9 reads the RAW `h`, and not the kept target. Step 4 already drops
   every unreachable pair, thus a C9 over the kept target would report
   100% reachable on every graph and the check would be dead. C6, C7 and
   C8 read the kept target. Agent A15 found this on 2026-08-22.
5. The hop histogram is in `sizes` as `hop_hist`. A reader sees the star
   of C7 even when the guard is off.

### B11. `tasks/__init__.py`, `evaluator/__init__.py`

`TASKS` is a plain dict of `name -> callable`. `register(name)` is a
decorator. A new task is one file plus one import line.

`evaluator/__init__.py` re-exports `link_prediction`, `dist_approx`,
`evaluate`, `Report`, `TaskResult`, `load_graph`, `load_embedding`,
`PROTOCOLS`, `TASKS`, and `DegenerateEvaluation`.

**Must satisfy.**
1. `import evaluator` costs less than 1.0 s and it imports no
   `sklearn`, no `networkit`, and no `gensim` at the top level.
2. `dir(evaluator)` holds every name above. `tests/test_api.py` writes the
   list literally.

### B12. `cli.py`, `__main__.py`

**Must satisfy.**
1. `python -m evaluator --help` exits 0 and it prints every flag of
   section 6.
2. `--task "link_prediction(test_size=0.3)"` gives the same configuration
   as `--link-prediction --lp-test-size 0.3`.
3. An unknown key in the sugar raises and it names the valid keys.
4. `--results out.jsonl` twice gives a file of two lines.
5. The exit code is 1 when a guard raises, and the message names the id.

### B13. `tests/`

`test_parity.py` is the gate of section 9. `test_api.py` freezes the
public surface. `test_guards.py` builds the star and it asserts the raise.
`test_cli.py` runs the module as a subprocess and it parses the `--json`
output. `test_structure.py` asserts the import graph of section 4.

Every test runs on cora. The whole file must finish inside 5 minutes on
this machine.

---

## 8. Cross-block invariants

**I1. One generator, one order.** A task receives ONE `rng`. Every draw of
that task comes from it, in the order that the docstring writes down. A
task never makes a generator. `seed` goes to sklearn (`random_state`) and
`rng` goes to the sampling; the two are separate and both are recorded.

**I2. A protocol is a name, not a default.** Every number in a record
carries the protocol that produced it. An overridden keyword sets
`protocol_modified`.

**I3. No dense `(n, n)`.** No block builds a matrix of that size at any
time. `hops.py` blocks its BFS; `pairs.py` blocks its rejection sampler.

**I4. The feature width is the protocol.** `distance` is 1 column and
`vector` is `n_dim` columns. A function must never fall back from one to
the other.

**I5. A guard raises.** It does not warn, unless the caller passed
`strict=False`. A silent fallback is the defect that this package exists
to stop.

**I6. Optional imports are inside a function.** `networkit`, `fodiwalk`
and `sklearn` are not imported at the top of `evaluator/__init__.py`.

**I7. One code path, two surfaces.** The CLI and the API reach the SAME
task functions. A surface holds argument parsing, printing and error
shaping, and no measurement. A number that one surface gives and the other
does not is a defect. `test_cli.py` proves it: the `--json` record and the
API `Report.to_dict()` must hold equal scores for one configuration.

---

## 9. The parity gate

This is the acceptance test of the whole project. The package is not done
until this table passes.

| # | Run | Reference | Tolerance |
|---|---|---|---|
| P1 | `evaluator` LP, cora, `otherge`, seed 42 | `bench_other_ge.task_link_prediction` | 1e-12 on all 5 |
| P2 | `evaluator` DA, cora, `otherge`, seed 42 | `bench_other_ge.task_sp_regression` | 1e-12 on rf and mlp |
| P3 | `evaluator` LP, pubmed, `otherge`, seed 42 | same | 1e-12 |
| P4 | `evaluator` DA, pubmed, `otherge`, seed 42 | same | 1e-12 |
| P5 | `evaluator` LP, cora, `fodined`, seed 42 | `reference/fodined_lp.link_prediction` | 1e-12 |
| P6a | `evaluator` DA, cora, `fodiwalk_dist`, seed 42 | `reference/fodiwalk_eval.task_hop(feature="distance")` | 1e-12 |
| P6b | `evaluator` DA, cora, `fodiwalk_vec`, seed 42 | `reference/fodiwalk_eval.task_hop(feature="vector")` | 1e-12 |
| P7 | `evaluator` LP+DA, `com_youtube`, `n2v1m`, seed 42 | the recorded log `node2vec_com_youtube_1M.log` | 5e-3 absolute |
| P8 | peak RSS of P7 | the same log | not above it |

P1 to P6b run on cora and pubmed, thus they cost minutes and they run in
CI. P7 and P8 cost hours and they run one time, by hand, at the end.

**WHY THE TOLERANCE IS 1e-12 AND NOT ZERO. Measured on 2026-08-22.** The
REFERENCE is not bit-stable against itself. Three identical back-to-back
runs of `reference/other_ge.task_sp_regression` on cora gave

    rf_mae = 1.5527613472344024
             1.5527613472344024
             1.552761347234402

a spread of 4.44e-16, about one ULP. The cause is
`RandomForestRegressor(n_jobs=-1)`: `predict` accumulates the per-tree
outputs into a shared array across threads, thus the summation ORDER
varies between runs. It is not a defect of this package and it cannot be
removed without `n_jobs=1`, which would change the recorded protocol.

Two rules follow, and `tests/test_parity.py` must hold both:
1. The tolerance stays 1e-12. That is four orders above the noise and
   still far below any difference a real defect would make.
2. NEVER assert bit-equality (`==`, or `np.array_equal`) on the output of
   any estimator built with `n_jobs=-1`. A test that does so passes on
   most runs and fails at random, which is worse than no test.

Bit-equality IS required, and IS achieved, on everything upstream of an
estimator: the pair arrays, the hop arrays and the feature matrices. P1,
P6a and P6b measured 0.000e+00 on 2026-08-22; P2 measured 4.441e-16, which
is the reference's own noise and not a divergence.

The reference of P1 to P6 is a FROZEN COPY of the source function, kept in
`tests/reference/`. The copy is verbatim and it is never "tidied". There
are two independent reasons for the copies, and both are real:

* `other_ge.py` and `n2v1m.py` -- the rewrite of section 10 DELETES the
  originals, thus the test would have nothing to compare against.
* `fodiwalk_eval.py` -- the original belongs to ANOTHER active session,
  which edited it on 2026-08-22. A reference that a second party can edit
  is not a reference.

All three copies were taken on 2026-08-22 and verified byte-identical to
their sources at that moment. `fodiwalk_eval.py` carries ONE deliberate
difference, written in its docstring: the default of `task_hop(feature=)`
is pinned to `"vector"`, the value it had when the protocol cells were
read. Every parity test passes `feature` explicitly, thus the pin changes
no number; it stops a later reader from thinking the default is the
protocol.

## 10. The rewrites

`experiments/other-ge/bench_other_ge.py` and
`experiments/large-graph-node2vec/bench_node2vec_1M.py` lose their task
code and they call `evaluator`. What each keeps:

* the loaders (they move to nothing; `bench_other_ge` calls
  `fodiwalk.make_graph.datasets`, which already holds the same code)
* the embedding methods (`uniform_walks`, `second_order_walks`,
  `embed_node2vec`, `embed_poincare`, `write_walks`)
* the argument parser of the METHOD knobs
* the printed table

What each deletes: `sample_non_edges`, `task_link_prediction`,
`task_sp_regression`, `poincare_distance`, and the inline scoring of
`bench_node2vec_1M.main`.

Each script must print the same numbers as its recorded log, for the same
seed --- WHERE THE METHOD ALLOWS IT. `results_2026-08-14.log` and
`node2vec_com_youtube_1M.log` are the references.

**MEASURED 2026-08-22: the node2vec rows of those logs were never
reproducible, and this is a property of the BASELINE, not of the
rewrite.** `gensim.models.Word2Vec` with `workers > 1` trains with
multi-threaded SGD whose update order is not deterministic. The default is
`os.cpu_count()`. Three runs with one fixed seed and byte-identical walks
gave three different embeddings:

    run 0   wv['0'][:3] = [ 0.139549 -0.383847 -0.485127]
    run 1   wv['0'][:3] = [ 0.247395 -0.520439 -0.435135]
    run 2   wv['0'][:3] = [-0.022310 -0.261014 -0.337324]

The ORIGINAL script, recovered from git and run unchanged on this machine,
gave a third accuracy that matched neither the log nor the rewrite. Thus a
node2vec row cannot be a parity reference for any refactor.

The rule that follows:

* **Poincare IS a parity reference.** `PoincareModel` trains
  single-threaded, thus it is reproducible. The rewritten script matched
  `results_2026-08-14.log` BIT-EXACT on every Poincare number, and that is
  what proves the `evaluator` wiring, the `metric="poincare"` route and
  the whole task path correct.
* **node2vec is a SANITY reference only.** Compare the deterministic parts
  --- the node and edge counts, `n_pairs`, the hop range, and the baseline
  MAE, which depend on the graph and the sampler and not on the embedding.
  Those matched exactly. The scores that depend on `Z` are compared for
  plausibility, never for equality.
* A future run that wants a reproducible node2vec must pass `--workers 1`,
  and it then produces a NEW reference log; it does not reproduce the old
  one.

## 11. Success criteria and measures

**Functional.**
| F1 | The parity table of section 9 passes P1 to P6. |
| F2 | `python -m evaluator cora Z.npy -lp -da` writes a valid record. |
| F3 | The API of section 5 works and `test_api.py` passes. |
| F4 | A new task needs one file and one registry line. Proven by a toy task in `test_api.py`. |
| F5 | Both rewritten scripts run. They reproduce their recorded numbers exactly for the reproducible method (Poincare) and on every deterministic quantity (counts, `n_pairs`, hop range, baseline MAE); the node2vec scores are compared for plausibility, because `Word2Vec(workers>1)` is not reproducible. See section 10. |
| F6 | `python -m evaluator ... --json` writes one JSON object to stdout and nothing else. `json.loads` of the captured stdout succeeds. |
| F7 | The `--json` record and the API `Report.to_dict()` hold equal scores for one configuration. |

**Quality.**
| Q1 | `import evaluator` under 1.0 s, and no heavy import at the top. |
| Q2 | Every public function has a docstring that says what it reads, what it returns, and why the method is the one it is. ASD-STE100. |
| Q3 | No cycle in the import graph. Proven by `tests/test_structure.py`. |
| Q4 | The lines of task code deleted from `experiments/` is at least 350. |

**Scale.**
| S1 | P7 finishes and its numbers hold. |
| S2 | P8: the peak RSS does not grow. |
| S3 | No block builds an `(n, n)` array. Proven by a read of `hops.py` and by S2. |

**Guards.**
| R1 | The star graph raises C7. |
| R2 | A Poincare `Z` with a norm of 1.0 raises C4. |
| R3 | An all-zero `Z` raises C3. |

## 12. Risks

**The RNG order.** Parity fails if one draw moves. The risk is high and the
answer is B3's written order plus P1 to P6.

**A protocol that is wrong on paper.** The table of B1 comes from a read of
four files. A wrong cell makes P1 fail, which is the point of P1.

**Scope creep.** More tasks (node classification, clustering,
visualization) are easy to add and they are NOT in this project. The
registry is the promise that they cost one file later.

**The `f1` rename.** `fodined` writes `f1-score`. A downstream reader that
looks for that key breaks. `report.py` records the change and the CATALOG
holds the entry.

---

## 13. The agent surface -- deferred, and why it stays cheap

The end state that the user named is a module that an AGENT calls as a
tool. That is NOT a block of this project. The reason is a measurement:
the `mcp` package is absent from `.venv/`, and this project adds no
dependency.

Nothing here blocks it, and this section records what it will cost.

**What a tool call needs, and where this design already gives it.**

| Need | Where it already exists |
|---|---|
| One entry point that takes flat arguments | `evaluate(graph, Z, tasks=..., protocol=..., seed=...)`, section 5.2 |
| A closed set of legal values | `PROTOCOLS`, `TASKS`, `DISTANCES`, `FEATURES` -- each a dict whose keys ARE the enum |
| A machine-readable answer | `Report.to_dict()`, section B8, and `--json`, section 6.4 |
| An error that a caller can read | `DegenerateEvaluation` carries an id (C1..C9), a measured value and a threshold, section B7 |
| No arbitrary code in the input | The graph is a name or a path; the embedding is a path. No expression is evaluated |

**What it will cost later.** One file, `tool.py`, of about 120 lines:

1. `TOOL_SCHEMA` -- a JSON Schema built FROM the dataclass fields of
   `config.py` and the registry keys. Generated, not written by hand, thus
   it cannot drift from the code.
2. `call(payload) -> dict` -- validate, call `evaluate`, catch every
   exception and return it as a field. This is the ONE place that catches;
   invariant I5 stands everywhere else.
3. A small answer: the scores and a one-line summary, with the sizes and
   the histogram left in the file that `save` names.

No block of section 7 changes. That is the point of I7: a surface holds no
measurement, thus a third surface is additive.

**The one rule to hold now.** Keep every legal value in a dict whose keys
are the enum, and keep `evaluate()` free of positional-only arguments.
A schema generator can then read the code. A hand-written `if name ==
"otherge"` chain cannot be read, and it would make the later file three
times as large.
