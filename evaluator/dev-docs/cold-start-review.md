# Cold-start API review — `evaluator` as a Python import

Date: 2026-08-23. Reviewer: a session with no part in the design, at the
request of the session that wrote the package. Method: `import evaluator`,
`help()`, and the code's own docstrings. `dev-docs/` was not read during the
test, and the CLI was not used. Graph: `cora` (2708 nodes).

This file is a RECORD. A cross-session message is addressed by a name that
the transport may reassign before it lands, and neither sender nor
recipient is told. The findings below reached no confirmed reader, thus
they are written here.

## 1. Defect — `Report.protocol_modified` is always False for a new task

`evaluate()` reads the flag off an attribute that `TaskResult` does not
declare:

```python
# evaluator/__init__.py:100
modified = modified or bool(getattr(res, "protocol_modified", False))
```

`TaskResult` (`report.py`) holds `task`, `cfg`, `scores`, `sizes`,
`seconds`, `warnings`. The flag lives at `res.cfg["protocol_modified"]`,
which `cfg_record` fills. The two shipped tasks work only because each one
PATCHES the object after it is built:

```python
# evaluator/tasks/link_prediction.py:94, and dist_approx.py:90
result.protocol_modified = bool(modified)
```

That line appears in no docstring: not in the task template at
`tasks/__init__.py:1-27`, not in `cfg_record`, not in `TaskResult`.

Measured. A third task written from the documented template, given an
explicit override:

```
task cfg["protocol_modified"] : True
report.protocol_modified      : False
table()                       : no leading `*`
```

PRD invariant I2 — a changed number must not claim the name of a recorded
baseline — therefore fails in silence for every task written from the
docs. The record still reads `protocol=default` and looks comparable.

Fix: read `res.cfg.get("protocol_modified")` in `evaluate()`, or promote
the flag to a real `TaskResult` field.

## 2. The Poincaré metric guards one direction only

`guards.check_metric` (line 93) returns at once unless
`metric == "poincare"`, then it demands `max|z| < 1`.

| embedding | metric | outcome |
| --- | --- | --- |
| Euclidean vectors | `poincare` | raises C4. Correct. |
| vectors in the unit ball | `euclidean` | no raise, no warning, a plausible WRONG number |

The second row is the DEFAULT path: all six protocols carry
`metric="euclidean"`. Measured on cora with a ball embedding (radius from
degree, max|z| = 0.95), `rf` scores:

```
poincare   mae=1.5308  r2=-0.027
euclidean  mae=1.5487  r2=-0.034
```

Both plausible, neither flagged. The right call is discoverable only
because the top-level docstring happens to show `metric="poincare"` in its
example. `metrics.py` exports no `METRICS` name, so the legal values
(`euclidean`, `poincare`, `cosine`, `dot`) are found by reading
`metrics.py:89`.

`link_prediction` cannot express a metric at all — `feature` is
`hadamard | l1 | concat | avg` — so a hyperbolic embedding cannot be
scored correctly on link prediction through this API. That gap is
unstated.

Suggestion: a reverse check. Every row inside the unit ball plus
`metric != "poincare"` is worth a warning.

## 3. "One file plus one registry line" — true for the mechanics

A toy third task (node-degree regression, 70 lines at
`evaluator/tasks/degree.py`, plus one line in `tasks/__init__.py`) was
picked up by `evaluate()`, the keyword routing, the unknown-keyword error,
`table()`, `to_dict()` and `write()` with nothing else touched. About 25
minutes of work, of which 20 s is the run. The file and its registry line
were removed afterwards; `ev.TASKS` holds the two original names.

Two qualifications:

* The file must also hold the undocumented line of finding 1. The real
  cost is one file, one registry line, and one line the author cannot know
  about.
* `config.PROTOCOLS` is a 2-TUPLE `(LPCfg, DACfg)` and `config.get(name)[0]`
  / `[1]` index it positionally. A third task cannot own a protocol config
  without editing all six protocol entries and both index sites. The
  workaround — a private cfg dataclass in the task's own file — works, but
  then `protocol="otherge"` reaches the record as a NAME whose settings the
  task ignores, and nothing warns. Either the template should say a task is
  protocol-free, or the registry needs one dict per task.

## 4. Smaller friction

* The package is not installed. `import evaluator` works only with the
  working directory at the repository root; a script elsewhere fails with
  `ModuleNotFoundError`. No `pip install -e .`, and no path note in the
  docstring.
* `dist_approx` cannot reach 5 of the 12 `DACfg` fields: `pair_draw`,
  `rf_estimators`, `rf_min_leaf`, `mlp_hidden`, `mlp_early_stop` have no
  keyword. `link_prediction` exposes all 7 of its own. A non-default
  classifier size is thus possible for LP (`n_estimators`) and impossible
  for the DA forest.
* A registered task is not exported. `ev.link_prediction` and
  `ev.dist_approx` are module-level names; a registered third task is
  reachable only through `ev.TASKS` or `evaluate(tasks=...)`.
* `help(ev)` buries the function list. `DegenerateEvaluation` brings the
  full `BaseException` boilerplate, about 60 lines, between the module
  docstring and `FUNCTIONS`.
* `dist_approx` returns `{model: {metric: value}}` and `link_prediction`
  returns `{metric: value}`. Documented, but a caller must special-case it.
  The `baseline` model is also insensitive to `metric` — it predicts the
  mean — which costs a confused run before one compares `rf` instead.

## 5. What held

* Import cost 0.30 s, with `sklearn`, `networkit` and `gensim` all absent
  from `sys.modules` afterwards. PRD B11.1 and invariant I6 hold.
* `evaluate()` reads the graph and the embedding one time; both tasks run
  off the loaded objects.
* The unknown-keyword error names the bad key AND lists the legal ones.
* `to_dict()` is `json.dumps`-clean; `write()` picks `.jsonl` append or
  `.json` from the suffix. Both verified.
* The provenance comments in `config.py` are the best documentation in the
  package. They make `protocol="otherge"` legible: both task configs take
  what `bench_other_ge.py` actually ran, so the number is comparable to
  that recorded row and to no other.

Nothing outright failed. Random-embedding sanity held throughout: LP
auc 0.529, DA r2 about 0.
