# evaluator — agent memory

Live state of the `evaluator/` package. Keep this current. Update it when
you verify something, fix something, find a defect, or get a report from
another session. Date every entry.

Last updated: 2026-08-26.

---

## Status

Built and working. 14 modules, two tasks, six protocols. Used by
`experiments/other-ge/bench_other_ge.py` and
`experiments/large-graph-node2vec/bench_node2vec_1M.py`, both rewritten to
call it instead of holding their own copies.

**No committed test suite.** Every parity result below came from a one-off
script. `evaluator/tests/` holds `reference/` and nothing else. This is the
largest open item.

**Nothing is in git.** `git ls-files evaluator/` returns 0. The whole
package exists only in the working tree. A backup of the full tree sits
outside the repo at `fdmap-backup-20260826T081231Z`, made and tested by
fdmap-b3. The user has not authorised a commit.

## Verified (2026-08-26, all on cora, random Z, seed 42)

| protocol | against | result |
|---|---|---|
| `otherge` LP | frozen `reference/other_ge.py` | 0.000e+00 |
| `fodined` LP | live `fodined/link_prediction.py` | 0.000e+00, all 5 scores, all 6 counts |
| `fodiwalk_dist` DA | frozen `reference/fodiwalk_eval.py` | 0.000e+00, all 15 values |
| `fodiwalk_vec` DA | frozen `reference/fodiwalk_eval.py` | 0.000e+00, all 15 values |

`pairs._far_pairs` vs `fodined.graph_augmentation.sample_far_pairs`:
identical pairs **and** identical generator state at 500 / 5278 / 20000.

Also holding: import 0.205 s with no sklearn, networkit or gensim loaded;
one shared graph load across tasks; `to_dict()` json-clean; `write()`
picks `.jsonl` append or `.json` by suffix; the unknown-keyword error
names the bad key and lists the legal ones.

Not verified: the `n2v1m` and `fodined` DA rows. `fodined` DA reproduces no
published number **on purpose** — `modular.py` reads a stored augmentation
weight, this package reads the graph's true distance. Different
measurements. Never compare across that row.

## Open defects, none fixed

- **B1 — worst one.** `Report.protocol_modified` is always False for any
  task written from the documented template. `evaluate()` reads
  `getattr(res, "protocol_modified", False)`; `TaskResult` declares no such
  field. The two shipped tasks work only because each patches the attribute
  on after construction — a line in no docstring. So invariant I2 fails
  silently and the record still looks comparable. **Fix chosen:** promote
  the flag to a real `TaskResult` field. Also fixes `cli.py:323` and
  removes the undocumented patch line. Found twice, independently.
- **H2** — guard warnings under `--no-strict` reach the saved record but
  never the printed table. `report.py:176` derives its mark from
  `protocol_modified` alone. Any legend added must cover both signals or it
  will imply the table shows warnings.
- **H3** — `PROTOCOLS` is a plain mutable dict. Wants `MappingProxyType`.
- **H5** — the Poincaré guard is one-way. It catches euclidean vectors
  declared `poincare`, and never catches ball vectors scored with the
  default `euclidean` — which is the default path for all six protocols.
  Measured on cora: rf r2 -0.027 vs -0.034, both plausible, neither
  flagged.
- **H1** — a high LP AUC on a degree-skewed graph can come from degree
  alone. Nothing flags it. Needs a design decision, not a patch.
- **F2** — five `DACfg` fields unreachable by keyword: `pair_draw`,
  `rf_estimators`, `rf_min_leaf`, `mlp_hidden`, `mlp_early_stop`. `LPCfg`
  exposes all seven of its own.
- **Protocol uninspectable from the CLI.** `--help` names six protocols and
  never says what any sets. No `--dry-run`, no `--list-protocols`. A user
  cannot confirm their protocol applied. This defeats the package's central
  claim from the command line. Fix: `--dry-run` printing the resolved
  settings — the data already exists as `tasks.*.cfg`.
- **`PROTOCOLS` is a positional 2-tuple** `(LPCfg, DACfg)`, indexed `[0]` /
  `[1]`. A third task cannot own a protocol config without editing all six
  entries and both index sites. No answer yet.
- **Not installed.** Repo root or `PYTHONPATH` only.
- Smaller: unknown graph name does not list valid ones and leaks
  `ValueError:` into the message; `--metric` filed under global options but
  task-scoped; `--da-models` takes a free string with no argparse
  validation; no `--version`; registered tasks not exported to the
  namespace; JSON layout undiscoverable from `--help` (the path is
  `tasks.<task>.scores.<metric>`, easy to guess wrong).

Source: `evaluator/dev-docs/cold-start-review.md`, written by fdmap-f4 —
an outside usability test, the best document in that directory. Plus a CLI
friction report from fdmap-b3.

## Facts other sessions need, and get wrong

**The `fodined` number.** A `fodined` link-prediction number that came out
of `evaluator` **before 2026-08-24** is wrong: accuracy 0.9777, hop R2
0.630. A number out of `fodined/link_prediction.py` is correct and always
was: 0.9744, 0.616. The module was never defective; this package's copy of
its settings was. Confirmed four ways — by the session that fixed it
(fdmap-69), by an independent full run (fdmap-c5), by the original
campaign log `results/g1_cora_walk_min_gap_plain_s42.log` (fdmap-e6), and
by direct comparison here.

Phrase it that way. "0.9777 is the defect" alone reads as an accusation
against `fodined/link_prediction.py` and caused a false alarm over 141
campaign runs.

**Which protocol is which.** A hop task with 200 BFS sources, 20,000 pairs,
`hop > 1`, one scalar distance feature and three models is
`fodiwalk_dist`, not `fodined`. `experiments/fdwalk/bench_fdwalk.py` runs
`fodined` LP and `fodiwalk_dist` DA — two different rows in one script.

## Ownership

The user made this session sole editor of `evaluator/` on 2026-08-26. All
six peer sessions were told and accepted. Others may read, run, and report;
only this agent edits. Entry in `AGENTS.md`.

fdmap-b3 owns `forcedirected/` under the same rule. `fodined/link_prediction.py`
belongs to fdmap-e6 — they will message before changing its sampling,
split, classifier or pair cap, because a silent drift there breaks this
package's `fodined` row with nothing on screen to warn anyone.

## Known-bad script patterns (each cost a run)

- `hop_sample` in the frozen reference returns **four** values
  `(u, v, d, h2)`, not three.
- The frozen reference names its models `"mean baseline"`, `"random
  forest"`, `"MLP"`. This package uses `baseline`, `rf`, `mlp`.
- `fodined.link_prediction.link_prediction` returns `(scores, info)`, and
  spells the harmonic mean `f1-score`. This package spells it `f1`.
