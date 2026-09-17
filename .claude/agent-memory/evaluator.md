# evaluator — agent memory

Live state of the `evaluator/` package. Keep this current. Update it when
you verify something, fix something, find a defect, or get a report from
another session. Date every entry.

Last updated: 2026-09-16.

**Identify a session by its ROLE. Not its name, and not its UUID.**

This session's role is `evaluator`, set with /rename on 2026-09-02. Two
weaker ids exist and both fail, in different ways:

- **The transport name churns.** This session went `fdmap-68 [54807a]` ->
  `fdmap-65 [20555e]` inside one hour, after six ownership notices had gone
  out under the old name. The peer list went 7 -> 4 in the same window with
  no mapping between them.
- **The transcript UUID is stable but NOT UNIQUE.** A forked session
  appends to its PARENT's transcript and carries the parent's sessionId, so
  two live sessions share one UUID. The `fodiwalk` session proved this on
  2026-09-02: it wrote a string no other session had written, searched
  every transcript under `~/.claude/projects/`, and found it in its
  parent's file. Verified here that `fd084f99-...jsonl` holds exactly one
  distinct sessionId, which is what that inheritance produces.

This session's UUID is `6f9d36c7-4699-47c9-add7-b451b41e4d84`. Record it as
a hint, never as a key.

Re-run `ListAgents` before addressing anyone. A name read more than a few
minutes ago is not trustworthy.

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

## Cross-method hop R2, for the degree question (2026-09-02)

The `fodiwalk` session found hop R2 splits by average degree near 3, not by
graph size, and called it "structural". It has since corrected that: the
evidence proves under-training is not the cause, not that the graph cannot
be embedded.

Numbers on the record, useful whenever someone reads a flat hop R2 as proof
about the graph:

| graph | avg degree | node2vec | fodiwalk | source |
|---|---|---|---|---|
| com_youtube d64 | 5.27 | 0.042 | 0.4084 | `experiments/fodiwalk-streaming/FINDINGS.md` s6 |
| com_youtube d128 | 5.27 | 0.043 | 0.428 | same |
| roadnet_ca | 2.82 | — | -0.0008 / -0.0043 | `experiments/fodiwalk-streaming/REPORT.md` |
| cora | 3.898 | — | 0.43-0.63 | measured here via `load_graph` |
| pubmed | 4.496 | — | 0.43-0.51 | measured here via `load_graph` |

**A flat hop R2 does not identify the graph as the cause.** node2vec is
near-flat at degree 5.27, where fodiwalk reaches 0.41. Ten times the epochs
moved roadnet_ca only -0.0008 -> -0.0043, hop MAE 117.4 -> 117.6.

Test designed here, NOT RUN, waiting on the fodiwalk user: score two
structurally different embeddings of one low-degree graph under one
protocol. Both flat means the graph; one not flat means the method.

    .venv/bin/python -m evaluator roadnet_ca <Z>.npy -da \
        --protocol fodiwalk_dist --max-nodes 200000

Split agreed: this session generates the spectral embedding and reports the
resolved `cfg` beside the scores; `fodiwalk` supplies its own `Z` and the
run parameters. Guard C7 may raise on a truncated road network — that is
correct behaviour, not a failed run.

## Other sessions, and what they hold

Identify a session by its transcript UUID. Names churn, badly: one session
reached this one twice in minutes as `fdmap-20` and `fdmap-43`, same UUID,
two sockets. A long-running session can outlive its own transport identity
mid-task, so its own earlier messages come back under a name it does not
recognise.

- `fodiwalk (parent)` — owns `fodiwalk/` plus `experiments/fodiwalk/` and
  `experiments/fodiwalk-streaming/`, shared with `dense-kernel` since the
  fork. Its work is finished and pushed at commit `5442889`. Uses
  `evaluator`, edits nothing in it. **Its dim-128 campaign lost 6 of 21 runs
  to GPU OOM** on `roadnet_ca` and `ncbi_taxonomy` (Z alone 960 MB and
  1434 MB against a 2048 MiB card); blank cells in its `REPORT.md` are
  missing, not zero. The `max_pairs=50_000` trap below is also its finding.
- `dense-kernel` — forked from the parent at 2026-09-02T02:00Z. Replacing
  SELL-C-sigma in `forcedirected/` with a flat batched kernel. Holds the
  low-degree hop R2 test, waiting on machine time; one job at a time on
  that box.
  Both carry UUID `fd084f99-dab1-4ce5-a2b5-7c74e42c28d8`. It identifies
  neither. `AGENTS.md` splits them at the fork timestamp, keyed on role.
- `fdmap-b3` (hex name, will churn) — owns `forcedirected/`.

**n2v1m trap.** The recorded node2vec baseline on the 1M graph used
`max_pairs=50_000`, an explicit override of the `n2v1m` default of 80,000.
A run at the protocol default is NOT comparable to it. `protocol_modified`
caught this in the field — the first outside confirmation that the flag
does its job.

## For when the test suite gets written (2026-09-02)

`forcedirected/PARITY.md` section 10 records a real gate run to size against:
**82 passed, 2 skipped, 3 deselected, exit 0, in 1869.86 s.** Thirty-one
minutes. Budget for that; a suite here will not be a few seconds.

Same section records the incident behind the `| tail` rule. On 2026-08-25 an
unfiltered run was piped to `tail -15`; bash returned tail's exit code so an
OOM kill showed as 0, and `pytest -q` prints no summary when killed, so 45
surviving progress dots were read as "45 tests passed". Every part of that
report was wrong. **Keep the incident attached to the rule** — a bare rule
does not survive a tired reader.

## Session age means nothing

A resume or a fork inherits its parent's transcript, so a session that
started an hour ago can be carrying days of work. Do not reason "this is the
only session old enough to be X". `dense-kernel` made that mistake, caught
it, and it is the same class as the UUID collision: the transport tells you
about the connection, never about the work.

Sockets die. `fdmap-b3` reached this session from
`uds:/run/user/1001/cc-socks/1204144.sock`; that socket no longer exists, so
the session that owned `forcedirected/` under that name is gone or
reconnected. Live sockets are enumerable at `/run/user/1001/cc-socks/`.

## Loader equivalence, settled 2026-09-02

`evaluator.io.load_graph` and `fodiwalk.make_graph.load` give the SAME node
numbering for a REGISTRY NAME. Verified on cora (n=2708) and pubmed
(n=19717): degree sequence in node order identical, `indptr` and `indices`
identical after `_finish`.

Structural, not luck. `io.py:104` calls `datasets.load(name, max_nodes,
seed)`, and `fodiwalk.make_graph.load is datasets.load` -> True. One
function, one numbering. `_finish` symmetrizes, zeroes the diagonal, sets
values to 1.0 and sorts; none of that renumbers.

Two limits, both real:
- `max_nodes` and `seed` must match. A different truncation is a DIFFERENT
  GRAPH, not a renumbering, and nothing warns.
- A path or an edge array goes through `_edges_to_csr`, which DOES renumber
  with `np.unique`. A stored `Z` from a registry load must never be scored
  against the same data read from an edge-list file.

Test to reuse: compare the degree sequence IN NODE ORDER. A renumbering
survives an nnz or shape comparison; it does not survive that one.

Score-name difference to expect: this package spells the harmonic mean
`f1`. `fodined/link_prediction.py:89` and `fodiwalk/misc/evaluation.py:113`
spell it `f1-score`. Anyone diffing a new record against a pre-`evaluator`
log reads the missing key as a missing measurement.

## Scoring the embeddings store (2026-09-04)

Scorer lives at `data_cache/evaluator/store-scores/eval_store.py`. Reports go
to `evaluator/reports/`. Raw rows to
`data_cache/evaluator/store-scores/<date>-all-scores.jsonl`.

**Large graphs need three adjustments, and without them the run FAILS, not
merely slows.** Applied 2026-09-04 for a set spanning 1.13M to 2.94M nodes:

- **Bound the hop sources above 200,000 nodes.** An unrestricted sample gives
  `hops.choose_backend` ~20,000 distinct sources; its rule only routes <=512
  sources to blocked BFS, so above that it asks for a PLL index over the
  whole graph, which does not fit. Draw 200 sources instead.
- **Hold one graph at a time.** Keeping a fixture per dataset means four
  multi-million-node adjacency matrices live at once.
- **kNN goes APPROXIMATE above `RECON_MAX_N` (200,000).** `metrics.knn`
  switches to pynndescent. Do not force the exact path; do not print the
  approximate value under the same column heading as an exact one. Agreed
  presentation with the `runner` session: a separate column or a value
  suffix, never a footnote, because a footnote is what a reader skips.

Both the source bound and the approximate kNN **change what the number
means**, not just the cost, so each row records `hop_sources`, `knn.path` and
`knn.queries`.

## The 0ba3b1b3 record, settled 2026-09-04

One record: `cora/128/260904-071825-fodiwalk_precomp-0ba3b1b3`, `k4=None`,
`hash_spec 2`, `git_dirty: true`, accuracy 0.9882. Its law is recorded ONLY
in the `notes` field: `Fr = -kr*h*exp(-x)`.

A twin at `260904-062153` once shared the fingerprint and ran a different law
(no decay, accuracy 0.9574). It is gone and **its removal is unaccounted
for** — `runner` did not knowingly delete it and has no log entry. The
2026-09-04 scorecard is now the best surviving record of it.

Lesson: `runner` asserted two records existed; counting showed one. Count,
do not trust the count in a message.

## node2vec q-sweep, confirmed twice (2026-09-04)

Geometry is monotone in `q`, favouring the DFS-like walk (`q < 1`). AUC does
not move at all across the range.

| q | cora rho | pubmed rho | cora hop R2 | pubmed hop R2 |
|---|---|---|---|---|
| 0.5 | 0.742 | 0.416 | 0.5301 | 0.2361 |
| 1.0 | 0.711 | 0.362 | 0.4674 | 0.2080 |
| 2.0 | 0.638 | 0.275 | 0.4056 | 0.1670 |

rho measured here; hop R2 measured independently by `runner`. Two metrics,
same ordering, both flat on wordnet (rho 0.011-0.019).

**Why it matters:** a tuned node2vec at its best `q` still loses to `fdhop` —
0.5301 against 0.7119 on cora, 0.2361 against 0.6194 on pubmed. That answers
the fair objection that the earlier comparison left node2vec at its defaults.

## Measurement variance on cora (cell A, 2026-09-16)

One fixed Z (cora, n_dim 64, lr 0.999, nbr_walk/fdlinear, 200 ep), scored
under eval seeds 42,56,88,101,7 with protocol `fodiwalk_dist`. Record:
`agentic-log/21.evaluator-cellA/`.

| metric | eval-only SD | p1 seed SD (Runner) |
|---|---|---|
| acc | 5.05e-03 | 2.92e-03 |
| auc | 1.34e-03 | 1.12e-03 |
| r2_dist | 2.90e-02 | 2.66e-02 |
| mae_dist | 5.38e-02 | 5.19e-02 |

**The evaluator alone produces the whole seed spread on these metrics.**
Any pin tighter than this is a trajectory pin, not a quality pin.

Why, measured:
- `max_pairs=50000` never binds on cora. 5,278 edges -> 10,556 LP pairs,
  test split 2,112. Binomial SD of acc at 0.975 on 2,112 is 3.4e-03.
- r2_dist pairs (~18,400) come from only 200 BFS sources; the source draw
  (`dist_approx.py:101`) dominates.

Split (same Z): sample arm (rng varies) vs classifier arm (seed varies).
acc 2.56e-03 vs 3.85e-03; auc 7.16e-04 vs 8.10e-04; r2_dist 3.56e-02 vs
1.81e-02. LP spread comes from both; hop spread mostly from the 200-source
sample. `seed` is not pure classifier: it also picks the test split. Lever
for hop variance is `n_sources`; for LP on cora, averaging eval seeds (all
edges are already used).

For method comparison fix BOTH `seed` and `rng` to an eval constant, or
average over several eval seeds. `rng` drives the sample; `seed` drives the
split, the forest and the MLP.

## Known-bad script patterns (each cost a run)

- `hop_sample` in the frozen reference returns **four** values
  `(u, v, d, h2)`, not three.
- The frozen reference names its models `"mean baseline"`, `"random
  forest"`, `"MLP"`. This package uses `baseline`, `rf`, `mlp`.
- `fodined.link_prediction.link_prediction` returns `(scores, info)`, and
  spells the harmonic mean `f1-score`. This package spells it `f1`.
