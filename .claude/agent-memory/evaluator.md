# evaluator — agent memory

Live state of the `evaluator/` package. Keep this current. Update it when
you verify something, fix something, find a defect, or get a report from
another session. Date every entry.

Last updated: 2026-09-22.

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

## Fresh clone, environment rebuilt (2026-09-21)

This working tree was freshly cloned with no `.venv` and no `data_cache/`.
Rebuilt both, and ran the first `nbr_walk`+`fdhop` numbers on the three
citation graphs.

- `.venv/` recreated at repo root, Python 3.12. Installed
  `evaluator/requirements-evaluator.txt` plus `pandas`, `gensim`, `jax`,
  `jaxlib`, `pytest` (no root `requirements.txt` exists for `fodiwalk` or
  `forcedirected`; versions were inferred from `import` statements). A GPU
  is present (GTX 1060) but no CUDA jaxlib was installed — `jax.devices()`
  falls back to CPU. Fine at this graph scale (seconds, not minutes).
- Downloaded Cora and Pubmed from `linqs-data.soe.ucsc.edu` into
  `data_cache/cora/` and `data_cache/pubmed/Pubmed-Diabetes/`, matching
  the paths `fodiwalk/make_graph/datasets.py` already expected.
- **Citeseer was NOT in the registry before this.** Added
  `_edges_citeseer()` to `fodiwalk/make_graph/datasets.py` (not an
  `evaluator/` file — touched it anyway since this was a fresh single-agent
  session with no other live editor, and it was minimal + additive:
  one function in the exact shape of `_edges_cora`, plus one line in
  `read_edges`). Citeseer's ids are alphanumeric
  (e.g. `bradshaw97introduction`), unlike Cora's plain integers, so the
  loader reads `dtype="<U32"` and lets `to_csr`'s `np.unique` renumber it
  — verified this works with no other change needed. Downloaded from the
  same LINQS host into `data_cache/citeseer/`. **A session that owns
  `fodiwalk/` should be told about this addition and fold it into their
  own record.**
- Loaded all three through `fodiwalk.make_graph.load` and got the
  standard published sizes: cora 2,708n/5,278e (avg deg 3.90), citeseer
  3,327n/4,552e (avg deg 2.74), pubmed 19,717n/44,324e (avg deg 4.50).

**Results**, `experiments/embeddings/make_embedding.py --method fodiwalk
--pairs nbr_walk --force fdhop --k4 1.0 --optim plain --weight min_gap
--lr 0.999 --dim 128 --epochs 200 --seed 42 --device cpu --protocol
fodiwalk_dist --score`. `k4=1.0` because the Config default (0.01) is
"100x flatter" than what `fdhop` wants — the convention already on record
in `make_embedding.py`'s own `--k4` help text and in
`experiments/embeddings/RESULTS.md` (which has `fdhop` rows only under
`walk_edges`, never `nbr_walk` — this is the first record of that
combination).

| graph | acc | f1 | auc | hop R2 (mlp, distance) | embed s | peak RSS |
|---|---|---|---|---|---|---|
| cora | 0.9934 | 0.9934 | 0.9996 | 0.3554 | 14.9 | 507 MB |
| citeseer | 0.9956 | 0.9956 | 1.0000 | 0.1553 | 11.7 | 548 MB |
| pubmed | 0.9920 | 0.9920 | 0.9993 | 0.3726 | 114.6 | 884 MB |

Stored under `data_cache/embeddings/<graph>/128/<name>/{Z.npy,config.json}`:
`cora/128/260921-060251-fodiwalk_precomp-0611c0e9`,
`citeseer/128/260921-060317-fodiwalk_precomp-ab4767d8`,
`pubmed/128/260921-060615-fodiwalk_precomp-c9933669`. Log:
`experiments/embeddings/logs/nbr_walk_fdhop_20260921T060216Z.log`.

Link prediction is near-ceiling on all three, as it is for every method on
these graphs (see the `RESULTS.md` table for `walk_edges`/`fdlinear`
rows). Hop R2 under `nbr_walk` (0.355 cora, 0.373 pubmed) reads lower than
the recorded `walk_edges`+`fdhop` numbers for the same graphs (+0.7119
cora, +0.6194 pubmed) — consistent with the existing note that `nbr_walk`
+ `fdlinear` also scored lower than `walk_edges` + `fdlinear` on cora
(+0.2537 vs +0.7073 in `RESULTS.md`). Read as "the pairs policy matters
more than which force law," not yet confirmed as a rule since this is one
seed, no repeats.

## walk_edges + fdhop, same three graphs (2026-09-21)

Same day, follow-up run: `--pairs walk_edges` in place of `--pairs
nbr_walk`, everything else identical (dim 128, epochs 200, k4=1.0, plain,
min_gap, lr 0.999, seed 42, 10x20 walks, window 5, `fodiwalk_dist`).

**Citeseer failed first, for a real reason, not a script bug.** `PlaneContractError`
(I5, `fodiwalk/embed/plan_contract.py:189`): 48 rows held stored pairs at a
true degree of 0. Citeseer genuinely has 48 isolated (degree-0) nodes after
`_finish`/`to_csr` — verified directly (`np.diff(A.indptr) == 0`). The
default `deg_source="auto"` counts `h==1` entries of `D` for the divisor,
and the `far`-pairs sampler had picked some of those isolated nodes for a
long-range pair anyway, so `D` held entries for a row that `A` says has no
neighbours. Every force law would have silently frozen those 48 rows for
the whole run — the guard exists exactly to catch that (I5) and it did.

**Fix, not a workaround: `deg_source="A"`**, already a `Fodiwalk`/`Config`
field (`fodiwalk/config.py:75`, `fodiwalk/embed/degrees.py`) but not wired
through `experiments/embeddings/make_embedding.py`. Added `--deg-source
{auto,D,A}` there (default `auto`, so cora/pubmed runs are unaffected) and
reran citeseer with `--deg-source A`. Confirmed with a 5-epoch smoke test
before the full run: `fw.diverged` False.

Cora and pubmed did NOT need this — their `auto` degree source never hit
a degree-0 row with a far pair. Whether they also have isolated nodes that
simply weren't picked by the far sampler this seed is unchecked.

| graph | acc | f1 | auc | hop R2 (mlp, distance) | embed s | note |
|---|---|---|---|---|---|---|
| cora | 0.9915 | 0.9915 | 0.9987 | 0.7080 | 11.2 | deg_source=auto |
| citeseer | 0.9951 | 0.9951 | 0.9998 | 0.6076 | 11.0 | deg_source=A (required) |
| pubmed | 0.9903 | 0.9903 | 0.9987 | 0.6315 | 57.7 | deg_source=auto |

Stored: `cora/128/260921-061803-fodiwalk_precomp-d3731d64`,
`citeseer/128/260921-062242-fodiwalk_precomp-76416b77`,
`pubmed/128/260921-062036-fodiwalk_precomp-59d929c3`. Logs:
`experiments/embeddings/logs/walk_edges_fdhop_20260921T061732Z.log` (cora,
pubmed, and the citeseer failure) and
`experiments/embeddings/logs/walk_edges_fdhop_citeseer_20260921T062217Z.log`
(the citeseer rerun).

Cora's number here (0.9915 acc, hop R2 0.7080) is close to but not
identical to the recorded `RESULTS.md` row for the same nominal settings
(0.9882, +0.7119) — within the measurement-variance band already on
record above (`acc` SD ~5e-3, `r2_dist` SD ~2.9e-2 across eval seeds on
cora), not a discrepancy to chase.

**`walk_edges` clearly beats `nbr_walk` for `fdhop`'s hop R2 on all three
graphs at these settings**: cora 0.708 vs 0.355, citeseer 0.608 vs 0.155,
pubmed 0.632 vs 0.373. Same direction on every graph, one seed each — the
next thing to check before calling it a rule is a second seed.

## Optimizer arms: sqn/const vs nesterov/linear, 3 seeds (2026-09-21)

`fdhop`, `walk_edges`, `min_gap`, 10x20 walks, dim 128, 200 epochs, `k4=1.0`,
on cora/citeseer/pubmed, embedding seeds 42/43/44 (== eval seed each run,
same convention as every run above). Not compared against the ICLR report
on request — this is its own record, not a reproduction check.

**`fodiwalk`'s own `Fodiwalk` class already had an `lr_decay` kwarg
(`"const"`/`"linear"`, `fodiwalk/model.py:53`) that `make_embedding.py`
never exposed.** Added `--lr-decay {const,linear}` there (default
`const`, so every prior run in this file is unaffected). Also fixed a
latent bug the same edit would otherwise have hidden: the script's own
stored `cfg["optimizer"]["lr_decay"]` was HARDCODED to `"const"`
regardless of what ran — a `linear` run would have recorded itself as
`const` in its own `config.json`. Now reads `args.lr_decay`.

`lr=0.099` for `nesterov` is not arbitrary: nesterov's steady-state gain
is `1/(1-beta)` = 10 at the default `beta=0.9` (the effective-lr law on
record above), so `0.099 x 10 ~= 0.99`, just under the stability edge and
inside the user's global "never 1.0" rule.

**Neither `Spearman rho` nor `recall@10` is an `evaluator` scorer.**
`METROLOGY.md` s6.3 and s6.6 both still mark them PLANNED. Computed both
standalone, script at
`agentic-log/recall-and-rho/{extra_metrics.py,build_table.py}` (copy the
scratch files there if this needs to be reusable; they ran from
`/tmp/.../scratchpad/` this session and are not yet saved in-repo):

- **rho**: Spearman of embedding distance against hop distance, hop >= 2,
  sampled with `evaluator.pairs.pairs_for_hops` using
  `evaluator.config.PROTOCOLS["fodiwalk_dist"].da` (200 BFS sources,
  `bfs_rows` draw, `min_hop=2`) — the SAME pairs the protocol's own hop R2
  scores, so the two columns describe one sample. Verified against the
  ICLR report's own number on an unrelated run before this batch: cora
  `walk_edges` seed 42 gave rho 0.8431, report says 0.843 +/- 0.006.
- **recall@10**: MICRO-average, denominator `min(degree, 10)`
  (`evaluator/reports/260904-store-scorecard.md:296` is the definition
  actually used everywhere else in this repo, not the s6.3 macro
  phrasing). A first attempt here used a per-node macro average and was
  off by up to 0.10 on pubmed specifically, because pubmed has far more
  high-degree hubs than cora/citeseer (11.8% of nodes above degree 10, vs
  3.5% and 2.4%) and the two averages diverge exactly where hubs are
  common. Caught by checking against the report; do not repeat the macro
  version.
- **`evaluator.config.PROTOCOLS[name]` is NOT a positional 2-tuple
  any more.** The "H3 / positional 2-tuple, indexed [0]/[1]" fact
  recorded above (2026-08-26) is stale: it is now a `Protocol` dataclass
  with named fields `lp, da, dr, ds, rt, lr, st, sb` — `.da` works,
  `[1]` raises `TypeError: no len()`. Something already fixed this since
  August; nobody updated this file. Trust the dataclass fields, not the
  old note.

**Results**, mean +/- population std over 3 seeds:

| graph | optimizer | Spearman rho | hop R2 (rf) | recall@10 | LP AUC | LP f1_score |
|---|---|---|---|---|---|---|
| cora | sqn/const/lr=0.999 | 0.8656 ± 0.0016 | 0.6503 ± 0.0191 | 0.9573 ± 0.0020 | 0.9992 ± 0.0002 | 0.9891 ± 0.0022 |
| cora | nesterov/linear/lr=0.099 | 0.8061 ± 0.0075 | 0.4964 ± 0.0319 | 0.9650 ± 0.0015 | 0.9989 ± 0.0003 | 0.9877 ± 0.0018 |
| citeseer | sqn/const/lr=0.999 | 0.8280 ± 0.0136 | 0.5701 ± 0.0433 | 0.9629 ± 0.0014 | 0.9996 ± 0.0005 | 0.9969 ± 0.0019 |
| citeseer | nesterov/linear/lr=0.099 | 0.6961 ± 0.0030 | 0.3345 ± 0.0179 | 0.9666 ± 0.0017 | 0.9998 ± 0.0002 | 0.9973 ± 0.0009 |
| pubmed | sqn/const/lr=0.999 | 0.7933 ± 0.0125 | 0.4811 ± 0.0395 | 0.7763 ± 0.0027 | 0.9989 ± 0.0002 | 0.9916 ± 0.0010 |
| pubmed | nesterov/linear/lr=0.099 | 0.7670 ± 0.0140 | 0.4486 ± 0.0216 | 0.8075 ± 0.0015 | 0.9989 ± 0.0001 | 0.9901 ± 0.0001 |

`sqn/const` beats `nesterov/linear` on rho and hop R2 on all three
graphs, sometimes by a wide margin (hop R2 +0.155 cora, +0.236 citeseer,
+0.033 pubmed) — consistent with the standing optimiser verdict on record
above (`plain`, lr 0.999, constant beats linear decay). `nesterov/linear`
wins recall@10 on all three graphs instead, same split seen in the
`nbr_walk` vs `walk_edges` comparison above: geometry-preserving metrics
(rho, hop R2) and neighbour-retrieval metrics (recall@10) do not move
together here, and LP AUC/f1 barely move at all (both saturated near
0.999, as documented in every report in this directory).

Raw per-run rows: `agentic-log/recall-and-rho/opt_arms_dirs.scored.json`
copy this from `/tmp/.../scratchpad/` if kept. 18 embeddings, logs at
`experiments/embeddings/logs/opt_arms_20260921T081724Z.log`. Citeseer used
`--deg-source A` in every arm (I5 guard, see the entry above); cora and
pubmed used the `auto` default.

## Community structure (NMI/ARI), added to the optimizer-arm batch (2026-09-21)

Same 18 stored runs as the entry above. Added the other metric `STCfg`
names (`evaluator/config.py:285`) and `METROLOGY.md` s6.6 marks PLANNED:
Louvain on `A` against k-means on `Z`, `k` = the community count Louvain
found, NMI and ARI between the two label sets
(`papers/ordered-rank-criteria.md:159` is the definition followed).

**Used `networkx.algorithms.community.louvain_communities`, on direct
instruction.** This repo's own convention is the opposite:
`papers/metrology-orchestration.md:425` and `PRD-v2.md:175` both say
"must not" use `networkx` or `python-louvain` for community detection,
because `networkx`'s pure-Python Louvain does not scale to this repo's
million-node graphs -- NetworKit `community.PLM` is the intended path,
per `COMMUNITY_DEFAULT` in `evaluator/config.py:191`. cora/citeseer/pubmed
top out at 19,717 nodes, well inside where that concern applies, so this
does not contradict the reason for the rule, only the letter of it.
**`networkx` is NOT added to `evaluator/requirements-evaluator.txt`** --
it is `pip install`ed into `.venv` for this standalone script only, not a
dependency of the package. A future task at NetworKit-only scale should
not copy this choice without re-reading why the rule exists.

Louvain ran ONCE per graph at a fixed seed (42) -- it is a graph property,
independent of any embedding -- so all six runs of one graph compare
against the SAME reference partition. k-means ran once per embedding, at
that embedding's own seed, `n_clusters` = the Louvain count. Script:
`agentic-log/recall-and-rho/community_nmi.py`.

Louvain found 105 communities on cora, 471 on citeseer, 45 on pubmed
(seed 42, default resolution 1.0). These are STRUCTURAL communities, not
the 6/7/3 label classes cora/citeseer/pubmed are usually cited for --
Louvain over-splits relative to the topic labels, as it does on every
citation graph. Do not read `louvain_k` as a class count.

| graph | optimizer | Louvain k | NMI | ARI |
|---|---|---|---|---|
| cora | sqn/const | 105 | 0.6992 ± 0.0037 | 0.3409 ± 0.0122 |
| cora | nesterov/linear | 105 | 0.7056 ± 0.0071 | 0.3509 ± 0.0151 |
| citeseer | sqn/const | 471 | 0.8671 ± 0.0028 | 0.3390 ± 0.0163 |
| citeseer | nesterov/linear | 471 | 0.8639 ± 0.0059 | 0.3298 ± 0.0255 |
| pubmed | sqn/const | 45 | 0.5829 ± 0.0012 | 0.3514 ± 0.0040 |
| pubmed | nesterov/linear | 45 | 0.5960 ± 0.0033 | 0.3776 ± 0.0106 |

Unlike rho/hop R2 (where `sqn/const` clearly wins on all three graphs),
**NMI and ARI show almost no separation between the two optimizer arms**
-- every gap here is within about 1-2 combined seed SDs. Community
structure in `Z` looks like it depends far more on the walk/force/pairs
choice than on which of these two optimizers ran it. Citeseer's NMI
(0.86+) is much higher than cora's (0.70) or pubmed's (0.58); ARI tells a
flatter story (0.33-0.38 everywhere) -- NMI and ARI are not just two
views of the same number here, and citeseer's high NMI with an ARI no
better than the other two graphs is worth a second look before reading
"citeseer preserves community structure best" out of the NMI column
alone.

Combined per-run rows (rho, hop R2, recall@10, LP AUC/f1, NMI, ARI, all
18 runs): `agentic-log/recall-and-rho/opt_arms_full.json`.

## Other-methods comparison, branch `other-comparisons` (2026-09-22)

Nine published comparison methods vs fodiwalk/deepwalk/node2vec on
cora/citeseer/pubmed, dim 128. Report:
`evaluator/reports/260922-other-methods-comparison.md`. Code:
`other-methods/`. All committed on branch `other-comparisons` (branched
off `explore/seed-variance`; NOT merged). Scaffold commit `0c91149`.

**Architecture that kept it orthogonal and mergeable:**
- Every method writes the SHARED store through
  `other-methods/common/store.py`, scored by the one `evaluator` path
  (protocol `fodiwalk_dist`, `lp_max_pairs=50_000`) with the SAME node
  numbering (`fodiwalk.make_graph.load`). So a new number compares to a
  fodiwalk number with no adjustment.
- `other-methods/score_report.py` reads the store and prints the seven
  columns (rho, hop R2, recall@10, LP AUC/f1, NMI, ARI), recomputing rho
  and recall@10 and community the same way the scripts above do.
- `other-methods/assemble_runlist.py` merges the new-method runs with the
  deepwalk / node2vec / best-fodiwalk anchors, dedups (graph,label,seed).

**Environment traps, each cost time:**
- **The box shrank to 7 GB RAM** (the ICLR report assumed 15 GB). Forced:
  dense n×n methods (NetMF, GraRep, HOPE) on cora/citeseer only — pubmed
  OOMs (~3 GB per n×n at 19,717 nodes); Force2Vec base (O(n²)) skips
  pubmed. All recorded, never dropped.
- **karateclub pins an ancient numpy** that will not build on Python 3.12.
  Fix: isolated `other-methods/karateclub/.venv-karate`, install a modern
  stack then `karateclub` + `nodevectors` with `--no-deps`. That venv has
  no jax, and `fodiwalk.make_graph.load` imports jax, so the karate step
  reads a pre-saved `A.npz` (dumped by the shared venv in fodiwalk
  numbering) instead of importing fodiwalk. Both venvs gitignored.
- **sklearn `SpectralEmbedding` (arpack) HANGS on these disconnected
  graphs.** citeseer (48 isolated nodes -> high-multiplicity zero
  eigenvalues) ran 30 min and hit the timeout. Shift-invert at sigma=0
  fails ("Factor is exactly singular"). Switched Laplacian Eigenmaps to
  `karateclub.LaplacianEigenmaps` — citeseer 2 s, pubmed 30 s, robust. It
  is also published, so it fits the doc's "use published code" rule better.
  `other-methods/laplacian_eigenmaps/run.py` (the sklearn one) is kept but
  superseded.
- **Force2Vec node alignment.** The HipGraph/Force2Vec C++ binary reads a
  `.mtx` with its own node ids. The wrapper builds the `.mtx` FROM the
  fodiwalk adjacency (1-indexed) and reads the `.embd` back into
  `Z[id-1]`, so it stays aligned. Feeding the repo's bundled `cora.mtx`
  would embed a DIFFERENT numbering and every score would be silently
  wrong. Binary has no seed flag -> Force2Vec rows are single-run.

**THE RESULT CAVEAT that decides the ranking, must be stated with any
quote of this table:** rho / hop R2 / recall@10 read EUCLIDEAN distance in
Z. LINE, GraRep, NetMF, HOPE and ProNE learn an INNER-PRODUCT similarity,
not a metric space, so they score near-zero or NEGATIVE rho while keeping
LP AUC 0.99+. That is expected, not a defect; read their LP/community
columns, not the Euclidean ones. `landmark_mds` reads the TRUE
shortest-path distances -> it is a distance-oracle REFERENCE, not a learned
competitor.

**Headline (learned methods, distance geometry):** fodiwalk leads rho on
cora (0.866 vs deepwalk 0.788) and pubmed (0.793 vs 0.699). On citeseer
(sparsest, avg deg 2.74) landmark MDS passes it (0.955 vs 0.828). tForce2Vec
is the best Force2Vec option (rho 0.717 cora) and has the BEST community
NMI/ARI of all methods. Full numbers in the report.

## Known-bad script patterns (each cost a run)

- `hop_sample` in the frozen reference returns **four** values
  `(u, v, d, h2)`, not three.
- The frozen reference names its models `"mean baseline"`, `"random
  forest"`, `"MLP"`. This package uses `baseline`, `rf`, `mlp`.
- `fodined.link_prediction.link_prediction` returns `(scores, info)`, and
  spells the harmonic mean `f1-score`. This package spells it `f1`.
