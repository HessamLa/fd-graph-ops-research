# evaluator -- Orchestration

Created 2026-08-22. Companion to `PRD.md`, which is the authority on WHAT
each block must contain and what it must satisfy. This document says WHO
builds each block, in what order, at what complexity, and how the output is
judged.

The `agentic-log/` tree of the repository root holds the record. This
campaign uses the ordinals `10` to `19`, thus it does not collide with the
`fodiwalk` campaign that holds `00` to `05`.

---

## 1. Rules that govern every subagent

**A subagent starts cold.** It has none of this conversation. Every brief
therefore QUOTES the invariant that it must not break; it does not point at
it. The brief also states what the agent must NOT touch.

**The first read is prescribed.** Every brief names the files to read, in
the order that builds understanding: the contract before the consumer of
the contract. An agent that reads `tasks/` before `config.py` writes a task
that carries its own defaults, and the protocol table then means nothing.

**Parity is the gate, not review.** A block is judged by the numbers of
`PRD.md` section 9 and by the criteria of its own block, and not by whether
its code reads well.

**Nothing is deleted before W5.** `experiments/other-ge/bench_other_ge.py`
and `experiments/large-graph-node2vec/bench_node2vec_1M.py` keep working
throughout. The new package is additive until the parity table passes.
`fodiwalk/misc/evaluation.py` is never changed by this campaign.

**Self-verification is required.** An agent must RUN what it built and
report the real output, and not a description of it. "It imports and it
returns finite numbers" is a weaker bar than "it equals the reference to
1e-12"; each brief says which bar it needs.

**Model selection.**
* `sonnet` -- a mechanical move with a clear target, or a wrapper over an
  existing function.
* `opus` -- a contract that must survive a restructuring, an RNG order that
  must be preserved, or a defect that would be SILENT.

**The redo loop.** After each agent returns: run its success criteria
myself. If any fails, do NOT patch it myself. Rewrite the brief to name the
specific failure and the specific expectation, and send it back to the SAME
agent with `SendMessage`, thus it keeps its context. Escalate `sonnet` to
`opus` after two failed attempts on one criterion.

**Logging.** Each agent appends to `agentic-log/<ordinal>.<slug>/`. One
line for each entry, with the tags `START, PLAN, ACTION, OBSERVE, DISPATCH,
CLAIM, VERIFY, DEADEND, END`. A `CLAIM` is what the agent said; a `VERIFY`
is what I observed when I re-ran it. The two are never merged.

---

## 2. The complexity split

The table sorts every block by the cost of getting it wrong. That cost, and
not the line count, chooses the model.

| Block | Work | Cost of a defect | Level | Model |
|---|---|---|---|---|
| B1 `config.py` | 2 dataclasses + a table read from 4 files | HIGH. A wrong cell makes every parity test fail, and the failure looks like a bug of another block | **hard** | opus |
| B2 `io.py` | Loaders, mostly adapted from existing code | LOW. A wrong loader fails loudly | easy | sonnet |
| B3 `pairs.py` | The rejection sampler, RNG order preserved | HIGH. A moved draw changes every pair and the failure is a number, not an error | **hard** | opus |
| B4 `hops.py` | Three backends and the crossover rule | HIGH. A wrong sentinel gives a plausible wrong distance | **hard** | opus |
| B5 `metrics.py` | 4 distances, 4 features, all vectorized | MEDIUM. The Poincare formula is easy to get wrong and it fails silently | medium | opus |
| B6 `scoring.py` | Two dicts of sklearn calls | LOW | easy | sonnet |
| B7 `guards.py` | 9 checks with thresholds | MEDIUM. A guard that never fires is worse than no guard | medium | sonnet |
| B8 `report.py` | 2 dataclasses, JSON casting, a table printer | LOW | easy | sonnet |
| B9 `tasks/link_prediction.py` | The step order IS the parity contract | HIGH | **hard** | opus |
| B10 `tasks/dist_approx.py` | The same, plus the feature-width rule | HIGH | **hard** | opus |
| B11 `__init__.py`, `tasks/__init__.py` | The registry and the re-exports | LOW | easy | sonnet |
| B12 `cli.py`, `__main__.py` | argparse, the paren sugar, `--json` | MEDIUM. The sugar parser is the only new grammar of the project | medium | sonnet |
| B13 `tests/` | The parity gate and the frozen reference copies | HIGH. A test that is wrong hides a defect of every block | **hard** | opus |
| R1 rewrite `bench_other_ge.py` | Delete task code, call the package | MEDIUM. The recorded numbers must not move | medium | sonnet |
| R2 rewrite `bench_node2vec_1M.py` | The same, at 1.13M nodes | MEDIUM | medium | sonnet |
| D1 `CATALOG.md` | The entity catalog that `CLAUDE.md` requires | LOW | easy | sonnet |

Seven blocks are `hard`. That is the honest count: this package is small in
lines and large in contracts.

---

## 3. Dependency order

```
  W1   A10 config.py           A11 io.py          A12 metrics.py+scoring.py
       (opus, B1)              (sonnet, B2)       (opus, B5+B6)
          |                        |                    |
  W2   A13 pairs.py + hops.py                    A14 guards.py + report.py
       (opus, B3+B4)                             (sonnet, B7+B8)
          |____________________________________________|
                                |
  W3                A15 tasks/ + registry
                    (opus, B9+B10+B11)
                                |
          ______________________|______________________
         |                                             |
  W4   A16 cli.py + __main__.py                 A17 tests/ + parity gate
       (sonnet, B12)                            (opus, B13)
         |_____________________________________________|
                                |
  W5      A18 rewrite bench_other_ge.py    A19 rewrite bench_node2vec_1M.py
          (sonnet, R1)                     (sonnet, R2)
                                |
  W6                  A20 CATALOG.md  (sonnet, D1)
                                |
  W7                  Verification by fdmap-83 + two peer sessions
```

W1 runs three agents at the same time. They touch disjoint files and they
share no state. W2 runs two at the same time. W5 runs two at the same time.

**Why `config.py` is alone at the front.** Every other block reads the
protocol table. An agent that starts before the table exists invents its
own defaults, and the parity gate then fails for a reason that has nothing
to do with the block under test.

---

## 4. Assignments

Each brief below is the SUMMARY. The full brief goes to
`agentic-log/00.master-agent/prompts/<nnn>.<slug>.md` before dispatch, and
it holds the quoted invariants, the read order, and the non-goals.

---

### A10 -- `evaluator/config.py` -- **opus** -- PRD B1

**Goal.** The two frozen dataclasses and the `PROTOCOLS` table.

**Read first, in this order.**
1. `evaluator/dev-docs/PRD.md` sections 1, 7 (B1), 8, 9.
2. `experiments/other-ge/bench_other_ge.py`, the functions
   `task_link_prediction` and `task_sp_regression`.
3. `experiments/large-graph-node2vec/bench_node2vec_1M.py`, `main`.
4. `fodined/modular.py` lines 520-670.
5. `fodiwalk/misc/evaluation.py`, `link_prediction` and `task_hop`.

**The one hard part.** The table of B1 is a CLAIM that I made by reading
those four files. This agent must CHECK every cell against the code and
report each disagreement. The known conflict is written in B1: the live
`fodined/modular.py` uses a one-column distance feature, and the docstring
of `fodiwalk.misc.evaluation.task_hop` says that it uses the `n_dim` vector
form. Resolve it against the CODE and record the answer.

**Must not.** Import anything but `dataclasses`. Add a knob that no
recorded baseline needs.

**Success criteria.** PRD B1.1 to B1.3, plus: a written report of every
cell that disagreed with the PRD table, with a `file:line` for each.

---

### A11 -- `evaluator/io.py` -- **sonnet** -- PRD B2

**Goal.** `load_graph`, `load_embedding`, `write_report`.

**Read first.** PRD B2; `fodiwalk/make_graph/datasets.py` (this is the
registry, and `load_graph` DELEGATES to it -- it does not copy it);
`experiments/other-ge/bench_other_ge.py` `load_csr`.

**Must not.** Copy the dataset loaders. Import `fodiwalk` at the top of the
module.

**Success criteria.** PRD B2.1 to B2.5. The agent must run
`load_graph("cora")` and paste the real `n`, `nnz` and `dtype`.

---

### A12 -- `evaluator/metrics.py` and `evaluator/scoring.py` -- **opus** -- PRD B5, B6

**Goal.** The distance registry, the feature registry, the two score sets.

**Why opus for a small file.** The Poincare distance is the one formula of
this project that gives a plausible WRONG number when it is wrong. The
guard `C4` depends on it. `bench_other_ge.poincare_distance` is the
reference and it must be matched exactly, `np.maximum` clamps included.

**Read first.** PRD B5 and B6; `bench_other_ge.poincare_distance`;
`fodiwalk/misc/evaluation.classify`.

**Success criteria.** PRD B5.1 to B5.4 and B6.1 to B6.3, plus: a printed
comparison of `metrics.distance(..., "poincare")` against
`bench_other_ge.poincare_distance` on 10,000 random points of the ball,
with `np.max(np.abs(diff))` shown.

---

### A13 -- `evaluator/pairs.py` and `evaluator/hops.py` -- **opus** -- PRD B3, B4

**Goal.** The samplers and the ground truth. This is the block where the
parity of the whole project is decided.

**The quoted invariant.** From `fodiwalk/tests/harness.py`: "THE ORDER OF
THE GENERATOR IS PART OF THE PARITY." The draw order of `negatives` is
written in PRD B3 and it must be copied exactly: `u` before `v`, and the
batch size `(count - have) * 2 + 1024`.

**The three traps, quoted into the brief.** PLL returns `2**64 - 1` for an
unreachable pair and NetworKit does not document it.
`nk.GraphFromCoo` gave a segmentation fault on a symmetric matrix with a
`data` array; pass the upper triangle as two `uint64` arrays.
`scipy.sparse.csgraph.shortest_path` always returns a DENSE
`(len(indices), n)` array, thus `indices=` limits the rows and not the
memory.

**Read first.** PRD B3 and B4; `experiments/bench_shortest_path_gemsec.py`
(the whole file: it is the measured comparison of the five backends);
`bench_other_ge.sample_non_edges`;
`bench_node2vec_1M.sample_non_edges` (note the `sources` argument);
`fodiwalk/augment_graph/landmarks.py`;
`fodiwalk/augment_graph/far_pairs.py`.

**Must not.** Use a Python set in the rejection sampler. Build an `(n, n)`
array. Import `networkit` at the top of the module.

**Success criteria.** PRD B3.1 to B3.5 and B4.1 to B4.5. The agent must
paste the real output of the byte-for-byte comparison of B3.5.

---

### A14 -- `evaluator/guards.py` and `evaluator/report.py` -- **sonnet** -- PRD B7, B8

**Goal.** The nine checks and the record.

**The reason for the block, quoted into the brief.** From
`bench_other_ge._pick_subtree`: "on NCBI a BFS returns a star ... Every
method then reaches a perfect score, and the measurement says nothing about
a tree." `guards.py` exists to stop that run.

**Read first.** PRD B7 and B8; `bench_other_ge._pick_subtree`;
`fodiwalk/core/plan_contract.py` if it exists (the same pattern, for the
engine).

**Must not.** Warn where the PRD says raise. Add a fallback.

**Success criteria.** PRD B7.1 to B7.3 and B8.1 to B8.3. The agent must
build the 500-leaf star itself and paste the raised message.

---

### A15 -- `evaluator/tasks/` and the registry -- **opus** -- PRD B9, B10, B11

**Goal.** The two tasks, the registry, and `evaluator/__init__.py`.

**Why opus.** The step order of B9 and B10 IS the parity contract. A step
that moves changes the generator state and every following number.

**The quoted rule.** From `fodiwalk/misc/evaluation.task_hop`: "A model
with 128 features can win only because it has more of them, thus the two
rows must not be mixed." `feature="distance"` gives ONE column and
`feature="vector"` gives `n_dim` columns. Neither ever falls back to the
other.

**Read first.** PRD B9, B10, B11, and invariants I1 to I7. Then the
finished `config.py`, `pairs.py`, `hops.py`, `metrics.py`, `scoring.py`,
`guards.py`, `report.py`. Then `bench_other_ge.task_link_prediction` and
`task_sp_regression` as the reference behaviour.

**Must not.** Make an `rng`. Import `cli`. Put a default in a task that
belongs in `config.py`.

**Success criteria.** PRD B9.1 to B9.3 and B10.1 to B10.4. The agent must
run the cora parity check itself and paste both score sets side by side.

---

### A16 -- `evaluator/cli.py` and `__main__.py` -- **sonnet** -- PRD B12

**Goal.** The argument parser, the paren sugar, the printer, `--json`.

**Read first.** PRD sections 6 and B12; `bench_other_ge.main` for the
summary table format.

**The one new thing.** `parse_call` reads
`"link_prediction(test_size=0.2, neg_ratio=1)"`. An unknown key RAISES and
it names the valid keys. This is the only grammar that this project
invents; it needs its own unit test with 8 inputs, 4 good and 4 bad.

**Must not.** Put a measurement in `cli.py`. Let an exception reach the
user as a traceback; a guard failure prints its id and exits 1.

**Success criteria.** PRD B12.1 to B12.5.

---

### A17 -- `evaluator/tests/` -- **opus** -- PRD B13, section 9

**Goal.** The parity gate.

**Why opus.** A test that is wrong hides a defect of every other block. The
frozen reference copies are the delicate part: `tests/reference/` holds a
VERBATIM copy of `bench_other_ge.task_link_prediction`,
`task_sp_regression` and `sample_non_edges`, because W5 deletes the
originals. A copy that is "tidied" is not a reference.

**Read first.** PRD section 9 and B13; the finished package.

**Must not.** Change a package file to make a test pass. A failing parity
test is a finding, and it goes back to the block's agent.

**Success criteria.** P1 to P6 of PRD section 9 pass, and the agent pastes
the real `pytest -q` output. The whole suite finishes inside 5 minutes.

---

### A18 -- rewrite `experiments/other-ge/bench_other_ge.py` -- **sonnet** -- PRD section 10

**Goal.** The script keeps its embedding methods and its printed table, and
it calls `evaluator` for every measurement.

**Deletes.** `sample_non_edges`, `task_link_prediction`,
`task_sp_regression`, `poincare_distance`, and the `sklearn` and
`networkit` imports that only they used. The loaders may also go, because
`fodiwalk.make_graph.datasets` holds the same code; the agent decides and
it says why.

**Keeps.** `uniform_walks`, `second_order_walks`, `embed_node2vec`,
`embed_poincare`, the method knobs of the parser, and the summary table.

**The reference.** `experiments/other-ge/results_2026-08-14.log`. The
rewritten script must print the same numbers for the same seed.

**Success criteria.** A run of
`.venv/bin/python experiments/other-ge/bench_other_ge.py --graphs cora,pubmed
--methods node2vec,poincare` prints numbers equal to that log, and the
agent pastes both. At least 200 lines of task code are gone.

---

### A19 -- rewrite `experiments/large-graph-node2vec/bench_node2vec_1M.py` -- **sonnet** -- PRD section 10

**Goal.** The same, at 1.13M nodes, with the `n2v1m` protocol.

**Deletes.** `sample_non_edges` and the whole inline scoring of `main`.
**Keeps.** `load`, `write_walks`, `rss_mb`, and the staged timing report.

**The cost warning, in the brief.** A full run needs more than an hour. The
agent develops against `--graph com_youtube` with a `--max-nodes 50000`
truncation, or against cora, and it does NOT start a full run without
asking me first.

**Success criteria.** The script runs to the end on a truncated graph and
it prints a complete report. The full run is P7 and P8, and I start it
myself.

---

### A20 -- `evaluator/dev-docs/CATALOG.md` -- **sonnet** -- `CLAUDE.md` requirement

**Goal.** The entity catalog that `CLAUDE.md` demands: one entry for each
named entity, with an elaborate description, a code snippet, and a
`file:line` provenance link.

**The entities.** Every protocol (`otherge`, `n2v1m`, `fodined`,
`fodiwalk`, `default`); every task; every distance; every pair feature;
every hop backend; every guard C1 to C9; the `pos_draw` knob and the reason
it exists; the `f1` rename; the intentional change of the `fodined` hop
source from `D.data` to the true hop distance.

**The rule.** The catalog only grows. An update names the reason and the
version, and it keeps the earlier text.

**Success criteria.** Every entity above has an entry. Every entry has a
provenance link that resolves.

---

## 5. Verification, W7

The user's requirement: other sessions must use and evaluate `evaluator`
through BOTH surfaces, the CLI and the import. A verifier that only reads
the code has not tested the surfaces.

**V1 -- `fdmap-83`.** It already knows this design; it reviewed the plan
before the PRD was written. Its brief:
1. Read `evaluator/` and judge it against `PRD.md` sections 7, 8 and 11.
2. Confirm that `fodiwalk/misc/evaluation.py` is unchanged.
3. Report every place where the package repeats code that `fodiwalk`
   already holds.
4. Run the package as an IMPORT: score one embedding with
   `ev.evaluate(...)` and paste the record.

**V2 -- a second peer session, CLI only.** It receives the CLI section of
the PRD and NOTHING else, and it must score an embedding from a cold start
with the CLI alone. What it has to ask for is a defect of the `--help`
text.

**V3 -- a third peer session, API only.** The same, with the API section
only. It must add a THIRD task (a toy one, e.g. degree recovery) with one
file and one registry line. That proves F4.

Each verifier reports to me. I re-run every claim myself before I report to
the user; a `CLAIM` without my own `VERIFY` line is an open item.

---

## 6. What I do myself

* This document and `PRD.md`.
* Every brief. The A13 and A15 briefs get the most effort of the campaign,
  because those two blocks hold the parity contract.
* Every verification: re-run the tests, re-run the two benchmark scripts,
  read the real diff of `pairs.py`, `hops.py` and `tasks/`.
* P7 and P8, the 1.13M-node run.
* The report to the user, with the observed numbers and not the claims.

## 7. Risks of the plan itself

**A13 and A15 are one contract split over two agents.** The draw order of
`pairs.py` and the step order of `tasks/` must agree. The answer is that
BOTH orders are written in the PRD, and A15 reads the finished `pairs.py`
before it writes.

**W1 blocks everything.** Three agents run at the same time, and A10 is on
the critical path of all of them. If A10 finds that the protocol table is
wrong, A12 and A11 have still done useful work; only A13 onward waits.

**The 1.13M run is expensive.** P7 and P8 are the last thing, they run one
time, and a failure there costs hours. The mitigation is A19's truncated
development and the fact that the same code path already passed P1 to P6.
