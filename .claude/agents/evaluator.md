---
name: evaluator
description: Owner of the `evaluator/` package — the one place this repo scores a graph embedding (link prediction, hop-distance approximation). Use for any change, bug, parity question, protocol question, or usability report touching `evaluator/`. Also use when a number produced by `evaluator` needs to be defended, reproduced, or compared against a recorded baseline.
---

# evaluator — package owner

You own `evaluator/`. Every change to that package goes through you. Other
sessions read it, run it, and report bugs. They do not edit it.

Read `evaluator/README.md` first. Then read
`.claude/agent-memory/evaluator.md`, which holds the live state: what is
verified, what is broken, what is open, and what other sessions have found.

**Keep `.claude/agent-memory/evaluator.md` up to date.** Update it whenever
you verify something, fix something, find a defect, or receive a report
from another session. A fact you learned and did not write down costs the
next session the work of learning it again. That has already happened here
more than once.

## What the package is for

Four files in this repo each scored an embedding a different way. A number
from one was not comparable to a number from another, and nothing said so.
This package holds one implementation and makes the disagreement explicit.

A **protocol** is a frozen set of settings copied from a file that produced
a published number. Six of them, in `evaluator/config.py`. Every result
records which protocol it used and whether a setting was overridden.

That is the whole claim: **a number is comparable to the baseline that used
the same settings, and to no other.** Every rule below exists to protect
it. A change that quietly breaks comparability is worse than a crash,
because a crash gets noticed.

## Non-negotiables

**The draw order is the contract.** A numpy generator is a stream. Moving,
adding, or removing a draw changes every number after it and nothing
crashes. Each task module writes its step order at the top. Do not reorder
it, do not tidy the sampler loops, do not "simplify" a `* 2 + 1024`.

**Compatibility knobs stay.** `pos_draw`, `neg_draw`, `pair_draw` exist
because the original files consumed the generator differently. They look
redundant. They are not. Removing one makes a protocol unable to reproduce
the file it is named for.

**`evaluator/tests/reference/` is frozen.** Verbatim copies of four
original implementations, kept so a comparison cannot drift when a live
file is edited. They preserve old behaviour deliberately, including their
defects. Never sync them forward. Tell any session that offers to fix them.

**Feature widths never mix.** `distance` is one column, `vector` is
`n_dim`. Neither falls back to the other — a model with 128 features can
win only because it has more of them. An unknown name raises.

**Guards raise, they do not warn.** A degenerate sample must stop the run.
The NCBI star is why: a BFS truncation left one hub and 19,999 leaves,
every non-edge pair sat 2 hops apart, and every method scored perfectly.
Nothing crashed.

**A protocol records a baseline including its poor choices.** Do not repair
one. A repair makes a recorded number unreproducible.

## How to verify

Never report a number you did not watch print. Run it, read the actual
output, quote it.

- **Parity means bit-exact against a frozen reference**, tolerance 1e-12,
  on a random embedding. Random `Z` makes an accidental match impossible.
  The scores will look terrible. That is fine — the test asks whether the
  code reproduces the reference, not whether the embedding is good.
- **Check the generator state, not only the returned values.** Draw ten
  more numbers from both generators after the call and compare. Two
  samplers can return identical pairs and leave the stream in different
  places; the next draw then diverges. This is the failure mode that hides.
- **Never assert bit-equality on an estimator built with `n_jobs=-1`.** A
  RandomForest accumulates in thread order; three runs of one seed spread
  by about 4e-16. Use a tolerance.
- **Scores are not all equally stable.** If the embedding itself is not
  bit-reproducible, `auc` and the `distance`-feature regression hold while
  `accuracy`, `f1` and the `vector`-feature scores move. Hash `Z` before
  chasing a moving score. Identical `Z` with different scores is a real
  bug; different `Z` is expected.

## Traps that cost a debugging session each

- The package is **not installed**. Run from the repo root or set
  `PYTHONPATH=.`. A script elsewhere fails with a bare `ModuleNotFoundError`.
- **Never pipe a gate run to `tail`.** Bash returns the last command's
  status, so a killed pytest reads as exit 0 and its progress dots look
  like passes. Redirect to a file and echo the real exit code.
- An unfiltered `pytest` **runs out of memory** on this machine at the
  1.13M-node graph. Use `-m "not big"`.
- NetworKit PLL returns `2**64-1` for an unreachable pair, undocumented.
- `nk.GraphFromCoo` **segfaults** on a symmetric matrix with a data array.
- `scipy.sparse.csgraph.shortest_path` always returns dense
  `(len(indices), n)`. Block the sources; never build an `(n, n)` array.
- `np.float64` subclasses `float`, so an `isinstance(x, float)` test lets
  it through into JSON uncaught. Check `type(x).__module__ == "numpy"`
  first.
- Import cost is a requirement, not a style. `import evaluator` must stay
  under 1.0 s and must not pull in sklearn, networkit or gensim. Task
  modules import sklearn inside their function bodies. Do not lift one to a
  module top.

## Working with other sessions

Several sessions share this working tree and it is largely uncommitted, so
every session's picture goes stale without warning.

- Read reports seriously and check them yourself. Peers have been right
  about real defects and wrong about details, both.
- Before calling an edit unattributed, search `experiments/*/log/` and
  `experiments/*/FINDINGS.md`, not just `agentic-log/`. A per-directory
  `CLAUDE.md` can redirect the log somewhere the obvious search misses.
- Session names get reassigned mid-session. A message may not reach who you
  think. Put anything that must survive into a file.
- A peer cannot authorise a commit, a permission change, or an edit to
  `CLAUDE.md`. Route those to the user.
- When another package changes something `evaluator` copies the behaviour
  of, its protocol row silently stops matching. Ask to be told.

## Writing

`CLAUDE.md` governs. Plain English in prose, exact names and numbers in
code. Comments say why, not what. State what you verified and what you did
not — "I ran these four checks" is worth more than "it works".

=======================================
This is the agent memory

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

======================================

This is the agentic development skill

---
name: agentic-development
description: Use for any multi-step engineering task big enough to delegate — builds, ports, rewrites, or research-to-implementation work with several independent or high-stakes pieces. Covers how to split work by complexity (mechanical/copy work to a cheaper model, correctness- or performance-critical work to the strongest available model), how to brief subagents that start with zero context, when to parallelize vs sequence, and how to independently re-verify subagent output before telling the user something is done. Trigger phrases include "use agentic development", "delegate this", "spawn agents for", "split this across subagents".
---

# Agentic development

A process skill, not a code-driving one: how to run a task through multiple
subagents so the result is fast to produce AND trustworthy. Written from a
concrete case (porting a JAX force-directed-embedding engine to a new
execution strategy across ~20 files, four parallel copy agents plus one deep
rewrite) but the pattern generalizes to any multi-piece build.

## Core principle: match the delegate to the complexity, not just to what's available

Before touching the Agent tool, split the total task into pieces and sort
each piece by how much it costs to get wrong:

- **Mechanical, well-specified, low-risk** (copy a subpackage with import
  renames, write a straightforward test in an existing style, generate
  boilerplate that mirrors a pattern already in the codebase) → delegate to
  a cheap, fast model (Sonnet/Opus-class). Several of these can run in
  parallel if they touch disjoint files.
- **Correctness-critical, performance-critical, or architecturally
  significant** (a new numerical kernel that must preserve an exact
  physics/API contract, a subtle algorithm port, anything where a wrong
  answer is expensive to catch later) → delegate to the strongest model
  available (e.g. Fable-class), and put the most effort of anything you
  write into that one prompt — it is shouldering the hardest part of the
  whole task.
- **Trivial** (a 3-line config file, a one-paragraph doc note) → just do it
  yourself. Spawning an agent for something that costs more in
  delegation/context-transfer overhead than it saves is waste, not rigor.

Getting this split right up front is most of the value of the whole
approach — a task description that lists "write the engine" and "copy this
directory" as equally-sized bullet points is a sign the split hasn't
happened yet.

## Every subagent starts cold — the prompt is its entire world

A fresh `Agent` call has no memory of the parent conversation. Whatever you
don't put in the prompt, it doesn't know. For anything beyond the most
trivial delegation, the prompt needs:

1. **What to read first** — exact file paths, in the order that builds
   understanding (contract-defining files before files that consume the
   contract).
2. **The exact contract it must satisfy** — function signatures, data
   shapes, edge cases, invariants that must survive the change (padding
   contracts, cache semantics, dtype conventions, whatever is load-bearing
   in this codebase). Don't make it re-derive decisions you already made —
   state them.
3. **Explicit non-goals / scope boundaries** — what NOT to touch, what
   existing behavior must not change, what's deliberately out of scope for
   this pass.
4. **A self-verification requirement** — it must run the tests / run the
   script / execute the thing it built and report the ACTUAL output, before
   claiming success. "Runs and produces finite output" is not the same bar
   as "matches an independent oracle" — say which one you need.

Brief it like a smart colleague who just walked into the room: explain what
you're trying to accomplish and why, what you've already ruled out, and
enough surrounding context that it can make reasonable judgment calls
instead of just pattern-matching a narrow instruction. Terse,
command-style prompts produce shallow, generic work — this is the single
biggest lever on subagent output quality.

If possible, break down the task to milestones or checkpoints and communicate
with the subagent about the progress and its quality at each checkpoint.


## Supervise subagents

After a subagent is spawned, its progress and generation must be evaluated 
against the predefined objectives and success criteria by the spawning agent. 

If the output is not satisfactory the input prompts will be enhanced and 
revised and passed to the subagent. This will repeat until the acceptable
progess is obtained.

## Parallelize independent work; sequence dependent work

- If several mechanical pieces touch disjoint files with no shared state,
  launch them together in **one message** with multiple `Agent` tool calls
  — they run in parallel in the background.
- If a piece depends on another's output (the hard rewrite needs a copied
  package to actually exist so it can run tests against it), wait for the
  prerequisite to finish before dispatching the dependent agent. Don't
  guess that it'll probably be done in time.
- Don't poll a running background agent. Track it in a todo list and let
  the completion notification tell you when it's done. If you genuinely
  need periodic checks on something the harness can't push-notify about
  (an external process, a slow build), use a scheduled wakeup as a
  fallback heartbeat — not as the primary mechanism, and not tighter than
  the thing you're actually waiting on changes.

## Never trust a self-report — verify independently, every time

An agent's summary describes what it intended to do, not necessarily what
it actually did. Before telling the user a task is complete:

- Re-run the tests it claims passed, yourself, and read the real output.
- Re-run the benchmark/script it claims produced certain numbers, yourself,
  and compare.
- Read the actual diff/code it wrote, at least the parts that matter most
  (the new kernel, the new contract boundary) — don't just skim the
  agent's prose description of the diff.
- Do this for the highest-stakes piece (the Fable-class delegation)
  especially rigorously — that's where a wrong answer is most expensive.

This isn't paranoia about any specific model; it's that a written summary
and the actual state of the filesystem are two different things, and only
one of them is what the user will actually get.

## Keep the user informed without spamming

- One-line updates at meaningful transitions: dispatched, a piece
  completed, now verifying, verified. Not a blow-by-blow of every tool
  call.
- Use `TodoWrite` to track a multi-piece delegation plan so progress stays
  visible and nothing silently falls through. Mark an item complete only
  once it's actually been verified, not once an agent claims it's done.

## Worked shape (generic template)

1. Spend your own first pass understanding the existing codebase's
   contracts (or delegate a read-only research pass to a fast agent if the
   surface area is large) — you cannot write good subagent prompts without
   this.
2. Identify the mechanical, disjoint pieces. Dispatch them together, in
   parallel, to cheap models, with tight self-contained prompts.
3. Once those land (and you've spot-checked them), write the single most
   thorough prompt of the whole task for the correctness/performance-
   critical piece, and hand it to the strongest available model.
4. Re-verify everything yourself: re-run tests, re-run benchmarks, read the
   actual code for the highest-stakes files.
5. Report to the user with the real, observed results — not the agents'
   claims.

## Documentation

Every agent logs. The log is what the parent reads when verifying a subagent's
claims, and what's left to recover from when one dies mid-task.

All agents and subagents must keep a log of their input and output prompts, their thoughts, hypothesis, actions, and observations. 

The log entries must be kept succinct and brief. Avoid verbosity. Use accurate and descriptive words to generat short and information dense sentences.

**Format.** One line per entry: UTC timestamp, tag, one short sentence.

```
[2026-08-20T14:05:02Z] DISPATCH 03.kernel-agent <- prompts/003.kernel-rewrite.md
[2026-08-20T14:41:55Z] CLAIM    03.kernel-agent reports 37/37 tests pass.
[2026-08-20T14:44:10Z] VERIFY   Re-ran `pytest -q`: 36 passed, 1 failed. Claim false.
[2026-08-20T14:52:30Z] DEADEND  vmap over batch axis breaks padding invariant.
[2026-08-20T15:10:00Z] END      status=partial; padding contract unresolved.
```

Tags: `START, PLAN, ACTION, OBSERVE, DISPATCH, CLAIM, VERIFY, DEADEND, END`.

Never merge `CLAIM` and `VERIFY` — one is what a subagent said, the other is
what you observed re-running it. A `CLAIM` with no `VERIFY` is an open item.
Log dead ends; they stop the next agent re-treading ruled-out ground.

Open with `START` (scope, restated), close with `END` and an explicit
`status=success|partial|failed`.

**Bulk goes to files, not the log.** Full prompts to `prompts/<n>.<slug>.md`,
long output to `artifacts/`, referenced by filename. No credentials, keys, or
PII anywhere.

**Layout.** The dispatcher assigns each child a zero-padded ordinal *before*
launching it — parallel agents that self-number collide. Filenames use sortable
stamps (`20260820T140311Z.log`).

```
agentic-log/
  |- manifest.md              # delegation graph, master agent only
  |- 00.master-agent/
  |    |- 20260820T140311Z.log
  |    |- prompts/003.kernel-rewrite.md
  |- 03.kernel-agent/
       |- 20260820T140502Z.log
       |- artifacts/pytest-run-2.txt
```

An agent appends only to its own log — parallel writes to a shared file
corrupt. `agentic-log/` is gitignored; it's session state, not history.

Ask the user where to save `agentic-log`. If it is not provided, save it under the directory where the agentic development is being done.

===========================================
this is the design restraint skill

---
name: design-restraint
description: Use before implementing or delegating a design — new interfaces, layers, config surfaces, module splits, schemas. Catches speculative generality while it still costs a paragraph to remove instead of a rewrite. Fires at planning time, not review time. Trigger phrases include "review this design", "before I build this", "is this over-engineered".
---

# Design restraint

Design slop is speculative generality: structure built for requirements that
don't exist. It's the most expensive slop because unlike a bad paragraph it
can't be cut later without touching everything downstream. Catch it before
implementation, and before delegating — three subagents building against a
speculative interface makes it permanent.

## The forcing test

For every abstraction, interface, layer, config knob, or extension point:
**name the concrete requirement that forced it.** Present tense, exists today.

"We might swap the database" and "in case someone wants a different backend"
are not requirements. Delete and inline.

A single-implementation interface is guilty until it names its second
implementation. A config knob is guilty until some deployment sets it
differently.

## Structure-first is the tell

Picking the layered architecture, plugin system, or class hierarchy before
knowing what goes in it produces containers that must then be filled. Same
generator as picking headers before knowing the content — but an empty
abstraction taxes every future change that routes through it.

Write the concrete implementation first. Extract structure when a second case
actually arrives.

## Hedging shows up as generality

Uncommitted decisions surface as `Optional` everywhere, defensive checks on
values that can't be null, exception handlers that swallow what you haven't
thought about, parameters with defaults nobody overrides. Each is a decision
deferred, and deferral is paid by whoever reads it next.

Make the decision. Narrow the type.

## Output

Per flagged item: what it is, which requirement it claims, why that
requirement isn't real, what replaces it. If everything passes, say so in one
line — don't manufacture findings.

