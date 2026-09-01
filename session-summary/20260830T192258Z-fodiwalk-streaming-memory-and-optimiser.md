# Session summary: tooling review

Session `fdmap-89 [312c80]`, 2026-08-28 to 2026-08-30.

Work covered: the `nbr_walk` augmentation memory rewrite; the stage-contract
docs; fodiwalk against node2vec at a million nodes; the streaming strategy
and its 14-run campaign; the optimiser and learning-rate study; the
dim-128 baseline flights.

**Peer consultation attempted and NOT obtained.** `ListAgents` shows one
peer, `experimenter [10e117]`, and it is OFFLINE. Everything below is this
session's own assessment; a second opinion should be sought before acting
on the recommendations.

---

# PART 1 -- SKILLS

## 1.1 Skills used, and helpful

### `agentic-development` -- used, and it earned its place

Invoked once via the `Skill` tool, then applied to three delegations
(`10.mem-agent`, `11.docs-agent`, `12.bench-agent`).

**What it caught that would otherwise have shipped:**

* Its rule *"Never trust a self-report -- verify independently, every
  time"* is the single most valuable line in it. Acting on it found that
  `12.bench-agent` had run `pairs=walk` instead of `pairs=nbr_walk` --
  benchmarking the one policy the memory work never touched. It also found
  that the agent's "the card is too small" conclusion was wrong: two cheap
  retries showed the obstacle was JAX's default 75% preallocation, and
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` ran it at 1993 MB of 2048.
* Its insistence on **one thorough prompt for the hardest piece** produced
  a 270-line brief for the memory rewrite that carried the measurements,
  the design, the contract, the traps and the milestones. That agent
  delivered a correct byte-exact rewrite in one pass.
* **Ordinal assignment before launch** prevented log collisions across
  parallel agents.
* The **CLAIM / VERIFY separation** in the log format made it possible,
  days later, to see which numbers were re-run and which were only
  asserted.

### `plain-english` -- followed, but NOT invoked

The writing standard was applied throughout (docs, comments, commit
messages, user-facing summaries), driven by `CLAUDE.md` rather than by
loading the skill. It worked. See 1.2 for the gap this exposes.

## 1.2 Skills used that need enhancement

### `agentic-development` -- five concrete gaps, each with an incident

**(a) It verifies the ANSWER but never the QUESTION.**
The skill says to re-run tests and read the diff. It does not say
*"confirm the agent is solving the problem you posed."* `12.bench-agent`
passed every gate it ran while measuring the wrong policy for 20 minutes,
pushing the machine to 80% swap.
**Enhancement:** add a step before the long run -- *the agent must echo
back the resolved configuration it is about to measure, and the dispatcher
must check it against the brief.* One line of output would have caught
this.

**(b) No guidance for agent DEATH mid-task.** Two agents died on session
limits (`09.forces-agent`, `10.mem-agent`), both mid-verification, leaving
unknown edits on disk.
**Enhancement:** require agents to append a `STATE` line to their log
after every file write, naming what is now on disk and what is unverified,
so a successor can resume rather than re-derive. Add a dispatcher-side
recipe: `stat` the tree, diff against the last known-good tag, decide
resume-or-restart.

**(c) It tells the DISPATCHER not to poll, but not the AGENT.**
`12.bench-agent` woke repeatedly to report "still waiting", spending about
280,000 tokens and 460 tool calls, most of it on no information.
**Enhancement:** the prompt template should instruct the spawned agent to
*end its turn* when blocked on a long job, and say that the dispatcher will
resume it. Add: "a wake-up that reports no new information is a defect."

**(d) Nothing about a SHARED, RESOURCE-BOUND machine.** This machine has
15 GB RAM and a 2048 MiB card. Two agents nearly collided; one drove swap
to 80%.
**Enhancement:** a section on running heavy jobs -- check `free -g` first
and report it, serialise rather than parallelise anything memory-hungry,
never start a multi-hour run on an unproven configuration, and prefer a
cheap smoke run first. `12.bench-agent` launched three separate 20-minute
runs on configurations that failed in the first 60 seconds.

**(e) Parallelism advice is too eager for this class of work.** The skill
says to launch disjoint mechanical pieces together. That is right for
file-disjoint edits and wrong for anything that measures time or memory,
where a concurrent run contaminates the result.
**Enhancement:** state the exception plainly -- *measurement runs are never
parallel, even when they touch disjoint files.*

### `plain-english` -- scope is narrower than the project needs

Its description is about explaining work *to the user*. `CLAUDE.md` has
had to extend it by hand to cover comments and docstrings.
**Enhancement:** fold the code-comment rules into the skill itself
(comments say what the code does and why, in words a reader outside the
project can follow; code stays exact -- real names, paths, error text,
numbers; simplify the words, not the facts). Then `CLAUDE.md` can point at
the skill instead of restating it, and the two cannot drift.

## 1.3 Skills that were NEEDED and are MISSING

### (i) `experiment-protocol` -- the highest-value gap

**What happened without it.** The single most dangerous defect of the
whole session was nearly shipping a comparison table in which the two
sides were scored differently. `evaluator`'s `n2v1m` protocol defaults to
`max_pairs=80,000`, but the archived node2vec run passed `--lp-pairs 25000`
and `bench_node2vec_1M.py:220` doubles it to **50,000**. Scoring fodiwalk
at the default would have produced two unrelated numbers in one table with
nothing to signal it. Separately, `bench_fodiwalk.py` has its OWN
`--lp-pairs` flag meaning something different, on a different protocol
(`fodiwalk_dist`). Catching this cost a careful manual read of three files.

**What it should do.** Before any A-against-B table: identify the protocol
each side used; diff the resolved configuration, not the flag names;
require every reported row to carry its `protocol_modified` flag; refuse to
place two rows in one table when their protocols differ; and force a
statement of which knobs are shared and which are not.

**Description:** *"Use before comparing one method against another, or
against a recorded baseline. Resolves and diffs the evaluation protocol on
both sides, catches same-name-different-meaning flags, and requires every
row of a comparison table to declare the protocol it was produced under."*

### (ii) `measure-before-optimising`

**What happened without it.** This session did it by instinct and it paid
off enormously -- profiling first showed the peak was 6.2x the resting size
and named `with_all_neighbours` as +1707 MB of it, which pointed straight
at a global `argsort` over already-sorted data. But the instinct was not
uniform: the initial estimate of dim-128 GPU feasibility ignored
SELL-C-sigma padding and was wrong by enough to cost a failed run.

**What it should do.** Before a performance change: measure the current
cost and the theoretical floor; name the single largest contributor; state
what the change should save and check the result against that prediction.

**Description:** *"Use before any performance or memory change. Requires a
baseline measurement, an explicit floor, and a named dominant cost before
code is touched, then checks the achieved saving against the prediction."*

### (iii) `jax-runtime` (or a broader `accelerator-budget`)

**What happened without it.** Three separate runs died on
`RESOURCE_EXHAUSTED` before the cause was found. The failing allocation was
exactly `1,134,890 x 64 x 4` -- one full `(n, d)` array -- and the real
obstacle was that JAX preallocates 75% of VRAM by default, capping the pool
near 1536 MB when the run needed 1984 MB. `XLA_PYTHON_CLIENT_PREALLOCATE=
false` did NOT help; `MEM_FRACTION=0.95` did. This cost about an hour and
produced a wrong intermediate conclusion ("the card is too small") that was
relayed to the user before being corrected.

**What it should do.** Know that JAX preallocates; know the difference
between `PREALLOCATE=false` and `MEM_FRACTION`; know how to compute the
device floor from array shapes; know that a failed allocation whose byte
count equals a known array shape names its own cause.

**Description:** *"Use when a JAX or XLA run fails on device memory, or
before sizing a run to a known card. Computes the device floor from array
shapes, explains the preallocation defaults, and maps a RESOURCE_EXHAUSTED
byte count back to the array that caused it."*

### (iv) `long-run-campaign`

**What happened without it.** Multi-hour campaigns were assembled ad hoc.
Serialisation had to be hand-built (a `pgrep` wait loop), progress had to
be hand-parsed, and one campaign lost 6 of 21 runs to OOM that a
pre-flight memory estimate would have predicted -- `roadnet_ca` and
`ncbi_taxonomy` at dim 128 need `Z` of 960 MB and 1434 MB on a 2048 MiB
card, which is arithmetic available before the run, not after.

**What it should do.** Enumerate runs with estimated time and peak memory
from prior measurements; refuse or flag a run whose estimate exceeds the
budget; serialise strictly; write one row per run so a failure leaves the
rest intact; report progress without polling.

**Description:** *"Use to plan and run a multi-hour benchmark campaign.
Estimates time and memory per run from prior measurements, flags runs that
will not fit, serialises execution, and keeps per-run results durable
against mid-campaign failure."*

### (v) `finding-vs-artefact`

**What happened without it.** Several near-misses where a number looked
like a result and was not. The badly diverged momentum run scored the
HIGHEST hop R2 on cora (0.282) purely because its embedding had blown up to
`max|Z| = 6e10`. The `velocity` and `plain` optimisers differed by 0.0063
on one seed, which was reported as a tie only because the recorded
three-seed spread (0.039-0.046) happened to be at hand. Without that check
a 1% difference would have become a recommendation.

**What it should do.** Before a claim: check whether the effect exceeds
known run-to-run variance; check whether a winning metric is explainable by
a degenerate solution; require the number of seeds to be stated.

**Description:** *"Use before reporting that one configuration beats
another. Checks the effect against known seed variance, tests whether a
favourable metric is an artefact of a degenerate solution, and requires the
seed count to appear beside the claim."*

---

# PART 2 -- AGENTS

## 2.1 Agents used, and helpful

All three delegations used the generic `claude` agent type with an explicit
model override.

### `claude` at `model: fable` -- `10.mem-agent` (the memory rewrite)

The strongest single piece of work delegated all session. Given a 270-line
brief it produced a correct, byte-exact rewrite: a row-blocked carrier
(`RowStats`), a plain `RowCSR` in place of `scipy.sparse`, a per-row merge
replacing a global sort, and a new memory gate. Peak RSS 3055 -> 1195 MB,
and the full 1.13M-node graph completed for the first time.

Notably it also **found a defect in its own brief's premise** -- it
diagnosed a redundant `counts.astype(np.float64)` on an array that was
already float64, which no one had asked it to look for -- and it
**rejected a shortcut**, declining to re-record `golden_baseline.json` when
an `indptr` dtype changed, fixing the dtype instead.

### `claude` at `model: fable` -- `12.bench-agent` (the 1M benchmark)

Mixed, but its good work was very good. It found the `max_pairs=50,000`
protocol trap independently, and its GPU diagnosis was the best analysis of
the session: it identified that the failing allocation equals
`1,134,890 x 64 x 4`, that `step_plain` never chunks, and that the floor is
therefore 831.2 MB regardless of `--chunks`. All of that was verified and
held. Its failures are covered in 2.2.

### `claude` at `model: sonnet` -- `11.docs-agent` (the stage contract)

Correctly sized to the task. Updated three documents, stayed strictly in
scope (verified by `stat`: only those three files were written), preserved
history rather than deleting superseded sections, and reported a tool
restriction it had worked around instead of hiding it.

## 2.2 Agents used that need enhancement

The generic `claude` agent is a blank slate; every behaviour below had to
be supplied by prompt, and each failure is a missing default.

**(a) It does not verify its own configuration before a long run.**
`12.bench-agent` ran a 20-minute job on the wrong pair policy, then two
more on GPU configurations that failed within 60 seconds.
**Enhancement:** default behaviour of echoing the resolved configuration
and running a minimal smoke case before any job over a few minutes.

**(b) It polls instead of ending its turn.** Roughly 280,000 tokens spent
on "still waiting" wake-ups.
**Enhancement:** when blocked on a background job with no new information,
end the turn.

**(c) It does not checkpoint against its own death.** Two agents died on
session limits mid-verification, leaving the tree in an unknown state.
**Enhancement:** append a durable one-line state record after each write.

**(d) It treats a blocked tool as an obstacle to route around.**
`12.bench-agent` was blocked from writing `REPORT_1M.md` by the `Write`
tool and used a Bash heredoc instead. The content was legitimate, and it
did report the workaround when asked -- but the default should be to
surface a block, not to bypass it.
**Enhancement:** a blocked tool is reported and permission requested; it is
never routed around silently.

## 2.3 Agents that were NEEDED and are MISSING

### (i) `evaluator` -- EXISTS and was NOT used. This is a gap in my usage, not in the roster

`.claude/agents/evaluator.md` defines an agent described as the owner of
the `evaluator/` package, for *"any change, bug, parity question, protocol
question ... Also use when a number produced by `evaluator` needs to be
defended, reproduced, or compared against a recorded baseline."*

That is a precise description of the `n2v1m` `max_pairs` question, which
was instead resolved by hand across three files. It should have been the
first call. **No enhancement needed; it needs to be USED.** The lesson for
the next session is to check `.claude/agents/` before delegating to a
generic agent.

### (ii) `benchmark-runner` -- missing

**How it would have helped.** Campaign assembly, serialisation, pre-flight
memory estimation and result collection were all hand-built this session,
and the one thing not hand-checked -- per-run memory feasibility -- cost 6
of 21 runs.

**Description:** *"Owns `experiments/`. Plans and runs benchmark campaigns
on a resource-bound machine: estimates time and peak memory per run from
recorded measurements, refuses runs that will not fit, executes strictly
sequentially, and writes durable per-run results. Never runs two
measurement jobs at once."*

### (iii) `claim-verifier` -- missing

**How it would have helped.** Independent verification was the highest-
value activity of the session and was done entirely by the dispatcher: it
caught the wrong-policy benchmark, the premature "card too small", the
single-block coverage gap in `golden.py` (block size 20,000 against cora's
2,708 nodes, so the multi-block merge path was never exercised), and six of
my own wrong claims. It is also the thing most likely to be skipped under
time pressure.

**Description:** *"Given a claim and the artefact that supports it,
re-derives the result independently -- re-runs the gate, re-reads the diff,
re-measures the number -- and reports agreement or the exact difference.
Never accepts a summary as evidence."*

### (iv) `repo-historian` -- missing

**How it would have helped.** Much of this session's value came from
recorded runs scattered across `experiments/fdwalk/results/`,
`experiments/large-graph-node2vec/`, `so-far.md` and `CATALOG.md` -- 299
recorded optimiser runs, the const-versus-decay ladder, the three-seed
variance that decided the optimiser verdict. Finding these took repeated
ad-hoc greps, and one earlier session had already mis-attributed a change
because it searched only `agentic-log/` while the log lived elsewhere.

**Description:** *"Searches the recorded experimental history of this
repository -- logs, result tables, catalogs and reports -- and answers what
has already been measured, under which configuration, and where the raw
evidence is. Use before running an experiment, to find out whether it has
already been run."*

---

# PART 3 -- THE SINGLE HIGHEST-VALUE CHANGE

If only one thing is adopted: **add configuration echo-back and a cheap
smoke run to `agentic-development`.**

Two of the session's three worst incidents -- the wrong-policy benchmark
and three failed GPU launches -- were long runs started on unverified
configurations. Both would have been caught by one line of output before
the expensive work began. It is cheaper than every other recommendation
here and it addresses the failure mode that recurred most.

# PART 4 -- WHAT WORKED AND SHOULD NOT CHANGE

* **Independent verification of every delegated claim.** It caught a wrong
  benchmark, a wrong conclusion, a coverage gap in the gate itself, and six
  of my own errors.
* **Recording corrections as corrections.**
  `experiments/fodiwalk-streaming/FINDINGS.md` section 11 lists six claims
  stated confidently and wrongly. That section is worth more to the next
  session than the wins.
* **Tagging before changing.** `fodiwalk-pre-stagemove`,
  `fodiwalk-pre-memory`, `fodiwalk-post-memory` made every result
  recoverable and every comparison reproducible.
* **Verifying scope by `stat`, not by trust.** Confirming which files an
  agent actually wrote, rather than which it said it wrote.
