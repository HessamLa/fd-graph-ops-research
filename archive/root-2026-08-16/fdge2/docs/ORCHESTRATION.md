# fdge2.0 — Orchestration Document

Companion to `RPD.md`. Defines the agents building fdge2, what each owns,
which model tier runs each, and how work is sequenced so the three-stage
modularity (see `ARCHITECTURE.md`, `RPD.md` §5) is enforced rather than
eroded by whoever happens to be implementing a given piece.

## 1. Principle: complexity decides the tier, not the module

Tier assignment follows task complexity from `RPD.md` §8, not which stage
a task belongs to:

- **Opus** — tasks that require resolving an open design question,
  touch the hot path (perf/memory-sensitive Numba kernels), or carry a
  correctness subtlety already known to have bitten this codebase once
  (e.g. `reduceat` on empty segments, `fastmath` nondeterminism, the
  dense-hop-fill `O(n^2)` trap). Getting these wrong is expensive to
  detect later.
- **Sonnet** — well-specified porting/implementation work where the
  target behavior is already fully described in the docs (a reference
  implementation exists to port from), plus test/benchmark harnesses
  that follow an already-documented methodology.
- **Haiku** — thin wrappers, re-exports, and doc synchronization with no
  judgment calls.

## 2. Agent Roster

| Agent | Tier | Owns (RPD task IDs) | Notes |
|---|---|---|---|
| **Lead/Architect** | Opus | T0.1 (core skeleton, Q1–Q3), reviews all other tasks | Only agent authorized to resolve open design questions or approve an import-boundary exception. Owns merge decisions. |
| **Core-Utils** | Sonnet | T0.2 (CSR/indexable utilities) | Well-specified port of existing `graph_to_csr` pattern; no open questions attached. |
| **Graph-Building** | Haiku | T1.1 (`make_graph` wrapper) | Pure re-export/thin adapter over `graphmaking/registry.py`. No new graph-construction logic. |
| **Augmentation** | Sonnet | T2.1 (dense hop-fill port) | Direct port of `get_hops_csr` + shell counts onto the `augment_graph` contract; reference behavior fully specified already. |
| **Sparse-Research** | Opus | T2.2 (sparse augmentation design/prototype, Q4/Q5) | Open-ended: no reference implementation exists yet, and the repulsion-collapse risk (Q4) needs real judgment, not porting. |
| **Embedding-Kernel** | Opus | T3.1, T3.2 (`forces`, `_scalar_force` port, `updateGradient` seam) | Hot path; perf/memory guardrails (RPD §3) are load-bearing and easy to silently violate. |
| **Validation** | Sonnet | T4.1, T4.2, T4.3 (correctness harness, regression parity, perf/memory benchmarks) | Methodology already fully worked out in `IMPLEMENTATION.md` — this is rigorous implementation, not open design. Escalates to Lead if a benchmark result contradicts a success metric. |
| **Docs-Sync** | Haiku | T5.1 (fold resolved questions back into architecture docs) | Mechanical: transcribe Lead's decisions into the existing docs. |

## 3. Sequencing

```
T0.1 (Lead, Opus)  ──┬──> T1.1 (Graph-Building, Haiku)
T0.2 (Core-Utils,    │
      Sonnet)  ──────┼──> T2.1 (Augmentation, Sonnet) ──> T2.2 (Sparse-Research, Opus)
                      │         │
                      └─────────┼──> T3.1 (Embedding-Kernel, Opus) ──> T3.2 (Embedding-Kernel, Opus)
                                │                                │
                                └──> T4.2 (Validation, Sonnet) <─┘
                                          T4.1, T4.3 (Validation, Sonnet), depend on T3.1 only
                                                                  │
                                                                  └──> T5.1 (Docs-Sync, Haiku)
```

Rules of the sequencing, not just a picture:

- **T0.1 blocks everything.** No stage-specific agent starts until the
  Lead has resolved Q1–Q3 and the core skeleton exists — otherwise
  `graph_building/`, `graph_augmenting/`, and `embedding/` agents are each
  guessing at a contract that isn't settled yet, which is exactly the
  cross-stage coupling this whole design exists to avoid.
- **T1.1 and T2.1 can run in parallel** once T0.1/T0.2 land — they don't
  depend on each other (per the import discipline in RPD §5, graph
  building doesn't feed augmentation's *code*, only its output value at
  runtime).
- **T2.2 (Sparse-Research) does not block T3.1.** The embedding stage is
  built and validated against the dense reference `D` first; sparse
  augmentation is a second, independent augmentation implementation that
  must satisfy the same `augment_graph` contract, not a prerequisite.
- **T4.2 (regression parity) needs both T2.1 and T3.1** — it's comparing
  the *whole* new pipeline against the old monolithic one.
- **T5.1 runs last**, after the Lead has actually ratified decisions
  arising from T2.2/T3.1 (not just the T0.1-time answers to Q1–Q3).

## 4. Import-Boundary Enforcement

This is a review gate, not an honor system:

- Every PR from **Augmentation**, **Embedding-Kernel**, or
  **Sparse-Research** is checked by the **Lead** against RPD §5 before
  merge: does this PR import anything sideways (`embedding/` importing
  `graph_augmenting/` internals) or backward (`core/` importing a stage
  package)?
- **Validation** is exempt (it's allowed to import everything) but its
  own code must not be imported *by* any of the four pipeline packages —
  it's a leaf, not a dependency.
- If an agent finds it *needs* a sideways import to get something done,
  that's treated as a signal the contract in `core/` is underspecified —
  the agent stops and escalates to the Lead (§5) rather than adding the
  import.

## 5. Escalation Rules

- **Sonnet/Haiku agents do not resolve open design questions.** If
  Augmentation, Graph-Building, Validation, or Docs-Sync hits an
  ambiguity that traces back to Q1–Q5 in `RPD.md` §6 (or a new one not
  yet listed), they stop and hand it to the Lead rather than guessing —
  guessing here is exactly how the sideways-coupling this design exists
  to prevent creeps back in.
- **The Lead is the only merge authority for `core/`.** Changes to the
  base `ForceDirected`/`FDModel` skeleton or the CSR/indexable contract
  ripple into every other module, so they get Opus-level review even if
  originally drafted by a Sonnet agent (e.g. Core-Utils' T0.2, which the
  Lead reviews against the contract T0.1 defines).
- **Sparse-Research escalates Q4 explicitly** rather than shipping a
  sparse `augment_graph` that quietly changes repulsion semantics —
  per RPD success metric #7, Q4 needs a written decision (or an explicit
  "deferred" note), not a silent implementation choice.

## 6. Definition of Done (per task)

A task is done when:

1. It satisfies its RPD §8 entry and the relevant success metric(s) in
   RPD §7 (cite which metric(s) in the PR description).
2. It passes the Lead's import-boundary check (§4).
3. For **Embedding-Kernel** and **Augmentation** tasks specifically: a
   correctness check exists (even a minimal one) before the Validation
   agent's harness is available — don't wait for T4.1 to discover a
   `reduceat`-style bug.
4. Any open question it touches (§6 of RPD) has either a written
   resolution or an explicit "deferred, tracked" note — not silence.
