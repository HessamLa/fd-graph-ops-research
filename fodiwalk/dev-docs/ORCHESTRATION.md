# fodiwalk -- Orchestration

Created 2026-08-18T21:20:00-07:00. Companion to `PRD.md`, which is the
authority on what each block must contain and what it must satisfy. This
document says WHO builds each block, in what order, and how their output is
judged.

## 1. Rules that govern every subagent

**A subagent starts cold.** It has none of the campaign's context. Every
brief therefore states the invariant it must not break, quoted, rather than
pointing at it. The four silent defects of 2026-08-17..18 all came from the
plane/law coupling, and a cold agent will reproduce them unless told.

**Parity is the gate, not review.** A block is judged by the numbers of
`PRD.md` section 8, not by whether its code reads well. A refactor that
changes a number has failed even when the code is better.

**Nothing is deleted.** `experiments/fodiwalk/` keeps working throughout. The
new package is additive until the parity table passes.

**Model selection.** `sonnet` for a mechanical move with a clear target;
`opus` where a contract has to be preserved through a restructuring, or
where a defect would be silent.

**The redo loop.** After each agent returns: run its success criteria; if
any fails, do NOT patch it myself. Rewrite the brief to name the specific
failure and the specific expectation, and send it back to the same agent
with `SendMessage` so it keeps its context. Escalate `sonnet` to `opus`
after two failed attempts on the same criterion.

## 2. Dependency order

```
    W1  A1 core/csr.py            A6 misc/                (parallel, no deps)
        |                             |
    W2  A2 core/sell_c_sigma + forces + plan_contract      (needs A1)
        |
    W3  A3 core/force_directed    A4 augment_graph/mechanical  (needs A2 / A1)
        |                         A5 augment_graph/walks.py
        |
    W4  A7 fodiwalk.py Fodiwalk                                (needs A2,A3,A4,A5,A6)
        |
    W5  A8 rewire experiments + parity                     (needs A7)
```

`core` imports nothing from the package except `core`. The dependency runs
one way and any cycle is a defect.

## 3. Assignments

---

### A1 -- `core/csr.py` and `make_graph/` -- **sonnet**

**Goal.** Move `fodined/core/csr.py` verbatim to `fodiwalk/core/csr.py`, and
`experiments/fodiwalk/datasets.py` to `fodiwalk/make_graph/datasets.py`. Add
the `__init__.py` files. Change imports only.

**Must not.** Change any function body. "Tidy" a docstring. Add a feature.

**Success criteria.** PRD B1 and B5.
1. `row_of` matches `np.repeat(np.arange(n), np.diff(indptr))` on 20 random
   CSRs including ones with empty rows.
2. `load("cora")` gives `n = 2708`, symmetric, zero diagonal, sorted
   indices.
3. `fodiwalk/core/csr.py` imports nothing from `fodiwalk`.

**Judged by.** A diff against the source that shows import lines only.

---

### A2 -- `core/sell_c_sigma.py`, `core/forces.py`, `core/plan_contract.py` -- **opus**

**Goal.** Move the plan builder and the force laws, and write the ONE new
module of this project: the contract asserter.

**Why opus.** This block is where every silent defect of the campaign
lived. The laws unpack their planes POSITIONALLY and nothing checks the
count, the order or the meaning. Moving them apart without a registry would
make that worse.

**Must build the registry.**

```python
FORCE_PLANES = {
    "v1":               ("shell_coeff", "h"),
    "v2":               ("shell_coeff", "h"),
    "fdlinear":         ("h", "freq"),
    "fdlinear_3plane":  ("shell_coeff", "h", "freq"),
    "fdlinear_fused":   ("w",),
}
```

and `plan_contract.check(law_name, planes, D)` must raise -- never warn,
never fall back -- when the count or the shape disagrees.

**The invariants to assert, quoted from PRD section 7:** I1 planes are
`(D.nnz,)` and aligned to `D.indices`; I2 the count and ORDER match the
registry; I3 a pad cell has every plane at 0.

**Success criteria.** PRD B2 and B3.
1. `make_plan` raises `ValueError` on a plane whose shape is not `(D.nnz,)`.
2. `pad_frac` on the recorded com_youtube plan equals
   `0.15407662320379392`.
3. `fdlinear_fused` reproduces `fdlinear` on Cora: `||dZ|| = 0.5016`,
   `auc = 0.9962`, `r2_dist = 0.253`.
4. `plan_contract.check("fdlinear", (shell, h), D)` RAISES -- this is the
   2026-08-18 defect, where a missing `freq` made `fdlinear` read
   `shell_coeff` as `h` and the run diverged to NaN with no error.
5. A pad cell contributes exactly 0 for a law with a constant term.

---

### A3 -- `core/force_directed.py` -- **opus**

**Goal.** `Callback_Base` and `class ForceDirected` with exactly the
methods the brief names: `forces`, `updateGradient`, `attach_callback`,
`notify_callback`, `get_embeddings`, `get_embeddings_df`, `Th`, `updateZ`,
`embed`. Source is `fodined/core/fodined.py` plus the `Fodiwalk` overrides in
`bench_fdwalk.py` lines 486-600.

**Why opus.** The chunking logic and the batching interact: `embed` slices
`dZ` by the same row range the plan chunk uses, and the two must stay the
same object. Getting that wrong gives a `dZ` that is silently summed twice.

**The invariant to preserve, quoted:** a chunk is a ROW RANGE, never an
arbitrary set of pairs, because `_step` writes `dZ.at[rows].add(...)` and
only disjoint rows make the parts additive.

**Success criteria.** PRD B4.
1. A recording callback sees `train_begin`, then per epoch `epoch_begin`,
   per batch `batch_begin`/`batch_end`, `epoch_end`, then `train_end`.
2. `embed(batch_count=k)` equals `embed(batch_count=1)` to float32 rounding.
3. `updateZ` dispatches through `misc/optim.py`, default `plain`.
4. A non-finite `Z` is RECORDED and returned, never raised through an
   evaluator (invariant I7).

---

### A4 -- `augment_graph/` mechanical parts -- **sonnet**

**Goal.** Move `pairs.py` (`cap_per_node`, `row_cap`, `to_csr`,
`to_csr_directed`, `split_key`), `weights.py`, `buckets.py`,
`far_pairs.py` (from `fodined/graph_augmentation.py`: `degree_table`,
`sample_far_pairs`), `landmarks.py`. Imports only.

**The invariant to preserve, quoted:** a stored weight is an INTEGER in
`[1, window]` and weight 1 means adjacency. A continuous weight gives every
pair its own shell, `degrees_from_D` returns 0 for every row, and the whole
force vanishes with no error -- AUC fell to 0.55 with `||dZ|| = 0.000`.

**Success criteria.** PRD B6 items 4, 5, 6.
1. Every weight rule returns integers in `[1, window]`.
2. `row_cap(m)` gives max row width exactly `m` and exactly `n*m` entries.
3. `sample_far_pairs` rejects a pair stored in EITHER direction.

---

### A5 -- `augment_graph/walks.py` -- **opus**

**Goal.** Move the walk functions: `uniform_walks`, `node2vec_walks`,
`make_walker`, `walk_pair_stats`, `walk_rows`, `with_all_neighbours`,
`with_neighbours_low_deg`, `row_buckets`.

**Why opus.** Three subtleties that a mechanical move loses.

1. `make_walker(A, n, 1.0, 1.0)` must return `uniform_walks` ITSELF, not
   the rejection sampler with trivial weights, so a default run reproduces
   every earlier number BIT-EXACTLY rather than only distributionally.
2. `walk_rows` has NO prune BY DESIGN -- its bound IS `n_walks*walk_len`
   per row. `walk_pair_stats` DOES prune and its prune is an
   approximation. Do not "unify" them.
3. `_pairs_of` collapses direction at the source with `min*n + max`. Any
   attempt to make it directed doubles the accumulator, which is the
   structure that OOM-killed three runs at 1.13M nodes.

**Success criteria.** PRD B6 items 1, 2, 3.
1. `make_walker(A, n, 1, 1)` returns `uniform_walks` itself; `walk_rows`
   with it produces a BIT-IDENTICAL key array to the default.
2. `node2vec_walks` at `p=q=1` matches `uniform_walks` distributionally
   within the seed spread (it is NOT bit-exact: the rejection loop draws
   one extra number per accept test).
3. All four of `p<1, p>1, q<1, q>1` move the mean distinct-nodes-per-walk
   in the direction node2vec predicts.

---

### A6 -- `misc/` -- **sonnet**

**Goal.** Move `optim.py` (all eight rules), `fodined/embedding/drop.py`,
and build `evaluation.py` from `fodined/link_prediction.py` plus
`hop_sample` and `task_hop` in `bench_fdwalk.py`.

**Success criteria.** PRD B7.
1. All eight rules run 20 epochs on Cora and return a finite `Z` or record
   a divergence.
2. `state_arrays("sqn", 3) == 8`, `state_arrays("plain") == 0`.
3. `link_prediction` caps at `max_pairs` unique pairs and never samples a
   stored edge as a negative.

---

### A7 -- `fodiwalk/fodiwalk.py` -- **opus**

**Goal.** `class Fodiwalk(ForceDirected)` with `make_graph` (stub),
`graph_walk`, `augment_graph`, `embed`, `fit` (raises).

**Why opus.** This is where the 896-line script's branching becomes an API.
The plane list must come from A2's registry, never from an `if` chain --
the whole point of the refactor.

**Success criteria.** PRD B8.
1. `augment_graph` builds planes from `FORCE_PLANES` and RAISES when a
   required plane is unavailable.
2. `graph_walk` reproduces `walk_rows` for the same seed.
3. `fit` raises `NotImplementedError`.

---

### A8 -- rewire the experiments, and prove parity -- **opus**

**Goal.** Change `experiments/fodiwalk/bench_fdwalk.py` to import `Fodiwalk`
from the package. Keep every command-line flag and the RESULT line format
byte-for-byte, so the existing drivers, `update_results.py`,
`grid_table.py` and the `results` session keep working untouched.

**Success criteria -- the parity table of PRD section 8**, which is the
gate for the whole project:

| # | Scenario | Must reproduce |
| --- | --- | --- |
| P1 | Cora `nbr_walk/min_gap/fdlinear` d64 200ep lr1.0 s42 | `dz=0.5016 auc=0.9962 r2_dist=0.253` |
| P2 | Cora `walk/min_gap/v1/plain` d64 2000ep lr0.999 s42 | `auc=0.9986 r2_dist=0.523` |
| P3 | Cora `walk/buckets+far/fdlinear` 3 seeds | `r2_dist = 0.636 ±0.024` |
| P4 | Cora `nbr_walk p=2.0` 3 seeds | `r2_dist = 0.452 ±0.016` |
| P5 | `row_cap(16)` com_youtube | width 16, `n*16` entries |
| P6 | eight optimizers, Cora 60ep d32 lr1.0 | the 2026-08-18 ranking |

P1, P2 and P5 must be EXACT. P3, P4 and P6 must fall inside the recorded
seed spread.

---

## 4. How I evaluate each return

For every agent, in this order:

1. **Run its success criteria as code.** Not read the summary -- run them.
   An agent's report of its own success is evidence, not proof.
2. **Diff against the source.** For a `sonnet` move, the diff must be
   import lines. Anything else is investigated before it is accepted.
3. **Check the invariant it was told to preserve.** I1..I7 of PRD section 7.
4. **On failure:** rewrite the brief naming the exact criterion that failed
   and the exact expected value, and `SendMessage` it back to the same
   agent so its context survives. Escalate `sonnet` to `opus` after two
   failures on one criterion.
5. **Nothing merges until A8's parity table passes.** A green block with a
   red parity table is not done.

## 5. Definition of done

1. `from fodiwalk import Fodiwalk` works from the repository root.
2. All six parity scenarios reproduce.
3. `experiments/fodiwalk/bench_fdwalk.py` runs on the package with unchanged
   flags and an unchanged RESULT line.
4. `fodiwalk/tests/` passes.
5. `experiments/fodiwalk/` is untouched and still runnable, so the campaign
   is never blocked by the refactor.

---

## 6. The SECOND orchestration -- the split of the god class, 2026-08-20

Section 3 records how the package was BUILT. This section records how it
was RESHAPED. The plan is [REFACTOR.md](REFACTOR.md): ten defects, a target
tree, thirteen traps, ten gates G1-G10 and five milestones M0-M4.

**The difference from the first orchestration.** The first one had a
parity table as its gate and it built new code from a script. This one
moves EXISTING code and must not change one number, thus the gate had to
be cheap enough to run between two edits. That gate is `tests/golden.py`,
and the master wrote it BEFORE any agent was dispatched.

### 6.1 M0 -- the master, before any dispatch

| what | why |
| --- | --- |
| `tests/golden.py` and `golden_baseline.json` | the byte-exact record of 16 augment cases and 4 CPU embed runs, taken from the tree as it was. A refactor compares against it, and not against itself |
| the commit `89abaf2`, tag `fodiwalk-pre-refactor` | the revert point. The package was untracked in git until then |
| `dev-docs/REFACTOR.md` | the brief every agent reads |
| `tests/check_api.py`, gate G3 | written by a peer session and verified by the master. It planted three ghost names as a negative control, and the check reported them |

**One DEADEND is recorded and it shaped the gate.** A sha1 of `Z` is too
sharp a tool: the same run differs in the last bit between two processes on
the CPU backend. The embed half therefore compares five scalars at
`rtol = 1e-4`, and the augment half -- pure NumPy -- stays byte exact.

### 6.2 The five agents

| agent | owns | model | gate |
| --- | --- | --- | --- |
| A1 `01.augment-agent` | `augment_graph/`: `result.py`, `policies.py`, `policy_walk.py`, `policy_buckets.py`, `policy_nbr_walk.py`, `merge.py` | opus | its own equivalence test against the old methods on the 16 golden cases, plus G1 unchanged |
| A2 `02.embed-agent` | `embed/`: `planes.py`, `degrees.py`, `planner.py`, and `core` publishes `step` | opus | the same, on the plane, degree and plan seams |
| A3 `03.integration-agent` | `config.py`, `model.py`; `fodiwalk.py` and `models/` deleted | opus | G1 to G8 |
| A4 `04.structure-agent` | `tests/`: the new shape as a TEST | opus | G4, G5, G6, G7, G8 as tests |
| A5 `05.docs-agent` | `CATALOG.md`, `README.md`, `BUILD.md`, `ORCHESTRATION.md` | opus | G9, and every snippet it writes must RUN |

The master log records the model at each dispatch. A4 and A5 were
dispatched together at M3, and A5 wrote this section while A4 ran; correct
the A4 row from the master log if it differs.

### 6.3 The structure -- what ran in parallel, and what could not

```
M0  master        golden.py, the tag, REFACTOR.md, check_api.py
                              |
M1  A1 -----------------------+---------------------- A2      PARALLEL
    augment_graph/            |                       embed/
    NEW FILES ONLY. fodiwalk.py stays and still runs the package.
                              |
M2  A3                        |                                SEQUENTIAL
    wires both, deletes the god class. The gates then say
    whether the wiring is right.
                              |
M3  A4 -----------------------+---------------------- A5      PARALLEL
    the structure tests       |                       the documents
                              |
M4  master        independent re-run of G1-G9, and the GPU gate G10
```

**Why M1 is parallel and M2 is not.** A1 and A2 got DISJOINT files and no
shared state. Each one proved its new code EQUALS the old method, on the
same 16 golden cases, while the old method was still there to compare
against -- 17 equivalence tests for A1 and 66 for A2. Nothing was wired,
thus neither agent could break the other or the package. M2 is one agent
because deleting the god class is one atomic step: it is the step where the
refactor lands or breaks.

**Why M3 is parallel.** The tests and the documents read the same tree and
write different files. Neither changes a `.py` file of the package.

### 6.4 How the master judged a return

The rule of section 4 held: **an agent's report of its own success is
evidence, and not proof.** Every claim was re-run by the master.

| the claim | the master's verification |
| --- | --- |
| A2: 66 equivalence tests pass, `core` touched only to publish `step` | re-ran: 66 passed, `GOLDEN OK`. Read `git diff fodiwalk/core/`: a rename plus a `_step = step` alias, and nothing else |
| A1: 17 equivalence tests, 136 in the suite, the far merge written once | re-ran: 136 passed, `GOLDEN OK`, `check_api` OK, `fodiwalk.py` untouched. Read `merge.py`: the COO order kept and documented |
| A3: the wiring is complete and the behaviour is unchanged | re-ran after the delete: `GOLDEN OK (both)`, `53 passed, 2 skipped, 9 deselected`, `check_api` exit 0, `model.py` 200 lines, no dispatch chain in `model.py` |

**Two open items came out of the returns and both are recorded, not
hidden.** `augment_graph/walks.py` is 422 lines and breaks the 300-line
rule of G4; it is pre-existing and belonged to no agent of this
orchestration. And REFACTOR.md section 5.1 said `Config` had 41 fields; the
tree and the tag both hold 39, thus the DOCUMENT was wrong and it was
corrected. `CATALOG.md` section 19.10 lists every place where the code and
REFACTOR.md disagree.

### 6.5 Definition of done, for the split

1. `GOLDEN OK (both)` -- the augment half BYTE EXACT.
2. The suite passes with no failure and no error.
3. `check_api` exits 0: `Config`, the methods, the attributes and the key
   sets did not move.
4. `fodiwalk/fodiwalk.py` and `fodiwalk/models/` are gone, and no module
   imports a `_private` name of another module.
5. `CATALOG.md` holds a NEW numbered section, and nothing was removed from
   it.
6. The GPU parity gate G10, by the master, at the end.
