# fdwalk -- Orchestration

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

**Nothing is deleted.** `experiments/fdwalk/` keeps working throughout. The
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
    W4  A7 fdwalk.py FDWalk                                (needs A2,A3,A4,A5,A6)
        |
    W5  A8 rewire experiments + parity                     (needs A7)
```

`core` imports nothing from the package except `core`. The dependency runs
one way and any cycle is a defect.

## 3. Assignments

---

### A1 -- `core/csr.py` and `make_graph/` -- **sonnet**

**Goal.** Move `fodined/core/csr.py` verbatim to `fdwalk/core/csr.py`, and
`experiments/fdwalk/datasets.py` to `fdwalk/make_graph/datasets.py`. Add
the `__init__.py` files. Change imports only.

**Must not.** Change any function body. "Tidy" a docstring. Add a feature.

**Success criteria.** PRD B1 and B5.
1. `row_of` matches `np.repeat(np.arange(n), np.diff(indptr))` on 20 random
   CSRs including ones with empty rows.
2. `load("cora")` gives `n = 2708`, symmetric, zero diagonal, sorted
   indices.
3. `fdwalk/core/csr.py` imports nothing from `fdwalk`.

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
`embed`. Source is `fodined/core/fodined.py` plus the `FDWalk` overrides in
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

### A7 -- `fdwalk/fdwalk.py` -- **opus**

**Goal.** `class FDWalk(ForceDirected)` with `make_graph` (stub),
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

**Goal.** Change `experiments/fdwalk/bench_fdwalk.py` to import `FDWalk`
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

1. `from fdwalk import FDWalk` works from the repository root.
2. All six parity scenarios reproduce.
3. `experiments/fdwalk/bench_fdwalk.py` runs on the package with unchanged
   flags and an unchanged RESULT line.
4. `fdwalk/tests/` passes.
5. `experiments/fdwalk/` is untouched and still runnable, so the campaign
   is never blocked by the refactor.
