# fdwalk -- Product Requirements Document

Created 2026-08-18T21:10:00-07:00.

## 1. Purpose

`experiments/fdwalk/` holds 2,902 lines of working research code in 15 flat
scripts, of which `bench_fdwalk.py` alone is 896 lines and mixes argument
parsing, augmentation, the embedding loop, evaluation and reporting. Every
result of the 2026-08-15..18 campaign came out of it, and none of it is
importable.

This project turns that code into a package, `./fdwalk/`, with the tree and
the usage of `./archive/root-2026-08-16/fdge_jax/`, and then rewires the
experiments to import `FDWalk` from it.

**This is a refactor with a parity requirement, not a redesign.** The
physics, the augmentation rules and the numbers do not change. A run of the
new package must reproduce a run of the old script to the digits recorded
in `experiments/fdwalk/RESULTS.md`.

## 2. Non-goals

* No new force law, augmentation policy, or optimizer.
* No change to any numeric result. A difference is a defect, not a finding.
* `make_graph` is a stub in this project. It exists in the API and returns
  the graph it is given.
* `fit` is not implemented. It raises `NotImplementedError`.
* No new experiment. The campaign continues on the existing scripts until
  parity is proved.

## 3. Reference structure

`archive/root-2026-08-16/fdge_jax/` is the model:

```
fdge_jax/core/csr.py                 row_of, n_rows
fdge_jax/core/force_directed.py      Callback_Base, the engine class
fdge_jax/embedding/shell_force.py    the force law
fdge_jax/graph_augmenting/*.py       hopfill, sparse_hops
fdge_jax/graph_building/*.py         strategies, registry
fdge_jax/models.py                   the user-facing class
fdge_jax/validation/*.py             parity and behaviour tests
```

## 4. Target tree

```
fdwalk/
  __init__.py            exports FDWalk
  fdwalk.py              class FDWalk: make_graph, augment_graph, embed,
                         fit, graph_walk
  core/
    __init__.py
    csr.py               row_of, n_rows
    force_directed.py    Callback_Base, class ForceDirected
    sell_c_sigma.py      build_ladder, make_plan, _step, PlanCache
    forces.py            the force laws and the plane builders
  make_graph/
    __init__.py
    datasets.py          read_edges, to_csr, induced, load
  augment_graph/
    __init__.py
    walks.py             the walk functions and the pair statistics
    pairs.py             cap_per_node, row_cap, to_csr, to_csr_directed
    weights.py           flat, min_gap, mean_gap, pmi
    buckets.py           bucket_sample, budget, row_buckets
    far_pairs.py         degree_table, sample_far_pairs
    landmarks.py         pick, distances, pair_distance
  misc/
    __init__.py
    optim.py             the eight update rules, RULES, state_arrays
    drop.py              drop_steady_rate, FallbackKeys
    evaluation.py        link_prediction, hop_sample, task_hop
  docs/
    PRD.md  ORCHESTRATION.md
  tests/
    test_parity.py  test_contracts.py  test_smoke.py
```

## 5. What else belongs in core -- the answer to the open question

The brief lists `csr`, `ForceDirected` and `sell_c_sigma` + `Callback_Base`.
**Two more belong in core, and one does not.**

### 5.1 `core/forces.py` -- BELONGS IN CORE (recommended)

It holds `shell_force`, `shell_force_v2`, `fdlinear`, `fdlinear_3plane`,
`fdlinear_fused`, `fuse`, and the plane builders `shell_counts`,
`shell_coeff_data`, `degrees_from_D`.

**Why core and not `misc/`.** The force law and the plan are one contract,
not two components: `make_plan` takes a tuple of planes and `_step` hands
that tuple to `force_fn`, which unpacks it POSITIONALLY. Nothing checks the
count, the order, or the meaning. **Four separate silent defects of the
2026-08-17..18 campaign came from exactly this coupling** -- a third plane
passed to a two-plane law, a two-plane law reading `shell_coeff` as `h`, a
missing `freq` falling back to different physics, a policy that emptied the
`h = 1` rows and froze them. Splitting the law from the kernel across a
package boundary would make that coupling harder to see, not easier.

### 5.2 `core/plan_contract.py` -- NEW, and it is the reason this refactor pays

A single module that asserts the invariants of section 7 at the seam. It is
new code, and it is the only new code in this project. It exists because
every defect above was silent: the run produced a number, not an error.

### 5.3 `misc/evaluation.py` -- DOES NOT belong in core

Link prediction and the hop-distance regression are measurement, not
engine. `fdge_jax` put them under `validation/`. They go in `misc/` per the
brief's description of `misc/`, and a later move to `fdwalk/eval/` is a
rename, not a redesign.

**Open question for the user:** the evaluation currently lives in
`fodined/link_prediction.py` and in `bench_fdwalk.py`. Options are (a) copy
it into `fdwalk/misc/evaluation.py`, (b) keep importing `fodined`, or (c)
make `fdwalk` depend on nothing outside itself. **This PRD assumes (c), a
self-contained package**, because a package that imports its predecessor is
not a module. Say so if (b) is wanted instead.

## 6. Blocks

Each block below has: contents, the public API it must expose, the contract
it owes its callers, and the success criteria that decide whether it is
done.

---

### B1. `core/csr.py`

**Contents.** `row_of(indptr)`, `n_rows(D)`. Verbatim from
`fodined/core/csr.py`.

**Contract owed.** `row_of` returns, for a CSR of `nnz` stored entries, an
array of length `nnz` whose entry `i` is the row that owns stored entry
`i`. It is the inverse of `indptr` and every plane builder depends on it.

**Success criteria.**
1. `row_of` matches `np.repeat(np.arange(n), np.diff(indptr))` on 20 random
   CSR matrices, including ones with empty rows.
2. No import from any other `fdwalk` module (it is the bottom of the tree).

---

### B2. `core/sell_c_sigma.py`

**Contents.** `build_ladder`, `_to_csr`, `make_plan`, `_step`, `PlanCache`.
Verbatim from `fodined/embedding/sell_c_sigma.py`.

**Contract owed -- the plane contract. This is the most defect-prone
interface in the codebase and it must be stated exactly.**

```
make_plan(D, planes, degrees, b_cells, k_max, ladder_base)
    -> (plan, inv_deg_ext, stats)

planes : a SEQUENCE of arrays, each of shape (D.nnz,), aligned 1:1 with
         D.indices in D's OWN pre-split, pre-sort CSR order. Each becomes
         one padded (nb, R, k) float32 tile per rung, IN THE ORDER GIVEN.
         make_plan never interprets a plane.
degrees: (n,) the per-row divisor. NOT derived here. "What counts as a
         degree" is a force-law question.
```

**The padding contract.** A pad cell carries every plane at exactly 0, and
its neighbour index is the row's own id, thus `x = 0` exactly. A force law
built from the planes must vanish there. The kernel guards it a second
time.

**The chunking contract.** A chunk is a ROW RANGE, never an arbitrary set
of pairs, because `_step` writes `dZ.at[rows].add(...)` and only disjoint
rows make the parts additive. The global quantities -- `shell_coeff_data`
and the force-law degree -- are computed over the WHOLE `D` and then
sliced. A degree counted on one chunk is not the degree of the node.

**Success criteria.**
1. `make_plan` raises `ValueError` when any plane's shape is not `(D.nnz,)`.
2. For a `D` with a row of width above `k_max`, `stats["n_split"] > 0` and
   the reconstructed row set equals the original.
3. `pad_frac` on the com_youtube plan equals 0.15407662320379392, the value
   recorded on 2026-08-17.
4. A pad cell contributes exactly 0 to `dZ` for a force law with a constant
   term.

---

### B3. `core/forces.py`

**Contents.** The laws `shell_force`, `shell_force_v2`, `fdlinear`,
`fdlinear_3plane`, `fdlinear_fused`; the host-side builder `fuse`; the
plane builders `shell_counts`, `shell_coeff_data`, `degrees_from_D`.

**Contract owed -- the law/plane table.** Every law declares the planes it
reads, in order, and the registry carries it:

| law | planes, in order | attraction |
| --- | --- | --- |
| `shell_force` (v1) | `(shell_coeff, h)` | `h == 1` only |
| `shell_force_v2` | `(shell_coeff, h)` | `h == 1` only |
| `fdlinear` | `(h, freq)` | `h <= 1` only |
| `fdlinear_3plane` | `(shell_coeff, h, freq)` | `h <= 1` only |
| `fdlinear_fused` | `(w,)`, `w = -1` at `h=1`, `h/freq` at `h>=2` | `w < 0` |

**Success criteria.**
1. A registry `FORCE_PLANES` maps each law name to its plane tuple, and
   `FDWalk` builds the plane list FROM the registry, never from a branch.
2. `fdlinear_fused` reproduces `fdlinear` to float32 rounding on Cora:
   `||dZ|| == 0.5016`, AUC `0.9962`, hop R2 `0.253` (2026-08-17 record).
3. `shell_counts` never allocates a table wider than the number of DISTINCT
   values in `D.data`.
4. `degrees_from_D` returns the count of `D.data == 1` per row.

---

### B4. `core/force_directed.py`

**Contents.** `Callback_Base` and `class ForceDirected`, with the methods
the brief names: `forces`, `updateGradient`, `attach_callback`,
`notify_callback`, `get_embeddings`, `get_embeddings_df`, `Th`, `updateZ`,
`embed`.

**Contract owed.** `ForceDirected` owns the epoch loop, the batching, the
callback events and the `Z` update. It owns NO augmentation and NO force
law: `forces` dispatches to a law supplied by the caller, and
`updateGradient` fills `dZ` for a row range.

Callback events, in order: `train_begin`, then per epoch `epoch_begin`,
per batch `batch_begin` / `batch_end`, then `epoch_end`, then `train_end`.

**Success criteria.**
1. A recording callback observes exactly that event order for
   `epochs=3, batch_count=2`.
2. `Th(dZ)` returns the mean row norm and matches the old value.
3. `embed` with `batch_count=k` gives a `dZ` equal, to float32 rounding, to
   `batch_count=1` on the same `D`.
4. `updateZ` dispatches through `misc/optim.py` and defaults to `plain`.

---

### B5. `make_graph/`

**Contents.** `datasets.py` -- `read_edges`, `to_csr`, `induced`, `load`.

**Contract owed.** `load(name)` returns `(A, n)` where `A` is a symmetric
`scipy.sparse.csr_matrix` with no self loops and sorted indices, and `n` is
its row count. Datasets live under `data_cache/`.

**Success criteria.**
1. `load("cora")` gives `n = 2708`; `load("pubmed")` gives `n = 19717`;
   `load("com_youtube")` gives `n = 1134890` with `max degree 28754`.
2. `A` is symmetric, has a zero diagonal and `has_sorted_indices`.

---

### B6. `augment_graph/`

**Contents.** `walks.py` (`uniform_walks`, `node2vec_walks`, `make_walker`,
`walk_pair_stats`, `walk_rows`, `with_all_neighbours`,
`with_neighbours_low_deg`), `pairs.py` (`cap_per_node`, `row_cap`,
`to_csr`, `to_csr_directed`, `split_key`), `weights.py`, `buckets.py`,
`far_pairs.py`, `landmarks.py`.

**Contract owed -- three invariants, each of which has already caused a
silent defect.**

1. **The weight contract.** A stored weight is an INTEGER in `[1, window]`,
   and **weight 1 means adjacency**. A continuous weight gives every pair
   its own shell, `degrees_from_D` then returns 0 for every row,
   `inv_deg_ext` becomes 0.0, and the entire force vanishes with no error:
   AUC fell to 0.55 with `||dZ|| = 0.000`.
2. **The `h = 1` contract.** Attraction exists at `h = 1` only. Any policy
   that can leave a row with no `h = 1` entry MUST pass an explicit
   `degrees` array, or that row is frozen in silence.
3. **The budget contract.** `walk_rows` has NO prune: its bound IS
   `n_walks * walk_len` per row, thus the caller's walk budget is the
   memory bound. `walk_pair_stats` DOES prune, and its prune is an
   approximation that `prunes > 0` reports.

**Success criteria.**
1. `make_walker(A, n, 1.0, 1.0)` returns `uniform_walks` itself, thus a
   default run reproduces earlier keys BIT-EXACTLY.
2. `node2vec_walks` at `p = q = 1` matches `uniform_walks`
   DISTRIBUTIONALLY within the seed spread -- not bit-exactly, because the
   rejection loop consumes the stream differently.
3. `node2vec_walks` moves the mean distinct-nodes-per-walk in the direction
   node2vec predicts for all four of `p<1, p>1, q<1, q>1`.
4. `row_cap(m)` gives a maximum row width of exactly `m` and exactly
   `n * m` entries on a graph where every row has more than `m` candidates.
5. Every weight rule returns integers in `[1, window]`.
6. `sample_far_pairs` rejects a pair stored in either direction of `D`.

---

### B7. `misc/`

**Contents.** `optim.py` (eight rules, `RULES`, `state_arrays`), `drop.py`,
`evaluation.py`.

**Contract owed.** An update rule is
`step(Z, dZ, lr, state, epoch, **kw) -> (Z_new, state)`, pure in `Z` and
`dZ`, with its state in the dict the caller keeps.
`state_arrays(name, memory)` returns how many arrays of the shape of `Z`
the rule holds -- the number that decides what fits on the card.

**Success criteria.**
1. All eight rules run 20 epochs on Cora and return a finite `Z`, or record
   a divergence.
2. `state_arrays("sqn", 3) == 8` and `state_arrays("plain") == 0`.
3. `link_prediction` caps at `max_pairs` unique pairs and never samples a
   stored edge as a negative.

---

### B8. `fdwalk.py` -- `class FDWalk`

**Contents.** The user-facing class, subclassing `ForceDirected`.

```python
class FDWalk(ForceDirected):
    def make_graph(self, data, **kw)      # stub: returns data
    def graph_walk(self, A, n, **kw)      # -> walk statistics
    def augment_graph(self, G, **kw)      # -> D, and the plan
    def embed(self, D, epochs, **kw)      # inherited, plus the plan
    def fit(self, data, **kw)             # raises NotImplementedError
```

**Contract owed.** `augment_graph` builds the planes FROM the force
registry of B3, never from an `if` chain, and it fails loudly when a law's
planes cannot be produced.

**Success criteria.**
1. `graph_walk` returns the same statistics dict as `walks.walk_rows` for
   the same seed.
2. `augment_graph` raises when the law needs `freq` and the policy did not
   build one -- the defect of 2026-08-18, which silently ran `v1` physics
   under an `fdlinear` label.
3. `fit` raises `NotImplementedError` with a message naming the project.

---

## 7. Cross-block invariants

These hold across the whole package. `core/plan_contract.py` asserts them.

| # | Invariant | What breaks in silence if it is violated |
| --- | --- | --- |
| I1 | planes are `(D.nnz,)` and aligned to `D.indices` | wrong physics, no error |
| I2 | the plane COUNT and ORDER match the law's registry entry | a law reads one plane as another |
| I3 | a pad cell has every plane at 0 | pad cells contribute force |
| I4 | stored weights are integers, and 1 means adjacency | the whole force vanishes |
| I5 | a row with no `h = 1` entry gets an explicit degree | the row freezes, forever |
| I6 | a chunk is a row range | `dZ` is summed twice or not at all |
| I7 | a non-finite `Z` is RECORDED, never evaluated | a crash inside sklearn, no measurement |

## 8. Parity requirement

The project is DONE when the new package reproduces these recorded runs.
The numbers come from `RESULTS.md` and `FINDINGS.md`.

| # | Scenario | Must reproduce |
| --- | --- | --- |
| P1 | Cora, `nbr_walk/min_gap/fdlinear`, dim 64, 200 ep, lr 1.0, seed 42 | `dz=0.5016 auc=0.9962 r2_dist=0.253` |
| P2 | Cora, `walk/min_gap/v1/plain`, dim 64, 2000 ep, lr 0.999, seed 42 | `auc=0.9986 r2_dist=0.523` |
| P3 | Cora, `walk/buckets + far/fdlinear`, 3 seeds | `r2_dist = 0.636 ±0.024` |
| P4 | Cora, `nbr_walk`, `p=2.0`, 3 seeds | `r2_dist = 0.452 ±0.016` |
| P5 | `row_cap(16)` on com_youtube | max row width 16, `n*16` entries |
| P6 | the eight optimizers on Cora, 60 ep, dim 32, lr 1.0 | the ranking of 2026-08-18 |

**Tolerance.** Exact for a bit-exact path (P5, and P1/P2 which use
`uniform_walks`). Within the recorded seed spread for a multi-seed row
(P3, P4). A difference outside that is a defect and blocks the project.

## 9. Risks

| Risk | Mitigation |
| --- | --- |
| A subagent "improves" the physics while moving it | Parity table of section 8 is the gate; any numeric change fails it |
| The plane/law coupling is broken during the split | B3 registry plus `plan_contract.py`; I2 is asserted, not assumed |
| `bench_fdwalk.py` is rewritten instead of rewired | The experiment scripts keep their flags and their RESULT line format |
| The campaign is blocked by the refactor | `experiments/fdwalk/` keeps working until parity passes; nothing is deleted |
| Import cycles between `core` and `augment_graph` | `core` imports nothing from the package except `core`; the dependency runs one way |
