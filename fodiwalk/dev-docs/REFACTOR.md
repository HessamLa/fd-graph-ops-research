# fodiwalk -- the refactor plan of 2026-08-20

`fodiwalk/fodiwalk.py` is 698 lines and it holds four jobs. This document
says what is wrong, what the tree becomes, what must NOT change, and the
gates that decide if a step is done. It is the brief that every subagent
reads.

Read first: [fodiwalk-module.md](fodiwalk-module.md) -- the three
categories that the package must show. Then [PRD.md](PRD.md) section 7 and
[CATALOG.md](CATALOG.md) sections 2, 3, 14, 15.

---

## 1. The goal

One rule, from `fodiwalk-module.md`: **three categories, and each one is a
place in the tree.**

1. graph construction -> `make_graph/`
2. graph augmentation -> `augment_graph/`
3. embedding -> `embed/` on top of `core/`

A stage takes a small, named input and gives a small, named output. A
category does not reach into the state of another category. The model class
wires the three and holds nothing else.

**Behaviour does not change. Not one number.** This is a code move, and the
gates of section 5 measure exactly that.

---

## 2. The defects, with the evidence

| # | defect | evidence |
| --- | --- | --- |
| D1 | God class. `Fodiwalk` holds the config, the walk dispatch, three `_build_D_*` policies, the plane builder, the degree source, the plan builder and the force kernel call. | `fodiwalk.py`, 698 lines, 4 jobs |
| D2 | Policy dispatch by `if` chain on a string, in two methods, while the package already solved that with registries for the laws, the weights and the update rules. | `_build_D`, `graph_walk`, `_plane` |
| D3 | The far-pair CSR merge is copied THREE times, each copy slightly different. `np.concatenate` appears 26 times in one file. | `_build_D_nbr_walk`, `_build_D_walk`, `_build_D_buckets` |
| D4 | Hidden temporal coupling. `_build_planes` reads `self.freq`, which some earlier method must have set. 26 attribute writes, 5 of them to `self.freq` and `self.stats` from four methods. | `self.freq =` at lines 179, 244, 539, 625, 691 |
| D5 | Parameter tunnels. `_build_D_nbr_walk(self, A, n, rng, info, t0)` passes a log dict and a wall clock into the algorithm. | signature |
| D6 | Cross-module import of a private name: `from .core.sell_c_sigma import make_plan, _step`. | line 64 |
| D7 | Stage leak. Stage 2 (`augment_graph`) builds `self.params`, which is physics. Stage 3 machinery (`jax.jit`, `device_put`, chunking) lives in the model class. | `augment_graph`, `_build_plan` |
| D8 | `stats` is an untyped bag whose keys change shape as it flows; `_take` is a private helper of the wrong module. | `_take` at the file end |
| D9 | Cryptic locals against the naming rule of `CLAUDE.md`: `fw`, `fq`, `cc`, `fc`, `nk`, `cl`, `rr`, `k1`, `k2`. `fw` means "far weight" here and "the model" in the README. | `_build_D_walk` |
| D10 | An empty `models/` directory in the tree. | `fodiwalk/models/` |

---

## 3. The target tree

**Section 3 is CORRECTED as of 2026-08-21.** M0-M4 built `embed/planes.py`
and `embed/degrees.py`; the stage boundary then MOVED, and this section now
describes the tree as it stands, not the tree of the first split. The
reason, and what M0-M4 actually built before the move, are section 8.

```
fodiwalk/
  __init__.py           Fodiwalk, Config              (surface unchanged)
  config.py             Config (FLAT, unchanged fields) + the stage specs
  model.py              class Fodiwalk -- the wiring only. <= 200 lines
  make_graph/           STAGE 1. datasets.py                    unchanged
  augment_graph/        STAGE 2 -- the recipe's DATA
      result.py         `Augmentation` -- what stage 2 gives stage 3
      policies.py       `POLICIES` registry, and `build(A, n, spec, rng)`
      policy_walk.py    `walk`, `walk_edges`, the `cap` policy
      policy_buckets.py the `buckets` policy of the undirected pairs
      policy_nbr_walk.py the directed policy, `cap` and `buckets`
      merge.py          `add_far_pairs` -- the ONE far/freq CSR merge
      planes.py         `PLANE_BUILDERS` registry, `ForceSpec` (moved here
                        2026-08-21; was `embed/planes.py`)
      degrees.py        `resolve_degrees` (moved here 2026-08-21; was
                        `embed/degrees.py`)
      walks.py pairs.py weights.py buckets.py far_pairs.py landmarks.py
                                                              unchanged
  embed/                STAGE 3 -- CONSUMPTION ONLY
      planner.py         `build_plans` -> `PlanSet`, `PlanSpec`
  core/                 the engine, the kernel, the laws, the asserter
                                                              unchanged
  misc/                 optim, drop, evaluation               unchanged
  tests/                + golden.py, golden_baseline.json, test_golden.py
  dev-docs/
```

`fodiwalk/fodiwalk.py` and `fodiwalk/models/` are REMOVED.

**The dependency runs one way, and a cycle is a defect.**

```
core        <- imports core only.               (unchanged rule)
make_graph  <- numpy, scipy.
augment_graph <- numpy, scipy, core (the plane/degree contract). No embed,
              no model. (WIDENED 2026-08-21: `planes.py` reads
              `core.forces.planes_of`/`fuse`, `core.plan_contract`.)
embed       <- core only. It imports NO augment_graph, NO model.
model.py    <- config, make_graph, augment_graph, embed, core, misc.
```

`core` must NOT learn about `Config`. That is why the plane, degree and
force-param builders live in `augment_graph/`, and the plan/kernel
assembly in `embed/`, and neither lives in `core/`.

### 3.1 `Augmentation` -- the seam of stage 2 to stage 3

**Widened 2026-08-21** with `planes`, `degrees` and `params`: they are
stage-2 output (data-preparation of the recipe), not stage-3 state, thus
the seam object names them even though a policy's own `build()` cannot
fill them (it does not know the law) -- `Fodiwalk.augment_graph` fills
them once the law is known, still calling `augment_graph` code.

```python
@dataclasses.dataclass
class Augmentation:
    D: sp.csr_matrix        # the weighted matrix. `D.data` is the `h` plane
    freq: np.ndarray | None # (nnz,), aligned to D.indices, or None
    stats: dict             # the walk statistics, and `freq` as a CSR
    info: dict              # the counts and the timings a log prints
    planes: tuple | None = None      # filled after `build`, once law is known
    degrees: np.ndarray | None = None
    params: dict | None = None
```

### 3.2 The stage specs -- a narrow config for each category

`Config` stays FLAT and its field names do not change: it is the public
surface, `experiments/fodiwalk/*.py` builds one, and `harness.py` reads
`fw.cfg.chunks`. The narrowing happens at the seam instead. Each stage
function takes its own spec and cannot reach the knobs of another stage:

```python
AugmentSpec.from_config(cfg)   # pairs, policy, walks, walk_len, window,
                               # cap, row_cap, p, q, edge_rule, weight,
                               # freq_mode, far*, bucket*, landmarks,
                               # prune_*
ForceSpec.from_config(cfg)     # force, fuse_planes, k1, k4, kr,
                               # fdlinear_sign, no_deg_norm, deg_source,
                               # random_drop_rate, drop_strategy
                               # -- stage 2 since 2026-08-21; `augment_graph.planes`
PlanSpec.from_config(cfg)      # b_cells, k_max, ladder_base, chunks,
                               # chunk_host, check_planes, check_padding
                               # -- stage 3; `embed.planner`
```

---

## 4. What must NOT change -- the traps

Each one has already caused a SILENT defect: a wrong number and no
exception. Sources: `CATALOG.md`, and the session that ran the campaign.

1. **`make_walker(A, n, p, q)` returns `uniform_walks` ITSELF at
   `p = q = 1`.** The rejection sampler is distributionally the same and
   NOT bit-identical: it draws one more number for each accept test, thus
   the same seed gives other walks. Every number recorded before
   2026-08-18 came from `uniform_walks`. Do not unify the two paths.
2. **The order of the generator is part of the result.** ONE
   `np.random.default_rng(seed)` flows through the augmentation, then the
   link prediction, then the hop sample. A moved call, an extra draw or a
   second generator changes the pairs and looks like a defect of the
   physics.
3. **The planes are POSITIONAL.** `make_plan` takes a tuple and
   `force_fn` unpacks it. The plane list comes from
   `core.forces.FORCE_PLANES` and never from an `if` chain, and
   `plan_contract.check` asserts it at the seam.
4. **`degrees_from_D` counts the `h == 1` entries of a row.** A row with
   none gets degree 0, `inv_deg_ext` becomes 0.0, and the kernel zeroes
   the WHOLE force of that row. The node freezes and nothing raises.
5. **A stored weight is an INTEGER in `[1, window]`, and 1 means
   adjacency.** A continuous weight gives AUC 0.55 with `||dZ|| = 0.000`.
6. **`walk_rows` has NO prune by design** (its bound IS
   `walks * walk_len` for each row). `walk_pair_stats` DOES prune, and the
   prune is an approximation. Do not unify them.
7. **`_pairs_of` collapses the direction at the source** with
   `min * n + max`. A directed form doubles the accumulator that already
   OOM-killed the 1.13M-node runs.
8. **A chunk is a ROW RANGE and never a pair set.** `_step` does
   `dZ.at[rows].add(...)`, thus only disjoint rows make the parts additive.
9. **`nbr_walk` order**: `row_cap` bounds the WALK partners and it runs
   BEFORE the neighbours are added, thus a neighbour is never dropped.
10. **The far filter of `nbr_walk` stays as it is.** It drops a far pair
    that `near` holds in EITHER direction, AFTER `sample_far_pairs` has
    drawn. `directed=True` inside the sampler gives the same property and
    OTHER pairs, thus it breaks parity. `CATALOG.md` 8.2.
11. **A stage split must not make a SECOND generator.** One
    `np.random.default_rng(seed)` flows walks -> far pairs -> link
    prediction -> hop sample. A second one, or a moved call, changes every
    recorded number. This is trap 2 seen from the other side: it is the
    failure mode a refactor produces, and not the one a rewrite produces.
12. **`embed()` calls `augment_graph()` itself.** A new module that calls
    it AGAIN to time the stage makes the run embed a `D` that is not the
    `D` it reports, and `D.nnz` still agrees, because the pair counts are
    stable. Time the stage with a `train_begin` callback.
13. **`Z` on the GPU is not bit-reproducible when `n_split > 0`**
    (`CATALOG.md` 15). The golden gate therefore hashes the augmentation,
    which is pure NumPy and exact, and compares the embedding by numbers
    with `rtol = 1e-4` on the CPU backend.

---

## 5. The gates -- how a step is measured

Every gate is a command. A subagent reports the ACTUAL output, and the
master re-runs it.

| id | gate | command | pass |
| --- | --- | --- | --- |
| G1 | behaviour | `.venv/bin/python -m fodiwalk.tests.golden --check` | `GOLDEN OK (both)`. The augment half is BYTE EXACT |
| G2 | the suite | `.venv/bin/python -m pytest fodiwalk/tests -q -m "not parity and not big"` | NO failure and NO error, and no EXISTING test file is edited. The count moves as the agents add tests: 52 before, 136 after M1. M2 deletes `fodiwalk.py`, thus the two equivalence tests SKIP by their `importorskip` guard -- that is correct, and they retire with the thing they compared against |
| G3 | the surface | `.venv/bin/python -m fodiwalk.tests.check_api` | every name of section 5.1 exists and behaves |
| G4 | the size | `wc -l` over `fodiwalk/**/*.py` | no module > 300 lines except `core/sell_c_sigma.py`; `model.py` <= 200 |
| G5 | no dispatch chain | `grep -n '"walk"\|"nbr_walk"\|"buckets"\|"fdlinear"' fodiwalk/model.py` | no hit outside a docstring |
| G6 | no duplication | `grep -c 'sp.csr_matrix((np.concatenate' fodiwalk/augment_graph/*.py` | the far/freq merge lives in `merge.py` only |
| G7 | the direction | `pytest fodiwalk/tests/test_contracts.py -k imports` | `core` imports core; `augment_graph` imports no `core`/`embed`; no cross-module import of a `_name` |
| G8 | pure stages | read | every policy builder is a module function `(A, n, spec, rng) -> Augmentation`. No `self` |
| G9 | the docs | read | `CATALOG.md` has a NEW numbered section with the reason for the update. Nothing removed |
| G10 | the parity gate | `pytest fodiwalk/tests/test_parity.py -k p1 -m parity -q` | passes, on a GPU. Master only, at the end |

### 5.1 The public surface, which G3 checks

- `from fodiwalk import Fodiwalk, Config`
- The field names of `Config`: the 39 of today, spelled the same, with
  the same default values. `tests/check_api.py` holds them literally.
- `Fodiwalk(n_dim=, lr=, seed=, verbosity=, optim=, lr_decay=, eta=,
  sgd_frac=, sqn_memory=, **cfg)`; an unknown name raises `TypeError`.
- The methods: `make_graph`, `graph_walk`, `set_D`, `augment_graph`,
  `forces`, `embed`, `fit` (raises), `get_embeddings`, `attach_callback`,
  `set_rule`, `Th`.
- The attributes after `embed`: `cfg`, `law`, `rng`, `D`, `stats`, `freq`,
  `info`, `plan_stats`, `plans`, `steps`, `inv_deg_ext`, `chunk_rows`,
  `resident`, `params`, `diverged`, `dZ`, `Z`.
- `Fodiwalk._build_planes(D)` and `Fodiwalk._build_degrees(D, A)` stay
  callable: `tests/golden.py` calls both. They may become thin wrappers of
  `embed/`.
- The keys of `info`, of `plan_stats` and of `stats` are a SUBSET rule: a
  lost key fails, a new key passes. A lost key breaks a log or a `RESULT`
  line; a stage split may legitimately add one. `stats` is surface in
  practice -- `tests/golden.py` and `experiments/fodiwalk/bench_fodiwalk.py`
  read `stats["key"]` and `stats["mn"]` to build the hop-gap CSR.

---

## 6. The milestones

| M | what | who | gate |
| --- | --- | --- | --- |
| M0 | the golden baseline, the snapshot, this document | master | G1 records |
| M1 | `augment_graph/` policies, and `embed/` assembly. NEW files only; `fodiwalk.py` untouched | A1, A2 in parallel | the equivalence test of each agent, plus G1 unchanged |
| M2 | `config.py`, `model.py`; `fodiwalk.py` and `models/` removed | A3 | G1-G8 |
| M3 | the structure tests, and the docs | A4, A5 in parallel | G7, G9 |
| M4 | independent verification, and the GPU parity gate | master | G1-G10 |

**M1 gives the two agents disjoint files and no shared state.** Each one
proves its new code EQUALS the old method, on the 16 golden cases, before
anything is wired. M2 then deletes the old code, and the golden gate says
whether the wiring is right.

---

## 7. What is out of scope

- The physics, the force laws, the walk algorithms, the weight rules, the
  update rules. Not one line of `core/sell_c_sigma.py`, `core/forces.py`,
  `augment_graph/walks.py`, `misc/optim.py` changes behaviour.
- New features, new policies, new datasets.
- `fodined/`, `experiments/`, and every file outside `fodiwalk/`, except
  the import lines of `experiments/fodiwalk/*.py` if a name moves.
- Performance. A refactor that gets faster is fine; a refactor that changes
  a number to get faster is a defect.

---

## 8. UPDATE 2026-08-21 -- the stage boundary moves

**M0-M4 built `embed/planes.py` and `embed/degrees.py`.** That build is
correct and is what sections 1-7 above describe as delivered; this entry
records what changed AFTER it, and why. Nothing in sections 1-7 was wrong
at the time; the module specification `dev-docs/fodiwalk-module.md` was
then sharpened, and the tree follows.

**Reason for the update.** `dev-docs/fodiwalk-module.md` gained one
sentence: "This stage [embedding] shall not do any graph analysis or data
preparation. It must only consume the data. Its main goal is to apply the
force function on the input data using the best implementation to optimize
resource utilization." A plane, a degree and a force param are each a
choice made about DATA -- which values a law needs, which row freezes
without an explicit degree, which scalar a law reads -- and that choice is
a property of the RECIPE (one pair policy plus one force law), not of the
kernel that later applies it. Building them in `embed/` put data
preparation in the consumption stage.

**What moved.** `embed/planes.py` -> `augment_graph/planes.py`
(`ForceSpec`, `PLANE_BUILDERS`, `build_planes`, `force_params`).
`embed/degrees.py` -> `augment_graph/degrees.py` (`resolve_degrees`). No
line of a function body changed; only the module and the import lines
that reach `core.forces`/`core.plan_contract` moved with them.

**What widened.** `augment_graph.result.Augmentation` gained `planes`,
`degrees` and `params` (section 3.1). `augment_graph`'s allowed imports
gained `core` (section 3), because the moved code reads
`core.forces.planes_of`/`fuse` and `core.plan_contract.PLANE_CHECKS` to
know a law's plane contract -- knowing the contract is itself part of
preparing that law's data. `embed/` now imports `core` for the kernel and
the plan alone (`make_plan`, `step`, `plan_contract.check_plan`), and
builds no plane and resolves no degree.

**What did not move.** `embed/planner.py` (`PlanSpec`, `PlanSet`,
`build_plans`, the `jax.jit`/`device_put` plumbing): the plan and the
jitted step are consumption, not preparation, and stay stage 3.

**What did NOT change.** The physics, the numbers, `Config`'s 39 fields,
the public surface of `Fodiwalk` (`tests/check_api.py` still reports
`API OK`), and `_build_planes`/`_build_degrees` as callable model methods.
The golden gate (`tests/golden.py`) is BYTE EXACT across the move.

**The RECIPE, named.** One augmentation policy plus one force law, plus
the data they exchange, is a RECIPE. Baseline Fodiwalk's recipe is
`nbr_walk`/`walk`/`walk_edges` pairs feeding `fdlinear`'s planes
`(h, freq)`, or `fdlinear_fused`'s single plane `(w,)`. A different
augmentation and a different law are a different recipe with a different
data set; `augment_graph` is where a recipe is assembled end to end, and
`embed` is the one engine every recipe shares. `dev-docs/CATALOG.md` holds
the full entry.

**Gates re-run after the move**: G1 (`GOLDEN OK (both)`), G2 (`66 passed,
2 skipped`), G3 (`check_api` OK), G4-G8 (`test_structure.py`, 13 passed,
with `test_augment_graph_imports_no_embed_no_model` widened to allow
`core`). Two negative controls, run against scratch copies and not the
live tree: an `embed -> augment_graph` import still fails
`test_embed_imports_only_core_no_augment_graph_no_model`, naming the file.

**Adopted at.** The fodiwalk package immediately after M4 (commit that
lands the M0-M4 split), same session.

**Provenance.** [augment_graph/planes.py](../augment_graph/planes.py),
[augment_graph/degrees.py](../augment_graph/degrees.py),
[augment_graph/result.py](../augment_graph/result.py),
[embed/planner.py](../embed/planner.py), [model.py](../model.py),
[dev-docs/fodiwalk-module.md](fodiwalk-module.md).
