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

```
fodiwalk/
  __init__.py           Fodiwalk, Config              (surface unchanged)
  config.py             Config (FLAT, unchanged fields) + the stage specs
  model.py              class Fodiwalk -- the wiring only. <= 200 lines
  make_graph/           STAGE 1. datasets.py                    unchanged
  augment_graph/        STAGE 2
      result.py         `Augmentation` -- what stage 2 gives stage 3
      policies.py       `POLICIES` registry, and `build(A, n, spec, rng)`
      policy_walk.py    `walk`, `walk_edges`, the `cap` policy
      policy_buckets.py the `buckets` policy of the undirected pairs
      policy_nbr_walk.py the directed policy, `cap` and `buckets`
      merge.py          `add_far_pairs` -- the ONE far/freq CSR merge
      walks.py pairs.py weights.py buckets.py far_pairs.py landmarks.py
                                                              unchanged
  embed/                STAGE 3, the assembly
      planes.py         `PLANE_BUILDERS` registry (was `Fodiwalk._plane`)
      degrees.py        `resolve_degrees` (was `Fodiwalk._build_degrees`)
      planner.py        `build_plans` -> `PlanSet` (was `_build_plan`)
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
augment_graph <- numpy, scipy. It imports NO core, NO embed, NO model.
embed       <- core. It imports NO augment_graph, NO model.
model.py    <- config, make_graph, augment_graph, embed, core, misc.
```

`core` must NOT learn about `Config`. That is why the plane, degree and
plan assembly go to `embed/` and not to `core/`.

### 3.1 `Augmentation` -- the seam of stage 2 to stage 3

```python
@dataclasses.dataclass
class Augmentation:
    D: sp.csr_matrix        # the weighted matrix. `D.data` is the `h` plane
    freq: np.ndarray | None # (nnz,), aligned to D.indices, or None
    stats: dict             # the walk statistics, and `freq` as a CSR
    info: dict              # the counts and the timings a log prints
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
PlanSpec.from_config(cfg)      # b_cells, k_max, ladder_base, chunks,
                               # chunk_host, check_planes, check_padding
ForceSpec.from_config(cfg)     # force, fuse_planes, k1, k4, kr,
                               # fdlinear_sign, no_deg_norm, deg_source,
                               # random_drop_rate, drop_strategy
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
11. **`Z` on the GPU is not bit-reproducible when `n_split > 0`**
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
| G2 | the suite | `.venv/bin/python -m pytest fodiwalk/tests -q -m "not parity and not big"` | `53 passed`, and NO existing test file is edited |
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
- The field names of `Config`: the 41 of today, spelled the same.
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
- The keys of `info` and of `plan_stats` do not change.

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
