# fodiwalk -- the build record

2026-08-19. What was built, what it reproduces, and what it does not do.
Read [PRD.md](PRD.md) for the requirement, [ORCHESTRATION.md](ORCHESTRATION.md)
for the block plan, and [CATALOG.md](CATALOG.md) for the entities.

**Section 2 reports the gate as it first ran, BEFORE the shell-averaged
laws were removed** (CATALOG.md section 13). The gate was then run AGAIN on
the reduced package, and it passes: `6 passed in 1056s`, plus
`3 passed in 108s` for the `big` scenarios. P3 and P4 return floats
identical to the last bit -- `0.6687732889210574, 0.621311007487906,
0.6170849543166956` for P3 -- thus the removal moved no number, which is
the only claim that matters. P6 alone could not be re-run: it used `v1`.
Section 7 says what replaced it.

---

## 1. What exists

The package of PRD section 4, complete except where section 4 of this
document says otherwise:

```
fodiwalk/
  __init__.py  fodiwalk.py
  core/        csr, force_directed, sell_c_sigma, forces, plan_contract
  make_graph/  datasets
  augment_graph/  walks, pairs, weights, buckets, far_pairs, landmarks
  misc/        optim, drop, evaluation
  dev-docs/    PRD, ORCHESTRATION, CATALOG, BUILD
  tests/       conftest, harness, test_contracts, test_smoke, test_parity
```

`from fodiwalk import Fodiwalk` works from the repository root.

**How the blocks were built.** The mechanical blocks are FILE COPIES with
the imports rewritten, thus a `diff` against the source shows the import
lines and a provenance comment and nothing else -- `csr.py`,
`sell_c_sigma.py`, `drop.py`, `optim.py`, `weights.py`, `landmarks.py`,
`buckets.py`, `datasets.py`. The split blocks (`forces.py`, `walks.py`,
`pairs.py`, `far_pairs.py`, `evaluation.py`) were assembled with an AST
extractor that copies a function body character for character, thus no
body was retyped. `plan_contract.py` is the only new module, and
`fodiwalk.py` is the only rewritten one.

---

## 2. The parity gate -- PRD section 8

Run: `.venv/bin/python -m pytest fodiwalk/tests/test_parity.py -m parity -q`

| # | scenario | recorded | the package |
| --- | --- | --- | --- |
| P1 | Cora `nbr_walk/min_gap/fdlinear` d64 200ep lr1.0 s42 | `dz=0.5016 auc=0.9962 r2_dist=0.253` | **EXACT**, and the fused plane gives the same numbers |
| P2 | the lr ladder at 0.999, 2000 ep | `dz=0.1274 auc=0.9986 r2_dist=0.523` | **EXACT** |
| P3 | Cora `walk/buckets+far/fdlinear`, 3 seeds | `r2_dist = 0.636 +-0.024` | **0.6357 +-0.0234** |
| P4 | Cora `nbr_walk p=2.0`, 3 seeds | `r2_dist = 0.452 +-0.016` | **0.4518 +-0.0166** |
| P5 | `row_cap(16)` | width 16, `n*16` entries | **EXACT** in the rule's exact form; see section 3 on `n*16` |
| P6 | eight optimizers, Cora 60ep d32 lr1.0 | the ranking of 2026-08-18 | **EXACT**, every rule to three decimals -- then RETIRED, see section 7 |

`6 passed in 1052s` on 2026-08-19, with a GPU. The three `big` scenarios
pass too: `3 passed in 95s`, including the whole 1.13M-node augmentation
and its plan.

P1 goes further than "inside the tolerance": `||dZ|| = 0.501619` against
the recorded `0.501619`, `pad_frac = 0.20059361671282838` against the
recorded value to every digit, and even `r2_vec`, which depends on an MLP
fit over 64 features, matches the recorded 0.606 (three planes) and 0.612
(fused). The augmentation, the plan, the kernel and the generator order are
therefore identical, not merely equivalent.

**A correction to the PRD, and it is a label and not a number.** Section 8
writes P2 as `walk/min_gap/v1/plain`. The run that produced
`auc = 0.9986, r2_dist = 0.523` is `nbr_walk/min_gap/fdlinear/plain` at
`k4 = 1.0` (`results/lrladder/const_plain_lr0.999.log`, `run_lrladder.sh`).
No `walk/v1` row carries those numbers. The test uses the configuration
that produced them.

---

## 3. The numbers

**P3, `walk/buckets + far`, three seeds.** `0.6688, 0.6213, 0.6171`, thus
mean `0.6357 +-0.0234` against the recorded `0.6357 +-0.024`. The mean
agrees to four digits.

**P4, `nbr_walk` at `p = 2.0`, three seeds.** `0.4713, 0.4306, 0.4536`,
thus mean `0.4518 +-0.0166` against the recorded `0.452 +-0.016`.

**P6, the eight optimizers at 60 epochs, dim 32, lr 1.0.** Hop R2, and the
recorded value of 2026-08-18 in brackets:

| rule | hop R2 | recorded |
| --- | --- | --- |
| `plain` | 0.373 | 0.373 |
| `sqn` | 0.357 | 0.357 |
| `velocity` | 0.356 | 0.356 |
| `sgd` | 0.268 | 0.268 |
| `nesterov` | 0.157 | 0.157 |
| `momentum` | 0.117 | 0.117 |
| `adam` | -0.010 | -0.010 |
| `fa2` | diverged | crashed inside sklearn |

Every rule reproduces its recorded value to three decimals, thus the
ranking is not merely preserved -- the runs are the same runs. `fa2` still
produces a non-finite `Z`; the difference is that the package RECORDS the
divergence (I7) and the run of 2026-08-18 died inside `MLPRegressor` with
"Input X contains NaN", which carried no time, no memory and no `RESULT`
line.

**Two augmentation-only checks, in the fast suite** (seconds, no GPU):
`walk` gives `D.nnz = 75,068` and `walk_edges` gives `75,072`, against
`results/fdlinear/fdl_cora_{walk,walk_edges}_lr1.0_s42.log`. They guard
the walk statistics, `cap_per_node`, the symmetric CSR build and the far
pairs without paying for an embedding.

**The com_youtube plan, PRD B2 criterion 3.** The walk budget of the
1.13M-node run of 2026-08-17 is in no surviving script. `walk_rows` at
`2 walks x 10 steps` gives `raw = 19,668,991`, which is that log's recorded
value exactly (`1 x 20` gives `21,082,318`), thus the configuration is
recovered. The package then rebuilds the whole augmentation and reproduces
every recorded field:

| field | value |
| --- | --- |
| raw pairs | 19,668,991 |
| unique pairs | 18,624,811 |
| entries at `h = 1` | 5,975,248, thus `A.nnz` exactly |
| far pairs at `deg^0.75` | 2,269,516 |
| `D.nnz` | 23,163,843 |
| `n_virtual` / `n_split` / `rungs` | 141,943 / 50 / 14 |
| **`pad_frac`** | **0.15407662320379392** |

That is the strongest single check in the suite: it exercises the walks,
the neighbour merge, the biased far-pair draw with its post-filter, the
hub split and the eight-chunk plan at the size the campaign runs on. It
also settles the question the far-pair filter raised: the recorded far
count `2,269,516` of `2,270,000` asked for comes out only with the
post-filter, and not with `directed=True` inside the sampler.

**P5 and the `n * 16` count.** PRD section 8 writes P5 as "max row width
16, `n*16` entries". The count is exact only when EVERY row finds more than
16 partners -- the qualifier PRD B6.4 carries and section 8 drops. Cora
gives 208 short rows of 2,708, and com_youtube gives 18,154,004 entries of
the 18,158,240 that `n * 16` asks for, thus 99.98%. The rule itself is
exact on both: `width == min(16, what the row found)`, and the tests assert
that form. A synthetic graph where every row qualifies asserts the strict
`n * m` count
([test_b6_row_cap_gives_every_row_the_same_width](../tests/test_contracts.py)).

---

## 4. What this build does NOT do, and why

1. **A8 -- `experiments/fdwalk/bench_fdwalk.py` is NOT rewired.** The
   instruction for this session was "only work on the directory
   `./fodiwalk/`", thus nothing outside it was touched. The experiment
   scripts keep working, which is what the PRD's risk table asks for
   anyway, and the rewiring is now a small change: `fodiwalk/tests/harness.py`
   already does what `bench_fdwalk.py` does, in the same order and with the
   same generator, and `Fodiwalk(**cfg)` takes the flag names with
   underscores. The `RESULT` line format is untouched because the file that
   prints it is untouched.

2. **`--pairs ball` is not in the package.** It needs `k_hop_ball`, and the
   PRD's module list for `augment_graph/` does not carry it. The control
   stays in the experiment script. `Fodiwalk` raises a `ValueError` that
   names the alternative.

   > **ADDED 2026-08-19 on request, then REMOVED 2026-08-20 on request.**
   > `ball` and `sampled` were built (CATALOG section 16), measured against
   > the walk policies (section 17), and removed when the walk won
   > (section 18). The PRD's scope, quoted above, is where the package
   > stands again -- by measurement this time, and not only by scope. The
   > two tree datasets of section 16.4 stayed: they are loader work.

3. **`sample_far_pairs(..., directed=True)` is available and unused.** PRD
   B6 criterion 6 asks the sampler to reject a pair stored in either
   direction. The flag does that, and a test asserts it. The augmentation
   does NOT use it: a changed rejection changes how many pairs one batch
   accepts, thus it changes the draw of the next batch and the far set of
   every recorded run. `Fodiwalk` keeps the post-filter of
   `bench_fdwalk.py`, which gives the same property with the same numbers.
   See [CATALOG.md](CATALOG.md) section 8.2.

4. **The three com_youtube scenarios are marked `big` and are not in the
   fast suite.** They pass -- see section 3 -- and they cost 95 s and a few
   GB, thus a caller asks for them: `pytest fodiwalk/tests -m big`.

5. **`fit` raises**, `make_graph` returns its argument. Both are PRD
   non-goals, and the tests assert them.

---

## 5. Changes that were made deliberately, and their reasons

| what | why | where |
| --- | --- | --- |
| `make_walker` returns `functools.partial(uniform_walks, A)` in place of a lambda | PRD B6.1 asks for an assertion of identity, and a lambda carries none. The call and its numbers do not change. | [walks.py](../augment_graph/walks.py) |
| `datasets.DATA` is derived from the file and overridable by `FDMAP_DATA` | an absolute path of one machine must not be compiled into a package | [datasets.py](../make_graph/datasets.py) |
| `updateZ` dispatches through `misc/optim.py`, and `plain` is the default | PRD B4.4. `plain` IS the one line the base class had, thus no number moved. | [force_directed.py](../core/force_directed.py) |
| `embed` records a non-finite `Z` on `self.diverged` | I7. A crash inside sklearn carries no time, no memory and no RESULT line, thus a real measurement was lost as a stack trace. | [force_directed.py](../core/force_directed.py) |
| `get_embeddings_df` falls back to the row index when `G` has no `.nodes()` | the package hands a CSR to `embed`, and a CSR carries no labels | [force_directed.py](../core/force_directed.py) |
| `sample_far_pairs` gained `directed=False` | PRD B6.6, without moving a recorded number | [far_pairs.py](../augment_graph/far_pairs.py) |
| `core.set_rule` imports `misc.optim` inside the function | it keeps the module-level import graph one-way, which the PRD's risk table asks for, while `updateZ` still dispatches through `misc/optim.py` | [force_directed.py](../core/force_directed.py) |

---

## 6. The test suite

```
.venv/bin/python -m pytest fodiwalk/tests -q -m "not parity and not big and not slow"
.venv/bin/python -m pytest fodiwalk/tests/test_parity.py -q -m parity
.venv/bin/python -m pytest fodiwalk/tests -q -m big        # com_youtube
```

`test_contracts.py` holds one test for one criterion of PRD sections 6 and
7. `test_smoke.py` holds B4 and B8 on a 60-node graph. `test_parity.py`
holds section 8. `harness.py` runs one scenario end to end and gives the
fields of the `RESULT` line back.

**The order of the generator is part of the parity.** The script makes ONE
`np.random.default_rng(seed)` and passes it to the augmentation, then to
the link prediction, then to the hop sample. A second generator, or another
order, gives different pairs and a different number -- and the difference
would look like a defect of the physics. `Fodiwalk` keeps that generator as
`self.rng`, and `harness.run` passes it on after the augmentation has
advanced it.

---

## 7. UPDATE 2026-08-19 -- `shell_coeff` is removed, and P6 is retired

The user decided to keep the linear laws only. [CATALOG.md](CATALOG.md)
section 13 holds the full entry: what left the package, why, and what did
not change. In short: the plane `shell_coeff`, its two host builders, the
three laws that read it (`v1`, `v2`, `fdlinear_3plane`), the three
parameters only those laws read (`k2`, `k3`, `h_shift`), and one value test
in `plan_contract`.

**The order matters and it was deliberate.** The removal came AFTER the
parity gate of section 2 passed. The gate therefore proves the package
against the campaign, and section 3 keeps the numbers it produced. The
removal is a separate, later, recorded step.

**No surviving function body changed.** An AST comparison of every
surviving function against its frozen source in `experiments/fdwalk/` and
`fodined/` reports one difference, and it is the `directed` parameter of
`sample_far_pairs` from section 5 of this document.

**P6 is the one scenario that could not survive.** Its recorded run used
`v1`. A package with no `v1` cannot reproduce it, thus the parity claim
retires with the law; the table in section 3 stands as the record that it
DID reproduce, to three decimals for all eight rules, before the removal.

What replaces it is a baseline of the same eight rules under `fdlinear`, at
the same 60 epochs, dim 32, lr 1.0 on the `walk` policy, measured on
2026-08-19 and pinned in `test_parity.py` as `FDLINEAR_ROSTER`. **It is a
regression baseline and not parity with the campaign**, and the test says
so in its docstring.

| rule | hop R2, `fdlinear` | final `\|\|dZ\|\|` | hop R2, `v1` (2026-08-18) |
| --- | --- | --- | --- |
| `plain` | 0.609 | 1.8 | 0.373 |
| `sqn` | 0.607 | 3.5 | 0.357 |
| `velocity` | 0.606 | 1.9 | 0.356 |
| `sgd` | 0.522 | 3.4 | 0.268 |
| `adam` | 0.213 | 14.8 | -0.010 |
| `momentum` | 0.085 | 1.8e4 | 0.117 |
| `nesterov` | -0.000 | 1.3e6 | 0.157 |
| `fa2` | diverged | nan | diverged |

**The shape survived the change of the law**, which is the property the
roster exists to measure: `plain` leads, `sqn` and `velocity` follow within
0.003, `sgd` is one step behind, and `fa2` gives a non-finite `Z`. Two
things moved. The whole leading group rose from 0.373 to 0.609 -- the
linear law simply fits the hop distance better at 60 epochs. And `adam`
crossed above `momentum` and `nesterov`, which at lr 1.0 are inside a
near-divergent regime: their `||dZ||` ends four and six orders of magnitude
above `plain`'s, thus their hop R2 is near zero because the layout blew up
and not because a fit came out badly.

The two columns are DIFFERENT EXPERIMENTS and only their shape is
comparable. Do not quote the `fdlinear` column as a campaign result.

The other five scenarios are untouched: P1 through P4 pass `force="fdlinear"`
explicitly, and P5 is an augmentation rule with no force law in it.
