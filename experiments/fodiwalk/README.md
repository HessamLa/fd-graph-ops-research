# experiments/fodiwalk

The experiments that run on the `fodiwalk` PACKAGE, and not on the flat
scripts of `experiments/fdwalk/`.

`experiments/fdwalk/` is frozen. It holds the 2026-08-15..18 campaign and
every number that campaign recorded. Nothing here changes it.

| file | what it does |
| --- | --- |
| `verify_baseline.py` | proves the two claims the baseline rests on |
| `verify_modular.py` | the walk policies against `fodined/modular.py` |
| `bench_fodiwalk.py` | one run on one graph, with the report |
| `results/` | the `RESULT` lines, one for each run |

```bash
.venv/bin/python experiments/fodiwalk/verify_baseline.py
.venv/bin/python experiments/fodiwalk/verify_modular.py
.venv/bin/python experiments/fodiwalk/bench_fodiwalk.py --graph cora \
    --out experiments/fodiwalk/results/baseline_cora.tsv
```

---

## The baseline, and what makes it one

Two decisions, and `verify_baseline.py` checks both. It exits non-zero when
either fails, thus it is a gate and not a comment.

**1. The walk is the walk of node2vec.** `make_walker(A, n, 1.0, 1.0)`
returns `uniform_walks` itself. That function is the first-order walk of
DeepWalk, which IS node2vec at `p = q = 1`. The check is not a reading of
the code: it drives `fodiwalk`'s function and the `uniform_walks` of
`experiments/other-ge/bench_other_ge.py` -- the node2vec baseline of this
repository -- with the same graph, the same start nodes and the same seed,
and it compares the two arrays cell by cell. 27,080 x 20 cells, bit-exact.

Above `p = q = 1` the walker becomes `node2vec_walks`, the second-order
rejection sampler, thus `--p` and `--q` reach the walk.

**One difference to state.** The walk FUNCTION is shared; the walk BUDGET
is not. `fodiwalk` defaults to 10 walks x 20 steps, and
`experiments/other-ge/` defaults to 10 x 40. The window is 5 on both. A
head-to-head against node2vec must set the length on one side.

**2. The law is `fdlinear`.** `Config().force` is `"fdlinear"`, the model
binds `forces.fdlinear` itself, and its planes are `(h, freq)`:

```
h == 1:  Fa = k1 * x        Fr = -kr * exp(-k4 * x)
h >= 2:  Fa = 0             Fr = -(h / freq) * exp(-k4 * x)
```

Since 2026-08-19 no other family is registered -- the shell-averaged laws
were removed (`fodiwalk/dev-docs/CATALOG.md` section 13), thus `fdlinear`
and its fused form are the whole registry and the baseline cannot silently
be something else.

---

## The report the bench prints

Per stage: the wall time and the peak resident memory. Then link prediction
(accuracy, F1, AUC), the hop-distance regression (R2), and the final
`||dZ||`. The last line is a `RESULT` line in the `key=value` shape that
`experiments/fdwalk/bench_fdwalk.py` prints, thus one collector reads both.

A diverged run does not crash. It RECORDS the divergence, writes a `RESULT`
line with `diverged=1`, and exits 0 -- the measurement "this rate diverges"
is kept, and not thrown away as a stack trace.

Large artifacts belong in `./data_cache/`. This directory holds the scripts
and the `RESULT` lines only.

---

## Reading `||dZ||`, and the one trap in this bench

`||dZ||` is the mean row norm of the LAST epoch's step. It reads one step,
thus it carries every difference the run accumulated. The scores read `Z`,
which is 2000 steps added together, thus they average the differences out.

**A plan with `n_split > 0` is not bit-reproducible on a GPU.** The kernel
ends with `dZ.at[rows].add(...)`, a scatter-add. A hub row wider than
`k_max` becomes several virtual rows with the SAME owner id, thus several
writes land on one row of `dZ` and a GPU does not fix their order. Measured
on this baseline: three runs of one seed give `||dZ||` 0.117347, 0.117157
and 0.118518, while the same runs on the CPU agree to nine decimals.

The bench prints `n_split` in its `plan:` line. Read `||dZ||` against it:

| `n_split` | `||dZ||` |
| --- | --- |
| 0 | exact. A difference is a defect. |
| > 0 | about 1% between runs. Only a larger difference is evidence. |

`D`, the plan and `pad_frac` are exact either way -- the augmentation is
pure NumPy. `fodiwalk/dev-docs/CATALOG.md` section 15 holds the full entry.

**The trap, and it cost a wrong first report.** `embed` calls
`augment_graph` itself. A script that ALSO calls `augment_graph` to time
the stage runs the walks a SECOND time, on an `rng` the first call already
advanced -- thus the run embeds a different `D` than the one the script
reports, and `D.nnz` still matches, because the pair counts are stable.
Time the stage with a `train_begin` callback instead, which is what
`Stopwatch` in `bench_fodiwalk.py` does.

---

## Against `fodined/modular.py`, on the WALK policies

`fodined/modular.py` is the reference pipeline of this repository: load,
augment, embed on the SELL-C-sigma kernel, then link prediction and a hop
regression. It augments with a MEASURED 2-hop ball.

The augmentation of `fodiwalk` is the WALK family, thus the question is not
"is it the same matrix" -- it is not, and it is not meant to be. The
question is whether a walk reaches the result of a measured ball.

**It does, and `nbr_walk` exceeds it** -- which is why the exact policies
were removed on 2026-08-20. Cora, dim 128, 2000 epochs, seed 42,
`fdlinear`, `plain`:

| policy | `D.nnz` | accuracy | AUC | hop R2 dist | H2 exact |
| --- | --- | --- | --- | --- | --- |
| `modular.py` (2-hop ball, `shell_force`) | 115,478 | 0.9777 | 0.9953 | -- | -- |
| `walk` (the baseline) | 75,334 | 0.9702 | 0.9956 | 0.648 | 98.1% |
| `walk --cap 32` | 127,092 | 0.9702 | 0.9950 | 0.659 | 94.2% |
| `walk_edges` | 75,340 | 0.9706 | 0.9956 | 0.650 | 98.1% |
| **`nbr_walk`** | 209,542 | **0.9801** | **0.9982** | **0.676** | 7.3% |

```bash
.venv/bin/python experiments/fodiwalk/verify_modular.py
.venv/bin/python experiments/fodiwalk/bench_fodiwalk.py --graph cora \
    --pairs nbr_walk --out experiments/fodiwalk/results/walk_policies_cora.tsv
```

`verify_modular.py` proves the shared surface. It compares no `D`: the two
files no longer hold a policy in common, and they are not meant to. It drives `fodiwalk` and
`fodined` side by side with the same graph, seed and generator, and it
compares the arrays cell by cell: `sample_far_pairs`, `degree_table`, the
link prediction scores, the Hadamard feature, and every constant. Then it
asserts that each walk `D` keeps the contracts the law needs -- I4, I5 and
the plane contract. Nine checks, non-zero exit on a failure.

**What a walk shares, and what it does not.** The LONG-RANGE term is
`modular.py`'s own code: `n log10(n)` pairs at `far_weight = 100`, drawn by
`sample_far_pairs`. So are the engine, the regularizer, the loader, the
link prediction and the constants. The NEAR term is not: `modular.py`
measures the distance, and a walk takes the walk GAP, an upper bound.

**The law differs too, and it is a decision.** `modular.py` runs
`shell_force`; that family was removed on 2026-08-19
(`fodiwalk/dev-docs/CATALOG.md` section 13) and `fdlinear` replaces it.

### Do not read H2 as a quality score

H2 is how TIGHT the walk gap is against the true distance. `walk` is exact
on 98.1% of its stored pairs and `nbr_walk` on 7.3% -- and `nbr_walk` wins
every score. The layout does not need `h` to be the distance; it needs the
ORDER of `h` and enough partners (77.4 for each node against 27.8).
`--cap 32` is the control: same policy, more pairs, H2 falls to 94.2% and
the hop R2 RISES. `CATALOG.md` section 17.4.

### The exact policies are REMOVED

`--pairs ball|sampled` are gone from the package (2026-08-20), together
with `hop_from_D` and the `--k-hop` / `--hop-protocol` flags. Section 17 is
the reason: a walk does not lose to them, thus one family of `h` and one
meaning for every number. `fodiwalk/dev-docs/CATALOG.md` section 18.

`--pairs` now takes `walk`, `walk_edges` or `nbr_walk`, and an unknown
value raises with that list.

### The hop regression is out of sample

`hop_sample` draws its pairs by BFS from `--hop-sources` sources. Most of
them are pairs `D` never held, thus the number is OUT OF SAMPLE and it is
the harder one.

The in-sample protocol of `modular.py` -- score the pairs `D` STORES, with
their stored value as the target -- is gone with the exact policies. It
needs `D.data` to be a measured distance, and a walk stores a gap. It was
also EMPTY at `modular.py`'s own `K_HOP = 2`: the sentinel drops out,
`--hop-min 2` drops the ones, every target is then 2, and the reference run
prints `R2 = 1.000` for the mean baseline, which is the tell.
`fodiwalk/dev-docs/CATALOG.md` section 16.6 keeps that finding.
