# Removing SELL-C-sigma: a flat batched kernel, measured and rejected

**Date:** 2026-09-02. **Verdict: the change was reverted.** The flat kernel
is 4.1x to 4.3x SLOWER and uses 59% to 72% MORE card memory than the
SELL-C-sigma layout it replaced. Embedding quality is unchanged.

The code that was built is kept at
`forcedirected_removing_sell-C-sigma_attempt/`, with the `fodiwalk` half of
the change as `fodiwalk-side.patch` inside it. Nothing of it is live.

---

## 1. What was tried, and why

`forcedirected/sell_c_sigma.py` (603 lines) implements SELL-C-sigma: it
sorts rows by how many stored neighbours they have, groups them into
chunks, pads each chunk up to a rung of a geometric ladder, and splits a
hub row across several virtual rows so no one tile is enormous.

The project owner judged it no longer needed. Their reasoning: `fodiwalk`
already embeds a bounded batch of ROWS that fits on the card, and that is
plain batch processing, so the sorted-and-tiled layout earns nothing.

**One correction was made to the plan before any code was written, and it
still holds.** A naive dense replacement -- pad every row in a batch out to
the widest row in that batch -- would be WORSE than SELL-C-sigma on a graph
with a few very high-degree nodes, because one hub row forces every other
row in the batch up to the hub's width. The owner agreed: "a plane dense
patch is clearly not the right way. the flat list idea is good."

So the build was the flat form: **one flat array of `(row, partner)` pairs
for a batch, plus a segment id per pair, reduced with
`jax.ops.segment_sum`.** No row is padded to any other row's width.

## 2. The measurement

Identical driver, identical seeds, one job at a time on an idle machine.
The "before" arm ran first, against the unmodified tree, so nothing of the
rewrite could touch it.

    graph      pubmed, 19,717 nodes, D nnz 2,238,752
    pairs      nbr_walk        law      fdlinear
    rule       plain           lr       0.999      epochs   200
    driver     experiments/fodiwalk/bench_fodiwalk.py
    runner     experiments/fodiwalk-densekernel/run_bench.sh
    rows       sellc.tsv / dense.tsv, card peaks in *.gpu.tsv

`--lr 0.999`, never 1.0: the driver's own default of 1.0 breaks the owner's
standing rule, so the runner always passes it.

The driver's `rss` field is HOST memory only, so the runner samples
`nvidia-smi` once a second and records the card peak beside it.

### dim 16, five seeds (42-46)

| | SELL-C-sigma | flat batch | change |
| --- | --- | --- | --- |
| embed | 3.46 s | **14.30 s** | **+313%** |
| augment + plan | 1.14 s | 1.00 s | -12.3% |
| host RSS | 1274 MB | 1384 MB | +8.6% |
| GPU peak | 115 MiB | **183 MiB** | **+59%** |
| accuracy | 0.9747 | 0.9745 | -0.0002 |
| F1 | 0.9744 | 0.9743 | -0.0002 |
| AUC | 0.9962 | 0.9962 | 0.0000 |
| hop R2 | 0.3382 | 0.3384 | +0.0002 |

### dim 128, three seeds (42-44)

| | SELL-C-sigma | flat batch | change |
| --- | --- | --- | --- |
| embed | 11.33 s | **48.30 s** | **+326%** |
| augment + plan | 1.07 s | 1.00 s | -6.2% |
| host RSS | 1281 MB | 1391 MB | +8.6% |
| GPU peak | 179 MiB | **307 MiB** | **+72%** |
| accuracy | 0.9839 | 0.9838 | -0.0001 |
| F1 | 0.9838 | 0.9837 | -0.0001 |
| AUC | 0.9984 | 0.9984 | 0.0000 |
| hop R2 | 0.4447 | 0.4447 | 0.0000 |

**The slowdown is not noise.** Embed time was 14.3 s on all five 16d seeds
and 48.3 s on all three 128d seeds, with no spread at all.

**Quality is intact.** Every score differs by at most 2e-4, `||dZ||` matches
to four decimals, and `dnnz` is identical seed for seed. The physics was
right; the performance was worse.

## 3. Why it lost -- and it revises the premise

**The rung ladder was doing two jobs, and the sorting was the less
important one.**

    lax.scan over rungs   BOUNDED the live intermediate to one rung
    the (R, k, d) reduce  a dense sum along k, which a GPU does well

The flat kernel keeps neither. It runs ONE `segment_sum` over a whole
chunk's 2,238,752 pairs, so it materialises an `(m, d)` intermediate and
pays a scatter-add for every row. That cost grows with `d`, which is
exactly why 128 dimensions is hurt more than 16 (+326% against +313%, and
+72% card against +59%).

So the layout was earning its keep. Not through the part that looked
elaborate -- the width sorting and the hub split -- but through the scan
that bounded memory and the dense reduce that a scatter-add cannot match.

**The one thing the flat form did win** is the host-side build: augment +
plan fell 12.3% at 16d and 6.2% at 128d, because no ladder is built, no
rows are sorted and no tiles are packed. That saving is about one tenth of
a second and it is bought at four times the embed cost.

## 4. What was learned that outlives the attempt

**A pre-existing defect, found by the rewrite and confirmed from source.**
The old `fodiwalk/model.py::forces()` did:

```python
i = min(row_start // self.chunk_rows, len(self.plans) - 1)
full = self.steps[i](...)          # only chunk i's rows are non-zero
out = drop_steady_rate(full[row_start:row_end], ...)
```

`ForceDirected.embed` defaults `batch_count=1`, so it calls
`forces(0, n)` once. That picks chunk 0, and every row belonging to chunks
1..k comes back ZERO -- every epoch, for the whole run. Those rows never
move off their random start. The old kernel never crashed on it because
`sell_c_sigma.step` always allocated a full `(n, d)` `dZ`, so the slice
always fit. The flat kernel returned only its own chunk's rows and the
shape mismatch raised immediately.

The engine's own docstring says "the batch and the chunk are the same
object", but NOTHING couples `batch_count` to `cfg.chunks`. The invariant
was written down and never enforced.

`fodiwalk/tests/golden.py`'s `walk_chunks2` case runs `chunks=2` with the
default `batch_count=1`, so **`golden_baseline.json` records the frozen-row
output as the expected answer.** It was not re-recorded -- doing so would
erase the evidence. This is open and it is the owner's call.

**The measured comparison is unaffected by that defect.** Both arms ran
`--chunks 1`, where it cannot fire.

**Four traps, from the session that wrote `bench_stream.py`.** All were
carried into the build and all held:

* the pair axis MUST be bucketed to a power of two, or every distinct pair
  count is a new XLA shape and a fresh compile;
* `jax.ops.segment_sum` needs a STATIC `num_segments`;
* `fodiwalk/embed/planner.py:87` and `:117` put EVERY chunk's plan on the
  card and leave it there, so `--chunks` alone does not bound device
  memory. A flat kernel that inherited that shape would have been blamed
  for it;
* `forcedirected/sell_c_sigma.py:308` casts every plane to `float32` on
  purpose while `D.data` is `float64`. A `float64` flat kernel would differ from
  the tiles for that reason alone.

**Parity was reached and then stopped mattering.** A 20-epoch cora oracle
matched at 8.155e-07 relative on `Z` (independently recomputed from the
saved arrays, not taken on report). The owner then ruled that bit-exactness
was never the bar: "the main objective is to gain improvements on memory
usage and run time. negligible differences between other metrics such as
accuracy and R2 are acceptable." By that bar the quality columns above pass
and the performance columns fail.

**Long-run drift is sensitive dependence, not a defect.** Running both
kernels by hand against the same `D`, seed and `lr`:

    epochs   rel_dz     rel_Z
        50   1.65e-04   8.30e-03
       200   2.08e-03   4.76e-02
       500   2.05e-03   1.20e-01
      1000   1.23e-03   3.30e-01
      2000   2.28e-03   3.43e-01

`rel_Z` grows smoothly and plateaus; it never jumps early. A per-epoch gap
of about 8e-07 compounding through a nonlinear relaxation explains it.

## 5. The fix that was NOT tried

The loss is in the reduction, not in the flat layout. **Scan the flat pair
array in fixed-size BLOCKS.** That restores the bounded intermediate and
the reused compiled shape, while keeping everything the removal was for: no
geometric ladder, no hub split, no per-row padding, no width sorting.

It is a change inside `flat_batch.step` and not a return to
`sell_c_sigma.py`. It was proposed and not built -- the owner reverted
first. Anyone picking this up should try it before concluding that the flat
form itself is wrong, because this campaign did not measure the flat form
with a bounded reduction.

## 6. Gate results, for the record

The suite was run against the flat kernel. `forcedirected/tests/test_parity.py`
was deleted by the rewrite -- every test in it exercised the removed layout
directly (ladder values, `make_plan` determinism, the padding contract).

Failures and what each one was:

| test | what it was |
| --- | --- |
| `test_golden_augment_and_embed_reproduce` | the `walk_chunks2` defect above; augment half stayed byte-identical on everything except the three SELL-C-sigma-only stat fields |
| `test_p1_fdlinear_on_cora_200_epochs` | `KeyError: 'n_split'` -- a removed layout statistic used as a proxy for owner multiplicity |
| `test_p2_the_learning_rate_ladder_at_0999` | `dz` 0.1392 against a recorded 0.1274 +/- 5e-5, at 2000 epochs; sensitive dependence, see section 4 |
| `test_b4_updateZ_dispatches_through_optim_and_defaults_to_plain` | an `np.allclose` miss at about 6e-08 on values near 0.59, which passes `rtol=1e-5`; the miss is the `atol=1e-8` floor on an element near zero |
| `test_model_py_is_at_most_200_lines` | a real regression: the chunk loop pushed `model.py` to 234 lines against a 200 budget. Fixed by moving the loop into `embed/planner.py`, which brought it to 173. |

None of these survives the revert. They are recorded because the next
attempt will meet the first four again.

## 7. Provenance

    forcedirected_removing_sell-C-sigma_attempt/flat_batch.py   the kernel
    forcedirected_removing_sell-C-sigma_attempt/fodiwalk-side.patch
                                                                the fodiwalk half
    forcedirected_removing_sell-C-sigma_attempt/oracle_*.npz     the 20-epoch oracle
    forcedirected_removing_sell-C-sigma_attempt/p2_epoch_growth_check.py
    experiments/fodiwalk-densekernel/run_bench.sh               the runner
    experiments/fodiwalk-densekernel/{sellc,dense}.tsv          the RESULT rows
    experiments/fodiwalk-densekernel/{sellc,dense}.gpu.tsv      the card peaks
    experiments/fodiwalk-densekernel/logs/                      per-run logs
    agentic-log/00.master-agent/                                the decision log
    agentic-log/01.kernel-agent/                                the rewrite log
