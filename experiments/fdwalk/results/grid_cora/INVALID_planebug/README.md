# Invalid: the `freq` plane defect of 2026-08-18

These logs are KEPT and not deleted, as the documentation rule requires.

**Why they are invalid.** The `walk` + `--policy buckets` branch of
`build_D` returned no `freq` plane. `augment_graph` then fell back to
`planes = (shell_coeff_data(D), D.data)`, and `fdlinear` unpacks
`h, freq = planes`, thus it read `shell_coeff` -- a float in (0, 1] -- AS
the hop distance `h`, and `D.data` as the frequency. The physics was wrong
and **nothing raised an error**. The runs diverged to NaN at 2000 epochs,
and only the crash of the sklearn evaluator exposed it.

Every run here used `--force fdlinear` with `--policy buckets`, thus every
number in them is meaningless. The runs are re-done after the fix.

**The fix, 2026-08-18:** the branch builds `freq` on the same sparsity as
`near`, and a guard now STOPS any `fdlinear` run whose augmentation did not
build a `freq` plane, rather than falling back to different physics.
