# Contaminated by parallel execution, 2026-08-18

Kept, not deleted.

The Cora grid B/C re-run and the lr-decay runs OVERLAPPED on one machine.
Per the rule added to `CLAUDE.md` on 2026-08-18, a concurrent run makes the
runtime and the peak memory of BOTH runs meaningless.

**What is still valid in them:** the quality metrics. AUC, F1, accuracy,
hop R2 and the final `||dZ||` are deterministic given the seed and do not
depend on what else the machine was doing.

**What is NOT valid:** every `t_aug`, `t_embed` and `rss` number.

Both sets are re-measured alone, serially. This file records the reason the
first measurement was discarded.
