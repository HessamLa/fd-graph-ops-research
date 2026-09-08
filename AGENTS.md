The following is the schema of this file
```
---
session-id:[session-id]
titl:[latest title of the session]
purpose:[the main tasks assigned to the session]
summary:[a timestamped and very concise summary of everything done under the session. One line per summary. A summary entry may include one or two more lines to point to files. timestamp at the beginning of the entry.]
```

## AGENTS

---
session-id: `evaluator` — the ROLE, claimed 2026-08-26, set with /rename on 2026-09-02. The role is the key. Two weaker ids, both recorded only as hints: the transport name churns (this session was fdmap-68 [54807a], then fdmap-65 [20555e], within one hour), and the transcript UUID 6f9d36c7-4699-47c9-add7-b451b41e4d84 is stable but NOT UNIQUE — a forked session appends to its parent's transcript and carries the parent's id, so two live sessions can share one UUID (proved by the fodiwalk session, 2026-09-02). Address by role.
title: evaluator
purpose: Own the `evaluator/` package. Build it, verify it, and control every change to it. One place to score a graph embedding, so a number from one run compares to a number from another.
summary:
- OWNER OF `evaluator/`. All changes to that package go through this session. Other sessions may read it, run it, and report bugs or slow paths. They must not edit it. Announced to every session in the repo on 2026-08-26.
- Built the package from the plan in `evaluator/dev-docs/PRD.md` and `evaluator/dev-docs/ORCHESTRATION.md`, using subagents A10-A20 (logs are dated 2026-08-22 and 2026-08-23).
- 14 modules. Two tasks: `link_prediction` and `dist_approx`. Same code from the command line and from Python.
- Six named protocols in `evaluator/config.py`. A protocol is a frozen set of settings copied from a file that produced a published number. Every result records which one it used, and whether a setting was overridden.
- Froze verbatim copies of the four original scoring implementations in `evaluator/tests/reference/`, so a comparison cannot drift when a live file is edited. Do not "fix" them.
- Rewrote `experiments/other-ge/bench_other_ge.py` (565 -> 293 lines) and `experiments/large-graph-node2vec/bench_node2vec_1M.py` to call the package instead of holding their own copies.
- Wrote `evaluator/README.md` on 2026-08-24.
- Ran the plain-english pass over every docstring: 1023 -> 835 lines, 18% cut against a 30% target. Kept the trap notes, the draw-order contracts and the measured numbers. Proved no code changed by comparing the parsed syntax tree with docstrings stripped.
- Verified `otherge` link prediction against the frozen copy: max difference 0.000e+00.
- Verified another session's `neg_draw` repair, twice: `pairs._far_pairs` matches `fodined.graph_augmentation.sample_far_pairs` in both the pairs and the generator state at 500 / 5278 / 20000 pairs, and the whole `fodined` task matches `fodined/link_prediction.py` at 0.000e+00 on all five scores and all six counts.
- CORRECTION ON THE RECORD: for `fodined` link prediction, accuracy 0.9777 and hop R2 0.630 are the DEFECT. 0.9744 and 0.616 are correct. Do not quote 0.9777 as a baseline.
- OPEN, none started: four fixes (H2 guard warnings missing from the printed table; B1 `protocol_modified` dead for a new task; H3 `PROTOCOLS` is a mutable dict; H5 the metric guard only checks one direction). A `--dry-run` to print resolved settings. A committed test suite, including the untested `far_pairs` path. H1, degree leakage, needs a design decision.
- The package is not installed. Run it from the repository root or set `PYTHONPATH`.

---
session-id: fdmap-b3 [edf61e]
title: Fodiwalk refactoring
purpose: Restructure `fodiwalk/` from a 698-line god class into stage packages, then lift the shared engine and GPU kernel into a reusable root package. No behaviour change: not one number.
summary:
- OWNER OF `forcedirected/`. Changes to that package go through this session. Other sessions may read it, run it, import it, and report defects. Ask before editing.
- Split `fodiwalk/fodiwalk.py` (698 lines, four jobs) into stage packages: `config.py`, `model.py` (200 lines, wiring only), `augment_graph/` (policies behind a registry), and the stage-3 assembly. The god class and the empty `models/` directory are deleted. Plan, defects, traps and gates: `fodiwalk/dev-docs/REFACTOR.md`.
- Built `forcedirected/` at the repo root on 2026-08-26: `ForceDirected` (renamed back from `ForceDirectedEmbedding`, no alias kept), the SELL-C-sigma kernel absorbed from `sellcsigma/`, plus `csr.py` and `optim.py`. `sellcsigma/` is removed. `fodiwalk/core/` keeps `forces.py` and `plan_contract.py` only.
- `forcedirected/` imports numpy, scipy, jax and its own modules ONLY. Nothing of this repository. So every dependency points at it, no cycle is possible, and `fodined` does not depend on `fodiwalk` to use the kernel. Keep it that way.
- Five forwarders hold no algorithm and exist so no consumer needed an edit: `fodiwalk/core/{force_directed,sell_c_sigma,csr}.py`, `fodiwalk/misc/optim.py`, `fodined/embedding/sell_c_sigma.py`.
- Wrote `fodiwalk/tests/golden.py` and `golden_baseline.json`: 16 augmentation cases hashed byte-exact plus 4 short CPU runs, in 16 seconds. It is the cheap gate for a code move. The embed half compares numbers at rtol 1e-4, because a hash of `Z` differs in the last bit between processes.
- Verified bit-identical by an independent session: `make_plan` exact, `step` rel diff 0.000e+00, Cora 2000 epochs all nine fields exact, kernel sha256 0406d2ec...bebe unchanged.
- Hardened two gates past the original plan. `test_structure.py` gained a rule that `fodiwalk` may import no root package but `forcedirected` — it reads the package list off the filesystem and counts a directory with no `__init__.py`, because Python imports one of those as a namespace package and the gate was blind to `fdwalk`, `experiments` and `archive`. `test_p1` gained `assert plan["n_split"] == 0`, a conservative proxy for the real threshold (max in-batch owner multiplicity under 3).
- GATES, and the command that is safe to quote: `.venv/bin/python -m pytest fodiwalk/tests forcedirected/tests -q -m "not big"`. Do NOT run unfiltered — `test_p5_row_cap_on_com_youtube` loads 1.13M nodes and gets OOM-killed when the machine is busy. Three tests carry `@pytest.mark.big`.
- NEVER pipe a gate run to `tail`. Bash returns the last command's exit status, so a SIGKILLed pytest reads as exit 0, and `pytest -q` prints no summary when killed — the surviving dots look like passes. Redirect to a file and read pytest's own exit code.
- `forcedirected/sell_c_sigma.py:152` names a dead `sellcsigma/PARITY.md` path ON PURPOSE. The recovery script `reconstruct_pre_unification.py` keys on that file's sha256 and its regex matches that exact string. Comment, regex and `SOURCE_SHA_2026_08_25` must change together in one edit, then the gate re-run.
- Wrote the `plain-english` rule for comments and docstrings into the repo `CLAUDE.md` under Coding, on the user's instruction.
- A full tree backup, tested end to end, is outside the repo at `/home/h/gnn/fd-graph-embedding/fdmap-backup-20260826T081231Z`. The tag `fodiwalk-pre-refactor` resolves to HEAD and means "before ALL of 2026-08-20 onward" — it is NOT a revert point for this working tree. Do not `git checkout`, `git stash` or revert to it.

---
session-id: fd084f99-dab1-4ce5-a2b5-7c74e42c28d8 -- SHARED, NOT AN IDENTIFIER.
  A fork appends into the parent transcript under the parent sessionId, so this
  UUID does not name one session. Parent, whose work this entry records through
  2026-08-31: transport ref [6bad1b]. Fork, from 2026-09-02T02:00Z: transport ref
  [f9c582], its own entry below. Refs churn; the UUID is shared; neither alone
  identifies a session. Proved 2026-09-02 by writing a unique string and finding
  it in the parent's transcript, which holds exactly one sessionId.
title: fodiwalk (parent)
purpose: Own the `fodiwalk/` package. Cut its memory use so a million-node graph runs, then measure walk policies, force functions, optimisers and schedules against node2vec.
summary:
- `fodiwalk/` is SHARED between the parent [6bad1b] and the fork [f9c582] as of 2026-09-02. The parent's work is finished and pushed at `5442889`; nothing of it is in flight. Neither owns `forcedirected/` (fdmap-b3, gone) or `evaluator/` (the evaluator session).
- 2026-08-28: rewrote the augmentation carrier row-blocked. `augment_graph/rows.py` (RowStats, RowCSR) and `augment_graph/row_merge.py` replace the global argsort with a per-block searchsorted rank-merge. Peak RSS 3055 -> 1195 MB at 150k nodes, and about 2x faster. The full 1.13M-node graph finished for the first time: 4630 MB, 91 s.
- 2026-08-28: the user ruled that `D` is not stored. Per node: three flat arrays (partner id, min hop, frequency), each with its own data type. Two strategies, precomputed and streaming. Bit-exactness is NOT a gate for this line of work; judge by peak RSS and by the score.
- 2026-08-29: streaming driver `experiments/fodiwalk-streaming/bench_stream.py`. Memory stops tracking graph size: 922 MB at 2,708 nodes, 2,749 MB at 1,696,415 nodes / 11,095,298 edges.
- 2026-08-29: dim-128 against node2vec on the 1M graph. node2vec 764.6 s / 2212 MB / acc 0.9776 / f1 0.9780 / auc 0.9984 / hop R2 0.043. fodiwalk 50 epochs const lr 1018.1 s / 1975 MB / 0.9771 / 0.9769 / 0.9970 / 0.428.
- THE EFFECTIVE-LEARNING-RATE LAW: effective lr = lr x the DC gain of the update rule, and the stability edge is 1.0. Gains: plain/sgd/velocity 1; momentum and nesterov 1/(1-beta) = 10 at beta 0.9; generalised momentum m = beta*m + alpha*dZ has gain alpha/(1-beta), so alpha+beta = 1 IS velocity. The law predicted all 16 recorded outcomes with no misses, divergences included.
- GLOBAL RULE from the user: no learning rate is ever 1.0. Use 0.999 or less, and a decay schedule starts at 0.999 or less. This binds the effective lr too.
- BASELINE DIMENSIONS from the user: 128d for accuracy, f1, auc and hop R2 — that is, for walk methods and force functions. 64d for optimisers, training schedules and convergence rates. 32d or 16d only for a quick look at a new idea.
- Hop R2 splits by average degree near 3, not by graph size, and it is structural: 10x the epochs did not move `roadnet_ca`.
- Optimiser verdict: `plain`, lr 0.999, constant. Linear decay hurts all six rules by the same amount at 50 and at 200 epochs, so it is not under-training.
- GPU is a GeForce GTX 950 with 2048 MiB. JAX preallocates 75% by default; `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` is the fix. `XLA_PYTHON_CLIENT_PREALLOCATE=false` does not help. Six dim-128 runs (`roadnet_ca`, `ncbi_taxonomy`) still get OOM — Z alone is 960 MB and 1434 MB. Not yet re-run on CPU.
- Records: `experiments/fodiwalk-streaming/FINDINGS.md` (includes a corrections section), `REPORT.md`, `experiments/fodiwalk/REPORT_1M.md`, `fodiwalk/dev-docs/CATALOG.md` section 29, `fodiwalk/dev-docs/IDEA-hierarchical-freeze.md` (a future project, not built).
- Tags on origin: `fodiwalk-pre-stagemove`, `fodiwalk-pre-memory`, `fodiwalk-post-memory`. `fodiwalk-old/` is the user's untracked reference clone — read it, write nothing.

---
session-id: transport ref [f9c582]. Forked 2026-09-02T02:00Z from the transcript
  fd084f99-dab1-4ce5-a2b5-7c74e42c28d8, which it SHARES with its parent [6bad1b]
  and therefore cannot use as a key. The role name is the key; the ref and the
  UUID are hints only.
title: dense-kernel
purpose: Remove SELL-C-sigma from `forcedirected/`, put a flat batched kernel in its place, and measure what that costs in memory and in time.
summary:
- Owns the post-fork work only. Everything before 2026-09-02T02:00Z in the entry above belongs to the parent [6bad1b]; this session inherited the record of it, not the doing of it.
- WHY: the owner ruled that a bounded batch of rows that fits on the GPU is plain batch processing, so the sorted-and-tiled layout earns nothing.
- THE DESIGN, and the correction that shapes it: a NAIVE dense batch, padding every row out to the widest row present, would be WORSE than SELL-C-sigma, because one hub row drags the whole batch to its width. The build is a flat pair array per row batch plus a segment id, reduced by `segment_sum`. No padding to the widest ROW -- that is the win over tiles. The pair AXIS is still bucketed to `PAD_TO = 1 << 16`, because in JAX a new pair count is a new shape and a fresh compile.
- BASELINE, measured 2026-09-02 on the SELL-C-sigma code before any edit, and the before side of the comparison: pubmed, `nbr_walk`, `fdlinear`, `plain`, lr 0.999, 200 epochs. 16 dimensions over seeds 42-46: embed 3.46 s, host RSS 1274 MB, GPU peak 115 MiB, acc 0.9747, f1 0.9744, auc 0.9962, hop R2 0.338. 128 dimensions over seeds 42-44: embed 11.33 s, host RSS 1281 MB, GPU peak 179 MiB, acc 0.9839, f1 0.9838, auc 0.9984, hop R2 0.445. Runner: `experiments/fodiwalk-densekernel/run_bench.sh`.
- `bench_fodiwalk.py` defaults `--lr` to 1.0, which breaks the owner's global rule that no learning rate is ever 1.0. Every run passes `--lr 0.999`.
- The driver's `rss` field is HOST memory only. The runner samples `nvidia-smi` once a second so the card peak is recorded beside it.
- `forcedirected-old/` renamed to `forcedirected_old/` so Python can import it (a hyphen cannot be). Nothing inside is edited; it is the frozen before side. `fodined` is DEPRECATED by the owner and its forwarder now reads `forcedirected_old.sell_c_sigma`.
- FOUR TRAPS, from the parent [6bad1b], all forwarded to the rewrite: the pair axis MUST be bucketed or JAX recompiles every batch; `segment_sum` needs a static `num_segments` (`bench_stream.py:287`); `fodiwalk/embed/planner.py:87` and `:117` put EVERY chunk's plan on the card and leave it, so `--chunks` does not bound device memory and a flat kernel that inherits that shape will be blamed for it; `forcedirected/sell_c_sigma.py:308` casts planes to float32 on purpose, so a float64 flat kernel fails a 1e-3 parity gate for a reason that is not a defect.
- OPEN: deleting `sell_c_sigma.py` breaks `reconstruct_pre_unification.py`, which keys on that file's sha256. That is fdmap-b3's design, b3 is gone, and the decision is the owner's.
