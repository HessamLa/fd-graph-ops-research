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
session-id: fdmap-68 [54807a]
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
