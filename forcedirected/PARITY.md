# PARITY.md -- the evidence that the shared kernel is the same algorithm

**2026-08-25.** Repository at `89abaf2`. Milestone 1 of the unification.
Machine: NVIDIA GeForce GTX 950, jax 0.6.2, numpy 1.26.4, python of `.venv/`.

This document is the permanent record of ONE comparison that cannot be
repeated. It ran while `fodined/embedding/sell_c_sigma.py` and
`fodiwalk/core/sell_c_sigma.py` still held the algorithm. Milestone 2 turns
both into forwarders, thus the same script would then compare a module
against itself and pass without meaning. Script:
[`tests/m1_old_vs_new.py`](tests/m1_old_vs_new.py).

## 1. The strongest evidence is not a test

`sellcsigma/sell_c_sigma.py` was copied from `fodiwalk/core/sell_c_sigma.py`.
The copy was BYTE IDENTICAL before any edit (that file is
`forcedirected/sell_c_sigma.py` today -- section 9 -- and its sha256 has
not moved since):

| file | sha256 at the copy |
| --- | --- |
| `fodiwalk/core/sell_c_sigma.py` | `cfdbe266898a2ad3503bd3dbe14cbe5c17b29e9b6c71414c4e2e958c4152318d` |
| `sellcsigma/sell_c_sigma.py` (as copied) | `cfdbe266898a2ad3503bd3dbe14cbe5c17b29e9b6c71414c4e2e958c4152318d` |

Against the OTHER original, `fodined/embedding/sell_c_sigma.py`, the
algorithm region is also identical. Measured with `ast.unparse` after every
docstring is stripped, thus a comment or a docstring cannot hide in the
result. The whole difference is the rename that `fodiwalk` already made:

```
-def _step(Z, plan, inv_deg_ext, params, n, force_fn):
+def step(Z, plan, inv_deg_ext, params, n, force_fn):
+_step = step
```

**A refactor that changes no byte cannot change a result.** Everything
below only confirms it.

## 2. The edit made after the copy, enumerated

`to_csr` is the public name of `_to_csr`, and `_to_csr` stays as an alias.
Forced: `fodined/embedding/shell_force.py:69` imports `_to_csr`, and gate D6
(`fodiwalk/tests/test_structure.py:137`) forbids a plain module to import a
private name of another module. A shared package must offer a public one.
Same pattern as `step`/`_step`.

The complete delta against the copy: one `def` line renamed, its two call
sites renamed, one alias line, four comment lines, and a dated provenance
header. No expression changed.

## 3. `build_ladder`

Exact equality across all three modules, 45 cases:
`k_max` in {1, 2, 3, 7, 16, 64, 256, 1024, 4096} by
`base` in {1.1, 1.3, 1.5, 2.0, 3.0}.

## 4. `make_plan` -- exact, and it covers most of the file

Host side, numpy and scipy only, thus fully deterministic. Every plan array,
`inv_deg_ext` including its dtype, and the whole `stats` dict were compared
element by element. Old against new, both originals, one process.

| graph | n | result | stats |
| --- | --- | --- | --- |
| path-64 | 64 | EXACT | `cells 126, n_virtual 64, n_split 0, rungs 2, pad_frac 0.000` |
| star-1000 | 1006 | EXACT | `cells 2000, n_virtual 1004, n_split 1, rungs 2, pad_frac 0.0119` |
| cora | 2708 | EXACT | `cells 10556, n_virtual 2708, n_split 0, rungs 12, pad_frac 0.1069` |
| cora, `k_max = 16` | 2708 | EXACT | `cells 10556, n_virtual 2772, n_split 40, rungs 7, pad_frac 0.0809` |

`D` here is the raw symmetric adjacency, thus every stored pair is `h = 1`.
Planes are the real ones: `(shell_coeff_data(D), D.data)`, and `degrees` is
`degrees_from_D(D)`, which is what all seven live call sites pass.

## 5. `step` -- exact where exactness is available, and measured where it is not

`experiments/fdwalk/FINDINGS.md` lines 1100-1182 records that `step` is not
bit-reproducible run to run, by itself and independent of any change. A hub
row wider than `k_max` becomes several virtual rows that carry the SAME
owner id, thus `dZ.at[rows].add(F, mode="drop")` accumulates several values
on one address. Float32 addition is commutative but not associative. The
measured threshold is exactly 3 addends inside ONE batch.

This script therefore MEASURES the maximum in-batch owner multiplicity for
each plan, and it asserts exactness only below 3. It does not trust
FINDINGS for the number. Old and new ran in the SAME process, on the SAME
device plan and the SAME `Z` (`PRNGKey(42)`, `(n, 128)` float32), with
`shell_force` as the law and `k1 0.999, k2 1.0, k3 10.0, k4 0.01,
h_shift 1.0`.

| graph | `n_split` | max in-batch owner multiplicity | claim | relative difference |
| --- | --- | --- | --- | --- |
| path-64 | 0 | 1 | **EXACT** | `0.000e+00` |
| cora | 0 | 1 | **EXACT** | `0.000e+00` |
| star-1000 | 1 | 4 | tolerance `<= 1e-6` | `1.378e-09` |
| cora, `k_max = 16` | 40 | 10 | tolerance `<= 1e-6` | `3.219e-09` |

**Read the two exact rows as the result.** Where the scatter is
deterministic, old and new give the same bits. Where it is not, they differ
by about 1e-9 relative, which is three orders below the stated tolerance and
is the GPU's accumulation order, not the refactor.

### The `D` the recorded numbers actually came from

The four rows above use a synthetic or a raw-adjacency `D`. This case uses
the AUGMENTED `D` that `fdwalk/modular.py` embeds, rebuilt from its lines
130-165: walk pairs at `min_gap`, seed 42, 10 walks x 20 steps, window 5,
cap 16, `PAIR_BLOCK = 100_000`, plus `n * log10(n)` far pairs at weight 100.
`old` is `git show HEAD:fodined/embedding/sell_c_sigma.py` (sha256
`b4a500a2...`) loaded BY PATH, so the forwarder cannot stand in for it.

    D.nnz = 75334, max row width 498   (498 > k_max 256, thus it SPLITS)
    stats, both:  {'cells': 75334, 'n_virtual': 2711, 'n_split': 3,
                   'rungs': 12, 'pad_frac': 0.17065921000484388}
    make_plan, every array and every dtype and inv_deg_ext:  EXACT
    max in-batch owner multiplicity: 2
    step, same Z (PRNGKey(42), (n, 128) f32), shell_force:
        bit-identical: True     relative difference: 0.000e+00

**This is the strongest of the five cases.** It is the graph the recorded
numbers came from, it DOES split hub rows, and old against new is still
bit-identical -- because the multiplicity stops at 2, and two addends
commute. It also reproduces FINDINGS' `n_split = 3` and multiplicity 2
exactly, which the raw-adjacency Cora row does not and cannot.

Derived twice, independently: by this session and by `fdmap-f4 [d170fb]`,
from separate scripts. Every figure agrees.

### A note on Cora, so the next reader does not misread the table

FINDINGS reports Cora at `n_split = 3` with a max in-batch multiplicity of
2. This table reports `n_split = 0` for Cora. Both are right: they are
DIFFERENT `D`. FINDINGS embeds an AUGMENTED `D` whose rows are wider;
this script uses the raw adjacency, whose widest row is 168 and thus below
`k_max = 256`. The `k_max = 16` row is the same Cora forced to split, and it
is what gives a real graph above the threshold of 3.

## 5b. End to end: `fdwalk/modular.py` on Cora, against the recorded table

The whole pipeline through the forwarders, 2000 epochs, defaults
(walk/min_gap/plain, seed 42, n_dim 128, `PAIR_BLOCK = 100_000`), against
the recorded row of
[`experiments/fdwalk/results/g1_cora.tsv`](../experiments/fdwalk/results/g1_cora.tsv).

| field | recorded | this run |
| --- | --- | --- |
| `dnnz` | 75334 | 75334 |
| `dz` (final `\|\|dZ\|\|`) | 0.6301 | 0.6301 |
| `auc` | 0.9958 | 0.9958 |
| `r2_dist` | 0.616 | 0.616 |
| `mae_dist` | 0.930 | 0.930 |
| `acc` | 0.9744 | 0.9744 |
| `f1` | 0.9743 | 0.9743 |
| `r2_vec` | 0.789 | 0.789 |
| `mae_vec` | 0.686 | 0.686 |

**Every field, exact.** Plan stats matched too: `n_split = 3`,
`pad_frac = 0.17065921000484388`.

No tolerance was needed, and that is worth explaining rather than
celebrating. FINDINGS says `acc`, `f1`, `r2_vec` and `mae_vec` are not
reproducible to four decimals on this machine -- but it says so of runs in
general, and it also records that CORA specifically is bit-reproducible
because its maximum in-batch owner multiplicity is 2. Section 5 measures
that same 2 on this exact `D`. Thus an exact match on Cora is what the
mechanism predicts, and it is NOT evidence that the four unstable fields
are stable in general. On PubMed they would not be.

Only the timing and memory fields moved, as they must:
`t_embed` 7.7 s against 7.9 s, `t_aug` 0.4 s against 0.5 s, `rss` 1045 MB
against 1024 MB. The run was alone on the machine.

## 6. The provenance of section 1, and how to re-derive it

**The `fodiwalk` original was never committed.** Git `HEAD` (`89abaf2`)
holds an OLDER version of `fodiwalk/core/sell_c_sigma.py`
(sha256 `c9d3af95...`, before the `_step` -> `step` rename). The file this
package was copied from stood in the working tree only. No commit, no
stash, and `../fdmap_backup` holds `fodined` and `archive` only. The
forwarder has since overwritten it.

**It is re-derivable even so, and that is stronger than a witness.** Every
edit made after the copy is enumerated in section 2 and each one is
invertible. Reversing all three -- the docstring correction of section 7,
the provenance header, and the `to_csr` rename with its alias -- gives a
file whose sha256 is `cfdbe266...318d`. Collisions being infeasible, the
reconstruction IS the original, byte for byte.

    .venv/bin/python forcedirected/tests/reconstruct_pre_unification.py
    -> reconstructed sha256:   cfdbe266...318d
       pre-unification sha256: cfdbe266...318d
       MATCH

The match is evidence and not circularity: an edit left out of the
enumeration would break it.

**Two independent paths agree.** `fdmap-f4 [d170fb]` measured the sha256 on
the live file before the overwrite, and ran the reconstruction afterward.
Both give `cfdbe266...318d`.

**A caution on the session names in this document, 2026-08-26.** The
attributions to `fdmap-f4 [d170fb]` above name a session as it identified
itself at the time. That name has since been observed to resolve to a
DIFFERENT session, and refs were reported rotating as well. Thus the names
here are not durable identifiers and must not be used to route anything or
to re-open a question.

It does not weaken a single claim, because no claim in this document rests
on an attribution. The sha256 of section 1 is re-derived by the
reconstruction script; the augmented-D case of section 5 was derived in
this session from its own script; the artifact below was verified in this
session by reading and hashing the file. A second party agreeing is
recorded as corroboration, never as the evidence.

**The artifact was materialized once, 2026-08-25.** `fdmap-f4` ran the
script with an output path and got the recovered file: 558 lines, sha256
`cfdbe266...318d`. Thus the original has existed as a FILE since the
unification, and not only as a derivation. It was written to a session
scratchpad, which is temporary; this document deliberately does not name
that path, because a scratchpad is not a record. To get the file again,
run the script with an output path. To make it durable, commit the tree.
Verified in this session on 2026-08-26 by reading the file directly: 558
lines, sha256 `cfdbe266...318d`. It was still present at that moment.

**The other original needs none of this.** `fodined/embedding/sell_c_sigma.py`
IS in git at `89abaf2`, sha256 `b4a500a2...`, and a reader can check that
half directly at any time.

**The script is not a tripwire.** It pins the SOURCE hash as well
(`0406d2ec...bebe`, 2026-08-25). A later, legitimate edit to
`forcedirected/sell_c_sigma.py` makes these reversals inapplicable; the script
then says so and exits 0. It fails only when the source is still the
2026-08-25 file and the reconstruction stops matching -- which means an
edit went unrecorded, and the record is what to fix.

**Nothing here is committed.** The implementation, this document and the
gate are all working-tree files. Until the tree is committed the whole
chain is one `git checkout` from gone. Committing is the user's decision
and was not taken.

## 7. The docstring correction, 2026-08-25

The module docstring said the batch reduce has "no scatter, no atomics, one
predictable shape per batch". The clause "no scatter, no atomics" was FALSE
for the kernel as a whole: `step` ends in `dZ.at[rows].add(F, mode="drop")`,
which IS a scatter, and a split hub row puts several addends on one address.
A reader who believed it would call the kernel deterministic and would then
hunt a defect in any run that failed to reproduce.

It is rewritten to state what the layout actually gives -- contention once
per hub SPLIT and not once per stored pair -- and the old wording is kept
verbatim in a dated note above the provenance header, with the reason and
the pointer to `experiments/fdwalk/FINDINGS.md`. Nothing was removed.

**Sequenced after section 1 on purpose.** The zero-changed-lines diff was
taken and recorded against the VERBATIM copy. The correction was applied
afterward and re-checked: `ast.unparse` with every docstring stripped, against
`git show HEAD:fodiwalk/core/sell_c_sigma.py`, still gives only the two
renames and the two alias lines. The correction changed no code.

## 8. What is NOT claimed

* No bit equality of a long run. FINDINGS shows a 2000-epoch feedback loop
  amplifies one 1.0 ULP scatter to about 5.4e-06 relative, and that happens
  with no change to the code at all.
* Nothing about `archive/`. It was read for the study and never modified.

## 9. The move into `forcedirected`, 2026-08-26

`sellcsigma/` was ABSORBED into `forcedirected/`, the root package that now
holds the engine `ForceDirected` as well. The reason is scope: the kernel
has exactly one caller, `ForceDirected`, thus a separate root package for
it carried a package's cost and none of its benefit. The import rule that
put it at the root is UNCHANGED and is the whole point of the new package
-- `forcedirected` reads numpy, scipy, jax and its own modules only, thus
`fodined` and `fodiwalk` both read it and neither depends on the other.

Nothing in sections 1 to 8 is withdrawn. The move changed no byte of the
implementation:

| file | sha256 |
| --- | --- |
| `sellcsigma/sell_c_sigma.py` (before) | `0406d2ecfd74572878f6c0cf7e22036d4a89b58fa3a67f65f1707f77fd40bebe` |
| `forcedirected/sell_c_sigma.py` (after) | `0406d2ecfd74572878f6c0cf7e22036d4a89b58fa3a67f65f1707f77fd40bebe` |

Thus the reconstruction of section 6 still applies with no change to its
reversals; only `SOURCE` in the script moved. It was RUN from the new home
and printed `cfdbe266...318d`, `MATCH`, exit 0.

`sellcsigma/` was then removed. It was never committed, thus the removal is
permanent and the sha256 above is the record that nothing was lost. Where
sections 1 to 8 name `sellcsigma/sell_c_sigma.py` as a historical fact --
the copy of 2026-08-25 and its hash -- the name is kept, because that is
where the file stood on that day. Paths a reader FOLLOWS today are
repointed.

ONE stale name survives on purpose: `sell_c_sigma.py` line 152 still reads
`Parity record: sellcsigma/PARITY.md`. Its sha256 is the anchor of the
reconstruction in section 6, thus an editorial fix inside that file would
break the one route back to the pre-unification original. The comment is
wrong and the file is right; this note is the correction.

## 10. How to run the gates, and a trap that reads as a pass

**2026-08-26.** The suite, on the machine of section 1:

    .venv/bin/python -m pytest fodiwalk/tests forcedirected/tests -q -m "not big"
    -> 82 passed, 2 skipped, 3 deselected in 1869.86s (0:31:09)     exit 0

The 2 skips are `test_embed_equivalence.py` and `test_policy_equivalence.py`,
which retire themselves with `pytest.importorskip("fodiwalk.fodiwalk")` now
that the god class is gone. The 3 deselected carry `@pytest.mark.big`
(`test_contracts.py:413`, `test_parity.py:160`, `test_parity.py:246`).

**`-m "not big"` is not optional here.** The three `big` tests load
com_youtube, 1,134,890 nodes, "a few GB" by their own docstring. This
machine has 15 GB total and about 8 free, thus an unfiltered run is
SIGKILLed by the OOM killer at the first of them. Measured twice, both
times dying after exactly 45 tests.

**The trap, recorded because it produced a false report on 2026-08-25.**
An unfiltered run was piped: `pytest ... | tail -15`. Two things then hid
the kill:

* bash reports the exit status of the LAST command of a pipeline, thus the
  observed `0` belonged to `tail` and said nothing about pytest. A killed
  pytest still leaves `tail` exiting 0.
* `pytest -q` wraps its progress dots at 72 per line and prints NO summary
  line when it is killed. The surviving 45 dots were read as "45 tests
  passed". They were the 45 tests that ran before the kill.

The result was reported as "45 tests, all passed, exit 0". Every part of
that was wrong except the absence of a failure, which was not evidenced.
**Never pipe the gate run.** Redirect to a file and read pytest's own exit
code and its own summary line.

The kill is not a defect of the kernel. `test_p5_row_cap_on_com_youtube`
calls `make_graph.load`, `walks.walk_rows` and `pairs.row_cap`, and reaches
neither `make_plan` nor `step`.
