# The fdhop force-law family, and the embedding store

2026-09-01 to 2026-09-08. Session `Runner`, transport ref `[6bad1b]`,
transcript UUID `fd084f99-dab1-4ce5-a2b5-7c74e42c28d8`.

Two things came out of this stretch: a store that keeps every embedding
with the settings that made it, and six new force laws measured against
node2vec and DeepWalk at their published parameters.

---

## 1. The store

`data_cache/embeddings/<dataset>/<n_dim>/<datetime>-<method>-<hash>/`, one
directory per run, holding `Z.npy` and `config.json`. The spec is
`data_cache/embeddings/README.md`. 60-odd embeddings across cora, pubmed,
wordnet and com_youtube.

`<hash>` fingerprints the SETTINGS, not the output, so a repeat is visible
at a glance. `hash_spec` versions the field list: adding a config field
does not silently invalidate old hashes.

**Three defects the store found in itself, all recorded rather than hidden:**

- **Spec 1 could not tell two runs apart.** Both `fdhop` runs of 2026-09-04
  fingerprinted `eb56e1fb` while the law between them had changed, because
  the change lived in uncommitted code. Spec 2 adds `method.git_commit` and
  fills `force.params`. It does not save those two: the durable fix is to
  commit before a run.
- **`environment.peak_rss_mb` was never a peak.** It was ONE reading taken
  after the embedding, so it missed augmentation, the hungriest phase. A
  com_youtube run measured 5.34 GB with `ps` while writing 1392 MB into its
  config. A sampler thread fixes it from 2026-09-07. **Every record before
  that date understates its memory.**
- **A law can be added and its scalars not recorded.** `fdhop_all_freq`
  wrote `force.params = {}` for four runs because the driver's `LAW_PARAMS`
  had no entry. Back-filled from the run log, with a `notes` line saying so.

**One record left without a trace.** The cora `fdhop` run `260904-062153`
(no-decay law, accuracy 0.9574) is not in the store and there is no log
entry for its removal. Recorded in `runs.log` as unaccounted for.

---

## 2. The laws

All at dim 128, 200 epochs, `walk_edges`, `plain`, lr 0.999 constant,
`min_gap`, 10 walks x 20 steps, window 5, seed 42.

| law | attraction | cora acc | cora hop R2 | pubmed acc | pubmed hop R2 |
| --- | --- | --- | --- | --- | --- |
| `fdlinear` | h == 1 | 0.9692 | 0.7073 | 0.9712 | 0.5170 |
| `fdhop` | h == 1 | 0.9882 | **0.7119** | **0.9915** | **0.6194** |
| `fdhop2` | h == 1 | 0.9872 | 0.6927 | 0.9830 | 0.4435 |
| `fdhop_min` | h == 1, deg | 0.9493 | 0.1899 | 0.9316 | 0.2663 |
| `fdhop_all` | every h | 0.9905 | 0.6593 | 0.9914 | 0.5840 |
| `fdhop_all_freq` | every h, freq | **0.9948** | 0.2749 | 0.9898 | 0.2992 |

**`fdhop` is the one to keep.** It wins hop distance on both graphs and is
within 0.007 of the best accuracy. Nothing has beaten it at the task the
project cares about.

`fdhop_all_freq` sets the best accuracy in the store on cora and loses 61%
of the hop R2. `fdhop_min` loses half the neighbours (55% kept on cora,
51% on pubmed) and loses two thirds of the hop R2 with them.

---

## 3. THE STABILITY LAW, and it is measured, not argued

A force that is LINEAR in `x` makes each row a spring of constant

    k1 * sum over v of (the attraction weight) / deg(u)

and the engine divides the row sum by `deg(u)` at
`forcedirected/sell_c_sigma.py:485`, where `deg` counts `h == 1` entries
only. **Above 1.0 the map expands and `Z` goes non-finite.**

    fdhop            gain exactly 1.00 on every row, by construction:
                     it attracts on h == 1 and divides by that same count.
    fdhop_all        attracts on ~28 pairs per row, still divides by ~4.
                     k2=1.0: cora 2.85, pubmed 3.58  -> DIVERGED
                     k2=2.0: cora 0.47, pubmed 0.56  -> ran
    fdhop_all_freq   freq also multiplies the attraction; freq averages 44
                     on cora and reaches 1020.
                     k2=1.0: cora 375.2, pubmed 264.2
                     k2=7.0: cora 0.93,  pubmed 0.55 -> the first stable k2

**`k2` is the lever, not `k1`.** 86% of stored pairs are far, the far sum
is dominated by `h = 2`, and doubling `k2` cuts that weight by
`exp(-2) = 7.4`. Raising `k1` from 0.999 to 1.0 moves the gain by 0.1%:
measured, hop R2 changed by 0.0001.

Three divergences were predicted from this arithmetic before the run and
all three happened. `experiments/embeddings/runs.log` holds them.

The division itself is inherited, not invented: the same line is in
`archive/root-2026-08-16/fdge_jax_sell_c_sigma/embedding/sell_c_sigma.py:349`.
`experiments/fdwalk/log/2026-08-16T0020.md:586` records why -- without it
the law goes to NaN at `lr >= 0.1`, because the step is then proportional
to the degree. It costs geometry: a hub sits about 20x further from its
neighbours than a leaf. `no_deg_norm` turns it off and **no stored run has
ever used it.**

### The architecture question this raises

The user recorded it in `my-journal.md` on 2026-09-08 and committed it as
`5803987`: **the parameters and functions are not defined where they
belong.**

This session's own findings are the evidence for that. Three places:

- **`1 / deg(u)` is force-law policy living in the kernel.** It is applied
  at `forcedirected/sell_c_sigma.py:485`, and `fodiwalk/embed/degrees.py:41`
  says outright that `no_deg_norm` gives the law "that the fdlinear
  specification writes" -- so the division is NOT in the specification. A
  law cannot opt out of it without a flag that reaches across two packages,
  and `fdhop_all` diverged precisely because the divisor counts a set the
  law does not use.
- **The force factor is computed in stage 2, not stage 3.**
  `model.augment_graph` calls `_build_planes` at `fodiwalk/model.py:147`,
  so `exp`, `h / freq` and the `deg_le` mask are all built while the graph
  is being augmented. `fdlinear_fused` makes the cost explicit: fusing
  turns `h / freq` into a BUILD-time quantity, and a law of the form
  `h / freq**beta` then cannot sweep `beta` without rebuilding `D`.
- **A new law needs edits in three files that must stay in step** --
  `FORCE_PLANES` and `FORCE_FN` in `embed/forces.py`, `PLANE_BUILDERS` in
  `embed/planes.py`, `PLANE_CHECKS` in `embed/plan_contract.py`. The
  asserts catch a mismatch, which is why `fdhop_min`'s 1/0 mask was caught,
  but the split is the reason the mismatch is possible.

Nothing here was changed on that account. It is recorded because the laws
added this week are what surfaced it.

---

## 4. The baselines, at their published parameters

The first baselines were run at 10x20/window 5 to match fodiwalk's walk
budget. That is not a published setting for either method, so all six were
DELETED and re-run:

    node2vec   Grover & Leskovec 2016: 10 walks x 80 steps, window 10,
               1 epoch, negative sampling, p and q as named
    deepwalk   Perozzi et al. 2014: 80 x 40, window 10, 1 epoch,
               hierarchical softmax

**This changed the conclusion.** node2vec's hop R2 on cora went 0.2305 ->
0.4674 on the parameters alone. The earlier comparison flattered fodiwalk.

**node2vec's search bias was never being tested.** Every early run used
`p = q = 1`, at which the second-order walk degenerates to a uniform one
and `node2vec_walks` never executes. Fixed: the method name now carries the
bias, `node2vec_<p>_<q>`, and `q` in {0.5, 1.0, 2.0} was measured.

`q = 0.5` (DFS-like) beats `q = 1.0` beats `q = 2.0` on hop R2, on all four
graphs. The `evaluator` session's Spearman rho gives the same ordering
independently. A tuned node2vec still does not reach `fdhop`: 0.5301
against 0.7119 on cora, 0.2361 against 0.6194 on pubmed.

**`fdhop` with a biased walk does nothing.** cora, pubmed and com_youtube
at `q` in {0.5, 1.0, 2.0}: every span is inside the seed spread. The walk
bias helps node2vec and not this law.

**DeepWalk beats both fodiwalk laws on hop distance at a million nodes**
-- com_youtube 0.6236 against `fdlinear` 0.5420 and `fdhop` 0.4658. The
claim "fodiwalk beats the baselines on hop distance everywhere" is FALSE
and was withdrawn. `fdhop` still takes accuracy there, 0.9856 against
0.9551.

**The walk setting does almost nothing.** 10x20/w5 against 10x80/w10, four
graphs, both laws: the largest gap is 0.0060 hop R2.

---

## 5. What was tried and taken back out

`fdunit` and `fdunit_freq` reached the store and were then rewound out of
the code on the user's instruction. Their records remain and are NOT
reproducible from the current tree.

`fdunit`'s first form was wrong -- one signed magnitude per row -- and
collapsed to 0.69 accuracy. The corrected form reached 0.9593 on cora,
still below `fdhop`. Neither could run on the kernel: the far term is a
per-ROW aggregate, so it does not fit `force_fn(x, planes, params) ->
per-pair magnitude`. A host loop ran it, and that loop reproduces the
kernel's accuracy to 0.0004 but not its hop R2 (0.7512 against 0.7119),
which is float32 accumulation order compounding over 200 epochs.

---

## 6. Two contracts that caught defects

**I3, the pad convention.** `fdhop_min`'s `deg_le` plane was first written
as a 1/0 mask. `check_plan` rejected it: a real pair with `deg(u) > deg(v)`
carried 0 alongside a non-zero `h`, which reads as a corrupt pad. The plane
is now +1/-1 with 0 reserved for padding -- the sentinel trick
`fdlinear_fused` already used.

**The 300-line cap.** `embed/forces.py` passed it at seven laws. The cap is
raised to 400 for that file with the reason written into
`SIZE_EXCEPTIONS`: the D1 defect this gate stops was a file doing FOUR
JOBS, and this file does one.

---

## 7. What is NOT established

- **One seed everywhere.** The recorded seed spread in hop R2 is 0.039 to
  0.046, which is wider than several gaps quoted above.
- **Nothing converged at 200 epochs.** Every config in the recorded grid
  gained +0.12 to +0.22 hop R2 from 50 to 200.
- **`fdhop` has only ever been measured with `walk_edges`.** There is no
  `nbr_walk` + `fdhop` run, and `nbr_walk` is the augmentation the method
  description names.
- **`k4 = 1.0` is not a swept optimum.** It is the value that reproduces the
  law as first written. `kr` has never been varied. `fdlinear` keeps
  `k4 = 0.01`, so that comparison carries a 100x difference in decay.
- **`as_skitter` has node2vec rows only** -- all four fodiwalk runs were
  killed by the OOM killer at exit 137. `roadnet_ca` and `ncbi_taxonomy`
  were scheduled and never ran.
- **Rank, stress and neighbour recall have no parity reference.** Link
  prediction and hop regression are verified bit-exact against frozen
  copies; the geometry columns are not.

---

## 8. Verification by another session

The `evaluator` session recomputed every recorded metric independently:
**246 comparisons across 41 records, maximum absolute difference
4.441e-16**, the `n_jobs=-1` thread-order noise floor. It also caught a
claim of mine that was wrong -- I reported two records sharing a
fingerprint when only one was left -- by counting instead of trusting the
message.

Records: `evaluator/reports/260904-store-scorecard.md` and
`260905-store-scorecard.md`.
