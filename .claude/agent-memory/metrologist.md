# metrologist — agent memory

Live state of the metrology work. Last updated: 2026-09-17.

## Identity

Role: **Metrologist**.

- Name: `Metrologist [571a41]` since the user's `/rename` on 2026-09-16.
  Earlier transport names: `fdmap-67 [1c9625]` (2026-09-07),
  `fdmap-93 [571a41]` (2026-09-16, before the rename). Names churn.
- Transcript id, a hint only:
  `4c4d8881-cb2a-4399-9bf6-776f2cdfeb61`. A forked session inherits its
  parent's id, so it is not unique.
- On 2026-09-16 the evaluator owner was `Evaluator [b2ebf5]` (it was `[20555e]` on 2026-09-02).

No entry for this role exists in `AGENTS.md` yet.

## The metrology-v2 campaign

Goal: extend `evaluator/` in place to implement `papers/metrology.md`, by
the plan in `papers/metrology-orchestration.md`. Record:
`agentic-log/A00.metrology-coordinator/` (log, `prompts/`, `artifacts/`)
and `agentic-log/manifest.md`. Agents use an `A` prefix (`A20`..`A45`)
because `20.catalog-agent` already held the bare ordinal.

### Built, verified by re-run, and committed

| Unit | File | Commit |
| --- | --- | --- |
| A20 | `config.py`: 6 cfg dataclasses, frozen `Protocol`, `PROTOCOLS` as `MappingProxyType`, `metrology_h/_s/_quick`, `FLOORS`, `APPLIES`, 7 PROVISIONAL constants | `9203324` |
| A21 | `requirements-evaluator.txt`; `dev-docs/METROLOGY.md`, `dev-docs/PRD-v2.md` | `9203324`; `c294ffc` |
| A22 | `context.py` (effective-lr law, checked against CATALOG 29.2); additive `io.py` | `9203324`; `aceffb9` |
| A23 | `metrics.knn` (sklearn brute; pynndescent above `RECON_MAX_N`) | `aceffb9` |
| A24 | `rank.py` | `c094012` |
| A25 | `stress.py` | `c094012` |
| A40a | `tests/test_parity_v1.py` + `conftest.py`, the v1 parity gate | `b34f58b` |

Notes:

- `config.get(name)` still returns the v1 `(LPCfg, DACfg)` pair, so the two
  frozen v1 task files stay byte-identical. New tasks use
  `config.protocol(name)`.
- `Context` carries `beta` and `alpha`, which `PRD-v2.md` 5.4 omits; the
  gain law needs them.
- `DSCfg.min_hop` defaults to 2 so `dist_stress` draws the same pairs as
  `dist_rank`. A different default makes the two tasks score different
  pair samples.
- A23 and A24 died on the session rate limit; the coordinator verified
  their work directly and reproduced A24's cora table cell for cell.

### Held

W2 onward (`profile.py`, `retrieval.py`, `align.py`, `scoring.py`, then
the tasks, CLI and gates) waits for the owner's `h_star` ruling. The owner
deferred it on 2026-09-07 to read further.

### Parity gate

`.venv/bin/python -m pytest evaluator/tests/test_parity_v1.py -q`,
about 2.5 min. Asserts arrays, the generator state and scores at 1e-12.

| Date | Result |
| --- | --- |
| 2026-09-02 | 8 passed, 2 skipped |
| 2026-09-07 | RED: `KeyError 'f1_score'`, a name mismatch, no score moved. Fixed in the harness with `ref_key()`. Then 8 passed, 2 skipped |
| 2026-09-16 | 8 passed, 2 skipped, exit 0, 197 s. Output: `agentic-log/A00.metrology-coordinator/artifacts/verify-parity-260916.txt` |

## Findings for the owner

Each is measured. Numbers 2 to 7 are independent of `h_star`.

1. **The `h_star` rule does not work on cora.** `rho_cum` peaks at h=3, the
   first shell it can be computed on, and the whole curve spans 0.09. The
   rule returns 3 at tolerance 0.01, 0.02 and 0.05 on all three seeds.
   `rho_shell` decays to noise by h=9 and `x_mean` flattens at h=9, so the
   horizon is 8 or 9. Table:
   `agentic-log/A24.rank-agent/artifacts/cora-per-shell.txt`. OPEN,
   deferred.
2. **`h_star_isotonic` is not a horizon.** On a planted case whose true
   horizon is hop 4, it reads 8.0 (equal shells) and 4.6 (decaying
   shells); no tolerance from 0.02 to 2.0 recovers 4. It now returns
   `h_star_isotonic_share`, which shows it rests on 0.06% to 0.25% of the
   pairs. The module is right; `PRD-v2.md` B10's claim about it is not.
   OPEN.
3. **`somers_d` has two implementations.** `rank.somers_d` calls
   `scipy.stats.somersd`, O(n²): 116 s on 16,723 pairs against 0.01 s for
   rho and tau-b. `eval_store.py:109-113` uses the exact identity and
   records `somers_d_via="identity"`. The identity matches scipy to
   0.000e+00. Sampling is no substitute: at 8,000 of 16,723 pairs the
   spread is 1.33x the 1.5% floor and above the seed spread. OPEN: switch
   `rank.py` to the identity.
4. **`PRD-v2.md` B9 has the `somersd` argument order backwards.** Proved by
   pair counting: the four-pair case gives `D_yx` = 0.5 = `somersd(x, y)`;
   the PRD's `somersd(y, x)` gives 0.6. Code is right. OPEN: text fix.
5. **"Restricted to pairs at hop h" cannot be computed**
   (`METROLOGY.md` 6.1, `PRD-v2.md` B9): `y` is constant inside a shell.
   `rank.per_shell` uses shell h with h-1. OPEN: bless or redefine.
6. **Distortion depends on a scale nobody fixed.** On one cora embedding:
   16.08 at the stress-optimal scale (what `eval_store.py:118` reports),
   7.31 at the distortion-optimal scale, 53.50 as the literature's
   scale-free `c = r_max / r_min`. OPEN: report `c`, and name the scale of
   every distortion figure.
7. **`METROLOGY.md:88` is wrong twice.** Its "hop R2 reads 0.00 to 0.02"
   for class S is contradicted on wordnet at 128d (fodiwalk 0.096 to
   0.156, baselines 0.022 to 0.042); the 64d-vs-128d confound is not
   isolated. And it calls `wordnet` a tree: it has 84,427 edges on 82,115
   nodes in one piece, 2,313 more than a tree. `ncbi_taxonomy` is a tree
   exactly. OPEN: text fix.

Better evidence for finding 7's intent: on wordnet, hop R2 and rho rank
the three node2vec arms in OPPOSITE orders (q=1.0: highest R2 0.042,
lowest rho 0.011), and node2vec q=1.0 beats deepwalk on R2 (0.042 vs
0.023) while deepwalk's rho is 15x higher (0.171 vs 0.011).
Source: `evaluator/reports/260905-store-scorecard.md:104-118`.

### Also seen

- Somers' D has a ceiling of `1 - tie_frac(y)` (0.847789 on the cora
  sample), so it does not remove the ceiling problem tau-b has.
- The `f1` -> `f1_score` rename is now committed (`3c261e4`). Records
  written before it carry `f1`.

## Measured facts other sessions reuse

- Cora pair sample (seed 42, reject draw, 20,000 pairs): 3,277 have no path
  and are dropped, 16,723 scored, `tie_frac_y` 0.152211, `tau_b_max`
  0.920755. `eval_store.py` gives the same numbers. Drop rate by graph:
  cora 16.4%, as_skitter 0.1%, pubmed, wordnet, com_youtube 0.
- Seed spread, cora 64d, three seeds, one pair sample: rho 0.4281, 0.4349,
  0.4339 (sd 0.0036). Do not transfer it to another graph or dimension.
- Shared embeddings: `data_cache/metrology-baseline/{cora,pubmed}_64d_s{1,2,3}.npy`.
  Streaming `nbr_walk`, `fdlinear`, `plain`, lr 0.999, effective lr
  0.999, 50 epochs. None diverged.
- Every dataset, measured: `data_cache/README.md` (2026-09-07).
- The store (`data_cache/evaluator/store-scores/`) is single-seed, under
  `n2v1m` with `max_pairs=50,000`, so every row is `protocol_modified` and
  a MEASUREMENT only. It calls `rank`, `stress` and `metrics.knn`
  directly; no metrology task exists yet.

## Divisor-change study (Runner, 2026-09-17)

The owner's position, 2026-09-17: a variance above 3% is considerable; the
metrics that matter most are acc, f1_score, auc and hit@10. Runner's data:
`/tmp/seeds_{old,new}.json`, table
`agentic-log/00.master-agent/artifacts/equivalence-percent.txt`, report
`.../seed-variance-report.md`. Earlier audit by this role:
`agentic-log/02.metrologist-audit/review.md`. Answer sent: msg `024975ce`.

- Runner's table reproduces exactly. At a margin of ±0.5% of the mean,
  acc, f1_score and auc still pass.
- **For near-1.0 metrics, judge on the error scale** (1 − metric). ±1.5% of
  the mean lets p2 auc error grow 11.7x. On that scale, p2 acc changes
  +10.00% [−1.03, +21.03] and f1_score +10.19% [−0.83, +21.21], about 3 of
  2,112 test pairs per seed. Unsettled, not an effect.
- The 3% line is read as seed CV (a statement about the metric). The
  refactor verdict uses TOST on the paired SD. CV is a poor scale for
  r2_dist, whose mean can sit near 0.
- `hits_at_k` exists only as a name (`config.py:630`, `:694`, `:718`).
  Three meanings: OGB link Hits@K; retrieval hits (metrology 6.3); the
  store's `rec@10` (recall). The owner must choose one. OPEN.
- The 20 embeddings of that study were not saved, so hit@10 needs a
  re-run. Recommended: 10 paired seeds, save Z, commit the NEW arm first,
  fix every definition and margin beforehand.
- The store's "−5.52% rf_r2" (cora fdhop, seed 42) is NOT the divisor
  change. Runner paired `c0fedf2b` (09-06, `ea9ae67` dirty, rf_r2 0.7458)
  with `7222527f` (09-14, rf_r2 0.7046). `7222527f` matches the 09-04
  builds `0ba3b1b3` and `6047dac7` (acc identical, rf_r2 +1.404e-07), so
  the 09-06 dirty build is the outlier. Runner chose the baseline as the
  old record with the HIGHEST accuracy: a selection bias (regression to
  the mean). Runner's NEW arm is now committed: `f9fa83f` on
  `explore/seed-variance`.
- `7222527f` IS post-change, settled by rebuild (2026-09-17). Its record
  says `712dece` dirty and it was built 9.5 h before `0cd5264` was
  committed, so the record alone could not show it. Clean rebuilds, by
  Runner and again by this role, bit for bit: both 09-04 builds equal clean
  `712dece`; both 09-14 `7222527f` builds equal clean `0cd5264`;
  `c0fedf2b` differs from all by 4.747. Pre vs post max |dZ| 9.298e-06, so
  the divisor move barely changes this config at 200 epochs. Records:
  `agentic-log/A00.metrology-coordinator/artifacts/rebuild-260917/`,
  Runner's report section 9a. CLOSED.
- The store records the code version under `method.git_commit` and
  `method.git_dirty`, but no digest of the uncommitted diff. The run hash
  (`make_embedding.py:142`) includes the commit only, so two different
  dirty trees on one commit share a hash. Demonstrated 2026-09-17: a
  clean `712dece` build was named `...-7222527f`, the name the store
  already uses for dirty post-change builds. One hash, two code states. The
  owner of `make_embedding.py` decides the fix.

## ICLR 2027 optimizer map, section 5.5 (review, 2026-09-17)

Data: `experiments/for-iclr2027/embeddings/{cora,pubmed}/64/` (dim 64,
fdhop walk_edges, 200 ep, const lr, seeds 42 7 56). Script and table:
`agentic-log/A00.metrology-coordinator/artifacts/map55-260917/`. Answer to
Runner: msg `b162b0a1`.

- The Evaluator's rule: blown up if final mean row norm > 10x the median of
  plain const 0.999 (cora 5.516, pubmed 5.287). Sound on cora: healthy
  0.707-2.08x, blown at 34.6x and above.
- **I read the pubmed norms while verifying, so the cut can no longer be
  set blind on pubmed.** Remedy given: freeze 10x as proposed. Under it,
  pubmed also blows up sqn 0.02 seed 56 (18.9x). "Any threshold 5x-30x
  gives the same split" fails on pubmed. Lesson: before reading a
  held-out set to verify a rule, freeze the rule, or verify on the
  training set only.
- **lr_eff = 1 is not the divergence edge.** momentum is healthy at lr_eff
  2 and blown at 5; nesterov is healthy at 5 and blown at 10. Same on both
  graphs, 3/3 seeds. The standing no-lr_eff-1 rule is a conservative
  margin. OPEN for the owner.
- nesterov 0.5 (rho 0.870-0.874, 3 seeds) beats plain const 0.999
  (0.827-0.844) only because of the larger step. At equal lr_eff about 1,
  plain, momentum and nesterov do not differ. Runner's "plain ~0.64"
  reference matched no plain cell.
- **Record defect:** `make_embedding.py:336` gives `dc_gain` 0.0 to every
  rule except plain, sgd and velocity, so momentum and nesterov store
  `effective_lr` 0.0; beta is not recorded. Plot lr_eff from the rule and
  lr. The owner of `make_embedding.py` decides the fix.

## Sessions briefed

- 2026-09-02, `evaluator` owner: the campaign, the gate, and findings 1,
  3, 4 and 5 (msg `8b1e6cd9`). Not yet told findings 2, 6 and 7.
- 2026-09-17, `Runner`: the divisor study (msgs `024975ce`, `97111360`, `2e6163c4`) and the 5.5 map review (msg `b162b0a1`), section above. Waiting for the owner's choices, then the pre-registration text.
- 2026-09-07, `fdmap-9c`, writing `papers/fodiwalk/fodiwalk.md`: citation
  guidance for its Metrology section (msgs `ed8c7016`, `6c05147e`,
  `44f39ed3`). It reports fdhop as single-seed measurements, dropped the
  "0.00 to 0.02" line, and deferred a wordnet 3-seed run.

## Record defect

Coordinator log lines from `[2026-09-02T09:36` to the clock correction
entry carry extrapolated times, not `date -u` values. Their order is right;
their clock times are not.

## Audit: seed-variance study of the 1/deg(u) move (2026-09-16T16:53:50Z)

Review: `agentic-log/02.metrologist-audit/review.md`. Measured on
`/tmp/seeds_{new,old}.json` (5 seeds x 2 arms, cora 64d, p1 and p2):
- TOST at the 1.5% floor passes on p1 (all metrics) and on p2 acc, f1 and
  auc. It FAILS on p2 r2_dist (90% CI up to +1.04e-02 against a margin of
  8.53e-03). About 16 paired seeds would close it.
- The acc step is 1/2112, not 1/1056.
- At seed 42 both arms pass every p1 pin. p2 NEW matches neither pin set,
  and p2 OLD matches the pre-09-14 pins.
- p2 cross-arm correlation: r2 0.976, dz 0.923, acc 0.258.
- Advised: p1 stays a replay test (acc/f1 tol 1e-3, auc 2e-4, mae 7.5e-4).
  p2 becomes a 3-seed quality gate with tolerances from the paired SD, and
  its dz pin goes.
- dz is one drop-mask draw. The claim that the mask sets most of its
  spread is UNVERIFIED (cell F).
- 2026-09-16T17:00:54Z follow-up (section 8, dz cells, lr 0.999):
  - My M6 mask hypothesis is REFUTED: row-norm CV is 0.84, and the last
    mask is about 20% of the dz seed variance. The drop-key stream is the
    largest one-factor source (SD 2.00e-02 of 2.90e-02).
  - dz is still falling at epoch 200 (-2.6e-03 per epoch).
  - Drop off: final dz is 9.317.
  - lr 0.999 raises final dz by about 3.6e-04 against lr 1.0.

## ICLR 2027 paper claims, checked for `fdmap-96` (2026-09-21T17:34Z)

`fdmap-96 [abb502]` writes `fodiwalk_iclr2027.tex` from `draft_v1.0` and
verifies every number against
`evaluator/reports/260917-iclr2027-preliminary.md`. It asked me first
whether I am the Evaluator (msg `93b4acb4`: no, I am the Metrologist), then
sent 6 comparative claims (msg `510dcd38`).

**RULING, reusable: there is no measured floor for `rho`.**
`evaluator/config.py:605` sets `"rho": _UNSET`, and so are `tau_b`,
`somers_d` and the rest of `dist_rank`. Only auc, accuracy, f1_score, r2
and mae have one, all at relative 0.015 (`config.py:585`). So under PRD-v2
5.6 EVERY rho comparison in this repo is a MEASUREMENT, never a WIN, at any
size and any seed count. 0.231 and 0.002 are the same class. Do not carry
0.015 across to rho: it was measured on scores that saturate near 1.0,
while rho spans 0.005 to 0.87 in this report.

Checked against the 17:27Z build of the report:
- Claim 1 (cora +0.101, pubmed +0.115): arithmetic correct. NOT budget
  matched — deepwalk's displayed cell is 3,200 steps per node, fodiwalk's
  is 200. Told them to print steps per node in the caption.
- Claim 2 (wordnet +0.185 at dim 128): correct. Sign reverses at dim 16
  (deepwalk leads by 0.223). 0.348 is not fodiwalk's best wordnet number:
  fdlinear 0.365, fdhop FLAT 0.393. fdlinear over fdhop is 1.5 sd = TIE.
- Claim 3 (flat weight): cora +0.187 and wordnet -0.045 match. **pubmed
  does not: the report says +0.231, the paper says +0.232** (rounded means
  give 0.232). Paper must quote the source.
- Claims 4 and 5 (budget span 0.002 < sd 0.006; q sweep <= 0.002): both
  correct, both TIE. Open for Runner: does plain `walk_edges` equal
  `walk_edges_pq` at q=1? If not, "q in {0.5,1,2}" names 2 members.
- Claim 6 (refuses significance tests): correct, keep verbatim.

**RULING on "standard deviations of the difference" (15, 33, 25.7, 61.6,
3.3 in the report): omit, do not caveat.** The evaluation seed is fixed at
42, so the spread is embedding-seed spread and says nothing about another
split; rho's pairs share nodes so no ratio is a valid test anyway; and the
rule uses the seed spread as a THRESHOLD, not a denominator. The honest
route to significance is to vary the EVALUATION seed and give a paired CI
or TOST against a pre-fixed margin. That measurement does not exist.

**DEFECT to report to the evaluator owner:** the report's section 4.1
prose says the flat weight "WINS" on wordnet by 0.045 at 3.3 sd. `rho` has
no floor, so that verdict is not available. The paper's own wording
("sensitivity to the weight rule") is correct and must not be changed to
match the report.

**Source-file warning given:** the report is generated by
`data_cache/evaluator/for-iclr2027/gen_report.sh`, is uncommitted, and was
rebuilt 2026-09-21T17:27Z — after fdmap-96 read it (it quoted 2,336 scored;
the file now says 2338 scored, 2381 made of 2,277 planned). Pin a copy or a
commit before verifying. The "more runs than planned" count needs its
author's explanation.

**Identity note:** this session now reports as `fdmap-a0 [7e24bf]`. The
`/rename Metrologist` label did not survive. Address by ROLE.

## Evaluator owner found again (2026-09-21T17:39Z)

`Evaluator [46dd7a]`, Opus 5, was `fdmap-cc` earlier the same day, and ran
on Haiku 4.5 earlier in its own session. It owns `evaluator/` and scores
the ICLR 2027 runs. Its rule for quoting its report: **quote a table cell,
never a sentence.** The tables are generated; the prose is typed, and it
found two stale prose figures on 2026-09-21. It reports 2,338 runs scored,
0 failures, cora/pubmed/wordnet complete at dims 16/32/64/128, no large
graph yet, on branch `explore/seed-variance`.

Sent it (msg `81c66a96`): the "WINS" defect in section 4.1 with suggested
replacement wording; a request to confirm pubmed +0.231 against the paper's
+0.232; a request to name the two prose figures it fixed, because
`fdmap-96` verified against the older build; a request to commit or pin the
report, since it regenerates under a session that is quoting it; the advice
to drop or rename the "sd of the difference" column, which contradicts its
own "method spread, not sample spread" line; and the
`make_embedding.py:336` effective_lr defect.

## RULING: a threshold mark needs no floor (2026-09-21T17:45Z)

The Evaluator accepted all six points (msgs `81c66a96`, reply, `87ab3345`)
and asked whether its section 7 "x" mark — mean row norm above 10x the
plain reference — needs a measured floor.

**Ruling: no.** A floor governs a COMPARISON verdict and stops it being
read off noise. A condition mark on one run makes no comparison, so PRD-v2
5.6 does not reach it. The rule that DOES reach it is the gate rule
(`metrology.md:302-303`): fixed before the run, stated in the file. Three
conditions: freeze 10x dated in the generator; define the unit (per seed or
per cell mean); print the ratio beside the mark.

**Boundary to hold:** the mark is a label, not a filter — rho is printed
for marked cells and nothing is dropped. If a marked cell is ever excluded
from a mean or a ranking, the threshold becomes a data-selection knob and
needs pre-registration plus the effect shown both ways.

Re-stated my disclosure: I read the pubmed norms on 2026-09-17 before any
threshold was frozen, so the cut-off can no longer be chosen blind on
pubmed. Freeze 10x, which was set on cora.

**NEW DEFECT, measured: section 7 states no seed count and its cells are
3 seeds.** From `config.json` under
`experiments/for-iclr2027/embeddings/cora/64`, momentum: seeds 7, 42, 56 at
lr 0.999, 0.5, 0.2, 0.05, 0.02; 11 seeds (7, 42, 56, 88, 101, 123, 256,
512, 777, 1024, 2027) at lr 0.099 only. Section 6 has a seeds column,
section 7 has neither a count nor a spread. Told fdmap-96 not to quote any
section 7 number until it does.

**effective_lr defect, now measured exactly.** 296 of 2,381 run folders
record `effective_lr` 0.0: 74 each of adam, momentum, nesterov, sqn. Rule
counts: plain 1,147; velocity, sqn, nesterov, momentum, adam 74 each. The
gain table's `"sgd"` key matches nothing in this store. CORRECTION to my
own earlier advice: "multiply by 10" is valid ONLY for momentum and
nesterov (1/(1-0.9)); adam and sqn scale adaptively, no constant gain
exists, leave them blank. Not a risk in the Evaluator's tables: nothing it
publishes reads `effective_lr` (it confirmed by grep).

**The Evaluator's fixes, all in the generator so they survive a rebuild:**
verdict words removed from sections 1 and 4.1; a new first bullet in "How
to read these numbers" stating rho has no measured floor, citing
`config.py:605`; the "sd of the difference" column renamed "gap / seed
spread" with a bullet saying it is not a test statistic; the header now
reads run folders against cells.csv population, which explains the
2,336/2,338 split (disk count against CSV count, both honest, neither the
table population) and the "more runs than planned" count (owner ADDED cells
after the plan: matched-budget baselines, wordnet dimensions, deferred
large-graph q values).

**Four stale prose figures it fixed**, all from a 9-seed build never
re-typed at 11 seeds: 0.240 -> 0.239 at dim 16; 0.459 -> 0.462 at dim 16;
"deepwalk beats fdhop by 0.219" -> 0.223; the fdlinear/fdhop ratio 1.4 ->
1.5. It then checked every hand-typed figure against cells.csv; ten others
verified exactly. Relayed to fdmap-96 (msg `7836ac58`), with the confirmed
pubmed +0.231 (unrounded 0.778522 - 0.547236 = 0.231286; only pubmed
changes under rounding).

Pending: the Evaluator commits the report and cuts a pinned dated copy
under `evaluator/reports/`, then sends the path and build stamp.

## The blow-up mark, closed (2026-09-23T07:41Z)

The Evaluator implemented the ruling and answered the open unit question.

**The unit is PER CELL, and it always was.** `summarize.py` takes the
MEDIAN over a cell's seeds of each run's mean row norm, and divides by the
median of the reference cell (same graph, dim and force setting, under
plain/const/lr=0.999). So a cell can sit under the line while one seed sits
over it. That is my pubmed sqn lr 0.02 case exactly: the cell median clears
10x, seed 56 (18.9x) does not. The cell never disagreed with the rule. The
defect was that the unit was undocumented. It is now written down, and a
second column `ratio_max` prints the worst single seed.

Frozen at 10x, dated 2026-09-21, in both the code and the report. The
section 7 legend carries the unit, that the cut was set on cora (largest
healthy 2.08x, smallest blown 34.6x, so any cut 3x-30x agrees there), an
explicit DO NOT call the split threshold-insensitive in general with my
pubmed counter-example, and my 2026-09-17 disclosure as the reason it is
frozen rather than tuned. Ratios print wherever they reach 2x.

The label-not-filter boundary is now in the legend and the `summarize.py`
docstring. "diverged" is reserved: no non-finite values, `diverged.keys`
empty.

Section 7 now prints mean +/- std with the seed count per cell, and its
legend opens by stating most cells hold 3 seeds, that momentum lr 0.099 is
the only 11-seed row, and that plain 0.831 against velocity 0.836 is not a
separation. My seed lists matched its own.

Commit `6690fec` "evaluator: drop verdict words, freeze the blow-up line,
pin a copy". A rebuild runs behind Runner's roadnet_ca embedding at nice
19; the pinned path and build stamp come to me and fdmap-96 together.

**To tell Runner:** the dead `"sgd"` key in `make_embedding.py:336` means
the table silently defaults EVERY future rule to 0.0, not just today's
four.

## Identity (2026-09-23)

This session is `fdmap-64 [6fa355]`. It was `fdmap-a0 [7e24bf]` on
2026-09-21 and `Metrologist [571a41]` before that. A `/rename` on
2026-09-23 set the label to `Metrologis` (missing the final t). Transport
names churn and the label is unreliable. ADDRESS BY ROLE.
