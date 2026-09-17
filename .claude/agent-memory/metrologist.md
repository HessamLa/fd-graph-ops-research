# metrologist — agent memory

Live state of the metrology work. Last updated: 2026-09-16.

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

## Sessions briefed

- 2026-09-02, `evaluator` owner: the campaign, the gate, and findings 1,
  3, 4 and 5 (msg `8b1e6cd9`). Not yet told findings 2, 6 and 7.
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
