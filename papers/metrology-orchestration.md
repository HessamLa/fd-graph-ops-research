# evaluator v2 -- Orchestration

Created 2026-09-02. DRAFT v0.1. Companion to `PRD-v2.md`, which is the
authority on WHAT each block must contain and what it must satisfy. This
document says WHO builds each block, in what order, at what complexity,
with which model, and how the output is judged.

The `agentic-log/` tree holds the record. This campaign uses the ordinals
`20` to `39`. The v1 campaign holds `10` to `19` and the `fodiwalk`
campaign holds `00` to `05`.

Ownership: `evaluator/` belongs to session fdmap-68 (`AGENTS.md`,
2026-08-26). This campaign runs INSIDE that ownership. The coordinator of
this campaign is fdmap-68 or a session that fdmap-68 has named in
`AGENTS.md` before W0 starts. No other session edits `evaluator/` during
the campaign, and a bug report still goes to `.claude/agents/evaluator.md`.

---

## 1. Rules that govern every subagent

The v1 rules hold. They are repeated here because a subagent starts cold.

**A subagent starts cold.** It has none of this conversation. Every brief
QUOTES the invariant it must not break; it does not point at it. The brief
states what the agent must NOT touch.

**The first read is prescribed.** Every brief names the files to read, in
order: the contract before the consumer of the contract. For v2 the
contract is `config.py` (the cfg dataclasses, `FLOORS`, `APPLIES`) and
`PRD-v2.md` section 3.1 (the library policy). An agent that has not read
3.1 writes a merge-sort Kendall count, and that is a redo.

**The library is the implementation.** From `PRD-v2.md` 3.1: a routine
statistic is a library call. An agent that writes own code must cite
reason (a), (b) or (c) of 3.1 in the docstring. A brief that assigns a
block names the library call the agent must use, from 3.2. Deviating from
the named call is allowed only with a written reason in the log, and the
coordinator decides.

**The planted case is the gate, not review.** Every block is judged by
its planted test in `PRD-v2.md` B9 to B19 and by the v1 parity gate. Not
by whether the code reads well.

**Nothing v1 moves.** `tasks/link_prediction.py`, `tasks/dist_approx.py`,
`tests/reference/` are not edited. `pairs.py` and `hops.py` are edited
ADDITIVELY: a new draw or a new policy goes at the END of the draw order,
and the v1 reference tests run byte-exact after every commit that touches
them. `fodiwalk/misc/evaluation.py` is never touched.

**Self-verification is required.** An agent RUNS what it built and
reports the real output. "It imports" is a weaker bar than "the planted
path graph gives stress1 = 3.2e-9"; each brief says which bar.

**No `(n, n)`.** From `PRD-v2.md` section 8, invariant 4. An agent that
allocates an all-pairs array, even at cora size, has a defect. The test
for every block that touches all pairs runs under `tracemalloc` and
asserts the peak against `n * block * 8`.

**Model selection.**
* `haiku` -- a transcription, a file copy, a wrapper of one library call
  with a fixed signature, a docstring pass. The output is checkable by
  diff against a table in the PRD. A defect is loud.
* `sonnet` -- a block with a clear target and a planted test, where the
  library does the arithmetic and the agent does the plumbing, the
  argument order and the record shape. A defect fails a test.
* `opus` -- a contract that must survive a restructuring, an RNG order
  that must be preserved, a rule with a threshold the PRD marks
  `PROVISIONAL`, or a defect that would be SILENT (a plausible wrong
  number, a guard that never fires, a scale path that works at cora and
  dies at com_youtube).

**The redo loop.** After each agent returns: run its success criteria
myself. If any fails, do NOT patch it myself. Rewrite the brief to name
the failure and the expectation, and send it back to the SAME agent with
`SendMessage`. Escalate `haiku` to `sonnet` after one failed attempt on
one criterion; `sonnet` to `opus` after two.

**Token discipline.** A brief carries the PRD sections it needs and
nothing else. No agent reads `archive/`, `fdwalk/`, `fodined/` or the v1
`ORCHESTRATION.md`. A decision that an agent makes is written once as an
`ADR` line in its log (`DECIDE` tag, below), so the next agent reads the
line and not the reasoning.

**Logging.** Each agent appends to `agentic-log/<ordinal>.<slug>/`. One
line per entry, tags `START, PLAN, ACTION, OBSERVE, DISPATCH, CLAIM,
VERIFY, DECIDE, DEADEND, END`. A `CLAIM` is what the agent said; a
`VERIFY` is what the coordinator observed on re-run. The two are never
merged.

---

## 2. The complexity split

The table sorts every unit of work by the cost of getting it wrong. That
cost, not the line count, chooses the model.

| Unit | PRD | Work | Cost of a defect | Level | Model |
|---|---|---|---|---|---|
| U1 `config.py` | B1, B12, B16 | 6 cfg dataclasses; new fields on 6 existing protocols with `protocol_modified=False` semantics; 3 new protocols; `FLOORS`; `APPLIES`; `MappingProxyType` | HIGH. A wrong default on an existing protocol makes a v1 name score a new task differently from its record, and nothing raises | **hard** | opus |
| U2 `requirements-evaluator.txt`, `dev-docs/METROLOGY.md` copy, `dev-docs/PRD-v2.md` copy | 3.2, tree | Three files, contents given | LOW | trivial | haiku |
| U3 `io.py` | B2 | `read_context`, profile cache read/write, graph hash | LOW. Loud failure | easy | sonnet |
| U4 `context.py` | B13 | A dataclass, `validate`, the gain table, derived `effective_lr` | MEDIUM. The gain table is copied from CATALOG 29.2; a wrong gain makes the stability-edge warning lie | medium | sonnet |
| U5 `metrics.py` knn wrapper | B5 | `sklearn` brute search under `working_memory`; `pynndescent` above the cap; poincare as a callable | MEDIUM. A wrong `working_memory` unit allocates `(n, n)` and dies only at T3 | medium | sonnet |
| U6 `rank.py` | B9 | `scipy.stats` calls, plus own `tie_frac`, `tau_b_max`, `per_shell`, `h_star`, bootstrap wrapper | HIGH. `somersd` argument order; the `h_star` rule is `PROVISIONAL` and it is the headline number of class S | **hard** | opus |
| U7 `stress.py` | B10 | stress-1 (3 lines), distortion, `IsotonicRegression`, Shepard subsample | LOW-MEDIUM. Planted path graph catches the scale | easy | sonnet |
| U8 `retrieval.py` | B11 | kNN via U5; BFS truth via NetworKit; `sklearn` AP and trustworthiness with `precomputed` blocks; reconstruction via radius graph and bisection | HIGH. Three scale traps (truth matrix per block, radius bisection, T3 index) and the T&C precomputed-rank input is easy to build wrong and still return a number in `[0, 1]` | **hard** | opus |
| U9 `align.py` | B18 | Two `scipy` calls and one call into U8 | LOW | trivial | haiku |
| U10 `profile.py` | B14 | CSR to NetworKit once; four NetworKit calls; csgraph fallbacks; bipartite BFS; class rule; cache | HIGH. The class decides the regime, `applies`, and the protocol. A wrong `avg_deg` (directed count, self-loops) misclassifies every S graph as H. The fallback path must give the same class as the NetworKit path | **hard** | opus |
| U11 `structure.py` | B15 | NetworKit PLM/PLP, `KMeans`, `sklearn` NMI/ARI, own per-cluster conductance, PageRank | MEDIUM-HIGH. Conductance is own code and a sign or a `min` error is silent; seeding two libraries | medium | opus |
| U12 `floors.py` | B17 | `scipy.stats.bootstrap`; the verdict; `floors set` logic | HIGH. This block IS the noise rule. `MEASUREMENT` vs `TIE` vs `WIN` and the AND of two conditions; PLAN.md's verdict table must reproduce | **hard** | opus |
| U13 `guards.py` | B7 | Five new checks, thresholds from the PRD | MEDIUM. A guard that never fires | medium | sonnet |
| U14 `scoring.py` | B6 | Thin dict wrappers over U6, U7, U8 | LOW | trivial | haiku |
| U15 `report.py` | B8, 5.7 | Six new record blocks; `SeedReport`; `Verdict.table()`; bracket rule; v1 record loads with `None` blocks | MEDIUM. The bracket rule and the `applies` lookup; a v1 record must still print as v1 did | medium | sonnet |
| U16 `pairs.py`, `hops.py` additive changes | B3, B4 | `stratified_by_hop`, `all_edges`, blocked `neg_ratio`; `cross_component` policy and count | HIGH. RNG order. The v1 reference tests are byte-exact and every new draw goes at the END | **hard** | opus |
| U17 `tasks/dist_rank.py` | 5.2 | The step order; per-source and per-shell paths; bootstrap; cross-component | HIGH. This is the primary task of the metrology, and its step order is the contract every later comparison relies on | **hard** | opus |
| U18 `tasks/dist_stress.py`, `tasks/link_rank.py` | 5.2 | Two tasks that call U7 and `sklearn.metrics`; same pair draw as U17 | MEDIUM. They must share U17's draw so the two distance tasks score the same pairs | medium | sonnet |
| U19 `tasks/retrieval.py`, `tasks/structure.py`, `tasks/stability.py` | 5.2, B19 | Three tasks over U8, U11, U9 | MEDIUM. `stability` takes a sequence and the pair enumeration order is part of the record | medium | sonnet |
| U20 `tasks/__init__.py`, `__init__.py` registry and reservations | 5, 13 | The new names; `needs networkit` marker; `NotImplemented` reservations for temporal and dijkstra | LOW | trivial | haiku |
| U21 `multi_seed()`, `compare()`, `compare_reports()` | 5.5, 5.6 | The two public calls over U12 and U15; the protocol and `n_dim` refusal | HIGH. This is the surface a paper table comes from | **hard** | opus |
| U22 `cli.py` | B20 | Parser generated from `dataclasses.fields`; multi-EMBEDDING; `--context`; `--compare`; `--profile-only`; `floors set`; exit codes | MEDIUM. The generated parser must reproduce every v1 flag exactly | medium | sonnet |
| U23 `tests/` planted cases | B21 | One test per U6 to U12, U16 to U19; the `tracemalloc` peak assertion; `somersd` swap test | HIGH. A wrong test hides a defect of every block | **hard** | opus |
| U24 `tests/test_parity_v1.py`, `test_compare.py` | B21, section 9 | The v1 parity gate against the v2 tree; the PLAN.md verdict table | HIGH | **hard** | opus |
| U25 `tests/test_scale.py` | B21, gate 4 | Every new task on com_youtube at 1.13M under the `n2v1m` peak; marked `big` | HIGH. Runs once, costs an hour, and is the only proof of G10 | **hard** | opus |
| U26 `floors set` on cora and pubmed | gate 7 | Run the subcommand for 5 metrics, 3 seeds each; paste into `config.py` | MEDIUM. A run, then a reviewed one-line edit per metric | medium | sonnet |
| U27 baseline re-score | gate 8 | The current best method on 3 graphs, 3 seeds, every task; records to `experiments/metrology-baseline/` | MEDIUM. A run; the numbers are the day-one baseline of every new metric | medium | sonnet |
| U28 docstring and `plain-english` pass | CLAUDE.md | Every new module's docstrings in ASD-STE100; each own-code function cites its 3.1 reason | LOW | trivial | haiku |
| U29 `CATALOG.md` section 30, `evaluator/README.md` update, `AGENTS.md` entry | CLAUDE.md | The named entities of v2; the README quick start for the new tasks | LOW | easy | sonnet |

Twelve units are `hard`. Six are `trivial`. The count is honest: the
libraries take the arithmetic, and what is left is contracts, RNG order,
scale, and one rule that decides what counts as a result.

---

## 3. Dependency order

```
  W0   A20 config.py (U1)           A21 requirements + doc copies (U2)
       (opus)                       (haiku)
          |
  W1   A22 io + context (U3, U4)    A23 knn wrapper (U5)    A24 rank.py (U6)    A25 stress.py (U7)
       (sonnet)                     (sonnet)                (opus)              (sonnet)
          |                             |                       |                   |
  W2   A26 profile.py (U10)         A27 retrieval.py (U8)   A28 align + scoring (U9, U14)
       (opus)                       (opus)                  (haiku)
          |                             |
  W3   A29 structure.py (U11)       A30 floors.py (U12)     A31 guards + report (U13, U15)
       (opus)                       (opus)                  (sonnet)
          |_____________________________|_______________________|
                                        |
  W4                 A32 pairs.py + hops.py additive (U16)
                     (opus)
                                        |
  W5   A33 tasks/dist_rank (U17)    A34 dist_stress + link_rank (U18)    A35 retrieval/structure/stability tasks (U19)
       (opus)                       (sonnet)                              (sonnet)
          |_____________________________|_______________________________________|
                                        |
  W6   A36 registry (U20)           A37 multi_seed + compare (U21)
       (haiku)                      (opus)
                                        |
  W7   A38 cli.py (U22)             A39 planted tests (U23)              A40 parity + compare tests (U24)
       (sonnet)                     (opus)                               (opus)
          |_____________________________|_______________________________________|
                                        |
  W8                 A41 test_scale.py, one run on the campaign machine (U25)
                     (opus)
                                        |
  W9   A42 floors set (U26)         A43 baseline re-score (U27)
       (sonnet)                     (sonnet)
                                        |
  W10  A44 docstring pass (U28)     A45 CATALOG + README + AGENTS (U29)
       (haiku)                      (sonnet)
                                        |
  W11                Verification by fdmap-68 + two peer sessions
```

W1 runs four agents at once; they share `config.py` (read only) and touch
disjoint files. W2, W3, W5, W7, W9, W10 run two or three at once.

**Why `config.py` is alone at W0.** Every block reads a cfg dataclass and
the `APPLIES` table. Same reason as v1.

**Why `rank.py` is in W1 and not later.** U6 is the primary metric and
the `h_star` rule is `PROVISIONAL`. It goes first so the coordinator can
show the owner a per-shell table on cora by the end of W1 and confirm
the rule before four tasks depend on it.

**Why `pairs.py`/`hops.py` (W4) come AFTER the metric blocks.** The
metric blocks take `(x, y)` arrays and do not need the new draws. The
tasks (W5) need both. Putting the RNG-sensitive edit late means it runs
once, against a stable set of consumers, and the v1 reference tests are
re-run once after it and again after W5.

**Why `floors.py` (W3) precedes `multi_seed`/`compare` (W6).** The
verdict logic is tested on synthetic `(a, b, spread)` inputs in W3
before any report object exists. U21 then wraps it.

**Why `test_scale.py` (W8) is alone.** It runs once, on the campaign
machine, and takes about an hour. Nothing in W9 starts until it passes,
because the floors and the baseline must be produced by the tree that
passed at scale.

---

## 4. Assignments

Each brief below is what the agent receives. The full text of each is
written at dispatch time; what is here is the skeleton the coordinator
fills. "Read first" lists are in the order that builds understanding.

### A20 -- `evaluator/config.py` -- **opus** -- PRD B1, B12, B16

**Goal.** Six new cfg dataclasses; the six existing protocols extended
with the new cfgs at their defaults; three metrology protocols; `FLOORS`
with the five v1 floors set and every new metric `unset`; `APPLIES`;
`PROTOCOLS` as `MappingProxyType`.

**Read first.** `PRD-v2.md` sections 3.1, 5.2, 7 (B1, B12, B16), 8, 12.
Then `evaluator/config.py` as it is. Then `METROLOGY.md` sections 6 and
8 for what each field means.

**The one hard part.** An existing protocol name must run a new task and
record `protocol_modified=False`. That means the protocol dataclass
GAINS fields with defaults, and the `override` logic that sets the flag
must distinguish "the caller passed a keyword" from "the field exists
with its default". Read the current `override` before touching it.

**Must not.** Change any existing field value. Import anything but
`dataclasses`, `types`. Add a knob the PRD does not name.

**Success criteria.** The v1 protocol tests pass unchanged. `PROTOCOLS["otherge"].dr` exists and equals `DRCfg()`. `PROTOCOLS["metrology_s"].dr.cross_component == "drop"`. Assigning to `PROTOCOLS["x"]` raises. `FLOORS["rho"].source == "unset"`. A written table of every `PROVISIONAL` constant with its section-12 row.

---

### A21 -- `requirements-evaluator.txt`, doc copies -- **haiku** -- PRD 3.2

**Goal.** The pinned ranges of 3.2 in one file; `METROLOGY.md` and
`PRD-v2.md` copied into `evaluator/dev-docs/`.

**Read first.** `PRD-v2.md` section 3.2, last paragraph.

**Must not.** Edit the copied documents. Install anything.

**Success criteria.** `diff` against the source files is empty. The
requirements file lists exactly `numpy`, `scipy>=1.11`, `scikit-learn>=1.3`,
`networkit>=11`, and `pynndescent` under an `# optional` comment.

---

### A22 -- `io.py` and `context.py` -- **sonnet** -- PRD B2, B13

**Goal.** `read_context`, the profile cache, the graph hash; the
`Context` dataclass, `validate`, the gain table, derived `effective_lr`.

**Read first.** `PRD-v2.md` 5.4, B2, B13. `fodiwalk/dev-docs/CATALOG.md`
section 29.2 (the gain table's source). `evaluator/io.py` as it is.

**Must not.** Compute `effective_lr` from a typed value; it is derived
or `None`. Guess a missing field.

**Success criteria.** `Context(lr=0.1, update_rule="momentum").effective_lr == 1.0` (beta default 0.9). `Context(lr=0.5, update_rule="adam").effective_lr is None`. `validate` raises on `n_dim=64` with a `(n, 128)` `Z`. `read_profile_cache` returns `None` on a miss and the same dict on a hit. Planted tests pass.

---

### A23 -- `metrics.py` knn wrapper -- **sonnet** -- PRD B5

**Goal.** `knn(Z, queries, k, metric, exact)` over `sklearn.neighbors.
NearestNeighbors(algorithm="brute", n_jobs=-1)` under
`sklearn.set_config(working_memory=512)`; `pynndescent` above
`RECON_MAX_N`; self excluded; `poincare` as a callable.

**Read first.** `PRD-v2.md` 3.1, 3.2 (row B11 kNN), B5, section 8
invariant 4. `evaluator/metrics.py` as it is.

**Must not.** Write a distance loop. Allocate `(len(queries), n)` outside
`sklearn`'s own blocking.

**Success criteria.** On cora, `knn` with `k=5` equals a brute `numpy`
reference (written in the TEST, not in the module) exactly. Under
`tracemalloc`, peak on a 200,000 x 64 synthetic `Z` with 2,000 queries
is under 600 MB. The `pynndescent` path returns the same top-1 for 95% of
queries on the same synthetic and the record carries `knn="pynndescent"`.

---

### A24 -- `rank.py` -- **opus** -- PRD B9

**Goal.** The `scipy.stats` wrappers; own `tie_frac`, `tau_b_max`,
`per_shell`, `h_star`; bootstrap via `scipy.stats.bootstrap`.

**Read first.** `PRD-v2.md` 3.1, 3.2 (rows B9), B9. `METROLOGY.md`
6.1. The rank-criteria note's worked example and its "-b" section.

**The one hard part.** `h_star` is `PROVISIONAL` (`H_STAR_TOL = 0.02`
on cumulative ρ). Implement it as one function that takes the tolerance,
and produce the per-shell table on cora at 64d from an existing
embedding so the coordinator can show the owner what the rule picks.
Also: `scipy.stats.somersd(y, x)` puts the RESPONSE first; state this in
the docstring and in the planted test.

**Must not.** Reimplement any coefficient `scipy` provides. Compute
`tau_b_max` more than once per `y`.

**Success criteria.** The four-pair example gives `tau_b == 0.5477`
(3/sqrt(30)). A perfect layout (`x = y + 1e-9 * arange`) gives `rho ==
1.0` and `tau_b == tau_b_max(y)` to 1e-12. Swapping `x` and `y` in
`somers_d` changes the number on the planted case and the test asserts
it. `per_shell` on a path graph of 50 nodes laid out on a line gives
`rho == 1.0` at every `h` and `h_star == 49`. Bootstrap at a fixed
`random_state` is reproducible across two calls.

---

### A25 -- `stress.py` -- **sonnet** -- PRD B10

**Goal.** `stress1` with optimal scale, `distortion`, `isotonic_fit` via
`sklearn.isotonic.IsotonicRegression`, `shepard_pairs`.

**Read first.** `PRD-v2.md` 3.2 (row B10), B10. `METROLOGY.md` 6.2.

**Must not.** Use `sklearn.manifold.MDS` (it does not expose stress on a
given layout). Return a Shepard subsample above 5,000 pairs.

**Success criteria.** A path graph on a line at scale 3.7 gives
`stress1 < 1e-6`, `scale == 1/3.7` to 1e-9, `distortion_max < 1 + 1e-6`.
The isotonic fit on the same gives `isotonic_r2 == 1.0`.

---

### A26 -- `profile.py` -- **opus** -- PRD B14

**Goal.** The `Profile`; NetworKit calls with csgraph fallbacks; the
class rule; the cache.

**Read first.** `PRD-v2.md` 5.3, B14, 3.2 (rows B14). `METROLOGY.md`
section 4. `fodiwalk/make_graph/datasets.py:load` for what `A` is
guaranteed to be.

**The one hard part.** `avg_deg = 2m / n` where `m` is the UNDIRECTED
edge count: `A.nnz / 2` on a symmetric CSR with a zero diagonal. Getting
`A.nnz` gives 7.8 on cora and classifies every S graph as H. The
fallback path (no NetworKit) must give the same class and the same
`n_components`; `diam_est` may differ and the record says which method.
`Diameter(algo=EstimatedRange)` runs on the giant component only;
extract it first.

**Must not.** Hold the NetworKit graph on the `Profile`. Call NetworKit at
package import.

**Success criteria.** The four planted graphs of B14 give the stated
class, tree flag, diameter and assortativity (path, star, 10x10 grid, BA
m=4). cora gives `("H", "L", "C")` with `n_components == 78`. With
NetworKit hidden (`sys.modules` trick in the test), the class and
component count on all five are unchanged. `profile("com_youtube")` on
the campaign machine runs under 10 s and the second call under 0.1 s.

---

### A27 -- `retrieval.py` -- **opus** -- PRD B11

**Goal.** kNN via A23; BFS truth via NetworKit; `sklearn.metrics`
AP; `sklearn.manifold.trustworthiness` with `precomputed` per block;
MRR, Hits@k, overlap; reconstruction via radius graph and bisection.

**Read first.** `PRD-v2.md` 3.1, 3.2 (rows B11), B11, section 8
invariant 4. `METROLOGY.md` 6.3. The finished `metrics.py` (A23).

**The one hard part.** `trustworthiness(X, Z, metric="precomputed")`
needs the FULL distance matrix of the queries against all `n` for `X`,
which is `(n_queries, n)` hop distances. Build it per block from
NetworKit BFS, call per block, and combine the per-query terms. Prove
with `tracemalloc` that the peak follows the block and not `n`. Second:
the reconstruction radius. Bisect on a 10,000-pair sample of distances
to find the radius that yields about `m` pairs, then run
`radius_neighbors_graph` once and count exact.

**Must not.** Build an `(n, n)` anything. Break ties in hop-kNN by
anything but node id.

**Success criteria.** The planted one-hot layout gives `precision_at_k
== 1.0` at `k_mode="degree"`. On cora, `map` equals a per-query
`sklearn.metrics.average_precision_score` reference computed in the test.
`trustworthiness` on a 2,000-node grid whose `Z` is its own coordinates
is `> 0.99`. `tracemalloc` peak on 100,000 synthetic nodes with 2,000
queries under 700 MB. `reconstruction` on cora returns a count within 2%
of `m` for the radius chosen.

---

### A28 -- `align.py` and `scoring.py` -- **haiku** -- PRD B18, B6

**Goal.** `procrustes` over `scipy.spatial.procrustes` (and
`orthogonal_procrustes` for `scale=False`); `knn_jaccard` over A27; the
three scoring wrappers.

**Read first.** `PRD-v2.md` 3.2 (rows B18, B6), B18, B6. The finished
`rank.py`, `stress.py`, `retrieval.py` public names.

**Must not.** Do arithmetic in `scoring.py`. Centre or scale by hand.

**Success criteria.** `procrustes(Z, R @ Z)` for a random rotation `R`
gives `rms < 1e-9`. `knn_jaccard(Z, Z)` gives 1.0. Each scoring wrapper
returns the score names of PRD 5.2 for its task, checked by a set
equality in the test.

---

### A29 -- `structure.py` -- **opus** -- PRD B15

**Goal.** NetworKit PLM and PLP; `KMeans`/`MiniBatchKMeans`; NMI, ARI;
own per-cluster conductance; PageRank; the norm correlations via A24.

**Read first.** `PRD-v2.md` 3.2 (rows B15), B15. `METROLOGY.md` 6.6.
The finished `profile.py` for the CSR-to-NetworKit conversion (reuse it;
do not write a second one).

**The one hard part.** Conductance: `cut(S) / min(vol(S), vol(V \ S))`
from one `A @ indicator` product per cluster. A sign error or a missing
`min` returns a number in `[0, 1]` that is wrong. Plant a two-clique
graph with one bridge and assert the exact value. Seeding: NetworKit's
`setSeed(seed, True)` and `KMeans(random_state=seed)`; both in the
record.

**Must not.** Use `networkx` or `python-louvain`. Run spectral above
50,000 nodes.

**Success criteria.** Two 50-cliques joined by one edge: PLM finds 2
communities, `conductance_max == 1/(50*49+1)` to 1e-12 with
`Z` = one-hot of the community. cora: `nmi > 0.3` against labels-free
`KMeans` on an existing 64d embedding (a sanity floor, not a claim).
Two calls at the same seed give identical `nmi`.

---

### A30 -- `floors.py` -- **opus** -- PRD B17

**Goal.** `bootstrap_floor` via `scipy.stats.bootstrap`; `verdict`; the
`floors set` logic (not the CLI wiring; that is A38).

**Read first.** `PRD-v2.md` 5.6, B17. `METROLOGY.md` section 8.
`experiments/fdwalk/PLAN.md` from the line `## UPDATE 2026-08-17T23:25`
to the end of the 1.5% update, INCLUDING the verdict table.

**The one hard part.** The verdict has four outcomes and two AND-ed
conditions, and `MEASUREMENT` is not a fourth `TIE`. `n_seeds == 1` on
EITHER side, or `floor.source == "unset"`, gives `MEASUREMENT` and the
record says which of the two caused it. `TIE` is recorded WITH the
number (the 2026-08-17 defect was a tie recorded without one).

**Must not.** Edit `config.py`. Round a delta before comparing it to the
floor.

**Success criteria.** The PLAN.md 2026-08-18 verdict table reproduces:
`+0.14%` AUC is `TIE`, `+3.9%` is `WIN`, `-64%` is `LOSS`, `-8.9%` peak
RSS with `higher_is_better=False` is `WIN`. A `WIN` that fails the spread
condition alone reports `passes_floor=True, passes_spread=False,
result="TIE"`. A 1-seed input gives `MEASUREMENT` with the reason.
`bootstrap_floor` at a fixed seed is reproducible and on a planted
`(x, y)` with known variance returns a half-width within 10% of the
analytic value.

---

### A31 -- `guards.py` and `report.py` -- **sonnet** -- PRD B7, B8

**Goal.** Five new guards; six new record blocks; `SeedReport`;
`Verdict.table()`; the bracket rule; v1 record compatibility.

**Read first.** `PRD-v2.md` 5.7, B7, B8, B16. `evaluator/guards.py` and
`report.py` as they are. `METROLOGY.md` section 11 for the header.

**Must not.** Change the name, order or type of any v1 record field.
Suppress a bracketed metric.

**Success criteria.** Each new guard fires on a planted input and passes
on a clean one, and each `strict=False` path writes to `warnings`. A v1
`.jsonl` line from `experiments/other-ge/` loads through
`Report.from_dict` and `table()` matches the v1 output byte-exact. A
class S profile prints `dist_approx.r2` in brackets with the note. The
section 6.4 example renders with the given layout.

---

### A32 -- `pairs.py` and `hops.py` additive -- **opus** -- PRD B3, B4

**Goal.** `draw="stratified_by_hop"`, `draw="all_edges"`, blocked
`neg_ratio`; `cross_component="drop"|"cap"` with the count.

**Read first.** `PRD-v2.md` B3, B4, section 8 invariant 2. The v1
`PRD.md` B3 and B4 IN FULL (the draw order). `evaluator/pairs.py`,
`hops.py`, `tests/reference/`.

**The one hard part.** Every existing draw keeps its position and its
`rng` consumption. A new draw is a new function, or a branch that runs
AFTER every existing draw in the function, so the v1 path consumes the
`rng` exactly as before. Run the reference tests before the first edit
(to have the baseline) and after every edit.

**Must not.** Reorder, refactor or rename anything in the v1 path. Make
`drop` silent: the count goes into `sizes`.

**Success criteria.** `tests/reference/` passes byte-exact. The v1 parity
numbers (0.9744, 0.616 on `fodined`) are unchanged. `stratified_by_hop`
on cora returns equal counts per shell up to the smallest shell.
`cross_component="cap"` on cora sets `y == diam_est + 1` on exactly the
pairs that `drop` removes, and the count matches.

---

### A33 -- `tasks/dist_rank.py` -- **opus** -- PRD 5.2

**Goal.** The primary task. The step order: load, guards, draw, hops,
distances, global scores, per-source, per-shell, bootstrap, record.

**Read first.** `PRD-v2.md` 5.2 (`dist_rank`), 5.7, section 8. The v1
`tasks/dist_approx.py` for the step order it must PARALLEL up to the
scoring step. The finished `rank.py`, `pairs.py`, `hops.py`, `guards.py`,
`report.py`.

**The one hard part.** The draw of `dist_rank` and the draw of
`dist_approx` under the same protocol and seed must be the SAME pairs,
so the model-free and model-based numbers describe one sample. Assert it
in the test: `dist_rank(...).pairs` equals `dist_approx(...).pairs`.

**Must not.** Add a draw before the hop draw. Carry a default in the
task; every default is in `config.py`.

**Success criteria.** On cora with `protocol="otherge"`, the pair set
equals `dist_approx`'s. The record has every score name of 5.2. With
`per_shell=True` the shell table has one row per observed `h`. With
`bootstrap=200` every score has a `_ci`. `protocol_modified` is `False`
with no keyword and `True` with `n_pairs=20000` even when 20,000 is the
protocol value. Runs on pubmed under 5 s.

---

### A34 -- `tasks/dist_stress.py`, `tasks/link_rank.py` -- **sonnet** -- PRD 5.2

**Goal.** Two tasks. `dist_stress` shares A33's draw. `link_rank` scores
pairs by `-distance` and uses `sklearn.metrics.roc_auc_score`,
`average_precision_score`, at `neg_ratio` up to 1000 via A32's blocked
negatives.

**Read first.** `PRD-v2.md` 5.2, B10, B3. The finished `dist_rank.py`
(copy its skeleton; do not redesign it).

**Must not.** Draw pairs in a different order from `dist_rank`.

**Success criteria.** `dist_stress(...).pairs == dist_rank(...).pairs`.
`link_rank` on cora with `neg_ratio=1` gives `dist_auc` within 0.02 of
`link_prediction`'s `auc` (a sanity band, not equality). `ap_at_ratio`
at 100 is lower than at 1, and the record carries `neg_ratio_used`.

---

### A35 -- `tasks/retrieval.py`, `tasks/structure.py`, `tasks/stability.py` -- **sonnet** -- PRD 5.2, B19

**Goal.** Three tasks over A27, A29, A28.

**Read first.** `PRD-v2.md` 5.2, B19. The finished `dist_rank.py` for the
skeleton.

**Must not.** Let `stability` accept one `Z`. Enumerate pairs of seeds in
any order but `(i, j), i < j`, ascending.

**Success criteria.** Each record has the score names of 5.2.
`stability([Z, Z])` gives `procrustes_rms < 1e-9`, `knn_jaccard_mean ==
1.0`. `stability` with three `Z` reports over exactly three pairs.

---

### A36 -- registry and reservations -- **haiku** -- PRD 5, 13

**Goal.** The new names in `tasks/__init__.py`; the `needs networkit`
marker; `NotImplemented` reservations with the reason string.

**Read first.** `PRD-v2.md` section 13, U20 row. `tasks/__init__.py`.

**Success criteria.** `ev.tasks.names()` lists the eight tasks.
`ev.tasks.get("drift")` raises `NotImplementedError` with the reason
text. Importing `evaluator` with NetworKit hidden succeeds.

---

### A37 -- `multi_seed()`, `compare()`, `compare_reports()` -- **opus** -- PRD 5.5, 5.6

**Goal.** The two public calls over A30 and A31. `SeedReport.write()`
with the summary record. The refusal on protocol or `n_dim` mismatch.

**Read first.** `PRD-v2.md` 5.5, 5.6, 5.7. The finished `floors.py`,
`report.py`, `tasks/stability.py`.

**The one hard part.** `compare_reports` reads `n_seeds` from the summary
record and must return `MEASUREMENT` for a metric with `floor.source ==
"unset"` even when both sides have three seeds, and say so.

**Must not.** Compute a verdict outside `floors.verdict`. Compare across
protocol names.

**Success criteria.** Three cora embeddings at three seeds give a
`SeedReport` whose `spread["rho"]` equals `max - min` of the rows. A
comparison of two such reports on `auc` gives the same `Verdict` as
`floors.verdict` called by hand. Different `n_dim` raises with both
values in the message. The `.jsonl` has 4 lines: 3 runs, 1 summary.

---

### A38 -- `cli.py` -- **sonnet** -- PRD B20, section 6

**Goal.** The parser generated from `dataclasses.fields`; every v1 flag
reproduced; multi-EMBEDDING; `--seeds`; `--context`; `--context-file`;
`--cut`; `--compare`; `--profile-only`; `floors set`; exit codes 0/1/2.

**Read first.** `PRD-v2.md` section 6, B20. `evaluator/cli.py` as it is.
`config.py` (the field names ARE the flags).

**Must not.** Hand-write a flag for a cfg field. Break the paren sugar.

**Success criteria.** `python -m evaluator --help` lists every v1 flag
with its v1 help text (diff against the v1 `--help` captured before the
edit). `--max-nodes 1000` without `--cut` exits 1 with the guard message.
The section 6.4 example command runs and renders. `--compare a.jsonl
b.jsonl` prints the verdict table and exits 2 on mismatched protocols.
`floors set rho cora Z1 Z2 Z3 --n-boot 50` prints a `config.py` line.

---

### A39 -- planted tests -- **opus** -- PRD B21

**Goal.** One test file per U6 to U12 and U16 to U19, with the planted
cases of the PRD; the `tracemalloc` peak assertion for every all-pairs
block; the `somersd` swap test; the NetworKit-hidden test.

**Read first.** `PRD-v2.md` B21, and the "Success criteria" of A22 to
A35 above (they ARE the test list). `fodiwalk/tests/test_structure.py`
for the `ast` import-rule pattern.

**Must not.** Test against the module's own output as the reference.
Every reference is a planted analytic value or an independent library
call written in the test.

**Success criteria.** The suite runs in under 60 s without `big`.
Each test names the PRD block it covers in its docstring. The import-rule
test asserts section 8 invariant 1 with `ast`.

---

### A40 -- `test_parity_v1.py`, `test_compare.py` -- **opus** -- PRD B21, section 9

**Goal.** The v1 parity gate run against the v2 tree, byte-exact on the
two v1 tasks; the PLAN.md verdict table as a test.

**Read first.** v1 `PRD.md` section 9 in full. `PRD-v2.md` gate 1 and 3.
`experiments/fdwalk/PLAN.md` from `## UPDATE 2026-08-18T01:30`.

**Success criteria.** Max difference `0.000e+00` on `otherge` link
prediction and on all five `fodined` scores. The verdict table has one
assertion per row.

---

### A41 -- `test_scale.py` and the one run -- **opus** -- PRD B21, gate 4

**Goal.** Every new task on `com_youtube` at 1,134,890 nodes, under the
peak RSS of the `n2v1m` protocol on the same machine; marked `big`. Then
run it, once, and record the peak and wall per task.

**Read first.** `PRD-v2.md` G10, gate 4, section 11 rows 2 and 3.
`experiments/fodiwalk-streaming/REPORT.md` section on cost by stage
(for what "the same machine" means). `CLAUDE.md` on `pytest` and `tail`.

**Must not.** Pipe the run to `tail`. Run it while another job holds the
machine.

**Success criteria.** Every task finishes. Each task's peak is under the
`n2v1m` `dist_approx` peak on the same graph, or the report names which
one is not and why. The per-task table goes into
`experiments/metrology-baseline/SCALE.md`.

---

### A42 -- `floors set` -- **sonnet** -- gate 7

**Goal.** Run `floors set` for `rho`, `somers_d`, `stress1`, `map`,
`dist_ap` on cora and pubmed, 3 existing seeds each, `n_boot=1000`. Paste
the larger of the two floors per metric into `config.py` with the date.

**Read first.** `PRD-v2.md` B17, gate 7. `METROLOGY.md` section 8.

**Must not.** Set a floor from one graph. Round.

**Success criteria.** Five `FLOORS` entries with `source` naming both
graphs and `n_boot`. `compare()` on those metrics no longer returns
`MEASUREMENT` for a 3-seed input. The printed floors and the command
lines are in the log.

---

### A43 -- baseline re-score -- **sonnet** -- gate 8

**Goal.** The current best method (`nbr_walk / min_gap / fdlinear /
plain / streaming`, lr 0.999, 64d) at 3 seeds on cora, pubmed,
com_youtube, every task, `protocol="metrology"`, with a full `Context`.
Records to `experiments/metrology-baseline/results.jsonl`.

**Read first.** `PRD-v2.md` gate 8, 5.4, 5.5. `experiments/
fodiwalk-streaming/run_campaign.sh` for how an embedding is produced and
where it lands.

**Must not.** Score an embedding produced at lr 1.0 (the pre-rule
campaign). Omit a `Context` field that the run log provides.

**Success criteria.** 9 run records and 3 summary records per graph
task set. Every record's `context.missing` is empty for `n_dim, epochs,
lr, update_rule, effective_lr, strategy, device, machine`. A one-page
`README.md` in the directory with the summary table.

---

### A44 -- docstring pass -- **haiku** -- CLAUDE.md

**Goal.** Every new module and function has a docstring in ASD-STE100
via the `plain-english` skill; every own-code function cites its 3.1
reason `(a)`, `(b)` or `(c)`.

**Read first.** `CLAUDE.md` style rules. `PRD-v2.md` 3.1.

**Must not.** Change code. Remove a number, a name or a threshold from a
comment.

**Success criteria.** A grep for `def ` in the new modules finds no
function without a docstring. A grep for `reason (` finds one hit per
own-code function listed in 3.2 as "own code".

---

### A45 -- `CATALOG.md` section 30, `README.md`, `AGENTS.md` -- **sonnet** -- CLAUDE.md

**Goal.** The named entities of v2 in the catalog (profile, class,
context, effective lr in the record, floor, verdict, the three
protocols, the eight tasks); the README quick start for the new tasks
and `compare`; the `AGENTS.md` line that this campaign ran inside
fdmap-68's ownership.

**Read first.** `fodiwalk/dev-docs/CATALOG.md` section 29 for the entry
format. `evaluator/README.md` as it is.

**Success criteria.** Every `PROVISIONAL` constant appears in the catalog
with its section-12 row. The README example commands run.

---

## 5. Verification, W11

**V1 -- fdmap-68 (or the named coordinator).** Judge the tree against
`PRD-v2.md` sections 7, 8, 9, 10. Re-run gates 1 to 8 and paste the
output. Confirm `tasks/link_prediction.py`, `tasks/dist_approx.py` and
`tests/reference/` are byte-identical to their state at W0 (`git diff
--stat` against the W0 commit).

**V2 -- a peer session, CLI only.** Receives `PRD-v2.md` section 6 and
nothing else. From a cold start: profile a graph, score an embedding
with `--primary`, score three embeddings as a multi-seed run, compare two
`.jsonl` files. What it has to ask for is a defect of `--help`.

**V3 -- a peer session, API only.** Receives section 5 and nothing else.
Must score an embedding on a class S graph (`roadnet_ca@100k/bfs`) and
report whether the output makes clear, without reading the PRD, which
metric is primary. Must then add a ninth task (toy: degree recovery)
with one file and one registry line, and confirm its flags appear in
`--help` with no edit to `cli.py`.

**V4 -- the library audit, haiku.** Grep every new module for
arithmetic that 3.2 assigns to a library, and list each hit with the
function name and the 3.1 reason cited or missing. A hit with no reason
is a defect.

Each verifier reports to the coordinator. Every `CLAIM` gets a `VERIFY`
line before the report to the owner.

---

## 6. What the coordinator does

* This document and `PRD-v2.md`.
* Every brief. A24, A26, A27, A30, A32, A33 get the most effort; those
  six hold the primary metric, the class rule, the scale contract, the
  noise rule, the RNG contract, and the primary task.
* After W1: show the owner the per-shell table from A24 on cora and get
  the `h_star` rule confirmed or changed before W2 dispatches.
* After W2: show the owner `profile()` on all seven registry graphs and
  get the class assignments confirmed.
* Every verification. Re-run the tests; re-run the v1 parity gate; read
  the real diff of `pairs.py` and `hops.py`; run the section 6.4 command.
* The A41 run, in person, on the campaign machine.
* The report to the owner, with the observed numbers.

---

## 7. Risks of the plan itself

**A32 and A33 are one contract split over two agents.** The draw order
of `pairs.py` and the step order of `dist_rank` must agree, and `dist_rank`
must draw the SAME pairs as `dist_approx`. Both orders are in the PRD,
A33 reads the finished `pairs.py`, and A33's test asserts pair equality.

**The `h_star` rule is a guess until the owner sees a table.** That is
why A24 is in W1 and why the coordinator stops after W1. If the rule
changes, one constant changes and A24's test is re-planted; nothing
downstream is dispatched yet.

**NetworKit's API moves between majors.** `Diameter(algo=...)` and
`PLM` signatures have changed before. The version is pinned in
`requirements-evaluator.txt`, the planted tests assert the numbers, and
A26/A29 record the version in `env`.

**Four `opus` agents in W1 to W3 read the same `config.py`.** They do
not write it. If any of them needs a field that is not there, it logs a
`DEADEND` and the coordinator adds the field through A20's agent, not
through the requester.

**A41 runs once.** An hour, on a shared machine. The mitigation is that
every block's `tracemalloc` test at 100,000 to 200,000 synthetic nodes
already passed in A39, so A41 is a confirmation and not a discovery.

**Haiku on `scoring.py` and the registry.** Both are transcriptions, but
a wrong score NAME propagates into every record and every table. The
success criteria use set equality against the PRD's name lists, and one
failed attempt escalates to `sonnet`.