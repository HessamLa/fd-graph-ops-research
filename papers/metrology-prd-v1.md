# evaluator v2 -- Product Requirements Document

Created 2026-09-02. DRAFT v0.1. Authority on WHAT the `evaluator` package
must contain to implement `METROLOGY.md`, what each block reads and
writes, and what each block must satisfy. A companion `ORCHESTRATION.md`
will say WHO builds each block and in what order; it is not written yet.

Relation to the existing package: `evaluator/` exists (PRD of 2026-08-22,
owner session fdmap-68, two tasks, six protocols, parity-verified). This
document does not replace that PRD. It EXTENDS the package. Every
signature that the 2026-08-22 PRD froze stays frozen, every protocol keeps
its numbers, and the parity gate keeps passing. What is new is stated as
new. A reader of the old PRD needs only this document to see the delta.

Source of the requirements: `METROLOGY.md` (2026-09-02) and the
rank-criteria note. Where this document and `METROLOGY.md` disagree,
`METROLOGY.md` wins and this document has a defect.

---

## 1. Purpose

One package measures the quality of a graph embedding and produces a
record that another session, another paper, or another year can read
without the person who ran it.

The existing package answers "is the number comparable" with the named
protocol. It does not yet answer the four questions the metrology adds:

1. **Which metric applies to this graph.** Hop R2 reads 0.00 on a road
   network or a tree at any epoch count, and AUC reads 1.0000 on the same
   graph. The package must know the class of the graph and print the
   metric that applies beside the one that does not.
2. **What sits between the layout and the number.** Both current tasks go
   through a trained model. A model-free score (rank correlation, stress,
   retrieval) is the primary evidence of the metrology, and none exists.
3. **When a difference counts.** The 1.5% floor and the seed-spread rule
   live in a Markdown file and in the head of whoever writes the report.
   The package must apply them and print the verdict.
4. **What context produced the number.** Dimension, epochs, seeds,
   effective learning rate, strategy, device, machine. Today `tags` carry
   these as free text, so nothing checks them.

The primary user is a Python caller: a benchmark script, a campaign
runner, an agent session. The CLI is the second surface, for a human at a
shell and for a caller that is not a Python process.

---

## 2. Goals

1. **G1 One code path, two surfaces.** Kept from v1. API and CLI reach the
   same task functions. A surface holds argument parsing, printing and
   error shaping, and no measurement.
2. **G2 Backward compatibility.** Every v1 signature, protocol, record
   field and CLI flag keeps its meaning. The parity gate of v1 section 9
   passes unchanged. A v1 caller that upgrades sees new fields in the
   record and nothing else.
3. **G3 The metrology's metric families as tasks.** Model-free distance
   fidelity (rank, magnitude), neighbourhood retrieval, model-free link
   prediction, structure without labels, stability. Each is one task
   module plus one registry line.
4. **G4 The graph profile.** Every record carries the five structural
   numbers of the metrology and the class they give.
5. **G5 The measurement context.** Every record carries the context
   fields of `METROLOGY.md` section 1. A field the caller does not give is
   `null`, never guessed, and the record says which fields are `null`.
6. **G6 The verdict.** A comparison of two records, or of two sets of
   seeds, returns `WIN`, `LOSS` or `TIE` under the noise-floor rule, with
   the floor and the spread that decided it.
7. **G7 Multi-seed as a first-class call.** One call scores `k` embeddings
   of the same method and returns the mean, the spread, and the
   per-seed rows. Seed spread is half the noise rule and today it is
   computed by hand.
8. **G8 Per-metric noise floors.** Each metric carries a floor, with the
   date and the bootstrap that produced it. A metric with no floor cannot
   produce a verdict; it produces a measurement.
9. **G9 Guards, extended.** Kept from v1 and extended by the traps of
   `METROLOGY.md` section 12: `dZ`-scale, degree-zero rows, cross-component
   policy stated, truncation cut named.
10. **G10 Scale.** Every new task runs on a 1.13M-node graph inside the
    peak memory of the existing `n2v1m` protocol on the same machine.
    A model-free score must be cheaper than the model-based one it
    replaces, or it names why.
11. **G11 Machine-readable, append-only.** One JSON record per run, one
    line per record in `.jsonl`. A record is never edited; a correction
    is a new record with a `supersedes` field.

---

## 3. Non-goals

- `evaluator` does not build embeddings. It reads them.
- `evaluator` does not own the dataset registry. `fodiwalk.make_graph.
  datasets` owns it. `evaluator` reads it when the graph is a name.
- `evaluator` does not plot. It writes the numbers a plot needs (the
  Shepard pairs, the per-shell table, the dimension sweep rows) to the
  record, and a separate script draws them.
- `evaluator` does not run campaigns. It scores one embedding per call, or
  `k` embeddings of one method in the multi-seed call. Sweeps over
  dimensions, epochs and methods are the caller's loop.
- `evaluator` does not do node classification. Decision of the project.
- `evaluator` does not score temporal graphs in this version. The
  temporal metrics of the metrology (drift, rows touched, future-link AP)
  are a later PRD. This document reserves their names and nothing else.
- `evaluator` adds no dependency beyond `numpy`, `scipy`, `scikit-learn`.
  `networkit` stays optional (the exact PLL backend). Community detection
  for section 6.6 uses `scipy.sparse.csgraph` and `sklearn.cluster`, not
  `networkx` or `python-louvain`; see B15 for the choice of algorithm.
- `evaluator` does not decide the open items of `METROLOGY.md` section 13.
  Where an item is open, this document names the default it ships with,
  marks it `PROVISIONAL`, and a change is one constant.

---

## 4. Target tree

New files are marked `+`. Changed files are marked `~`. Unmarked files do
not change.

```
evaluator/
  __init__.py              ~  exports the new tasks, profile(), compare(), multi_seed()
  __main__.py
  cli.py                   ~  new task flags, --context, --compare, --profile
  config.py                ~  new dataclasses, new protocol fields, FLOORS table
  io.py                    ~  reads a context sidecar; writes the profile cache
  pairs.py                 ~  adds draw="stratified_by_hop", draw="all_edges"
  hops.py                  ~  adds cross-component policy; returns inf or cap
  metrics.py               ~  adds pairwise distance for retrieval (block-wise)
  scoring.py               ~  adds rank_scores, stress_scores, retrieval_scores
  guards.py                ~  adds check_dz_scale, check_degree_zero, check_cut_named
  report.py                ~  adds the profile, context, floors, verdict blocks
  profile.py               +  the graph profile and the class
  context.py               +  the measurement context dataclass and its validation
  floors.py                +  the noise floors, the bootstrap, the verdict
  rank.py                  +  ρ, τ-b, Somers' D, tie fraction, attainable max
  stress.py                +  stress-1 with optimal scale, distortion, Shepard pairs
  retrieval.py             +  kNN in Z, precision@k, MAP, MRR, Hits@k, overlap, T&C
  structure.py             +  communities on A, k-means on Z, NMI, ARI, conductance
  align.py                 +  Procrustes; kNN Jaccard across two Z
  tasks/
    __init__.py            ~  the registry, with the new names
    link_prediction.py
    dist_approx.py
    dist_rank.py           +  section 6.1 of the metrology
    dist_stress.py         +  section 6.2
    retrieval.py           +  section 6.3
    link_rank.py           +  section 6.4, model-free
    structure.py           +  section 6.6
    stability.py           +  section 6.7, takes two or more Z
  tests/
    reference/             (frozen, unchanged)
    test_rank.py           +  the four-pair worked example; perfect layout = max
    test_stress.py         +  a planted metric; stress-1 = 0 at the right scale
    test_retrieval.py      +  a planted kNN; p@k = 1
    test_profile.py        +  known graphs: a path, a star, a grid, a tree
    test_floors.py         +  bootstrap reproducibility at a fixed seed
    test_compare.py        +  the verdict table of PLAN.md 2026-08-18
    test_parity_v1.py      +  the v1 parity gate, run against v2
  dev-docs/
    PRD.md                 (v1, unchanged)
    PRD-v2.md              +  this document
    METROLOGY.md           +  copied from the approved draft
```

---

## 5. The public API

### 5.1 The v1 surface, unchanged

```python
ev.link_prediction(graph, Z, *, protocol="default", seed=42, ...) -> TaskResult
ev.dist_approx(graph, Z, *, protocol="default", seed=42, ...)     -> TaskResult
ev.evaluate(graph, Z, *, tasks=(...), protocol="default", seed=42,
            tags=None, strict=True, **task_kw)                      -> Report
```

Signatures as frozen in the v1 PRD section 5.3. `evaluate()` gains one
keyword, `context=None` (section 5.4), and the new task names in `tasks`.

### 5.2 The new task functions

Every task takes `(graph, Z, *, protocol, seed, rng, strict, **cfg)` and
returns a `TaskResult`. A keyword left at `None` takes the protocol value;
a keyword given explicitly sets `protocol_modified`. Same rule as v1.

```python
def dist_rank(graph, Z, *, protocol="default", seed=42, rng=None,
              n_pairs=None, n_sources=None, min_hop=None, max_hop=None,
              metric=None, hops=None, pair_draw=None,
              per_source=None, per_shell=None, bootstrap=None,
              cross_component=None, strict=True) -> TaskResult
# scores: rho, tau_b, somers_d, tie_frac_y, tau_b_max, tau_b_ratio,
#         n_pairs_scored, and if per_source: rho_src_mean, rho_src_std;
#         and if per_shell: shell[h] = {n, rho, r2, x_mean, x_std} for each h;
#         and if bootstrap: <score>_ci = [lo, hi]

def dist_stress(graph, Z, *, protocol="default", seed=42, rng=None,
                n_pairs=None, n_sources=None, min_hop=None, max_hop=None,
                metric=None, hops=None, pair_draw=None,
                shepard=None, cross_component=None, strict=True) -> TaskResult
# scores: stress1, scale, distortion_mean, distortion_max, distortion_p99,
#         isotonic_r2, h_star; and if shepard: shepard = {x, y, fit} arrays

def retrieval(graph, Z, *, protocol="default", seed=42, rng=None,
              k=None, k_mode=None, n_queries=None, metric=None,
              truth=None, block=None, strict=True) -> TaskResult
# k_mode: "fixed" | "degree"   truth: "adjacency" | "hop_knn"
# scores: precision_at_k, recall_at_k, map, mrr, hits_at_k, knn_overlap,
#         trustworthiness, continuity, reconstruction

def link_rank(graph, Z, *, protocol="default", seed=42, rng=None,
              max_pairs=None, neg_ratio=None, metric=None, neg_draw=None,
              strict=True) -> TaskResult
# scores: dist_auc, dist_ap, dist_ap_at_ratio, neg_ratio_used

def structure(graph, Z, *, protocol="default", seed=42, rng=None,
              n_clusters=None, community=None, centrality=None,
              strict=True) -> TaskResult
# scores: nmi, ari, conductance_mean, conductance_max,
#         rho_norm_degree, rho_norm_centrality, n_communities

def stability(graph, Zs, *, protocol="default", seed=42, rng=None,
              k=None, metric=None, n_queries=None, strict=True) -> TaskResult
# Zs: a sequence of two or more (n, d) arrays, same method, different seeds
# scores: procrustes_rms, procrustes_rms_std, knn_jaccard_mean,
#         knn_jaccard_std, pairs_evaluated
```

### 5.3 The profile

```python
def profile(graph, *, seed=42, diameter_samples=None, cache=True) -> Profile
```

Returns the five numbers and the class (B14). It is called once by
`evaluate()` and once by the CLI, and cached by graph hash so a campaign
does not pay it per run.

```python
Profile(n, m, avg_deg, max_deg, diam_est, n_components, giant_frac,
        assortativity, is_tree, is_bipartite, has_weights, has_labels,
        cut, cls)          # cls in {"H", "S", "B", "L", "W", "D", "C", "T"} as a tuple
```

A graph can be in more than one class (`cora` is `("H", "L", "C")`). The
first element is the PRIMARY class and decides the regime and the primary
metric. Primary is `S` if `avg_deg <= 3` or `is_tree`; else `H`. `B`, `L`,
`W`, `C` are secondary flags. `D` and `T` are reserved and never set in
this version.

### 5.4 The context

```python
Context(method=None, n_dim=None, epochs=None, seeds=None, lr=None,
        update_rule=None, dc_gain=None, effective_lr=None, strategy=None,
        device=None, machine=None, peak_rss_mb=None, wall_s=None,
        regime=None, level=None, tier=None, campaign=None, supersedes=None)
```

The caller fills what it knows. `n_dim` is filled from `Z.shape[1]` and
checked against the caller's value; a mismatch raises. `effective_lr` is
computed as `lr * dc_gain` when both are given, and `dc_gain` is looked
up from `update_rule` when the rule is one of the eight known names
(table in B13). `regime` is derived from `n_dim` and the profile class
when not given. `level` is the caller's claim and is not checked.

`evaluate(..., context=Context(...))` or `context={"n_dim": 64, ...}`.
`tags` stays for anything that is not a context field.

### 5.5 The multi-seed call

```python
def multi_seed(graph, Zs, *, tasks=(...), protocol="default",
               seeds=None, context=None, tags=None, strict=True,
               **task_kw) -> SeedReport
```

`Zs` is a sequence of embeddings of the same method; `seeds` is the
parallel sequence of seed values, or `range(len(Zs))`. Runs `evaluate()`
once per `Z` and once `stability()` over all of them. Returns a
`SeedReport` with `rows` (one `Report` per seed), `mean`, `spread` (max
minus min), `std`, and `n_seeds`. `SeedReport.write()` appends every
per-seed record AND one summary record with `"kind": "seed_summary"`.

### 5.6 The comparison

```python
def compare(a, b, *, metric, floor=None, higher_is_better=None) -> Verdict
def compare_reports(a: SeedReport, b: SeedReport, *, metrics=None) -> dict[str, Verdict]
```

`a` and `b` are `SeedReport`s, `Report`s, or record dicts. `Verdict` is:

```python
Verdict(metric, a_mean, b_mean, delta, delta_rel, floor, floor_source,
        spread_a, spread_b, spread_max, passes_floor, passes_spread,
        result)   # result in {"WIN", "LOSS", "TIE", "MEASUREMENT"}
```

`result` is `"MEASUREMENT"` when either side has `n_seeds == 1` or when the
metric has no floor. A `MEASUREMENT` is never a `WIN`. The two conditions
of the noise rule are both reported so a reader sees which one failed.
`compare_reports` refuses two reports with different `protocol` names or
different `n_dim` and raises with both values in the message.

### 5.7 The result objects

`TaskResult` and `Report` are the v1 objects with new blocks. The v1
fields keep their names and positions. A record is a dict:

```
{
  "kind":       "run" | "seed_summary" | "comparison",
  "task":       ...,            v1
  "protocol":   ..., "protocol_modified": ...,      v1
  "cfg":        {...},          v1, resolved settings
  "scores":     {...},          v1 plus the new keys
  "sizes":      {...},          v1
  "timing":     {...},          v1
  "env":        {...},          v1
  "tags":       {...},          v1
  "profile":    {...},          NEW, B14
  "context":    {...},          NEW, B13, with "missing": [names]
  "floors":     {metric: {"floor": f, "date": d, "source": s}},   NEW
  "warnings":   [...],          NEW, the guards that passed with a note
  "applies":    {metric: true|false|"secondary"},   NEW, B16
  "supersedes": null | record_id,                   NEW
  "record_id":  sha1 of (graph hash, Z hash, task, cfg, seed)      NEW
}
```

`applies` says, per metric, whether the metrology considers it primary
for this graph's class. A report table prints a non-applying metric in
brackets. The number is still there.

---

## 6. The CLI

```
.venv/bin/python -m evaluator GRAPH EMBEDDING [EMBEDDING ...] [tasks] [options]
```

More than one `EMBEDDING` means the multi-seed call. `--seeds 1,2,3`
names them; absent, they are numbered.

### 6.1 Task flags

```
--link-prediction, -lp      v1
--dist-approx, -da          v1
--dist-rank, -dr            NEW  rank correlation, section 6.1 of the metrology
--dist-stress, -ds          NEW  stress and distortion
--retrieval, -rt            NEW  neighbourhood retrieval
--link-rank, -lr            NEW  model-free link prediction
--structure, -st            NEW  communities and centrality
--stability, -sb            NEW  needs two or more EMBEDDING paths
--all                       every task that applies to the graph's class
--all-including-secondary   every registered task, applying or not
--model-free                dist-rank, dist-stress, retrieval, link-rank
--primary                   the primary metric set for the graph's class only
```

### 6.2 Per-task options

Each option carries the task prefix, as in v1.

```
--dr-pairs N  --dr-sources N  --dr-min-hop N  --dr-max-hop N
--dr-per-source  --dr-per-shell  --dr-bootstrap N  --dr-cross-component drop|cap
--ds-pairs N  --ds-sources N  --ds-shepard  --ds-cross-component drop|cap
--rt-k N  --rt-k-mode fixed|degree  --rt-queries N  --rt-truth adjacency|hop_knn
--lr-max-pairs N  --lr-neg-ratio F  --lr-neg-draw reject|far_pairs
--st-clusters N  --st-community label_prop|spectral  --st-centrality degree|pagerank
--sb-k N  --sb-queries N
```

The paren sugar of v1 (`--task "dist_rank(n_pairs=20000, per_shell=true)"`)
works for every task.

### 6.3 Global options

```
--protocol NAME        v1 names, plus the new ones of B12
--metric NAME          v1
--seed N               v1
--seeds LIST           NEW  one seed per EMBEDDING
--max-nodes N          v1; the cut is recorded in the profile (B14)
--cut bfs|subtree      NEW  required when --max-nodes > 0. No default.
--results PATH         v1
--tag K=V              v1
--context K=V          NEW  repeatable; a Context field. Unknown key raises.
--context-file PATH    NEW  a JSON or YAML-free (JSON only) sidecar
--no-strict            v1
--quiet                v1
--json                 v1
--profile-only         NEW  print the profile, run nothing
--compare A.jsonl B.jsonl [--on METRIC,...]
                       NEW  read two seed_summary records, print verdicts
```

### 6.4 Output

The table gains a header line with the profile and the class, and a
bracket around a metric that does not apply:

```
  cora  n=2708 m=5278 avg_deg=3.90 max_deg=168 diam~19 comps=78 assort=-0.066  class=H,L,C
  protocol=default seed=42 n_dim=64 epochs=50 eff_lr=0.999 strategy=streaming
     dist_rank       1.1s  rho=0.612 somers_d=0.531 tau_b=0.498 (max 0.712) tie_y=0.31 pairs=20000
     dist_stress     0.9s  stress1=0.213 distortion_mean=1.41 h_star=4
     retrieval       2.4s  p@10=0.71 map=0.64 mrr=0.83 overlap=0.58 trust=0.91 cont=0.88
     link_rank       0.3s  dist_auc=0.981 dist_ap=0.977 ap@1:100=0.41
     link_prediction 0.6s  auc=0.9952 f1=0.9730
     dist_approx     4.0s  [rf.r2=0.436 rf.mae=1.35]      <- bracket: model-based, secondary
```

On a class S graph the bracket moves to `dist_approx.r2` and to `auc`
with the note `not the metric for class S`, and `dist_rank.shell` prints
the per-shell rows up to `h_star`.

`--compare` prints one line per metric:

```
  rho      a=0.612±0.011  b=0.598±0.009  Δ=+0.014 (+2.3%)  floor=1.5%  spread=0.011  WIN
  auc      a=0.9952       b=0.9948       Δ=+0.0004 (+0.04%) floor=1.5%  spread=0.002  TIE
  map      a=0.64         b=0.61         Δ=+0.03            floor=none               MEASUREMENT
```

Exit code: 0 on success, 1 when a guard raises, 2 when `--compare` finds
incompatible records.

---

## 7. Blocks

Each block states what it reads, what it writes, and what it must
satisfy. Blocks B1 to B13 of the v1 PRD keep their contract; the changes
below are additive.

### B1. `config.py` (changed)

Reads: nothing. Writes: the dataclasses and the protocol table.

New dataclasses: `DRCfg`, `DSCfg`, `RTCfg`, `LRCfg`, `STCfg`, `SBCfg`. Each
field has a default and a docstring that names its unit and its source.

New protocol fields on every existing protocol: the six new cfgs at their
dataclass defaults, so a v1 protocol name runs a new task and the record
carries `protocol_modified=False` (the protocol had no opinion on the new
task; the defaults are the protocol). This is a decision: it keeps every
v1 name usable for every task. The alternative, raising, would force a
new name for every old baseline.

`FLOORS`: a dict `metric -> Floor(rel=0.015, abs=None, date, source)`.
Ships with `auc`, `accuracy`, `f1`, `r2`, `mae` at `rel=0.015` from
PLAN.md 2026-08-18. Every new metric ships with `Floor(None, None, None,
"unset")` and `compare()` returns `MEASUREMENT` for it until B17's
bootstrap sets it. `PROTOCOLS` becomes a `MappingProxyType` (closes v1
open item H3).

Must satisfy: every field of every cfg is reachable from the CLI; the
config test asserts it by reading `keywords(name)`.

### B2. `io.py` (changed)

Adds `read_context(path)` for the sidecar; `write_profile_cache` and
`read_profile_cache` under `data_cache/evaluator/profile/<graph_hash>.json`.
The graph hash is sha1 of `(n, m, A.indptr[-1], A.indices[:1000],
A.indices[-1000:])`; it is a cache key, not an identity.

### B3. `pairs.py` (changed)

Adds `draw="stratified_by_hop"`: draws an equal count per hop shell up to
`max_hop`, for `per_shell`. Adds `draw="all_edges"` for retrieval truth.
Adds `neg_ratio` up to 1000 for `link_rank`, drawn in blocks so no
`(count, 2)` array above 2e7 rows is live.

Must satisfy: the v1 draws are byte-identical (the reference tests).

### B4. `hops.py` (changed)

Adds `cross_component="drop"|"cap"`. `drop` removes the pair. `cap` sets
`y = diam_est + 1`. Default `PROVISIONAL: "drop"`, and the record's
`sizes` carries `pairs_dropped_cross_component`. The v1 draws already drop
silently; now the count is printed.

### B5. `metrics.py` (changed)

Adds `pairwise_block(Z, rows, metric, block)`: the distances from a block
of query rows to every node, in blocks of `block` rows, so `retrieval`
never allocates an `(n, n)` array. `block` default 1024; at 1.13M nodes
and d=64 a block is 1024 x 1.13M x 4 B = 4.6 GB in float32, which is
too much, so the block is chosen from a memory budget: `block = floor(
budget / (n * 4))` with `budget` default 512 MB. Recorded in `cfg`.

### B6. `scoring.py` (changed)

Adds thin wrappers that call `rank.py`, `stress.py`, `retrieval.py` and
return dicts in the score-name convention. No arithmetic here.

### B7. `guards.py` (changed)

Adds:

- `check_dz_scale(dZ_norm, low=1e-3, high=1e3)`: takes the caller's last
  `||dZ||` from `context`; absent, it passes with a warning `dz_norm not
  given`. Catches the 1.11e9 blow-up that `check_finite` missed.
- `check_degree_zero(A)`: a node with degree 0 in `A` is reported; more
  than `max_frac` (default 0.05) raises.
- `check_cut_named(max_nodes, cut)`: `max_nodes > 0` and `cut is None`
  raises. The truncation trap of CATALOG 4.11.
- `check_ties(tie_frac_y, warn_above=0.6)`: a warning when the response
  is so tied that τ-b's ceiling is below 0.7.
- `check_rank_floor(metric)`: a warning on every rank score that has no
  floor, so a report never quotes it as a win by accident.

Every guard has the v1 shape: `strict=True` raises, `strict=False` writes
to `warnings`.

### B8. `report.py` (changed)

Adds the six new record blocks of section 5.7. `table()` prints the
profile header, the context line, the bracket rule. Adds `SeedReport` and
its `table()` with mean ± spread. Adds `Verdict.table()`.

Must satisfy: a v1 record read by v2 `Report.from_dict` loads, with the
new blocks `None`, and `table()` prints it as v1 did.

### B9. `rank.py` (new)

Reads: `x` (float array), `y` (int array). Writes: the rank scores.

- `spearman(x, y)`: Pearson on midranks. `scipy.stats.spearmanr` is the
  reference; the implementation is its own so the tie handling is stated
  in code and not in a library version.
- `kendall_tau_b(x, y)`: O(n log n) merge-sort count. Reference:
  `scipy.stats.kendalltau(variant="b")`.
- `somers_d(x, y)`: `(C - D) / (C + D + T_y)`, `y` as response.
  Reference: `scipy.stats.somersd`.
- `tie_frac(y)`: `Σ t_i(t_i - 1) / n(n - 1)` over the tie groups of `y`.
- `tau_b_max(y)`: τ-b of `x = y + tiny distinct noise`, so the layout is
  perfect and only the tie pattern of `y` limits it. Computed once per
  `y`.
- `per_shell(x, y)`: for each `h` in `sorted(unique(y))`: `n`, `ρ` of `x`
  against `y` restricted to shells `<= h` (cumulative), `x_mean`, `x_std`,
  and `R2` of a per-shell linear fit. `h_star` is the largest `h` at
  which cumulative ρ stays within 0.02 of its maximum
  (`PROVISIONAL`; the rule is one constant `H_STAR_TOL`).
- `bootstrap(fn, x, y, n_boot, rng)`: resample pairs with replacement,
  return the 2.5 and 97.5 percentiles of `fn`.

Must satisfy: the four-pair example of the rank note gives `τ-b = 0.55`
(C=4, D=1, T_y=1, T_x=0). A perfect layout gives `ρ = 1` and `τ-b =
tau_b_max(y)`. Reversed gives `-1`. A constant `x` raises
`DegenerateEvaluation` through `check_target_degeneracy`.

### B10. `stress.py` (new)

- `stress1(x, y)`: `s* = Σxy / Σx²`, then `sqrt(Σ(s*x - y)² / Σy²)`.
- `distortion(x, y, s)`: `max(s*x/y, y/(s*x))` per pair; mean, max, p99.
- `isotonic_fit(x, y)`: `sklearn.isotonic.IsotonicRegression` of `y` on
  `x`; `isotonic_r2`. The fit's flat regions give a second `h_star`
  estimate, reported as `h_star_isotonic`.
- `shepard_pairs(x, y, n_keep)`: a subsample of `(x, y, fit)` for the
  record, `n_keep` default 5,000, so the record stays small.

Must satisfy: a planted layout `Z` of a path graph on a line has
`stress1 < 1e-6` at the right `s`, `distortion_max < 1 + 1e-6`.

### B11. `retrieval.py` (new)

- `knn_in_Z(Z, queries, k, metric, block)`: via B5's block distances;
  `argpartition` per block; excludes self.
- `truth_adjacency(A, queries)`: the neighbour sets.
- `truth_hop_knn(A, queries, k)`: the `k` nearest by hop, BFS per query
  with ties broken by node id (stated, so it is reproducible).
- `precision_recall_at_k`, `average_precision`, `mrr`, `hits_at_k`,
  `jaccard_overlap`.
- `trustworthiness_continuity(A, Z, queries, k)`: the two DR errors;
  the rank in the graph is the hop rank with the same tie rule.
- `reconstruction(A, Z, metric, block)`: the `m` nearest pairs in `Z`
  against `E`. This one is O(n²) in distances and O(m log m) in the
  selection; it runs in blocks with a running top-`m` heap and is
  skipped with a warning above `n = 200,000` (`PROVISIONAL`).

`k_mode="degree"` uses `k = deg(u)` per query, floor 1, cap 100.

Must satisfy: `Z` = one-hot of `A`'s rows scaled so that neighbours are
nearest gives `p@k = 1` for `k = deg(u)`.

### B12. New protocols

| Name | Source | Note |
| --- | --- | --- |
| `metrology_h` | this document | class H defaults: 20,000 pairs, 200 sources, `min_hop=2`, `per_shell=True`, `bootstrap=1000`, `k=10`, `neg_ratio=100` for `link_rank` |
| `metrology_s` | this document | class S defaults: `min_hop=1`, `max_hop=diam_est`, `stratified_by_hop`, `per_shell=True`, `cross_component="drop"`, `k=degree` |
| `metrology_quick` | this document | L1/L3: 5,000 pairs, no bootstrap, `k=10`; for turnaround |

`evaluate(protocol="metrology")` resolves to `_h` or `_s` from the
profile's primary class and records which. The v1 names stay exactly as
they are.

### B13. `context.py` (new)

The `Context` dataclass of section 5.4, `validate(ctx, Z)`, and the gain
table:

| `update_rule` | `dc_gain` |
| --- | --- |
| `plain`, `sgd`, `velocity` | 1.0 |
| `momentum`, `nesterov` | `1 / (1 - beta)`, `beta` from `context.beta`, default 0.9 |
| `fa2` | 10.0 (the speed cap `k_max`) |
| `adam`, `sqn` | `None` (adaptive); `effective_lr` stays `None` |
| `gen_momentum` | `alpha / (1 - beta)` |

`effective_lr >= 1.0` writes a warning `effective lr at or above the
stability edge`. It does not raise; a caller may score a diverged run on
purpose to record it.

### B14. `profile.py` (new)

Reads: `A`, `n`, `cut`, and the optional `weights`, `labels`. Writes:
`Profile`.

- `n`, `m`, `avg_deg = 2m/n`, `max_deg`.
- `n_components`, `giant_frac`: `scipy.sparse.csgraph.connected_components`.
- `diam_est`: double-sweep BFS from `diameter_samples` (default 8) random
  nodes of the giant component, max eccentricity found. A lower bound;
  named `diam_est` for that reason.
- `assortativity`: Pearson correlation of the degrees at the two ends of
  every edge.
- `is_tree`: `m == n - n_components`. `is_bipartite`: 2-colouring by BFS.
- `has_weights`: `A.data` not all 1. `has_labels`: the caller passed
  labels.
- `cut`: `None`, `"bfs"`, `"subtree"`, from the caller. The profile of a
  cut graph carries the cut in its name: `cora@1000/bfs`.
- `cls`: the tuple; primary first (section 5.3).

Cost: one pass over `A` plus 16 BFS. At 1.13M nodes, under 10 s. Cached.

Must satisfy: a path of 100 nodes is `S`, tree, diam 99. A star is `S`,
tree, diam 2, assortativity -1. A 10x10 grid is `S`, not tree, diam 18,
bipartite. A BA graph at m=4 is `H`.

### B15. `structure.py` (new)

- Communities on `A`: `label_prop` (own implementation, 20 sweeps,
  seeded) as the default, `spectral` (`sklearn.cluster.SpectralClustering`
  on `A`, capped at `n <= 50,000`). Louvain is NOT shipped because it
  needs a dependency; the name is reserved and the record says which
  algorithm ran.
- Clusters on `Z`: `sklearn.cluster.KMeans`, `n_clusters` = the community
  count found, or the caller's.
- `nmi`, `ari`: `sklearn.metrics`.
- `conductance` of each `Z`-cluster on `A`; mean and max.
- `rho_norm_degree`: Spearman of `||Z_u - mean(Z)||` against `deg(u)`;
  `rho_norm_centrality` against PageRank (own power iteration, 50 steps).

### B16. `applies`

A table, in `config.py`, `APPLIES[cls][metric]` in `{True, False,
"secondary"}`, filled from `METROLOGY.md` section 6:

| metric | H | S |
| --- | --- | --- |
| rho, somers_d, tau_b | True | True (per-shell, up to h_star) |
| stress1, distortion | secondary | True |
| dist_approx.r2, mae | secondary | False |
| link_prediction.auc | secondary | secondary |
| dist_auc, dist_ap | True | True |
| retrieval.* | True | True |
| structure.* | secondary | False |

`False` prints in brackets with the class note. It never suppresses.

### B17. `floors.py` (new)

- `bootstrap_floor(metric, graph, Zs, n_boot, rng)`: for each `Z` in
  `Zs` (3 seeds), resample the scored pairs `n_boot` times, take the
  95% half-width; the floor is the max over seeds of the half-width,
  expressed relative to the mean. Written to `FLOORS` with the date, the
  graph names, and `n_boot`.
- `verdict(a, b, floor, spread)`: the two conditions and the result.
- A floor is set on `cora` AND `pubmed` and the larger is kept.
- `floors set` is a CLI subcommand:
  `python -m evaluator floors set rho cora emb/cora_s1.npy emb/cora_s2.npy
  emb/cora_s3.npy --n-boot 1000`. It prints the floor and the line to
  paste into `config.py`. It does not edit `config.py`; a floor is a
  reviewed change.

### B18. `align.py` (new)

- `procrustes(Z_a, Z_b)`: `scipy.linalg.orthogonal_procrustes` after
  centring and scaling both to unit Frobenius norm; returns the RMS
  residual.
- `knn_jaccard(Z_a, Z_b, queries, k, metric, block)`: via B11.

### B19. `tasks/stability.py` (new)

Takes `Zs`. For every pair `(i, j)`, `i < j`: Procrustes RMS and kNN
Jaccard on the same `n_queries` query set. Reports mean and std over
pairs. With two `Z` it is one pair.

### B20. `cli.py` (changed)

Adds the flags of section 6, the multi-EMBEDDING positional, `--compare`,
`--profile-only`, `floors set`. The parser is generated from the cfg
dataclasses, so a new cfg field is a new flag with no edit to `cli.py`
(closes a v1 fragility).

### B21. `tests/`

- The v1 reference tests run unchanged.
- `test_parity_v1.py`: the v1 parity gate (v1 PRD section 9) against the
  v2 tree, byte-exact on the two v1 tasks.
- One test per new block with the planted cases named above.
- `test_compare.py`: the verdict table of PLAN.md 2026-08-18 reproduced:
  +0.14% AUC is a TIE, +3.9% AUC is a WIN, -64% R2 is a LOSS, a 1-seed
  input is a MEASUREMENT.
- `test_scale.py`, marked `big`: every new task on `com_youtube` at
  1.13M nodes, under the peak RSS of the `n2v1m` protocol on the same
  machine. Not in the default gate.

---

## 8. Cross-block invariants

1. No block above `tasks/` imports a task. `rank.py`, `stress.py`,
   `retrieval.py`, `structure.py`, `align.py` import numpy, scipy,
   sklearn and each other only. `tests/test_structure.py` asserts it with
   `ast`, as `fodiwalk` does.
2. One `rng`, drawn in one order. The order of draws in a task is fixed
   and documented in the task's docstring. A new draw goes at the END so
   earlier draws keep their values.
3. No score is computed twice by two blocks. `rho` in `dist_rank` and
   `rho_norm_degree` in `structure` both call `rank.spearman`.
4. Nothing of size `(n, n)` is ever live. Every all-pairs operation goes
   through `metrics.pairwise_block`.
5. A record is append-only. `write()` never truncates a `.jsonl`.
6. `protocol_modified` reports intent: an explicit keyword sets it even
   when the value equals the protocol's. Unchanged from v1.
7. A metric name is unique across tasks. `auc` belongs to
   `link_prediction`; the model-free one is `dist_auc`.
8. Every `PROVISIONAL` default is one named constant in `config.py` with
   a comment that cites the open item of `METROLOGY.md` section 13.

---

## 9. The gates

The v2 tree is done when:

1. The v1 parity gate passes byte-exact.
2. Every planted test of B9 to B19 passes.
3. `test_compare.py` reproduces the PLAN.md verdict table.
4. `test_scale.py` passes on the campaign machine.
5. `python -m evaluator cora emb/cora.npy --all` prints the table of
   section 6.4 with every new task present.
6. The three metrology protocols exist and `evaluate(protocol=
   "metrology")` resolves by class on `cora` (H) and `roadnet_ca@100k/bfs`
   (S).
7. `floors set` runs on `cora` and `pubmed` for `rho`, `somers_d`,
   `stress1`, `map`, `dist_ap`, and the resulting floors are in
   `config.py` with their dates. Until then, `compare()` on those
   metrics returns `MEASUREMENT`, and that is correct behaviour, not a
   missing feature.
8. The current best method (`nbr_walk / min_gap / fdlinear / plain /
   streaming`, lr 0.999, 64d, 3 seeds) is re-scored with every new task
   on `cora`, `pubmed` and `com_youtube`, and the records are in
   `experiments/metrology-baseline/results.jsonl`. This is the day-one
   baseline for every new metric.

---

## 10. Success criteria

- A row in any campaign table can be traced to one record by
  `record_id`, and the record alone reproduces the row.
- A class S graph in a report shows a per-shell table and an `h_star`,
  and its hop R2 is in brackets.
- No verdict in a report from this date forward is written by hand. It is
  the output of `compare()`.
- The memory of `dist_rank` on `com_youtube` is below that of
  `dist_approx` under `n2v1m` on the same machine (no model is trained).
- A new task is one file plus one registry line, as in v1, and its flags
  appear in the CLI without an edit to `cli.py`.

---

## 11. Risks

| Risk | Mitigation |
| --- | --- |
| The bootstrap floor on a rank score is so small (20,000 pairs) that every difference is a WIN | The rule is AND: the seed spread still binds. And the floor is measured, not guessed; if it is small, that is the finding. |
| `retrieval` at 1.13M nodes is slow (n_queries x n distances) | `n_queries` default 2,000; block from a memory budget; recorded. `reconstruction` skipped above 200k. |
| `diam_est` on a road network is a poor lower bound | 8 double sweeps; report as an estimate; the per-shell table uses observed `y`, not `diam_est`. |
| Label propagation gives unstable communities | Seeded, 20 sweeps, and NMI/ARI are secondary for H and off for S. |
| The v1 tasks silently change under the new `pairs.py` draws | The reference tests are byte-exact and run in the default gate. |
| A caller fills `Context` wrong | `n_dim` checked against `Z`; `effective_lr` derived, not typed; the rest is recorded as the caller's claim and marked so. |
| Two evaluators drift (this and `fodiwalk/misc/evaluation.py`) | Out of scope; v1 non-goal kept. `fodiwalk`'s campaign scripts already import `evaluator`. |

---

## 12. Open decisions, carried from METROLOGY.md section 13

Each ships with a `PROVISIONAL` default so the tree can be built and
tested. Changing one is one constant.

| Item | Provisional default | Constant |
| --- | --- | --- |
| headline distance metric | `somers_d`, with `rho` beside it | `HEADLINE_RANK` |
| cross-component pairs | `drop`, counted | `CROSS_COMPONENT` |
| retrieval `k` | `10` fixed for H; `degree` for S | `K_MODE_BY_CLASS` |
| `h_star` rule | cumulative ρ within 0.02 of its max | `H_STAR_TOL` |
| `reconstruction` size cap | 200,000 nodes | `RECON_MAX_N` |
| community algorithm | label propagation | `COMMUNITY_DEFAULT` |
| bootstrap count | 1,000 | `N_BOOT` |

---

## 13. Out of this version, reserved

- Temporal tasks: `drift`, `rows_touched`, `future_link`. Names reserved
  in the registry as `NotImplemented` so a caller sees the name and the
  reason.
- Weighted-hop ground truth (`hops="dijkstra"`): reserved; the flag
  raises with the reason.
- An agent tool surface over `evaluate()` and `compare()`: still one file
  over the API, still deferred, for the reason the v1 PRD section 13
  gives.