# Citeseer: the complications, and how each was solved

Citeseer was the hardest of the three graphs to embed and score. Every
problem traces to two facts:

- it was **not in the dataset registry**, and
- it has **48 isolated (degree-0) nodes** — it is the sparsest, most
  fragmented graph here: n = 3,327, 4,552 undirected edges, average degree
  2.74 (cora 3.90, pubmed 4.50).

Each complication below gives the symptom, the cause, the fix, and where the
fix lives.

## 1. Citeseer was not in the dataset registry

**Symptom.** `fodiwalk.make_graph.load("citeseer")` failed — only cora,
pubmed, wordnet and the SNAP graphs were registered.

**Cause.** No loader existed for the LINQS citeseer files.

**Fix.** Added `_edges_citeseer()` to
`fodiwalk/make_graph/datasets.py`, in the exact shape of `_edges_cora`,
plus one dispatch line in `read_edges`. Citeseer's node ids are
**alphanumeric strings** (e.g. `bradshaw97introduction`), not integers like
cora, so the loader reads them as `dtype="<U32"` and lets `to_csr`'s
`np.unique` renumber them to `0..n-1`. Verified the standard published size
loads: 3,327 nodes, 4,552 edges.

## 2. fodiwalk raised `PlaneContractError` (invariant I5)

**Symptom.** Every citeseer fodiwalk run stopped with
`PlaneContractError: 48 rows hold stored pairs and a degree of 0`.

**Cause.** The far-pairs sampler drew long-range pairs that included some of
the 48 isolated nodes, so the augmented matrix `D` held stored pairs on
rows whose true degree is 0. The default degree source counts a row's
`h == 1` entries in `D`; for those rows it is 0, and every force law divides
the row sum by a degree of 0 guarded to 1 — so the row keeps its whole force
and never moves, silently. fodiwalk's I5 guard
(`fodiwalk/embed/plan_contract.py`) exists to catch exactly that, and it
did.

**Fix.** Run citeseer fodiwalk with **`--deg-source A`**, which uses the
true degree of the adjacency `A` instead of counting `D`'s `h == 1`
entries. cora and pubmed do not need it (their `auto` source never hit a
degree-0 row with a stored pair). The `experiments/embeddings/make_embedding.py`
`--deg-source {auto,D,A}` flag was added for this; the run scripts pass
`--deg-source A` only for citeseer
(`other-methods/run_expand_11seed.sh`, function `fw`).

## 3. Laplacian Eigenmaps (scikit-learn) hung on citeseer

**Symptom.** `sklearn.manifold.SpectralEmbedding` with the ARPACK solver
ran for ~30 minutes on citeseer and hit the run timeout. cora finished in
~12 s.

**Cause.** The 48 isolated nodes make the graph highly disconnected, so the
normalized Laplacian has a **high-multiplicity zero eigenvalue** (one per
connected component). ARPACK cannot separate that cluster of near-identical
smallest eigenvalues and stalls. Shift-invert at `sigma=0` is not an escape
either — the Laplacian is exactly singular there
(`RuntimeError: Factor is exactly singular`).

**Fix.** Use **karateclub's `LaplacianEigenmaps`** instead
(`other-methods/karateclub/run.py`, method `lapeig`). It solves the same
graph in ~2 s (pubmed in ~30 s). It is also a published implementation, so
it fits the "use published code" rule better than the scikit-learn wrapper.
The scikit-learn version is kept at
`other-methods/laplacian_eigenmaps/run.py` but is superseded and noted as
such.

## 4. Community detection is slowest on citeseer

**Symptom.** Scoring a citeseer embedding takes ~20 s, several times a cora
record — the community metric dominates.

**Cause.** Louvain finds **471 communities** on citeseer (105 on cora, 45
on pubmed), because a fragmented graph splits into many small communities.
The community metric then runs k-means at `k = 471` with `n_init=10`, which
is the expensive step.

**Fix.** Not a defect, so nothing to repair — but it is why the scoring
sequencer (`other-methods/eval_sequencer.py`) preprocesses each graph once
per experiment and caches the seven metrics to `eval.json` beside every
embedding. The costly k-means runs one time per embedding and is never
repeated.

## Results notes that follow from citeseer's structure

Not errors — expected consequences of a sparse, tree-like graph:

- **Landmark MDS beats fodiwalk on rho only on citeseer** (0.960 vs 0.839).
  Citeseer is where a walk-based hop estimate is noisiest and a
  shortest-path distance oracle gains most.
- **Landmark MDS on citeseer has high NMI (0.955) but near-zero ARI
  (~0.002).** The two community measures disagree here; do not read the NMI
  alone as "citeseer keeps its communities best".
- Isolated nodes are **unreachable from every landmark**, so landmark MDS
  places them at the shared far cap (`other-methods/landmark_mds/run.py`,
  recorded in each run's `notes`).

## Where these are also recorded

- `.claude/agent-memory/evaluator.md` — the live state, with the same fixes.
- `evaluator/reports/260922-other-methods-comparison.md` — the report, §4
  (methods/provenance) and the notes.
- Git history on `other-comparisons`: the citeseer loader, the
  `--deg-source A` fix, and the Laplacian solver switch.
