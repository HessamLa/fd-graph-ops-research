# other-methods — comparison embedding methods

Extra embedding methods to compare against fodiwalk, from the plan in
`evaluator/recommended-comparisons.md`. Each method uses a **published**
implementation (pip package or source build), not a from-scratch rewrite.

Every method writes to the SHARED store
(`data_cache/embeddings/<graph>/<dim>/<run>/`) through
`other-methods/common/store.py`, so a number from a new method compares to
a fodiwalk, deepwalk or node2vec number with no adjustment:

- one node numbering — every method loads the graph through
  `fodiwalk.make_graph.load`, so row `i` of `Z` is the same node;
- one scoring path — `evaluator` link-prediction + dist-approx, protocol
  `fodiwalk_dist`, `lp_max_pairs=50_000`, the same as this session's
  fodiwalk and baseline runs;
- one report — `other-methods/score_report.py` reads the store and prints
  the seven columns (rho, hop R2, recall@10, LP AUC, LP f1, NMI, ARI).

Priority graphs: **cora, citeseer, pubmed** (dim 128).

## Method catalog

| method | label | source | install | graphs | status |
|---|---|---|---|---|---|
| Laplacian Eigenmaps | `laplacian_eigenmaps` | scikit-learn `SpectralEmbedding` | shared `.venv` | cora, citeseer, pubmed | RUN |
| Landmark shortest-path MDS | `landmark_mds` | own, to the doc's spec (scipy BFS + classical MDS) | shared `.venv` | cora, citeseer, pubmed | RUN |
| ProNE | `prone` | karateclub | `.venv-karate` (`--no-deps`) | cora, citeseer, pubmed | RUN |
| GraRep | `grarep` | karateclub | `.venv-karate` | cora, citeseer (dense n×n OOMs pubmed) | RUN small |
| NetMF | `netmf` | karateclub | `.venv-karate` | cora, citeseer (dense n×n OOMs pubmed) | RUN small |
| Force2Vec (base) | `force2vec` | HipGraph/Force2Vec, option 1 | C++ build | cora, citeseer (O(n²) impractical on pubmed here) | RUN small |
| tForce2Vec | `tforce2vec` | HipGraph/Force2Vec, option 5 | C++ build | cora, citeseer, pubmed | RUN |
| rForce2Vec | `rforce2vec` | HipGraph/Force2Vec, option 7 | C++ build | cora, citeseer, pubmed | RUN |
| LINE | — | — | — | — | NOT RUN — no maintained CPU pip build; original is C++ with a heavy build |
| NetSMF | — | — | — | — | NOT RUN — C++ build (THUDM), not attempted this pass |
| GOSH | — | — | — | — | NOT RUN — GPU/CUDA only; this box has no CUDA jaxlib and a 6 GB card |

Reasons for NOT RUN follow the doc's rule: record why a result is missing
(unsupported size, memory, build cost, hardware), never drop the row.

## Machine limits that shaped the choices

7 GB RAM total (~4 GB free), 8 cores, GTX 1060 6 GB, no CUDA jaxlib. So:
dense `n × n` methods (GraRep, NetMF) run on the small graphs only; the
O(n²) Force2Vec base skips pubmed; GPU-only methods are out. Every skip is
recorded, not hidden.

## Layout

```
common/store.py          load graph, score with evaluator, save a record
common/store_handoff.py  score a Z built in another venv (karateclub)
score_report.py          read the store, print the per-dataset tables
laplacian_eigenmaps/run.py
landmark_mds/run.py
karateclub/run.py        ProNE / GraRep / NetMF, runs in .venv-karate
force2vec/run.py         wraps the HipGraph/Force2Vec binary
force2vec/src/           the cloned + built C++ source
run_all.sh               run every method serially, one embedding at a time
results/                 run lists and the scored json
```
