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
| Laplacian Eigenmaps | `laplacian_eigenmaps` | karateclub `LaplacianEigenmaps` | `.venv-karate` | cora, citeseer, pubmed | RUN |
| Landmark shortest-path MDS | `landmark_mds` | own, to the doc's spec (scipy BFS + classical MDS) | shared `.venv` | cora, citeseer, pubmed | RUN |
| ProNE | `prone` | nodevectors | `.venv-karate` (`--no-deps`) | cora, citeseer, pubmed | RUN |
| RandNE | `randne` | karateclub | `.venv-karate` | cora, citeseer, pubmed | RUN |
| LINE | `line` | karateclub `FirstOrderLINE`+`SecondOrderLINE` | `.venv-karate` | cora, citeseer, pubmed | RUN |
| NetMF | `netmf` | karateclub | `.venv-karate` | cora, citeseer (dense n×n OOMs pubmed) | RUN small |
| GraRep | `grarep` | karateclub | `.venv-karate` | cora, citeseer (dense n×n OOMs pubmed) | RUN small |
| HOPE | `hope` | karateclub | `.venv-karate` | cora, citeseer (dense n×n OOMs pubmed) | RUN small |
| tForce2Vec | `tforce2vec` | HipGraph/Force2Vec, option 5 | C++ build | cora, citeseer, pubmed | RUN |
| rForce2Vec | `rforce2vec` | HipGraph/Force2Vec, option 7 | C++ build | cora, citeseer, pubmed | RUN |
| Force2Vec (base) | `force2vec` | HipGraph/Force2Vec, option 1 | C++ build | cora, citeseer (O(n²) impractical on pubmed) | RUN small |
| NetSMF | — | — | — | — | NOT RUN — C++ build (THUDM), not attempted this pass; NetMF stands in |
| GOSH | — | — | — | — | NOT RUN — GPU/CUDA only; this box has a 6 GB card and no CUDA build |

Reasons for NOT RUN follow the doc's rule: record why a result is missing
(unsupported size, memory, build cost, hardware), never drop the row.

`laplacian_eigenmaps/run.py` (a scikit-learn `SpectralEmbedding` version)
is kept but superseded: sklearn's arpack solver hangs on these graphs' many
disconnected components (citeseer has 48 isolated nodes). The karateclub
Laplacian Eigenmaps solves the same graphs in seconds, so the run pipeline
uses it. LINE became available (karateclub) and is RUN, not the "no CPU
build" it first looked like.

**Results:** `evaluator/reports/260922-other-methods-comparison.md`.

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
assemble_runlist.py      merge new-method runs with deepwalk/node2vec/fodiwalk anchors
landmark_mds/run.py      shared .venv (scipy)
laplacian_eigenmaps/run.py  superseded sklearn version (see note above)
karateclub/run.py        lapeig / prone / randne / line / netmf / grarep / hope, in .venv-karate
force2vec/run.py         wraps the HipGraph/Force2Vec binary
force2vec/src/           the cloned + built C++ source (gitignored)
run_all.sh               run every method serially, one embedding at a time
results/                 run lists, scored json, comparison tables
```
