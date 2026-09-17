# Fodiwalk: Critical Literature Review and Comparison Agenda

## Assessment

Fodiwalk has a plausible research contribution, but the available evidence does not yet establish a broadly competitive graph embedding method. Its strongest candidate contribution is **using minimum observed walk separation to weight repulsive interactions on a sparse support, while retaining immediate-neighbor attraction**. Forces, direct coordinate updates, random walks, degree correction, and avoidance of all-pairs shortest paths are individually established ideas. The scientific burden is to show that this particular combination extracts useful information, beyond changing the amount of repulsion or the set of interacting pairs.

My reviewer judgment is that the current blueprint supports an interesting hypothesis and an experimental program, rather than a substantiated top-tier novelty or performance claim. A narrow geometry–cost result could become convincing. A general claim of superior unattributed representation learning would require substantially broader evidence.

This assessment concerns transductive node embeddings learned from graph topology without supplied node attributes or training labels. It distinguishes graph-distance preservation, community recovery, structural-role similarity, and link prediction. These are different targets; a method can improve one while damaging another. “Unattributed” does not mean unsupervised, and “structural” does not uniquely mean shortest-path geometry.

The project evidence is the retrieved *Fodiwalk_Paper_Blueprint.md*, revision 2, based on repository `HessamLa/fd-graph-ops-research` at commit `5c46a98c3f49dc34aa5bed796743b5fc7c75b6dc`. The shared “Repo Analysis Summary” transcript has now also been read at the [provided conversation link](https://chatgpt.com/share/6aa58382-dd5c-83e8-a22d-a15eb4906de3). It supplies the explicit force equations and update conventions discussed in the supplement below. Its repository observations were not independently rerun. Reported defects and preliminary results below are therefore attributed to that evidence, not new findings from executing code.

The literature search extends through September 12, 2026. It covers primary papers and proceedings on arXiv, OpenReview, ACM, IEEE, PMLR/ICML, NeurIPS, IJCAI, JMLR, Nature, Springer, PLOS, and graph-drawing literature, with author repositories and institutional records for additional verification. This is a targeted, extensive review, not a claim to have searched every research website or found every relevant paper. Recent preprints and unresolved publication records are identified separately.

## 1. The strongest objections

### 1.1 “Forces rather than neural training” is not a scientific distinction

Force2Vec explicitly derives attractive and repulsive updates, supports multiple force models, and includes rForce2Vec, whose attractive neighborhoods come from random walks. Its section III-C makes the overlap explicit. A comparison with node2vec alone leaves the closest methodological competitor untested. Fodiwalk must compare against both the neighbor-based and walk-based Force2Vec variants, including a distance-based force option where feasible.[^1]

Nor is direct coordinate optimization unique to force methods. NetMF gives a matrix-factorization interpretation of several familiar embedding objectives; this undermines a narrative that skip-gram embeddings are intrinsically opaque or lack a structural target. The objectives differ, but labeling one update a force does not establish superiority.[^2]

**Required response:** describe the sampled support, pair roles, scalar weighting, geometry, normalization, and solver separately. Attribute any demonstrated gain to the component that survives controlled ablations. Avoid “first physics-based,” “first walk-and-force,” and “forces instead of gradients.”

### 1.2 The closest predecessor may be the earlier work in the same research line

*Force-directed graph embedding with hops distance* and *Kinematic-Based Force-Directed Graph Embedding* already establish hop-aware force embedding in this author lineage. Fodiwalk needs an explicit account of what is inherited and what changes when exact hop information is replaced by sampled walk gaps and retained interactions.[^3][^4]

*Reducing Complexity of Force-Directed Graph Embedding* is an especially important unresolved comparison. Its indexed record describes limiting force calculations to selected node pairs. The full-text endpoint was intermittently indexed but subsequent access returned a verification challenge; its exact sampling rule, publication status, and relationship to the current project were not established. It should not be described as an independent accepted competitor without clarification, nor omitted from the novelty audit.[^5]

**Required response:** prepare a version-lineage table: force equation, exact versus observed distance, pair selection, edge completion, normalization, optimizer, asymmetry, memory representation, and experiments. If Fodiwalk is a successor or renamed version of that manuscript, say so internally and assess the increment over its actual contents. This is a novelty accounting issue, not an allegation of misconduct.

### 1.3 Nonedges are not necessarily dissimilar

The strongest recent challenge to Fodiwalk's repulsive interpretation is Liu et al.'s KDD 2025 work on dimension regularization. It derives a relationship between skip-gram repulsion and dimension centering near collapse and instantiates an alternative for LINE and node2vec. Its empirical results show that the need for repulsion depends on graph regime; they do not prove that repulsion can always be removed.[^6]

For Fodiwalk, a nonadjacent pair reached within a community can be graph-close and community-similar. Repelling it may improve local spacing or distance ranking while weakening the community signal. A walk does not certify semantic dissimilarity, and absence of an edge may reflect incomplete observation.

**Required response:** measure the proportion and total weight of repulsive pairs within planted communities, across communities, and across missing true edges in controlled graphs. Compare actual gaps with constant weights, within-source shuffled weights, and a carefully defined nonlocal-attraction control. Include the KDD 2025 LINE/node2vec variants when claiming an efficiency benefit over negative sampling.

### 1.4 Walk separation is a biased observation, not a distance guarantee

For an unweighted graph and a pair joined by an observed walk segment, the shortest-path distance satisfies

\[
\delta_G(u,v)\le h_W(u,v),
\]

where \(h_W\) is the minimum retained observed gap. This follows because every observed segment is a walk between the endpoints and can be shortened to a path. It does not imply unbiasedness, correct pair ordering, or small relative error. Assigned far-pair weights do not inherit this bound unless backed by actual paths.

An adjacent pair can be observed via a long detour. Two pairs can satisfy \(\delta_1<\delta_2\) but \(h_1>h_2\). More samples can reduce a retained pair's minimum gap, changing its force even if support stays fixed. Under pruning, the support itself also changes. Thus a larger walk budget is simultaneously a coverage intervention and a weighting intervention.

**Required response:** on feasible graphs, hold support fixed and replace observed gaps with exact BFS distances. Report gap error by source degree, true hop distance, and community membership. If exact distances do not improve results, the mechanism may be useful for something other than approximating shortest paths. If shuffling does not hurt, partner-specific gap information has not been established as the cause.

These are deductions from the stated Fodiwalk mechanism; they are not claims that such failures have already been measured.

### 1.5 Geometry claims require geometry competitors

A paper centered on graph-distance fidelity cannot establish its value using only context embeddings as opponents. Kamada–Kawai is the classical graph-distance layout reference. Sparse stress and stochastic stress optimization directly address computationally constrained distance-based layout; they are closer to the claimed target than another feature-based GNN.[^7][^8][^9]

The comparison should include a landmark-distance embedding and a sparse stress implementation, with preprocessing charged to the total cost. Use exact stress or Kamada–Kawai on small graphs as diagnostic references. Standard layout implementations are often two-dimensional; a higher-dimensional adaptation must be labeled and validated rather than represented as the native published implementation.

The attraction–repulsion spectrum literature also shows why a prettier layout is ambiguous: changing force balance changes the type of structure emphasized. UMAP, LargeVis, and PaCMAP are useful conceptual comparisons, but need a declared graph-derived input pipeline before numerical comparison.[^10][^11][^12][^13]

### 1.6 Avoiding all-pairs distances is too weak a scalability claim

DeepWalk, node2vec, LINE, and FastRP do not require an all-pairs shortest-path matrix. NetSMF supplies a sparse approximation with an explicit theoretical target; ProNE and FastRP provide efficient non-force alternatives. These methods make “no all-pairs distance matrix” a description of implementation requirements, not a competitive result.[^14][^15][^16][^17][^18][^19]

LightNE 2.0, SketchNE, GraphVite, and PyTorch-BigGraph raise the systems bar further. Their published scale demonstrations use different hardware and objectives and cannot be imported as fair speed comparisons. They nevertheless rule out treating million-node feasibility or GPU execution alone as novelty.[^20][^21][^22][^23]

For retained interaction count \(M\), a basic force sweep costs \(O(Md)\); storing coordinates requires \(O(nd)\) memory before optimizer state. Padded sparse formats require the actual padded interaction count. Generating \(R\) walks of length \(L\) per node already visits \(O(nRL)\) steps; enumerating all within-walk pairs can be much more expensive than window-limited enumeration. Deduplication, sorting, neighbor completion, and host/device transfers must be measured.

**Required response:** report quality against end-to-end wall time and peak CPU/GPU memory, including construction. Include a fast simple baseline and a strong scalable factorization baseline. Match hardware or state the resource difference explicitly. Compare achieved quality at resource budgets, not equal epoch counts.

### 1.7 Current evidence has specification and evaluation risks

The prior blueprint reports that `walk_edges` may retain a true edge with a gap above one when the edge was first encountered through a detour; adding missing edges does not necessarily overwrite such a gap. If attraction is keyed to gap one, that can contradict the advertised neighbor-attraction invariant. It also distinguishes this behavior from the `nbr_walk` merge policy. These are reported snapshot-specific observations, not a new audit.

The blueprint further identifies possible full-graph embedding in a link-prediction script, differing decoders across tables, single-seed comparisons, and uncertain peak-memory accounting. Those observations prevent treating the old scorecard as conclusive evidence. Full-graph embedding is appropriate for descriptive geometry and reconstruction, but it does not demonstrate prediction of unseen edges.

**Required response:** freeze one specification, assert every retained original edge has the intended role, and rerun affected results. If link prediction remains in scope, remove held-out edges before walks, degree computation, and embedding. The degree-bias study and HeaRT evaluation paper justify stronger negative sampling and protocol controls.[^24][^25]

## 2. Recent work that changes the comparison agenda

| Work | Date/status verified | Why it matters to Fodiwalk | Treatment |
|---|---|---|---|
| Bypassing Skip-Gram Negative Sampling: Dimension Regularization as a More Efficient Alternative for Graph Embeddings | KDD 2025; manuscript began in 2024 | Challenges the necessity and cost of node-pair repulsion | Mandatory discussion; benchmark its variants for an efficiency claim |
| Network community detection via neural embeddings | Nature Communications, November 2024 | Connects node2vec to community detectability under model assumptions | Mandatory for community claims; reproduce a relevant synthetic regime |
| Community Detection Guarantees Using Embeddings Learned by Node2Vec | NeurIPS 2024; preprint began in 2023 | Provides recovery guarantees for node2vec embeddings in SBM settings | Mandatory theory context for community claims |
| Implicit degree bias in the link prediction task | ICML 2025 | Explains why standard evaluation can reward degree information | Mandatory protocol citation if LP is retained |
| PieClam: A Universal Graph Autoencoder Based on Overlapping Inclusive and Exclusive Communities | ICML 2025; preprint began in 2024 | Makes decoder expressiveness and non-assortative structure explicit | Strong conditional comparator for reconstruction or broad expressiveness |
| Revisiting Modularity Maximization for Graph Clustering: A Contrastive Learning Perspective | KDD 2024 | Relates modularity to positive/negative examples | Relevant for the interpretation of repulsion and community objectives |
| Robustness of shallow graph embedding methods for community detection | arXiv 2405.00636, revised manuscript | Provides a direct shallow-embedding robustness agenda | Cite as a preprint unless final venue is separately verified |
| Dynamic Graph Embedding Through Hub-aware Random Walks | 2025 manuscript | Adds a recent hub-aware sampling direction | Conditional only if updates/dynamic graphs become a claim |
| NOMAD: Generating Embeddings for Massive Distributed Graphs | April 2026 preprint | Latest verified systems lead in this review; distributed LINE-based embedding | Systems context; benchmark only with a distributed-scale claim |
| FUSE: Fast Semi-Supervised Node Embedding Learning via Structural and Label-Aware Optimization | October 2025 preprint | Topology-based input still uses labels during training | Separate supervised track; not a matched unsupervised baseline |

Sources: the first row [6]; community results [26–27]; evaluation [24]; PieClam [28]; modularity [29]; robustness [30]; dynamic walks [31]; NOMAD [32]; FUSE [33].

The distinction between publication date, preprint date, and crawl date matters. A page recently indexed in 2026 is not necessarily a new 2026 paper. NOMAD is included as a verified recent preprint, not as evidence of an accepted new state of the art. Recent attributed, heterogeneous, knowledge-graph, and graph-level embedding papers were not automatically promoted to baselines because they address different inputs or outputs.

## 3. Required comparison families

### 3.1 Closest methodological and historical works

| Paper/family | Comparison question | Priority |
|---|---|---|
| Force2Vec and rForce2Vec [1] | Does walk-gap repulsion outperform established force/walk choices? | Essential experiments |
| Hop-distance and kinematic predecessors [3–4] | What is the scientific increment over the inherited force model? | Essential small/medium-scale experiments |
| Reducing Complexity… [5] | Has sparse force selection already been introduced in this lineage? | Resolve before a novelty claim |
| Fruchterman–Reingold [34] | Which classical attraction/repulsion components are inherited? | Historical citation; small-graph control |
| Kamada–Kawai [7], sparse stress [8], stress SGD [9] | Can a direct distance objective obtain better geometry at similar cost? | Essential for geometry-centered claims |
| Noack: Modularity clustering is force-directed layout [35] | Is the community interpretation new or an instance of known energy relationships? | Essential conceptual citation for CD |
| ForceAtlas2 [36] | Are degree-related forces and practical stabilization already established? | Context; visualization control if relevant |
| Attraction–repulsion spectrum [10] | Is the gain simply a different force balance? | Mechanistic citation and balance controls |

Noack's connection between modularity clustering and force layout makes “forces reveal communities” an insufficient contribution. It does not prove Fodiwalk optimizes modularity. That requires a derivation for Fodiwalk's own force system.[^35]

### 3.2 Unattributed embedding performance baselines

| Method | What it contributes to a fair suite | Priority |
|---|---|---|
| DeepWalk [14] | Standard walk-context reference | Essential |
| Tuned node2vec [15] | Walk bias and neighborhood flexibility | Essential |
| LINE, first/second-order configurations declared [16] | Local-proximity and efficient shallow reference | Essential |
| NetMF [2] | Explicit matrix target and spectral interpretation | Essential on feasible scales |
| NetSMF [17] | Sparse approximation of a meaningful matrix target | Strong scalability baseline |
| ProNE [18] | Fast factorization/propagation alternative | Essential efficiency reference |
| FastRP [19] | Cheap optimization-free topology baseline | Essential efficiency reference |
| Laplacian Eigenmaps [37] | Transparent graph-smoothness reference | Essential geometry/CD baseline |
| VERSE [38] | Explicit similarity-distribution target | Strong supplementary baseline |
| GraRep [39] | Higher-order transition information | Supplementary; not a shortest-path objective |
| Conditional Network Embeddings [40] | Conditions on known structural information such as degree | Strong LP/degree-confounding comparison |
| GraphWave and struc2vec [41–42] | Structural roles rather than proximity | Include if claiming role preservation |
| VGAE [43] | Graph reconstruction through a learned encoder/decoder | Optional topology-only neural control |

A VGAE comparison must state the input features. Identity, constant, random, and topology-derived features have different information and scalability consequences. Supplied text or attribute vectors belong in a separate track. Similarly, GCN or GraphSAGE trained with class labels cannot establish a fair unsupervised comparison merely because the original graph has no node attributes.

Role-preserving methods are important to delimit the claim, but should not be declared inferior solely because they place distant, structurally equivalent nodes together. That behavior is their intended target. If shortest-path distance is the primary outcome, label role methods as target-mismatch diagnostics rather than mandatory winners to beat.[^41][^42]

### 3.3 Large-graph systems and conditional geometry

Use LightNE 2.0 or SketchNE for a strong scalable non-force comparison, and GraphVite or PyTorch-BigGraph for optimized training context. FREDE is relevant when claiming an anytime quality–memory trade-off. NOMAD is relevant when claiming distributed scalability. Running every system is not necessary for a geometry paper on moderate graphs, but omitting the family while claiming general scalability is difficult to defend.[^20][^21][^22][^23][^32][^44]

For hierarchical graphs, include Poincaré embeddings as a conditional geometry comparison. Use each method's native distance and calibrate scales. Euclidean Fodiwalk should not be expected to preserve all graph metrics equally well at low dimension.[^45]

For visualization claims, UMAP, LargeVis, and PaCMAP require graph-derived inputs with charged preprocessing. For topology-preservation claims, Topological Node2vec is relevant because it explicitly introduces persistent-homology information. Shortest-path stress is not a substitute for a topological invariant.[^11][^12][^13][^46]

## 4. A sharper mathematical critique

### The representation target is underspecified

Increasing repulsion with graph gap does not imply equilibrium distance is proportional to graph distance. Every node balances many interactions, and each pair's separation depends on the surrounding graph. The paper needs either an explicit target objective, a restricted analysis, or a clearly empirical claim about geometric outcomes.

Three useful questions are distinct: Does the force field optimize a scalar objective? Does optimization approach a stationary configuration? Does that configuration preserve the desired graph statistic? Proving one does not prove the other two. An equilibrium for an isolated attractive/repulsive pair cannot establish global graph-distance fidelity.

### Degree normalization may be preconditioning

For reciprocal interactions derived from an energy \(E\), an update of the form

\[
\dot Z=-D^{-1}\nabla E(Z)
\]

with positive diagonal \(D\) is a preconditioned gradient flow. Along that continuous-time flow, \(dE/dt=-\nabla E^T D^{-1}\nabla E\le0\). This is a conditional mathematical observation, not a convergence proof for Fodiwalk. Directed supports, nonreciprocal weights, masking, singular forces, and finite step sizes require separate treatment.

This perspective suggests a useful ablation: compare degree normalization with row-width normalization and a tuned step size. Otherwise, the purported structural benefit may just be improved numerical conditioning. A decrease in displacement under a decaying learning rate is not sufficient evidence that the force residual is small.

### Sparse support leaves unobserved geometry weakly constrained

Pairs never retained have no direct force under the stated support-limited model. Their geometry is induced indirectly. On disconnected components without cross-component interactions, relative translations are not identified by within-component forces. Isolates require an explicit policy. Optional far sampling can resolve some constraints, but its contribution must be measured rather than folded into a claim about witnessed walk gaps.

Even a connected sparse interaction graph need not determine a unique embedding shape. Report invariance to initialization through distances or aligned coordinates, rather than comparing raw coordinate axes. Repeated embeddings may be equivalent under rotation or translation while differing numerically.

### Community separation and distance preservation can conflict

Consider a star: leaves share a structural role and are two hops apart. A role embedding may place them together; a distance-preserving embedding should distinguish them. In a bipartite graph, adjacent endpoints belong to different partitions while same-side vertices can be two hops apart. A universal rule of “adjacent attracts, nonadjacent repels” therefore does not directly encode partition membership.

These examples are proposed diagnostic cases, not predictions of exact output. PieClam's inclusive/exclusive-community formulation offers a recent reminder that graph structure includes both connectivity and systematic disconnection. Its decoder expressiveness results do not automatically establish a better shortest-path embedding, but they challenge unqualified claims about representing arbitrary graph structure.[^28]

## 5. Experiments that can settle the contribution

### 5.1 First run the decisive mechanism experiment

Use a citation graph, a social graph, and a tree/grid or road-like graph. Freeze the node order, initialization, support, original-edge mask, and update randomness where possible. Choose parameters on a development set, then report held-out evaluations.

| Intervention | Held fixed | What a benefit would establish |
|---|---|---|
| Actual observed gaps versus constant nonedge weights | Support and local-edge law | Value beyond uniform repulsion, subject to magnitude matching |
| Actual gaps versus within-source shuffled gaps | Per-row gap histogram and repulsion budget | Value of assigning particular gaps to particular partners |
| Observed gaps versus exact shortest-path gaps | Identical retained pairs | Effect of gap error |
| Walk support versus uniform nonedge support | All edges, interaction budget, force law | Value of walk-selected pairs |
| Gap repulsion versus specified nonlocal attraction | Pair support; matched parameter-search budget | Effect of role assignment |
| Native Fodiwalk versus native rForce2Vec | Same graph, dimensions, resource reporting | Competitive value of the complete method |
| Fixed support versus resampled support | Expected work and declared weighting policy | Whether freezing the sample limits quality |

A sign-flipped Fodiwalk control is not rForce2Vec. Reproducing the published baseline and isolating one mechanism are separate experiments. Report both fixed-parameter and validation-retuned ablations if the intervention changes force magnitude substantially.

If shuffled gaps perform similarly, the present central narrative should be revised. If exact gaps are consistently better, invest in better separation estimation. If only extra far sampling helps, the contribution belongs to coverage rather than minimum observed gap. If a constant-force baseline matches the method, the simplification may be the useful outcome.

### 5.2 Geometry evaluation

Use fixed evaluation sources/pairs independent of the sampled interaction support. Compute exact BFS distances for the chosen source set on unweighted graphs. Fit a nonnegative scalar calibration on validation pairs and evaluate normalized stress on test pairs:

\[
\operatorname{stress}=\sqrt{\frac{\sum_{(u,v)\in T}(s\|z_u-z_v\|-\delta_G(u,v))^2}{\sum_{(u,v)\in T}\delta_G(u,v)^2}}.
\]

Also report rank correlation, graph-neighborhood retrieval, and results by hop shell and source degree. State treatment of ties and unreachable pairs. A nonlinear decoder predicting hop count is a separate recoverable-information task, not proof of raw metric fidelity.

Add paths, cycles, stars, grids, balanced trees, assortative SBM, degree-corrected SBM, bipartite graphs, and rewired/shortcut variants. Vary density and mixing independently of size. This suite tests whether gains are limited to citation-like geometry and whether the social-graph reversal noted in the blueprint is reproducible.

### 5.3 Community detection, if it remains a primary claim

The recent community literature makes tuned node2vec a serious opponent rather than a historical checkbox. Kojaku et al. analyze detectability under model assumptions, while Davison et al. establish recovery results for node2vec followed by clustering. Neither guarantees best performance on every real graph.[^26][^27]

Use NMI and ARI with uncertainty on planted labels; measure estimated versus true community count. Include Leiden on the original graph, spectral clustering, and node2vec plus a common clustering procedure. A no-embedding baseline tests whether embedding adds value to this downstream task. Leiden is preferable to relying only on Louvain because the latter can return badly connected communities.[^47]

If applying Leiden or Louvain to an embedding-derived graph, specify k-nearest-neighbor construction, similarity function, symmetrization, and resolution. Running community detection on the original adjacency does not evaluate the embedding. Report embedding and clustering costs together, plus clustering sensitivity. Citation topics are not automatically structural ground-truth communities; label their interpretation accurately.

DMoN and the KDD 2024 modularity/contrastive work belong in a conditional modern clustering track with features controlled. Their contribution is relevant, but raw attributed results should not be compared directly with topology-only Fodiwalk.[^29][^48]

### 5.4 Link prediction, only if retained

Split edges before every graph-dependent training operation. Use one common decoder protocol and validation budget for embedding comparisons. Include degree-only and local-heuristic baselines, ordinary negatives and a separately specified hard/degree-controlled evaluation. AP, MRR, and Hits depend on candidate populations; disclose them. Do not merge reconstruction scores with held-out prediction.[^24][^25]

### 5.5 Replication and efficiency

Use at least five independent embedding seeds for principal moderate-size comparisons, with graph-generation replicates on synthetic data. Keep official benchmark splits fixed; use multiple splits where custom splits are part of the question. Pair evaluations across methods and report intervals over appropriate independent units, not millions of correlated node pairs treated as independent trials.

Use a main dimension such as 128 and a small dimension sweep. Log total construction/training/evaluation time separately; peak host memory and accelerator memory; interaction counts before/after padding; tuning trials; and failures. A Pareto plot of geometry quality versus total runtime is more informative than a table of unrelated epoch counts. Old papers' reported speedups are not reusable measurements of the current implementations.

## 6. Minimum defensible benchmark and paper positioning

For a geometry-centered paper, the practical minimum is Fodiwalk, the exact-hop predecessor on feasible graphs, Force2Vec, rForce2Vec, DeepWalk, tuned node2vec, LINE, Laplacian Eigenmaps, ProNE, FastRP, and sparse stress or a clearly specified landmark-distance baseline. Add NetMF at moderate scale and NetSMF or LightNE 2.0 at large scale. Add the KDD 2025 repulsion alternatives for a strong efficiency claim. These are selected for distinct objections, not to maximize the number of baseline names.

The paper should make the following bounded claim only if experiments support it: **sampled walk separation supplies useful partner-specific information for sparse geometric repulsion, improving a measured distance-fidelity–cost trade-off in identified graph regimes**. It should explicitly distinguish this from universal downstream superiority, exact graph-distance preservation, and a novel force-based learning paradigm.

The most valuable potential outcome is an explanation of when walk-gap repulsion works. For example, evidence may show benefits when distance variation is broad and sampling covers relevant routes, but weaker results when distances concentrate or community-affinity preservation dominates. Those are hypotheses to test, not conclusions to place in an abstract now.

Three gates should precede extensive benchmark expansion: resolve the predecessor/version relationship; repair and freeze the neighbor-role specification; and show an effect from partner-specific gaps under fixed-support controls. If those gates fail, polishing the abstract or adding unrelated GNNs will not solve the scientific problem.

## Transcript-based refinement

Reading the shared transcript confirms the overall assessment, but changes how the critique should be attributed. The earlier discussion already rejected “forces instead of gradients” as sufficient novelty, identified rForce2Vec and the hop-based predecessors, acknowledged the citation/social reversal, and proposed fixed-support gap interventions. Those are **recognized research obligations**, not omissions newly discovered by this review. The unresolved issue is whether the proposed experiments have been executed and support the claims; the transcript presents them as future work.

The transcript also makes the following analysis possible. Its equations are treated as the reported specification at the inspected snapshot, not as an independent verification of the current repository. The derivations below assume positive force parameters and, where stated, reciprocal fixed support.

### An explicit energy exists for the reciprocal fdhop force

With attraction positive along the unit vector from node u to node v, the reported magnitude is

\[
m(r,h)=\begin{cases}
k_1r-k_r,&h=1,\\
-k_rh\exp(-k_4r),&h\ge2.
\end{cases}
\]

For symmetric retained pairs with matching weights in both directions and positive distances, this is the negative coordinate gradient of the pair potential

\[
U_h(r)=\begin{cases}
\tfrac12k_1r^2-k_rr,&h=1,\\
\tfrac{k_rh}{k_4}\exp(-k_4r),&h\ge2.
\end{cases}
\]

Indeed, differentiating with respect to r gives m(r,h), and the coordinate gradient supplies the opposite unit direction. Summing once over each unordered retained pair gives a scalar energy. Degree division can then be interpreted as diagonal preconditioning, subject to a positive, fixed divisor. This strengthens the earlier conditional observation: an explicit potential can be written for this particular reciprocal force law. It does not establish convexity, discrete-time convergence, or applicability to asymmetric `nbr_walk` rows.

This gives a more useful theoretical task than debating whether Fodiwalk “has a loss”: state the exact assumptions under which the potential describes the implemented update, then analyze or measure what changes when symmetry, masking, or the optimizer changes.

### Gap weights do not specify nonlocal target distances

For h greater than one, U decreases as r grows; it has no finite preferred separation. The observed gap multiplies the potential rather than setting its minimum. Consequently, calling these interactions “distance constraints” would be stronger than the equation supports. Local attraction and the rest of the graph determine the eventual finite geometry.

At an equal force magnitude c for an isolated nonlocal contribution,

\[
r=\frac{1}{k_4}\log\left(\frac{k_rh}{c}\right),
\]

when that r is nonnegative. This is not a global equilibrium derivation. It shows that even a simple force-level interpretation yields logarithmic dependence on h, rather than a direct proportional target distance. A useful added ablation is a sampled stress objective on exactly the same support, with observed gaps as explicit target lengths. That tests whether gap-weighted repulsion is preferable to directly fitting the available distance proxy.

### The zero-distance rule weakens an unconditional anti-collapse claim

The transcript states that the kernel returns zero contribution at exactly zero embedding distance. Thus, with the plain update, placing all nodes at precisely the same coordinate is a fixed point of the implemented update, even though the limiting neighbor force for small positive r is repulsive. This is a mathematical edge case, not evidence that random Gaussian initialization typically collapses.

The distinction matters for a theorem or an unconditional statement that repulsion prevents collapse. Test exact duplicate coordinates and near-collisions, and document the numerical convention. The potential above is nondifferentiable as a function of coordinate differences at collision, so its smooth-gradient interpretation was deliberately restricted to positive distances.

### Default update masking changes the effective step

The transcript reports whole-node update suppression with probability 0.5 and no rescaling. For an independent keep-mask B and a force update g computed from the current state,

\[
\mathbb E[B\eta g\mid Z]=0.5\eta g.
\]

Therefore, a comparison against unmasked training at the same learning rate confounds stochasticity with a smaller expected step. Add an unmasked half-learning-rate control and a declared mean-step-matched masked control. Equal expected movement does not imply equal trajectories or convergence, but it removes the simplest alternative explanation.

### fdlinear versus fdhop is not a clean frequency ablation

According to the transcript, the reported default-sign fdlinear law uses k1*r − kr*exp(−k4*r) on neighbors and −(h/f)*exp(−k4*r) on farther pairs. fdhop uses k1*r − kr on neighbors and −kr*h*exp(−k4*r) on farther pairs. Switching between them therefore changes neighbor repulsion, nonlocal scaling, and frequency dependence simultaneously.

A full-law comparison is valid as a model comparison. It cannot isolate the effect of frequency. Add a common-law control that toggles only the frequency divisor while keeping the edge law, support, decay, and declared scale-matching policy fixed. This is a more specific experimental requirement than the general force-family comparison in the original report.

The transcript also confirms that the degree divisor usually counts stored h=1 neighbors, and that augmentation is normally fixed across epochs. Together with the reported neighbor-gap issue, this means a misclassified edge may change both its force role and its source's normalization. A repair can therefore alter multiple parts of the dynamics; affected historical results need identified reruns rather than relabeling.

## Sources and reading order

Read [1], [3–6], [8–10], [17–20], and [24–28] first. They bear most directly on novelty, target alignment, efficiency, and evaluation. Remaining entries define the broader comparison boundary. Publication years below distinguish conference/journal dates from earlier arXiv versions where verified. These references support the statements about prior work; Fodiwalk recommendations and deductions are analytical judgments.

[^1]: Md. Khaledur Rahman, Majedul Haque Sujon, and Ariful Azad. **Force2Vec: Parallel Force-Directed Graph Embedding.** ICDM 2020. [Paper](https://arxiv.org/abs/2009.10035), especially §III-C for rForce2Vec.
[^2]: Jiezhong Qiu et al. **Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec.** WSDM 2018. [Paper](https://arxiv.org/abs/1710.02971).
[^3]: Hamidreza Lotfalizadeh and Mohammad Al Hasan. **Force-directed graph embedding with hops distance.** 2023 preprint. [Paper](https://arxiv.org/abs/2309.05865).
[^4]: Hamidreza Lotfalizadeh and Mohammad Al Hasan. **Kinematic-Based Force-Directed Graph Embedding.** Complex Networks XV, 2024, pp. 137–149. [Publisher](https://link.springer.com/chapter/10.1007/978-3-031-57515-0_11).
[^5]: **Reducing Complexity of Force-Directed Graph Embedding.** OpenReview record; detailed contents and final status unresolved here. [Record](https://openreview.net/forum?id=1MjOlHwCE6). The indexed author profile lists Hamidreza Lotfalizadeh, Omar Yaqub, and Mohammad Al Hasan; verify against the manuscript before importing metadata.
[^6]: David Liu, Arjun Seshadri, Tina Eliassi-Rad, and Johan Ugander. **Bypassing Skip-Gram Negative Sampling: Dimension Regularization as a More Efficient Alternative for Graph Embeddings.** KDD 2025; 2024 initial preprint, June 2025 revision. [Paper](https://arxiv.org/abs/2405.00172), [author implementation confirming KDD status](https://github.com/dliu18/negative-sampling).
[^7]: Tomihisa Kamada and Satoru Kawai. **An Algorithm for Drawing General Undirected Graphs.** Information Processing Letters, 1989. [DOI](https://doi.org/10.1016/0020-0190(89)90102-6). Historical reference; publisher full text was not accessible in this review.
[^8]: Mark Ortmann, Mirza Klimenta, and Ulrik Brandes. **A Sparse Stress Model.** 2016 manuscript; expanded JGAA article 2017. [Paper](https://arxiv.org/abs/1608.08909).
[^9]: Jonathan X. Zheng, Samraat Pawar, and Dan F. M. Goodman. **Graph Drawing by Stochastic Gradient Descent.** 2017 manuscript. [Paper](https://arxiv.org/abs/1710.04626).
[^10]: Jan Niklas Böhm, Philipp Berens, and Dmitry Kobak. **Attraction-Repulsion Spectrum in Neighbor Embeddings.** JMLR 2022; 2020 initial manuscript. [Paper](https://arxiv.org/abs/2007.08902).
[^11]: Leland McInnes, John Healy, and James Melville. **UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.** 2018 manuscript. [Paper](https://arxiv.org/abs/1802.03426).
[^12]: Jian Tang et al. **Visualizing Large-scale and High-dimensional Data.** 2016, LargeVis. [Paper](https://arxiv.org/abs/1602.00370).
[^13]: Yingfan Wang, Haiyang Huang, Cynthia Rudin, and Yaron Shaposhnik. **Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization.** JMLR 2021. [Paper](https://jmlr.org/papers/v22/20-1061.html).
[^14]: Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. **DeepWalk: Online Learning of Social Representations.** KDD 2014. [Paper](https://arxiv.org/abs/1403.6652).
[^15]: Aditya Grover and Jure Leskovec. **node2vec: Scalable Feature Learning for Networks.** KDD 2016. [Paper](https://arxiv.org/abs/1607.00653).
[^16]: Jian Tang et al. **LINE: Large-scale Information Network Embedding.** WWW 2015. [Paper](https://arxiv.org/abs/1503.03578).
[^17]: Jiezhong Qiu et al. **NetSMF: Large-Scale Network Embedding as Sparse Matrix Factorization.** WWW 2019. [Paper](https://arxiv.org/abs/1906.11156).
[^18]: Jie Zhang et al. **ProNE: Fast and Scalable Network Representation Learning.** IJCAI 2019. [Proceedings](https://www.ijcai.org/proceedings/2019/594).
[^19]: Haochen Chen, Syed Fahad Sultan, Yingtao Tian, Muhao Chen, and Steven Skiena. **Fast and Accurate Network Embeddings via Very Sparse Random Projection.** CIKM 2019, FastRP. [Paper](https://arxiv.org/abs/1908.11512).
[^20]: Yuyang Xie et al. **Towards Lightweight and Automated Representation Learning System for Networks.** 2023 manuscript, LightNE 2.0. [Paper](https://arxiv.org/abs/2302.07084).
[^21]: Yuyang Xie et al. **SketchNE: Embedding Billion-Scale Networks Accurately in One Hour.** 2021 initial manuscript. [Paper](https://arxiv.org/abs/2110.12782).
[^22]: Zhaocheng Zhu et al. **GraphVite: A High-Performance CPU-GPU Hybrid System for Node Embedding.** 2019. [Paper](https://arxiv.org/abs/1903.00757).
[^23]: Adam Lerer et al. **PyTorch-BigGraph: A Large-scale Graph Embedding System.** 2019. [Paper](https://arxiv.org/abs/1903.12287).
[^24]: Rachith Aiyappa et al. **Implicit degree bias in the link prediction task.** ICML 2025. [Proceedings](https://proceedings.mlr.press/v267/aiyappa25a.html).
[^25]: Juanhui Li et al. **Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking.** 2023 manuscript, HeaRT. [Paper](https://arxiv.org/abs/2306.10453).
[^26]: Sadamori Kojaku, Filippo Radicchi, Yong-Yeol Ahn, and Santo Fortunato. **Network community detection via neural embeddings.** Nature Communications 15, 9446, November 2024. [Paper](https://www.nature.com/articles/s41467-024-52355-w).
[^27]: Andrew Davison, S. Carlyle Morgan, and Owen G. Ward. **Community Detection Guarantees Using Embeddings Learned by Node2Vec.** NeurIPS 2024; 2023 initial manuscript. [Paper](https://arxiv.org/abs/2310.17712).
[^28]: Daniel Zilberg and Ron Levie. **PieClam: A Universal Graph Autoencoder Based on Overlapping Inclusive and Exclusive Communities.** ICML 2025. [Proceedings](https://proceedings.mlr.press/v267/zilberg25a.html).
[^29]: Yunfei Liu et al. **Revisiting Modularity Maximization for Graph Clustering: A Contrastive Learning Perspective.** KDD 2024. [Record and abstract](https://openreview.net/forum?id=b2WIkJRG2E).
[^30]: **Robustness of shallow graph embedding methods for community detection.** arXiv 2405.00636, revised manuscript. [Paper and version history](https://arxiv.org/abs/2405.00636).
[^31]: **Dynamic Graph Embedding Through Hub-aware Random Walks.** 2025 manuscript, DeepHub. [Paper](https://arxiv.org/abs/2505.17764).
[^32]: **NOMAD: Generating Embeddings for Massive Distributed Graphs.** April 2026 preprint. [Paper](https://arxiv.org/abs/2604.09419).
[^33]: Sujan Chakraborty et al. **FUSE: Fast Semi-Supervised Node Embedding Learning via Structural and Label-Aware Optimization.** October 2025 preprint. [Paper](https://arxiv.org/abs/2510.11250).
[^34]: Thomas M. J. Fruchterman and Edward M. Reingold. **Graph Drawing by Force-Directed Placement.** Software: Practice and Experience, 1991. [Publisher](https://onlinelibrary.wiley.com/doi/10.1002/spe.4380211102).
[^35]: Andreas Noack. **Modularity clustering is force-directed layout.** Physical Review E, 2009; 2008 manuscript. [Paper](https://arxiv.org/abs/0807.4052).
[^36]: Mathieu Jacomy, Tommaso Venturini, Sebastien Heymann, and Mathieu Bastian. **ForceAtlas2, a Continuous Graph Layout Algorithm for Handy Network Visualization Designed for the Gephi Software.** PLOS ONE, 2014. [Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0098679).
[^37]: Mikhail Belkin and Partha Niyogi. **Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering.** NIPS 2001. [Proceedings](https://papers.nips.cc/paper/1961-laplacian-eigenmaps-and-spectral-techniques-for-embedding-and-clustering).
[^38]: Anton Tsitsulin et al. **VERSE: Versatile Graph Embeddings from Similarity Measures.** WWW 2018. [Paper](https://arxiv.org/abs/1803.04742).
[^39]: ShaoSheng Cao, Wei Lu, and Qiongkai Xu. **GraRep: Learning Graph Representations with Global Structural Information.** CIKM 2015. [Publisher](https://dl.acm.org/doi/10.1145/2806416.2806512).
[^40]: Bo Kang, Jefrey Lijffijt, and Tijl De Bie. **Conditional Network Embeddings.** ICLR 2019. [Author institutional record and manuscript](https://biblio.ugent.be/publication/8610843).
[^41]: Claire Donnat et al. **Learning Structural Node Embeddings Via Diffusion Wavelets.** GraphWave; 2017 initial manuscript. [Paper](https://arxiv.org/abs/1710.10321).
[^42]: Leonardo F. R. Ribeiro, Pedro H. P. Saverese, and Daniel R. Figueiredo. **struc2vec: Learning Node Representations from Structural Identity.** KDD 2017. [Paper](https://arxiv.org/abs/1704.03165).
[^43]: Thomas N. Kipf and Max Welling. **Variational Graph Auto-Encoders.** 2016 manuscript. [Paper](https://arxiv.org/abs/1611.07308).
[^44]: Anton Tsitsulin et al. **FREDE: Anytime Graph Embeddings.** PVLDB 14(6), 2021. [Author institutional record](https://pure.au.dk/portal/en/publications/frede-anytime-graph-embeddings/), [DOI](https://doi.org/10.14778/3447689.3447713).
[^45]: Maximilian Nickel and Douwe Kiela. **Poincaré Embeddings for Learning Hierarchical Representations.** NeurIPS 2017. [Paper](https://arxiv.org/abs/1705.08039).
[^46]: Yasuaki Hiraoka et al. **Topological Node2vec: Enhanced Graph Embedding via Persistent Homology.** 2023 manuscript. [Paper](https://arxiv.org/abs/2309.08241).
[^47]: V. A. Traag, L. Waltman, and N. J. van Eck. **From Louvain to Leiden: guaranteeing well-connected communities.** Scientific Reports 9, 5233, 2019. [Paper record](https://pubmed.ncbi.nlm.nih.gov/30914743/).
[^48]: Anton Tsitsulin, John Palowitch, Bryan Perozzi, and Emmanuel Müller. **Graph Clustering with Graph Neural Networks.** JMLR 24(127), 2023, DMoN. [Paper](https://www.jmlr.org/beta/papers/v24/20-998.html).

**Additional orientation:** Palash Goyal and Emilio Ferrara, *Graph Embedding Techniques, Applications, and Performance: A Survey*, Knowledge-Based Systems, 2018. [Manuscript](https://arxiv.org/abs/1705.02801). Useful taxonomy; not a substitute for checking the primary papers above.
