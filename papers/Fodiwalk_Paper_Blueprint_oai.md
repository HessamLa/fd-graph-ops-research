# Fodiwalk: Paper Blueprint

**Working title:** Fodiwalk: Graph Embedding through Walk-Derived Separation

**Revision 2:** Expanded abstract strategy, primary-source abstract study, ranked differentiators, sentence-level writing plan, and experiments that directly test the proposed mechanism. The method and results claims remain tied to the inspected repository snapshot.

**Purpose:** An abstract, paragraph-level manuscript plan, citation map, and experiment and discussion design for a top-tier machine-learning conference.

**Evidence base:** `HessamLa/fd-graph-ops-research`, commit `5c46a98c3f49dc34aa5bed796743b5fc7c75b6dc`. This is a writing and research plan, not a report of newly executed experiments. Proposed experiments, analyses, and claims are identified below.

## 1. Venue and central argument

ICLR 2026's abstract and full-paper deadlines were September 19 and September 24, 2025, respectively. Both have passed. This blueprint follows that venue's **nine-page initial-submission format**, rather than suggesting a new ICLR 2026 submission is possible. References and appendices were outside that limit; reviewers were not required to read appendices. For another cycle, check that cycle's rules. [ICLR 2026 Author Guide](https://iclr.cc/Conferences/2026/AuthorGuide).

**Proposed thesis:** Random walks reveal both co-occurrence and path separation. Fodiwalk uses the latter to weight nonlocal repulsion while retaining local attraction, constructing graph geometry through a sparse set of explicit interactions.

**Empirical hypothesis:** This particular assignment of geometric roles can improve the quality–cost trade-off for preserving graph separation. The mechanism is implemented; its comparative advantage must be established by the experiments below.

This is a hypothesis to establish, not a conclusion justified by the existing repository alone. The paper should explain when the construction helps, why it helps, and where it fails.

**Critical novelty boundary:** Random walks plus forces are not sufficient novelty. Force2Vec already includes **rForce2Vec**, which uses walk-reached vertices for attraction. Fodiwalk's proposed distinction is the role of retained nonlocal walk pairs: they receive gap-dependent repulsion, while immediate neighbors receive attraction. Compare the actual equations and sampling procedures, and evaluate this distinction directly. [Force2Vec, §III-C](https://arxiv.org/html/2009.10035v1).

**Three defensible contribution targets:**

1. A precisely specified sparse interaction construction and `fdhop` update, with complete neighbor retention and explicit walk-gap semantics.
2. A controlled account of how pair support, gap weighting, and normalization affect geometric fidelity and stability.
3. A reproducible quality–cost evaluation against strong walk, spectral, and force-based baselines, including failure regimes.

The implementation architecture is useful supporting material. Three Python stages and a GPU kernel are not, by themselves, the scientific contribution. Do not claim to invent force-directed embedding, random-walk neighborhoods, or SELL-C-sigma.

## 2. Abstract: the paper's central argument

### 2.1 What the reader should remember

**Essence:** Fodiwalk turns observed walk separation into nonlocal repulsion while retaining attraction at immediate neighbors.

**The distinctive idea in one sentence:** Fodiwalk gives walk-derived relationships explicit geometric roles: immediate neighbors attract, and retained nonlocal pairs repel with strengths weighted by their observed path separation.

The phrase **walk-derived separation** is the recommended organizing concept. Define it immediately as the minimum observed walk gap. This names the information entering the algorithm, rather than merely naming its implementation style. Use “walk gap” afterward for brevity.

The abstract should make four things memorable: the signal, its geometric role, the sparse construction, and the demonstrated consequence. It should not become a list of configuration knobs, metrics, force variants, or software packages.

### 2.2 Distinctive features, ranked by scientific importance

| Priority | Feature | Why it matters | Boundary of the claim | Abstract treatment |
|---|---|---|---|---|
| **1 — central** | Minimum observed walk gap weights nonlocal repulsion. | Path separation changes the magnitude of an interaction; context frequency alone does not specify this rule. | Hop-weighted forces have predecessors. The proposed advance is their use within a sampled walk-derived interaction construction, not the existence of repulsion. | Put this mechanism in sentences 2–3. |
| **2 — central** | Local and nonlocal pairs have different force roles. | Original neighbors retain attraction; nonlocal walk pairs contribute repulsion. This is a different inductive bias from using walk-reached vertices as positive attractive context. | Attraction and repulsion are not new. The exact role assignment and its consequences are the scientific question. | Explain it plainly, not through an abstract label such as “hybrid learning.” |
| **3 — central supporting** | Walks select interaction support and supply path-gap information. | One sampled graph process provides both which nonlocal pairs interact and how they are weighted, without an exact all-pairs distance table. | Optional sampled far pairs have assigned weights and are separate. Do not imply every interaction came from a witnessed path. | State the computational rationale in one sentence. |
| **4 — supporting** | Explicit neighbor completion protects the intended local attractive support. | Sampling should not silently remove original neighbors from the force system. | Retaining edges in D and assigning all of them h=1 are different invariants. The headline policy must satisfy both; see §2.10. | Mention neighbor retention, but do not call it a standalone invention. |
| **5 — supporting** | The fdhop force consumes h without a frequency term. | A simple magnitude law makes the separation hypothesis easier to interpret and test. | Walk frequency can still affect pair retention. The entire algorithm is not necessarily frequency-independent. | Usually leave frequency out of the abstract; explain in the method/ablation. |
| **6 — implementation/analysis** | Degree normalization, stochastic updates, sparse plans, and optional directed rows. | These affect dynamics, cost, and interpretation. | They are not established novel principles; asymmetry is policy-specific and does not make Euclidean distances directed. | Keep only a short update clause if space permits. |

**Do not build the abstract around “forces instead of gradients.”** Skip-gram updates also change node vectors; their gradients admit attraction/repulsion interpretations. Explicit Euclidean force magnitudes make Fodiwalk's design inspectable, but using force terminology does not prove novelty, speed, simplicity of optimization, or better geometry. With negative sampling, the relation between skip-gram and matrix factorization further cautions against claiming those methods lack structural meaning. [NetMF](https://arxiv.org/abs/1710.02971).

**The comparison that matters:** rForce2Vec uses walk-reached vertices for attractive neighborhoods; Fodiwalk's fdhop construction assigns retained nonlocal walk pairs gap-dependent repulsion. This distinction is read from the methods, not inferred from their titles. Force2Vec is therefore both a required baseline and a guard against an overly broad novelty claim. [Force2Vec, §III-C](https://arxiv.org/html/2009.10035v1).

### 2.3 Abstract study: what influential papers do well

I read the full abstracts on the linked primary manuscript pages. The sample includes six widely cited foundational papers—DeepWalk, node2vec, LINE, GraphSAGE, GCN, and GAT—plus Poincaré embeddings, NetMF, VERSE, and the especially relevant Force2Vec. This is a targeted writing study, not a ranked bibliometric survey; citation counts vary by index and date and are not used to score the prose. Influence does not establish that an abstract's wording caused a paper's success.

The notes below are **editorial interpretations** of the abstracts. Short quoted anchors locate the relevant language; they are not templates to copy. Each source contributes less than 25 words of quoted prose. Full abstracts remain at the linked sources.

| Paper / source | Short abstract anchor | How its abstract makes the contribution clear | Lesson applied to Fodiwalk |
|---|---|---|---|
| **DeepWalk** — Perozzi et al., KDD 2014. [Abstract](https://arxiv.org/abs/1403.6652). | “treating walks as the equivalent of sentences” | Gives a memorable conceptual translation, then connects it to tasks and concrete empirical benefits. | Explain the translation from walk gap to geometric force in equally plain terms. Do not borrow the language-model analogy. |
| **node2vec** — Grover & Leskovec, KDD 2016. [Abstract](https://arxiv.org/abs/1607.00653). | “a flexible notion of a node's network neighborhood” | Identifies neighborhood flexibility as the organizing advance and names biased walks as its mechanism. | Make the role of observed separation the organizing advance, not a broad claim about a new embedding framework. |
| **LINE** — Tang et al., WWW 2015. [Abstract](https://arxiv.org/abs/1503.03578). | “preserves both the local and global network structures” | Couples a structural objective with an optimization mechanism and a concrete scale statement. | Connect local attraction/nonlocal repulsion to a measured resource benefit. Do not equate LINE's “global” claim with exact shortest-path preservation. |
| **GraphSAGE** — Hamilton et al., NeurIPS 2017. [Abstract](https://arxiv.org/abs/1706.02216). | “sampling and aggregating features from a node's local neighborhood” | States a concrete limitation—generalization to unseen nodes—then gives a mechanism and evaluation that directly address it. | Align Fodiwalk's problem, mechanism, and geometry measurements. Do not inherit an inductive-learning claim: Fodiwalk currently learns per-node coordinates. |
| **GCN** — Kipf & Welling, ICLR 2017. [Abstract](https://arxiv.org/abs/1609.02907). | “a localized first-order approximation of spectral graph convolutions” | Names the approximation, identifies its computational consequence, and quickly reaches results. | State the exact approximation being used: observed path gaps replace the need to obtain all-pairs exact distances. Do not label them unbiased or accurate without measurement. |
| **GAT** — Veličković et al., ICLR 2018. [Abstract](https://arxiv.org/abs/1710.10903). | “different weights to different nodes in a neighborhood” | Translates an architectural mechanism into an understandable effect on how neighbors contribute. | Explain that a larger walk gap gives stronger repulsion at the same embedding distance. This is deterministic weighting, not learned attention. |
| **Poincaré embeddings** — Nickel & Kiela, NeurIPS 2017. [Abstract](https://arxiv.org/abs/1705.08039). | “simultaneously capturing hierarchy and similarity” | Motivates geometry through a specific structural mismatch, then matches the evaluation to that structure. | State the desired geometry before the solver. Scope Fodiwalk's benefit by graph regime; do not promise hierarchy preservation. |
| **NetMF** — Qiu et al., WSDM 2018. [Abstract](https://arxiv.org/abs/1710.02971). | “unified into the matrix factorization framework” | Leads with an analytical result and then derives an algorithmic consequence. | Include an analysis claim in Fodiwalk's abstract only when it is substantial and proved. An elementary gap upper bound alone is not a theoretical foundation for the whole method. |
| **VERSE** — Tsitsulin et al., WWW 2018. [Abstract](https://arxiv.org/abs/1803.04742). | “preserve the distributions of a selected vertex-to-vertex similarity measure” | Makes the representation target explicit and compares a scalable sampled variant to a fuller-information counterpart. | Name the signal Fodiwalk uses and add a same-support exact-distance control. Do not turn a force rule into a claimed distortion-minimization objective. |
| **Force2Vec** — Rahman et al., ICDM 2020. [Abstract](https://arxiv.org/abs/2009.10035). | “mapping its core computations to linear algebra” | Links computation to a measurable systems result and connects layout with predictive tasks. | State which cost is reduced and measure it end to end. The existence of parallel force-based embedding is already prior work. |

**Synthesis:** The strongest reusable pattern is a short causal chain: **specific representation problem → distinctive signal and mechanism → practical consequence → measured result**. This is my writing synthesis, not an empirical law about acceptance or citation impact. The final abstract should give its largest share of space to Fodiwalk's mechanism and evidence.

### 2.4 Recommended abstract — current evidence

**Status:** A polished working abstract for the fdhop-based Fodiwalk formulation. It reports existing results as preliminary, does not claim a completed new benchmark, and does not present the method as globally convergent. Neighbor-preserving wording describes the intended headline specification; the exact policy and h=1 invariant must be frozen as described in §2.10. The existing empirical evidence spans recorded variants, not a newly validated final configuration.

Random walks reveal both node co-occurrence and path separation, yet these signals need not imply the same embedding geometry. We introduce Fodiwalk, a graph embedding method that uses observed walk separation to construct sparse force interactions. The method retains immediate neighbors as attractive interactions and assigns each retained nonlocal walk pair a repulsive force weighted by its minimum observed walk gap. Neighbor repulsion resists collapse, while nonlocal repulsion decays with embedding distance. This construction gives local connectivity and nonlocal separation distinct roles in shaping node representations. Degree-normalized updates evolve the embedding over the retained pairs, without requiring an all-pairs shortest-path matrix. We assess geometric fidelity through shortest-path ranking and distance approximation, separately from edge discrimination. Preliminary experiments show favorable distance fidelity on citation networks, while DeepWalk remains stronger on a million-node social graph. Fodiwalk makes walk-derived separation an explicit geometric inductive bias, with benefits that depend on graph structure.

**Why this version is stronger:** It opens with the exact signal distinction, reaches the mechanism in sentence two, names what each pair type does, and ties the computation to a specific avoided requirement. It replaces “a framework for investigating” with a defined geometric hypothesis. The cautious empirical ending is a consequence of the available evidence, not a recommendation to keep provisional language in the submitted paper.

**Recommended lead if a more direct first sentence is preferred:** “Graph embeddings that recover local connectivity need not preserve separation over longer graph paths.” Use this if readers find the co-occurrence/separation opening too conceptual. Support the distinction with direct geometry and reconstruction results in the introduction; do not claim existing walk methods never preserve longer-range structure.

### 2.5 Submission-ready structure — fill only after final experiments

This is the stronger final form to aim for, not a statement that these results already exist. The brackets are deliberately conspicuous research-dependent fields. Replace them with verified facts, then remove every bracket before submission.

Random walks reveal both node co-occurrence and path separation, yet these signals need not imply the same embedding geometry. We introduce Fodiwalk, a graph embedding method that converts observed walk separation into sparse geometric interactions. Fodiwalk retains immediate neighbors under attraction and repulsion, while retained nonlocal walk pairs exert distance-decaying repulsion weighted by their minimum observed gap. Degree-normalized updates use this interaction graph without requiring all-pairs shortest-path computation. Across [N] graphs and [S] seeds, Fodiwalk achieves [primary geometry effect] relative to [strongest relevant baseline] under [matched budget]. Fixed-support ablations show [measured effect of replacing or permuting walk gaps], isolating the contribution of separation information from pair selection. End-to-end measurements demonstrate [verified time or memory result], with [named structural regime] remaining a limitation. These results establish [the narrow conclusion supported jointly by the geometry, ablation, and cost evidence].

**The result sentence should carry the paper.** A named baseline, a defined metric, a scope, and a cost condition are more informative than “outperforms state of the art.” Report a percentage reduction for an error metric only when the denominator is meaningful; use absolute changes for correlations or R² when that is clearer. A parenthetical uncertainty estimate may be appropriate if space allows. Do not insert an uncertainty interval from a different sampling unit or protocol.

**Alternative endings selected by evidence, not preference:**

| Final evidence | Suitable conclusion |
|---|---|
| Gap ablation improves direct geometry at matched cost across several regimes | “The results show that observed walk separation can guide sparse force interactions toward improved graph-distance fidelity.” |
| Competitive geometry uses materially less measured memory or time | “The results establish a practical trade-off between graph-distance fidelity and the cost of sparse interaction construction.” |
| Gains are concentrated in particular graph families | “The results identify the graph regimes in which walk-gap-weighted repulsion improves embedding geometry.” |
| Shuffling gaps leaves geometry unchanged | Do not conclude that separation information caused the gain. Reframe around interaction support or the measured negative result, and revise the title and opening accordingly. |

### 2.6 Sentence-level abstract architecture

Target roughly **150–200 words** for the final abstract. This is an editorial target, not a claimed venue word limit. Use a single paragraph with seven to nine sentences; avoid citations, equations, acronyms beyond the method name, and a list of dataset names unless they establish scope.

| Sentence role | One-line essence | Approximate space | Evidence / paper destination |
|---|---|---:|---|
| 1. Specific tension | Co-occurrence and separation need not imply the same geometry. | 18–25 words | Introduction I1–I3; context objectives and metric disagreement. |
| 2. Method | Fodiwalk uses observed walk separation to construct sparse geometric interactions. | 15–22 words | Method M1–M3. |
| 3–4. Mechanism | Neighbor attraction and gap-weighted nonlocal repulsion play distinct roles. | 35–45 words | Exact law M4–M6; primary differentiator. |
| 5. Consequence | Only retained pairs are processed and no all-pairs distance table is required. | 18–25 words | Cost A3; end-to-end measurements E8. |
| 6. Main result | Direct geometry improves or remains competitive under a declared comparison. | 25–35 words | Table 2, multi-seed effects and common protocol. |
| 7. Causal evidence | Fixed-support controls identify whether walk gaps matter. | 18–25 words | Table 3 and proposed Figure 4. |
| 8. Cost or boundary | Give the measured resource benefit or the clearest structural limitation. | 18–25 words | Figure 2 / structural sweep. |
| 9. Optional conclusion | State the precise implication in one short sentence. | 10–18 words | Discussion D1–D4; omit if it repeats the opening. |

Avoid making all nine sentences mandatory. If one result sentence can carry both scope and boundary, combine them. The target is high information density with a causal story, not a checklist disguised as prose.

### 2.7 Claim-to-evidence contract for the abstract

| Phrase the abstract may use | What must support it | What it does not establish |
|---|---|---|
| “weights repulsion by observed walk gap” | Exact fdhop equation and inspected implementation | Larger gaps always produce larger final Euclidean distances |
| “retains immediate neighbors as attractive interactions” | Every input edge is present with h=1 in the final policy | Every original edge is short in the resulting embedding |
| “without requiring all-pairs shortest paths” | Augmentation and training use observed gaps; optional exact-distance analyses are separately identified | Training is faster than every baseline or never computes a BFS |
| “preserves graph separation” as an empirical outcome | Direct shortest-path rank/stress on a declared test pair population | A nonlinear decoder's R² alone is proof of metric fidelity |
| “gap information improves geometry” | Fixed-support controlled gap interventions with uncertainty | Performance differences between two independently tuned full pipelines isolate causality |
| “scalable” or a quality–cost advantage | Whole-process time, host/GPU peaks, graph/support sizes, failures, hardware | Constant memory, unconditional linear time, or universal million-node feasibility |
| “generalizes to unseen edges” | Test edges excluded before graph-dependent training and augmentation | Classification of pairs whose edges were already embedded |
| “stable” or “convergent” | Explicit scope: bounded empirical trajectories or a proved theorem with applicable assumptions | A finite 200-epoch run establishes convergence |

### 2.8 Wording to replace

| Weak or misleading wording | Preferred treatment |
|---|---|
| “Unlike neural methods, we directly update node coordinates.” | Explain the specific interaction rule. Neural embedding methods update coordinates too. |
| “The first method to combine random walks and forces.” | Acknowledge rForce2Vec; focus on the role and weighting of nonlocal pairs. |
| “Without a loss function.” | Usually omit. A missing explicit scalar objective is not a benefit by itself, and restricted energy interpretations may exist. |
| “Preserves local and global topology.” | State the exact measured property: neighbor recovery, path-distance ranking, or calibrated stress. |
| “Walk distances.” | Use minimum observed walk gaps, distinguishing them from shortest-path distances. |
| “Global repulsion.” | Use nonlocal repulsion over retained pairs. A sparse sample is not a complete global constraint system. |
| “Frequency-independent embeddings.” | The fdhop force has no frequency term; pair selection may still depend on frequency. |
| “Orders nodes according to graph distance.” | Gap-weighted repulsion supplies a separation signal; ordering preservation must be measured. |
| “Interpretable, efficient, scalable, robust, and effective.” | Name one mechanism and one verified outcome instead of stacking adjectives. |
| “Experiments validate the effectiveness of our framework.” | Name the metric, comparator, scope, uncertainty, and meaningful limitation. |

### 2.9 Make the rest of the paper deliver the abstract

**Introduction I1–I3:** Open with a concrete example of the same two nodes appearing within walk context while being separated by multiple graph edges. Explain why contextual affinity and graph-distance fidelity are different targets; do not claim that one is inherently correct for every task. Then contrast the role of that pair under positive-context learning, rForce2Vec, and Fodiwalk.

**Figure 1 upgrade:** Use the same small graph in every panel. Highlight a direct edge u–v and a path u–a–b–w. Annotate the observed gaps, identify the direct neighbor's attractive/repulsive interaction and the nonlocal pair's repulsion, and show the resulting force directions. A companion mini-panel can show the positive-context role in rForce2Vec, explicitly labeled as a mechanism schematic rather than the final embedding. Do not depict node2vec's two-vector dot-product objective as identical to a Euclidean spring.

**Table 2 upgrade:** Lead with direct geometry and the strongest relevant competitor. Reconstruction and learned-decoder metrics are supporting panels. A paper whose abstract centers geometry should not headline a saturated reconstruction AUC.

**Table 3 upgrade:** Lead with exact same-support gap shuffling and constant-gap controls. Repeat from matched initial coordinates with controlled randomness. A full-pipeline comparison cannot establish that the gap information itself is responsible.

**New proposed Figure 4 / optional replacement for part of Figure 3:** Plot the gap-information intervention against direct geometry at fixed support: actual gaps, within-source shuffled gaps, constant nonedge gaps, and exact shortest-path weights. Show separate graph regimes and intervals. This is the visual evidence for the abstract's causal sentence; if it is more decisive than a broad sensitivity heatmap, give it the main-text space instead. Do not add pages to retain every planned figure.

**Discussion D1:** State which component survived the decisive controls. If fixed-support permutations erase no benefit, say so and change the abstract's claimed explanation. Mechanistic clarity is more important than defending the first narrative.

### 2.10 Implementation and evidence details that affect wording

**Neighbor-retention correction discovered during this revision:** In the inspected `walk_edges` builder, `add_edges` inserts missing edge keys with h=1 but does not explicitly reset h for a true edge already present in the walk statistics with a larger gap. Such a pair is possible, for example when a walk reaches an adjacent endpoint through another path without traversing the direct edge. Thus “every true edge is an attractive h=1 interaction” should not be asserted for all existing walk_edges runs solely because every edge key is stored. `nbr_walk` with `edge_rule='both'` uses a merge that sets direct-neighbor gaps to 1. The final method must check or enforce the intended invariant, and any changed code requires new identified runs. This task updates the paper plan, not repository code.

**Scope the family:** fdhop is the main force, but walk_edges and nbr_walk have different support and gap semantics. Use them as named variants until one is frozen. Stronger neighbor-completion wording describes a specification, not proof that all historical results used it exactly.

**Cost language:** Sparsity controls force evaluation, while fixed-support storage and Z still grow with graph size. Optional far samples can have assigned rather than observed weights. The abstract need not describe every option; the configuration reported in the method must match its simplified account.

**Current evidence:** The existing scorecard's citation-graph strengths and social-graph reversal can be summarized provisionally. They do not supply final multi-seed effects, causal gap evidence, or verified generalization on edges withheld before embedding. Replace provisional result sentences once the planned experiments are complete. No new training or benchmark was run for this writing revision.

## 3. Main-paper allocation

| Component | Target pages | Function |
|---|---:|---|
| Title and abstract | 0.40 | State mechanism and measured result |
| 1. Introduction | 1.00 | Establish problem, closest gap, contributions |
| 2. Related work | 0.65 | Position against direct predecessors |
| 3. Fodiwalk | 2.00 | Define the complete algorithm |
| 4. Properties and computational cost | 0.70 | Give bounded, checkable analysis |
| 5. Experiments | 3.35 | Answer the scientific questions |
| 6. Discussion and limitations | 0.65 | Explain mechanisms and boundaries |
| 7. Conclusion | 0.25 | State the supported contribution |
| **Total** | **9.00** | Figures and tables must fit inside these allocations |

Treat these as planning estimates, not a reason to compress typography. Keep the algorithm, main result, strongest ablation, and key limitation in the main text. Use appendices for full protocols, extra datasets, proofs, and comprehensive sweeps.

## 4. Paragraph-by-paragraph manuscript plan

Each row specifies one intended manuscript paragraph. The **essence** can become its topic sentence. Citation keys resolve to linked primary sources in §10. A dash means original method or original results: no external citation is needed. Cite each baseline at first appearance; repeated result paragraphs need only the relevant table or figure cross-reference.

### 1. Introduction — five paragraphs

| ID | One-line essence | What to write | Cite / placement |
|---|---|---|---|
| I1 | Walk co-occurrence and path separation are distinct signals for representing a graph. | Start with the graph example in §2.9. Explain which task needs contextual affinity and which needs distance fidelity. Do not claim all tasks require shortest-path preservation. | DW, N2V, VERSE for representation targets; BIAS for why edge scores alone are insufficient. |
| I2 | The geometric role assigned to a walk pair determines what the update encourages. | Explain positive context, rForce2Vec's attractive walk neighborhoods, and gap-weighted repulsion. Note that skip-gram also updates vectors and has structural interpretations. Neither family is universally superior. | N2V, NETMF, F2V; cite after the corresponding descriptions. |
| I3 | The open question is how walk-derived separation should control sparse forces. | Acknowledge earlier hop-based forces and rForce2Vec. Identify the proposed change: retain local attraction and use observed gap to weight nonlocal repulsion. Do not claim “first walk-based force embedding.” | FD23, KFD24, F2V. Add REDUCE only after its full manuscript is checked. |
| I4 | Fodiwalk converts sampled paths into a sparse, explicitly weighted force system. | Preview augmentation, neighbor completion, gap-based law, degree normalization, and repeated updates. State whether the headline variant is symmetric or directed. | Original method; insert Figure 1 after this paragraph. |
| I5 | Controlled experiments test the mechanism, its cost, and its failure regimes. | Give three contributions and one compact measured summary once experiments are complete. State a boundary alongside strengths. | Original results; refer to Tables 2–3 and Figure 2. |

### 2. Related work — four paragraphs

| ID | One-line essence | What to write | Cite / placement |
|---|---|---|---|
| R1 | Walk methods learn from sampled context, while spectral methods encode matrix structure directly. | Explain the particular contrast needed for this paper, including why minimum walk gap differs from frequency. Avoid a long catalog of unrelated GNNs. | DW, N2V, LINE, NETMF, LE. |
| R2 | Force-based graph representation has established precedents, including random-walk variants. | Make the closest comparison: objective, pair support, attractive versus repulsive roles, hop information, normalization, and computation. Earlier convergence claims do not automatically transfer. | FR, KK for origins; F2V, FD23, KFD24 for closest methods. |
| R3 | Geometry-preserving methods optimize different notions of neighborhood and global structure. | Position similarity preservation and manifold/hyperbolic approaches. Explain that data-input assumptions and geometry differ; do not compare raw-feature methods to topology-only Fodiwalk without a separate track. | VERSE, UMAP, LV, POINCARE. |
| R4 | Evaluation design can change the apparent ordering of graph embeddings. | Separate reconstruction, held-out prediction, direct distance geometry, and learned decoders. Motivate challenging negatives and degree controls. | HEART, BIAS. Insert compact prior-work comparison Table A1 in the appendix. |

### 3. Fodiwalk — eight paragraphs

| ID | One-line essence | What to write | Cite / placement |
|---|---|---|---|
| M1 | Fodiwalk maps a topology-only graph to Euclidean node vectors. | Define G, n, m, d, true shortest-path distance δ, interaction set S(u), and Z. State undirected input, treatment of disconnected components and isolates, and transductive scope. | —; notation box if space permits. |
| M2 | Walk sampling selects a limited set of candidate interactions. | Define walks per node R, length L, transition probabilities, window w if applicable, and the precise pair-retention procedure. The main algorithm must name one policy. | DW for uniform walks; N2V only if biased transitions are introduced. |
| M3 | Each retained walk pair carries its minimum observed path gap. | Define h separately from δ and frequency f. Explain the upper bound, path versus window semantics, pruning, and selected-pair bias. Assigned far weights are a separate category. | —; equations and toy example in Figure 1. |
| M4 | Explicit neighbor completion protects immediate graph connectivity. | Add every original neighbor at h=1 in both directions for the chosen main policy. Specify ordering relative to caps. State whether nonlocal support is symmetric. | —; refer to completeness ablation. |
| M5 | The force law attracts immediate neighbors and repels retained nonlocal pairs in proportion to their gap. | Give the exact piecewise fdhop magnitude, direction, parameter domains, and zero-distance rule. Define h=0 as absent/padding, not a real pair. | FD23/KFD24 for lineage, not for the new equation's validation. |
| M6 | Degree normalization and update masking are part of the method. | Define the divisor, plain update, learning-rate schedule, drop probability, and zero-degree policy. Report that dropout is not rescaled. These choices cannot be hidden as implementation details. | —; other optimizers belong in ablations. |
| M7 | Sparse execution evaluates only retained interactions. | Explain reuse of augmentation, row batching, representation dtype, and working memory. Briefly identify SELL-C-sigma as existing machinery; separate resident and host-streamed modes. | SELL; insert Algorithm 1 after M6–M7. |
| M8 | One frozen specification defines the headline method. | Give a compact configuration and stopping rule. Name walk_edges versus nbr_walk variants explicitly. Treat fdlinear and other force families as controls, not competing definitions of Fodiwalk. | —; full configuration table in Appendix B. |

### 4. Properties and computational cost — three paragraphs

| ID | One-line essence | What to write | Cite / placement |
|---|---|---|---|
| A1 | An observed walk gap upper-bounds shortest-path distance but does not guarantee its ordering. | State and prove the elementary bound for reached pairs; give a counterexample to global order preservation. Separate assigned long-range weights from this proposition. | Original observation; proof in Appendix C. |
| A2 | The force law has an interpretable local balance, but global convergence needs stronger assumptions. | Derive neighbor equilibrium; explain reciprocal versus directed interactions. A restricted energy interpretation may be possible, while a general global convergence theorem is not established. | KFD24 for prior context only; own derivation in Appendix C. |
| A3 | Computational cost depends on the retained interaction count and preprocessing. | Define M, padded count M̃, and augmentation cost. Give step and memory accounting with assumptions. Do not claim constant memory or unconditional linear complexity. | SELL for format; own accounting; reference scaling Figure 2. |

### 5. Experiments — thirteen paragraphs

| ID | One-line essence | What to write | Cite / placement |
|---|---|---|---|
| E1 | The evaluation separates geometric fidelity, prediction, and resource use. | State RQ1–RQ5 from §7 and predeclare the primary outcome. Explain that representation and predictive-decoder results are separate. | BIAS, HEART. |
| E2 | The graph suite varies structure as well as size. | Introduce dataset families, preprocessing, component policies, and exact graph versions. Define the structural rationale rather than listing familiar names. | Dataset-specific original sources; OGB if used. Insert Table 1. |
| E3 | Baselines cover the closest competing mechanisms. | Include Force2Vec/rForce2Vec, direct predecessors, DeepWalk, tuned node2vec, LINE, and spectral/NetMF. Explain restricted-scale baselines and shared information. | F2V, FD23, KFD24, DW, N2V, LINE, NETMF, LE. |
| E4 | Model selection is isolated from the reported test outcomes. | State dimensions, seeds, splits, tuning budgets, validation objective, stopping rules, and hardware. Distinguish each method's native settings from budget-matched controls. | —; Appendix B for full search spaces. |
| E5 | Direct distance metrics test what the embedding geometry itself preserves. | Report shortest-path rank correlation and scale-calibrated stress, with connected-pair coverage and uncertainty. Discuss neighborhood recall alongside global geometry. | Own results; insert Table 2, panel A. |
| E6 | A trained hop decoder measures recoverable information beyond raw distance fidelity. | Report scalar-distance RF R²/MAE and a simple calibration control; put vector-decoder results in the appendix. Analyze disagreements with E5 without calling a nonlinear-decoder gain a metric-preservation proof. | Own results; Table 2, panel B. |
| E7 | Held-out edges test generalization only when they were unavailable to augmentation. | Present true train-graph LP with random and challenging negative sets, plus degree-only controls. Keep the old full-graph results labeled reconstruction. | HEART, BIAS; Table 2, separate prediction panel or Table A2 if space is tight. |
| E8 | End-to-end cost determines whether a quality gain is practical. | Show preprocessing, compile, embedding, and scoring costs separately. Compare quality at matched elapsed budgets and memory limits, including OOM. | Own results; insert Figure 2. |
| E9 | Pair-selection controls isolate the contribution of walks. | Compare edges-only plus uniform nonedges, walk_edges, nbr_walk, and a symmetric/directed control with matched support budget where possible. Report realized M and neighbor coverage. | Own results; Table 3, panel A. |
| E10 | Fixed-support force ablations isolate pair role and walk-gap weighting. | Hold pairs and initialization fixed; intervene on the nonlocal role, replace gap weights by constants or permutations on nonedges, use exact distances on the same support, and compare fdhop with fdlinear under a matched decay search. Label new attractive-role controls as experimental modifications. | Own results; Table 3, panel B; optional Figure 4 from §2.9. |
| E11 | Update controls test whether improvements come from normalization or effective step size. | Compare degree normalization, no normalization, row-width normalization, and dropout with separately tuned step sizes. Mark modified variants as proposed controls. | Own results; Figure 3, panel A. |
| E12 | Controlled graph families reveal when the mechanism fails. | Sweep density, community mixing, and tree-like versus loopy structure independently. Show uncertainty; do not infer a universal degree threshold from a few real graphs. | Original generator references when finalized; Figure 3, panel B. |
| E13 | Checkpoints reveal whether a reported comparison reflects training progress or a stable plateau. | Plot geometry and displacement at 50/100/150/200 epochs and selected longer runs; compare against elapsed time too. A finite run is not proof of convergence. | Own results; Figure A3, with a main-text summary. |

### 6. Discussion and limitations — five paragraphs

| ID | One-line essence | What to write | Cite / placement |
|---|---|---|---|
| D1 | Walk-gap repulsion is useful only to the extent supported by fixed-support controls. | Interpret E9–E10: support, gap information, or their interaction? If shuffled gaps perform equally well, narrow the claimed mechanism. | Tables 2–3; F2V when contrasting mechanisms. |
| D2 | Better edge discrimination and better global geometry need not coincide. | Explain metric disagreements, especially citation/social/tree contrasts. Discuss degree effects and decoder flexibility. Do not select whichever metric favors the method. | BIAS; Table 2 and Figure 3. |
| D3 | Structural failure cases delimit the method's use. | Discuss sparse graphs, missing global constraints, Euclidean representation of hierarchies, and disconnected components. Distinguish measured causes from hypotheses. | POINCARE for geometric context; Figure 3. |
| D4 | Sparse computation trades interaction coverage for cost. | Explain augmentation memory, fixed versus regenerated walks, and hardware dependence. A million-node completed run does not imply all million-node graphs fit. | Figure 2; SELL only when discussing the format. |
| D5 | The current method is transductive and its general convergence is unresolved. | State no learned out-of-sample encoder, no demonstrated original-directed-input guarantee, stochastic and numerical variability, and parameter sensitivity. Identify only a few targeted next steps. | Own limitations; prior theorem assumptions only if compared explicitly. |

### 7. Conclusion — one paragraph

| ID | One-line essence | What to write | Cite / placement |
|---|---|---|---|
| C1 | Fodiwalk provides a testable sparse-force construction whose benefits depend on graph structure and evaluation objective. | Restate the exact contribution, strongest supported finding, and main boundary in 4–5 sentences. No new experiment or promise of universal superiority. | — |

## 5. Mathematical specification to settle before drafting

Use the final `fdhop` law as an integral part of Fodiwalk. The existing `papers/paper-description.md` already points in this direction. The augmentation still needs a single declared headline policy: the repository's directed design is `nbr_walk`, whereas the recorded fdhop comparisons principally use `walk_edges`. Do not describe one and report the other's scores. My recommendation is to carry both named variants through development, then freeze the main choice using validation data before new test evaluation.

### Pair semantics

For a source-walk variant, let Wᵤ be its sampled walks and define hᵤᵥ as the minimum positive step index reaching v. For windowed walk_edges, define hᵤᵥ as the minimum separation of u and v within retained walk windows. These are different estimators.

For either estimator, a retained observed path yields δ_G(u,v) ≤ hᵤᵥ. This holds for observed path gaps, not arbitrary assigned far weights. The bound alone gives no guarantee that hᵤᵥ < hᵤ𝓌 implies δ_G(u,v) < δ_G(u,w). Record minimum-gap error and pair coverage separately.

Neighbor completion should guarantee N_G(u) ⊆ S(u) and hᵤᵥ=1 for every original neighbor. Verify this invariant in the exact policy used for the paper. The library's plain `walk` option does not provide that completion. Policies using transformed PMI or flat weights also change the force's h=1 branch, so they are not clean gap-weight ablations.

### Main force and update

Let rᵤᵥ=‖zᵥ−zᵤ‖₂ and eᵤᵥ=(zᵥ−zᵤ)/rᵤᵥ for nonzero rᵤᵥ. The fdhop magnitude is

\[
g(r,h)=\begin{cases}
k_1r-k_r, &h=1,\\
-k_rh\exp(-k_4r), &h\ge2.
\end{cases}
\]

\[
F_u(Z)=\frac{1}{d_u}\sum_{v\in S(u)}g(r_{uv},h_{uv})e_{uv},
\qquad z_u^{t+1}=z_u^t+\eta_t b_u^t F_u(Z^t).
\]

For the default row-drop mechanism, bᵤᵗ is Bernoulli(1−p_drop), with no inverse-probability rescaling. Therefore E[bᵤᵗFᵤ | Zᵗ]=(1−p_drop)Fᵤ for a fixed force; this expectation does not equate the trajectories of dropped and undropped systems. Define dᵤ as the actual implemented count of h=1 entries or explicit override, not |S(u)|. Address zero degree explicitly; the implementation can freeze such rows. At r=0, the present kernel returns zero pair contribution.

Positive k₁,kᵣ,k₄ give a neighbor-pair zero-force distance kᵣ/k₁. Far pairs alone have no finite zero-force distance. Neither statement proves a whole-graph equilibrium or convergence. In particular, disconnected groups with only repulsion between them may keep separating while forces decay.

### Analysis that would be valuable and honest

For **fixed symmetric support and reciprocal weights**, candidate pair potentials are U_near(r)=k₁r²/2−kᵣr and U_far(r)=(kᵣh/k₄)exp(−k₄r), up to constants. Their radial derivatives give the magnitudes above. Under positive degree normalization, the undropped continuous dynamics can be interpreted as diagonally preconditioned descent on the summed pair energy. This is a proposed restricted derivation to verify, not a theorem already established for every code path. It excludes directed/asymmetric support, collision singularities, arbitrary update rules, and unsupported fixed-step convergence claims. Do not state that unequal degrees automatically eliminate every energy interpretation: preconditioning matters.

If adding a discrete-time theorem, supply explicit regularity, boundedness, and step-size assumptions, and prove them or restrict the theorem. The repository's empirical “effective learning-rate boundary” is a useful diagnostic, not a universal stability theorem. Keep operational learning rates below 1 as the project specifies; that setting alone does not establish stability.

### Cost accounting

Let M=Σᵤ|S(u)| count directed retained interactions and M̃ count cells actually processed after padding. A conceptual force epoch costs O(Md); the current padded kernel's arithmetic tracks O(M̃d). Embeddings alone require Ω(nd) storage. Sparse interaction storage adds O(M); plans, padding, augmentation temporaries, and optimizer state add costs that must be measured.

With constant-cost uniform transitions, R walks of length L per node cost O(nRL) to generate. Window-pair emission can add a factor w, followed by aggregation, pruning, and sorting. Biased rejection sampling and far-pair sampling require their own accounting. Report total time as T_walk+T_aggregate+T_plan+T_compile+T_embed. Do not infer O(n) end-to-end time merely from bounded walk length, especially when graph edges, far-pair budgets, or padded support grow differently.

**Algorithm 1 should contain:** input graph and all scientific parameters; walk generation; gap aggregation; pair budget; neighbor completion; optional far-pair rule; random initialization; degree construction; force aggregation; update masking; coordinate update; stopping and divergence reporting; output Z. Include any far-pair path actually used: `far=0` in a cap-based walk policy can request an automatic budget rather than disable sampling.

## 6. Evidence audit: what the repository can and cannot support

| Existing evidence or issue | Consequence for the paper |
|---|---|
| Recorded fdhop results favor some citation-graph geometry metrics; DeepWalk is stronger on recorded com_youtube distance metrics. | A regime-dependent claim is plausible. A universal win is not. Recompute using the final common protocol. |
| The store and narrative tables contain different decoder/protocol settings. | Never combine numerical cells solely because they share a label such as “hop R².” Record feature, model, sampling, and graph hash. |
| Many relevant comparisons use one seed. | They are preliminary measurements. Lack of significance is not evidence of a tie or equivalence. |
| fdlinear and fdhop were often compared with k₄=0.01 and 1.0, respectively. | A 100-fold decay difference confounds a force-law comparison; match or equally tune it. |
| `evaluator/tasks/link_prediction.py` receives an already learned Z and splits pair features. | It does not itself enforce an edge holdout before embedding. When Z used the full graph, label the result edge reconstruction/discrimination. |
| Some earlier memory records sampled RSS after embedding rather than at the peak. | Do not reuse those numbers as peak-memory evidence. Measure the full process lifetime. |
| The directed method description and recorded main fdhop policy differ. | Freeze and name variants; describe exactly the variant whose scores appear. |
| walk_edges inserts missing edge keys at h=1 but does not explicitly lower the gap of already-retained true edges. | Audit neighbor attraction, not just support coverage. A stronger neighbor-completion guarantee requires an invariant check or a corrected version with new runs; see §2.10. |
| Some runs were OOM or never executed, and several trajectories still improved at 200 epochs. | Report missingness and failures; do not convert an epoch budget into a convergence claim. |

Primary repository anchors: `fodiwalk/model.py`, `fodiwalk/augment_graph/policy_walk.py`, `policy_nbr_walk.py`, `fodiwalk/embed/forces.py`, `degrees.py`, `forcedirected/sell_c_sigma.py`, `evaluator/tasks/`, `papers/fodiwalk/fodiwalk.md`, and `session-summary/20260908T161112Z-force-law-family-and-the-embedding-store.md`. The prose drafts are useful context, but code and frozen run records take precedence when they disagree.

## 7. Experiments: hypotheses, protocols, and controls

### Research questions

| RQ | Question | Required evidence | What would weaken the claim? |
|---|---|---|---|
| RQ1 | Does Fodiwalk preserve graph geometry competitively? | Direct geometry on held-out evaluation pairs across multiple graph families | Gains disappear outside citation graphs or only occur with a flexible decoder |
| RQ2 | Do nonlocal pair roles and walk-gap weighting explain the gain? | Fixed-support role interventions plus true-gap, constant-gap, and shuffled-gap controls | Gap permutation has no effect or repulsive role offers no advantage after comparable tuning |
| RQ3 | Does the method generalize to unobserved edges? | Train-graph-only embeddings and degree-aware/challenging negative evaluation | Gains vanish after removing test edges or matching degree distributions |
| RQ4 | Is quality competitive at comparable cost? | End-to-end time, true memory peaks, and quality–cost curves | Better scores require much more compute or unmatched preprocessing |
| RQ5 | Which structures cause failure? | Controlled synthetic sweeps and per-regime real-data analysis | Broad performance claims are contradicted by tree/road/social regimes |

### Dataset plan

| Family | Candidate datasets | Why include it? | Required reporting |
|---|---|---|---|
| Citation | Cora, Citeseer, PubMed | Continuity with existing runs; manageable exact-distance audit | n, m, components, source/version, directed-to-undirected conversion |
| Social | com_youtube; one additional independent social graph | Million-node scalability and the observed baseline reversal | Degree quantiles, clustering, reachable-pair fraction, memory failures |
| Hierarchical | WordNet; taxonomy if resources permit | Stress Euclidean geometry and sparse branching | Exact relation subset, edge direction policy, giant-component policy |
| Spatial | roadNet-CA or a smaller road graph, power grid | Long paths and low local degree | Diameter estimates, isolate handling, scale of exact-distance evaluation |
| Controlled | Paths, cycles, trees, grids; SBM and degree-heterogeneous synthetic graphs | Vary structure without conflating it with graph size | Generator parameters, seeds, achieved rather than requested graph statistics |
| Standardized LP | One suitable OGB link-prediction dataset | Independently defined task and official splits | Follow dataset-specific feature, temporal, and negative-set rules exactly |

Use roughly 6–8 real graphs plus controlled families for the central story. Table 1 holds compact statistics; the appendix names exact source papers and dataset URLs after the final versions are selected. Do not cite a generic dataset collection as the original source for every dataset. A future extension can add community detection, but it is not necessary to establish this geometry-focused paper; if included, use identical clustering and report NMI/ARI on planted or verified labels.

### Baseline plan

**Essential:** DeepWalk, tuned node2vec, Force2Vec including rForce2Vec, the earlier hop-based force method where feasible, LINE, and a spectral or NetMF baseline. Force2Vec/rForce2Vec is a closer novelty test than simply adding more GNNs.

**Conditional:** VERSE; a Poincaré model for hierarchical datasets; UMAP/LargeVis only with a documented common graph-derived input and scalable preprocessing. Classical Kamada–Kawai or exact-hop force runs are small-graph references, not required million-node competitors. A feature-based GNN belongs in a separately labeled track if node features or supervision exceed what Fodiwalk receives.

**Controls:** Random Gaussian embeddings; degree-only edge scoring; common-neighbor and degree-normalized local heuristics for LP. A cheap baseline outperforming a learned embedding is scientifically informative and should remain visible.

Use native recommended baseline configurations plus a fair validation search. Separately run a matched-walk-budget or matched-time experiment. Equal walk counts alone do not equalize training objectives or total compute, and equal epoch counts across methods have no common computational meaning.

### Geometry protocol

1. Embed the full graph for the descriptive-geometry task. Seeing the graph is intended here; this is not unseen-edge prediction.
2. Fix evaluation source nodes and node pairs independently of each method's augmentation. Use a separate validation set for choosing hyperparameters and calibration. Reuse the test pairs across methods.
3. Obtain **true** shortest-path lengths by BFS on the selected graph. If exact computation is infeasible, disclose approximation separately; do not label landmark estimates exact ground truth.
4. Report rank fidelity using Spearman correlation with a stated tie convention. Add neighborhood recall with a fixed k or a degree-aware definition; document ties and approximate-index accuracy.
5. Use calibrated metric stress, e.g. sqrt[Σ(s·rᵢ−δᵢ)² / Σδᵢ²], with a nonnegative scale s fitted on validation pairs and then frozen. Specify the pair population, component weighting, and weighting across hop shells. This is a declared stress variant, not an unspecified universal “distortion” score.
6. Treat RF hop R²/MAE from scalar distance as a secondary learned-decoder task. Include a simple linear or monotone calibration control. Report vector-decoder results separately.
7. Exclude infinite distances from finite-distance metrics and report excluded fractions. State that the result is conditional on reachability. Stratify by hop length and source degree so common short pairs do not hide long-range failure.
8. If using source-based BFS sampling at scale, use the same scheme across methods and make its target distribution explicit. Where small and large graphs use different schemes, add a protocol-sensitivity comparison rather than presenting them as interchangeable.

### Held-out link-prediction protocol

Split positive edges into train/validation/test **before** walks, degree computation, interaction construction, or embeddings. Use official splits where available. For custom static splits, prescribe and report the procedure, including any connectivity-preservation bias. Validation selects settings; test edges remain absent from every training-graph-dependent operation. If a final refit on train+validation is allowed by the benchmark, keep test edges excluded and document the refit.

Report ROC-AUC and AP on a declared negative ratio; use MRR/Hits@K for a fixed documented ranking candidate protocol where appropriate. Include random nonedges and a separate degree-corrected or heuristic-hard negative track inspired by BIAS/HEART. The evaluator may consult the full known positive set to avoid labeling a withheld positive as a negative; that exclusion must not feed held-out topology into the embedding model. Keep edge reconstruction in a separate labeled panel if retained.

Hold the decoder and its search budget fixed for embedding comparisons. Supplement learned-decoder results with direct embedding-distance scores where meaningful. Report per-degree and edge-type results and a degree-only baseline. Record all negative sets and splits for replay.

### Replication and model selection

Use at least five embedding seeds on principal real-graph comparisons. For custom LP, ideally use three independent edge splits with five initialization/walk seeds nested within each split; preserve fixed official splits where applicable. On expensive large graphs, three seeds can be reported as a resource-limited exception with wider uncertainty, not silently pooled with five-seed results.

Pair method comparisons on the same splits and evaluation pairs. Report mean, standard deviation, and paired effect intervals; bootstrap over independent seeds/splits or source clusters as appropriate, not over millions of dependent pairs as though they were independent replications. Do not interpret a non-significant test as equivalence. Avoid averaging raw R² or stress across incomparable graph protocols without disclosing the aggregation.

Use d=128 for main geometry and predictive comparisons, matching the project's standard. Use d=64 for optimizer/schedule development, then validate selected choices at 128. Dimension sensitivity at 32/64/128/256 belongs in the appendix. Checkpoints at 50, 100, 150, and 200 epochs support the primary training analysis; selected longer runs test whether conclusions change. Sub-five-epoch runs support memory/runtime diagnostics only.

### Focused ablation matrix

| Factor | Main comparison | Hold fixed | Interpretation |
|---|---|---|---|
| Walk support | walk_edges vs nbr_walk vs uniform sampled nonedges plus all edges | d, initialization, force, approximate M, validation budget | Does walk-derived support matter? |
| Nonlocal pair role | fdhop repulsion vs an explicitly specified nonlocal-attraction control; native rForce2Vec separately | Exact S(u), initial Z, local-edge law, matched scalar search budget; record changed magnitude distribution | Does the nonlocal role matter beyond sampling? A sign change is a new control, not a reproduction of rForce2Vec. |
| Gap information | actual gaps vs shuffled nonedge gaps vs constant nonedge gap ≥2 | Exact support, edges at h=1, normalization, initial Z | Does the weight encode useful information? |
| Gap accuracy | observed gap vs exact shortest-path distance on the same support | Pair support and all training settings | Does path-length error limit geometry? Run on feasible graphs. |
| Neighbor completion | completion on/off | Raw walks and nonlocal support; record changed degree divisor | Are retained original edges essential? This is a joint support/normalization change unless separately controlled. |
| Force function | fdhop vs fdlinear | Support, equal search over k₄, initialization, step budget | Is removal of frequency helpful after decay is controlled? |
| Normalization | h=1 degree vs row width vs none | Same pairs and law; retune η per variant | Which stability/geometry effect belongs to normalization? |
| Update drop | p_drop=0, 0.25, 0.5 | Same support; both fixed-η and mean-step-matched controls | Does stochastic masking help beyond shrinking average movement? |
| Far sampling | prescribed budgets and assigned weights vs disabled sampling | Local support and force | Are extra global constraints responsible for gains? Verify how “disabled” is expressed in code. |
| Walk budget | R, L, w and p,q sweeps | Declared resource or interaction budget | Does more exploration improve quality per unit cost? |

Do not run an unstructured full Cartesian sweep of every knob. First identify mechanisms on a small diverse development suite, freeze the selected settings, and confirm on the complete suite. Report the number of trials and tuning compute for every method.

For gap interventions, preserve the edge/nonedge mask and degree divisor. A global permutation preserves the overall nonedge-gap histogram; a within-source permutation additionally preserves each row's total gap weight. Use the latter to test partner-specific information independently of row-level repulsion scale. A permutation within true-hop shells asks a narrower question and must be labeled separately; do not call all three the same control. Keep the same drop-mask stream when comparing seeded trajectories, and state when hardware nondeterminism prevents exact replay. Measure both fixed-parameter effects and validation-retuned effects: they answer different questions.

## 8. Tables, diagrams, and result narratives

| Item / location | Contents and layout | What the adjacent paragraph must explain |
|---|---|---|
| **Figure 1 — I4 / method opening** | Three panels: a small graph with highlighted walks; its retained pair table labeled original edge/observed gap/assigned far weight; force arrows and an example update. Use solid attractive edges and distinct dashed repulsive pairs, with a legend. | Trace one pair from sampled path to h to signed force. Show missing/unobserved pairs explicitly. Make clear that 2D illustration is not the 128D evaluation. |
| **Algorithm 1 — M6–M7** | Compact pseudocode for augmentation and one epoch loop, with all scientific choices visible. | Explain which data are fixed and which change every epoch. Identify normalization and masks. |
| **Table 1 — E2** | Dataset rows; columns n, m, mean/upper-tail degree, components, clustering or structural family, task coverage. | Why these graphs test different mechanisms; no unsupported dataset counts. |
| **Table 2 — E5–E7** | Main quality table split into geometry, scalar decoder, and held-out LP panels. Dataset rows, selected method columns, or vice versa for readability. Mean ± SD; full values in appendix. | Lead with paired effect sizes and graph-dependent reversals. Bold only point-estimate best values with a clear convention; do not imply significance from boldface. |
| **Figure 2 — E8** | Panels: quality vs total runtime; peak host/GPU memory vs graph scale; stage-wise runtime bars for representative graphs. | Where preprocessing dominates, which methods form a quality–cost frontier, and why runs fail. Include compile cost for cold starts and separately label warm runs. |
| **Table 3 — E9–E10** | Focused ablations on three representative graph regimes, with realized M and geometry change from the main method. | Which component produces the gain, which control refutes an explanation, and whether cost changed. |
| **Figure 3 — E11–E12** | Panels: normalization/step-size diagnostic; controlled structural sweep with confidence bands. Select one density and one mixing/tree regime rather than a wall of heatmaps. | Distinguish a measured transition from an asserted universal threshold. Discuss failures alongside successes. |
| **Table A1 — related-work appendix** | Method, pair selection, attractive role, repulsive role, distance signal, degree normalization, input assumptions. | Establish the exact difference from rForce2Vec and prior hop-based forces. Verify each source before filling entries. |
| **Figures A1–A3 — appendix** | Gap-error distributions and retained-pair coverage; per-degree/per-hop geometry and prediction; checkpoint and sensitivity curves. | Show robustness, not selected favorable settings. |
| **Optional Figure A4** | Native 2D embeddings of the same small synthetic graphs with identical styling. | Illustrate failure modes; never use visual separation as quantitative proof or claim a 2D projection validates 128D geometry. |

This is a menu constrained by nine pages: if Table 2 becomes too wide, move the secondary learned-decoder panel to the appendix before shrinking fonts. Keep the direct geometry and closest-baseline comparison visible. All numeric figures must be made from measured data with standard plotting tools; no illustrative numbers in result plots.

**Results-writing template for every result paragraph:** question → controlled comparison → measured difference and uncertainty → interpretation → boundary. For example: “With pair support fixed, shuffling nonedge gaps changes [metric] by [effect, interval] on [graphs]. This [supports/does not support] the hypothesis that gap information contributes beyond pair selection. The effect [does/does not] persist on [failure regime].”

## 9. Discussion, appendices, and execution order

The discussion should answer reviewer questions rather than repeat tables:

- **Is this rForce2Vec with another name?** Point to the different role and weighting of nonlocal pairs, and show a direct baseline plus fixed-support controls. If that difference does not yield a robust benefit, narrow the contribution to an empirical characterization.
- **Does it preserve shortest paths?** Distinguish observed-gap bounds, direct rank/stress metrics, and decoder predictions. None alone proves low distortion for all pairs.
- **Why do results reverse across graphs?** Use controlled structural sweeps; treat high-dimensional Euclidean limitations, coverage, and degree normalization as hypotheses until isolated.
- **Is it scalable?** State supported graph sizes and hardware, M/n, padding, full-process peaks, and OOM cases. Avoid “constant memory” because Z itself grows with n·d.
- **Does it converge?** Report empirical trajectories and restricted analysis. Do not import a prior proof across changed sampling, asymmetry, normalization, and updates without checking assumptions.

**Appendices:** A, detailed related-work comparison; B, datasets/preprocessing/splits/search spaces and final configurations; C, proofs and counterexamples; D, full results and uncertainty; E, ablations and training curves; F, resource accounting and failed runs; G, reproducibility artifacts. Include an LLM-usage statement if applicable under the chosen venue's policy. ICLR 2026 explicitly required disclosure of significant LLM contributions to ideation or writing. [Author Guide](https://iclr.cc/Conferences/2026/AuthorGuide).

**Recommended execution order:**

1. Freeze the scientific specification and graph/split/evaluator contracts. Resolve walk_edges versus nbr_walk, far-pair semantics, degree division, isolates, and `fdhop` constants.
2. Establish Force2Vec/rForce2Vec and predecessor baselines; inspect REDUCE's full text before making a novelty claim.
3. Run the smallest decisive fixed-support gap and matched-decay controls on citation, social, and tree/road regimes. Stop expanding claims if these controls contradict the mechanism.
4. Freeze hyperparameters using validation only; execute multi-seed main geometry and true held-out LP comparisons.
5. Measure cold-start and end-to-end quality–cost behavior, including large-graph failures and preprocessing memory.
6. Run structural sweeps and longer checkpoints to explain observed boundaries.
7. Populate tables, write result paragraphs, then revise the abstract and introduction to match the supported findings.

**Release records:** committed code revision, dependency versions, graph/node-order hashes, split and evaluation-pair files, final configuration, augmentation seed, embedding seed, decoder seed, embedding hash, hardware, total runtime, peak memory, and failure status. Provide anonymized artifacts for a double-blind submission rather than inserting an identifying repository link into the anonymous manuscript.

## 10. Citation key and placement registry

Sources below were checked through primary publication pages, author manuscripts, or proceedings. A citation supports the stated prior-work claim; it does not establish Fodiwalk's performance. The list is a focused starting bibliography, not an exhaustive novelty clearance. Venue/year details should be imported from the publisher's BibTeX when assembling LaTeX.

| Key | Paper and primary source | Role in this manuscript |
|---|---|---|
| DW | Perozzi, Al-Rfou & Skiena (2014), [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652). | I1, R1, M2, E3: random-walk context and baseline. |
| SAGE | Hamilton, Ying & Leskovec (2017), [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216). | Abstract-writing study §2.3; optional R3 when discussing input assumptions and transductive scope. Not automatically a matched-input baseline. |
| GCN | Kipf & Welling (2017), [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907). | Abstract-writing study §2.3: compact approximation-to-cost argument. Cite in the manuscript only if the technical comparison is actually discussed. |
| GAT | Veličković et al. (2018), [Graph Attention Networks](https://arxiv.org/abs/1710.10903). | Abstract-writing study §2.3: explain the effect of weighting. Fodiwalk's gap weights are not learned attention. |
| N2V | Grover & Leskovec (2016), [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653). | I1–I2, R1, M2, E3: biased walks and tuned baseline. |
| LINE | Tang et al. (2015), [LINE: Large-scale Information Network Embedding](https://arxiv.org/abs/1503.03578). | R1, E3: proximity-preserving baseline. |
| NETMF | Qiu et al. (2018), [Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec](https://arxiv.org/abs/1710.02971). | I2, R1, E3: objective interpretation and spectral comparison. |
| LE | Belkin & Niyogi (2001), [Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering](https://papers.nips.cc/paper/1961-laplacian-eigenmaps-and-spectral-techniques-for-embedding-and-clustering). | R1, E3: spectral geometry baseline; distinguish this proceedings paper from the expanded 2003 article. |
| FR | Fruchterman & Reingold (1991), [Graph Drawing by Force-Directed Placement](https://doi.org/10.1002/spe.4380211102). | R2: historical force-layout lineage. |
| KK | Kamada & Kawai (1989), [An Algorithm for Drawing General Undirected Graphs](https://doi.org/10.1016/0020-0190(89)90102-6). | R2: graph-distance-aware layout lineage. |
| F2V | Rahman, Sujon & Azad (2020), [Force2Vec: Parallel Force-Directed Graph Embedding](https://arxiv.org/abs/2009.10035). | I2–I3, R2, E3, D1: essential closest competitor, including rForce2Vec. The [2022 extended article](https://link.springer.com/article/10.1007/s10115-021-01634-9) is additional implementation/context reading. |
| FD23 | Lotfalizadeh & Al Hasan (2023), [Force-Directed Graph Embedding with Hops Distance](https://arxiv.org/abs/2309.05865). | I3, R2, M5, E3: direct hop-force predecessor; cite in third person when anonymized. |
| KFD24 | Lotfalizadeh & Al Hasan (2024), [Kinematic-Based Force-Directed Graph Embedding](https://link.springer.com/chapter/10.1007/978-3-031-57515-0_11). | I3, R2, A2, E3: kinematic formulation and prior analysis; verify proof assumptions from full text before relying on them. |
| REDUCE | [Reducing Complexity of Force-Directed Graph Embedding](https://openreview.net/forum?id=1MjOlHwCE6), OpenReview record. | Mandatory predecessor to investigate for I3/R2. Record found, but full text was blocked by a browser challenge in this review; author list, final status, and technical overlap remain unverified. Do not label it an accepted paper without confirmation. |
| VERSE | Tsitsulin et al. (2018), [VERSE: Versatile Graph Embeddings from Similarity Measures](https://arxiv.org/abs/1803.04742). | I1, R3: explicit similarity preservation; optional baseline. |
| UMAP | McInnes, Healy & Melville (2018), [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426). | R3: manifold-layout context, with input differences explained. |
| LV | Tang et al. (2016), [Visualizing Large-scale and High-dimensional Data](https://arxiv.org/abs/1602.00370). | R3: scalable graph-based layout of high-dimensional data. |
| POINCARE | Nickel & Kiela (2017), [Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039). | R3, D3: hierarchical geometry and conditional baseline. |
| HEART | Li et al. (2023 manuscript), [Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking](https://arxiv.org/abs/2306.10453). | R4, E1, E7: common splits, baseline tuning, hard-negative evaluation. Import final publication metadata separately. |
| BIAS | Aiyappa et al. (2025), [Implicit Degree Bias in the Link Prediction Task](https://proceedings.mlr.press/v267/aiyappa25a.html). | I1, R4, E1, E7, D2: degree bias and degree-corrected evaluation. |
| OGB | Hu et al. (2020), [Open Graph Benchmark: Datasets for Machine Learning on Graphs](https://arxiv.org/abs/2005.00687). | E2: standardized benchmark provenance if an OGB dataset is used. |
| SELL | Kreutzer et al., [A Unified Sparse Matrix Data Format for Efficient General Sparse Matrix-Vector Multiply on Modern Processors with Wide SIMD Units](https://arxiv.org/abs/1307.6209), 2013 manuscript. | M7, A3: credit SELL-C-sigma rather than presenting the format as Fodiwalk's invention. |

**Citation discipline:** Use original-source citations after the sentence they support, not a large list at the end of an unrelated paragraph. The abstract and original method/results paragraphs generally need no citations. Dataset papers and software versions are added only after their exact instances are selected. This bibliography is suitable to the requested historical ICLR 2026 framing; a future submission also needs a fresh search for relevant work released after its September 2025 deadline.

## 11. Revision record

**Revision 2 — abstract and mechanism emphasis:** Replaced the initial abstract section with an expanded strategy, a primary-source study of ten abstracts, a ranked feature map, a polished current-evidence draft, a results-dependent final template, and sentence-level guidance. Updated I1/I2/E10, the evidence audit, RQ2, and the ablation matrix so the body tests the abstract's core explanation. Added same-support nonlocal-role controls and distinct permutation designs. Added SAGE/GCN/GAT as writing-study sources without automatically expanding the baseline suite. Flagged the difference between retaining all neighbor keys and forcing all neighbor gaps to h=1. Original paper allocation, remaining paragraph plans, experiment protocols, figures, and discussion structure are retained.
