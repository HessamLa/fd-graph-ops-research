# Expanding the Fodiwalk Benchmark: Recommended Comparison Methods

Given Fodiwalk's two-stage design—walk-based graph augmentation followed by kinematics-based force-directed embedding—I suggest expanding the experimental comparison beyond DeepWalk and node2vec.

The benchmark should investigate three questions: how Fodiwalk compares with other force-directed embedding methods, whether its embeddings preserve graph-distance information relative to methods based on different mathematical principles, and how its computational requirements compare with scalable graph embedding algorithms.

I recommend adding nine comparison configurations, including closely related force-directed methods, random-walk and matrix-factorization approaches, and graph-distance-oriented baselines.

Below are the proposed methods, followed by revised tables that can be inserted into our Markdown manuscript.

## 1. Force-directed graph embedding methods

These comparisons are particularly relevant because they help distinguish the contribution of Fodiwalk's walk-based augmentation and hop-dependent force function from force-directed embedding more generally.

Direct methodological comparison

## 1. Force2Vec

Rahman, Sujon, and Azad · ICDM 2020

Force2Vec applies force-directed graph layout principles to graph representation learning. Its implementation supports multiple attractive and repulsive force models and parallel execution.

![](https://www.google.com/s2/favicons?domain=https://doi.org\&sz=32)

Science

+1

It provides a direct comparison with Fodiwalk's kinematics-based force calculation.

The relevant experimental question is whether incorporating walk-derived hop coefficients into the force function changes local neighborhood preservation and global distance fidelity.

Use the published implementation: Force2Vec GitHub repository .

Direct methodological comparison

## 2. tForce2Vec

This is the t-distribution-based variant of Force2Vec. Its implementation combines a t-distribution force model with negative sampling.

![](https://www.google.com/s2/favicons?domain=https://pubmed.ncbi.nlm.nih.gov\&sz=32)

PubMed

Comparing it with Fodiwalk helps examine how the form of the force law affects the resulting embedding geometry.

I would treat tForce2Vec as a separate experimental configuration while identifying it as part of the Force2Vec family.

Walk-based force-directed comparison

## 3. rForce2Vec

rForce2Vec is a variant of Force2Vec that combines a sigmoid-based force model with semi-random walks. Its implementation exposes this configuration separately from the t-distribution and standard sigmoid variants.

![](https://www.google.com/s2/favicons?domain=https://pubmed.ncbi.nlm.nih.gov\&sz=32)

PubMed

It is especially relevant because both rForce2Vec and Fodiwalk use graph walks in the embedding process.

The comparison can investigate the effect of using walks to obtain interaction information and the effect of how the respective force functions consume that information.

It is important not to assume that the two methods use random walks in the same way.

For reproducibility, the published Force2Vec implementation exposes tForce2Vec as option 5 and rForce2Vec as option 7. Its option 1 is the quadratic all-pairs Force2Vec configuration.

![](https://www.google.com/s2/favicons?domain=https://pubmed.ncbi.nlm.nih.gov\&sz=32)

PubMed

For the main benchmark, I would include a published scalable Force2Vec configuration rather than rely solely on the quadratic variant, which serves a different computational regime.

If your earlier force-directed graph embedding method is available as a separately executable implementation, I would also include it as an internal reference baseline. This would allow us to quantify the effect of replacing its original graph-interaction construction with Fodiwalk's walk-based augmentation.

## 2. Other graph embedding methods

The following methods provide comparisons based on different embedding mechanisms.

## 4. LINE

Tang et al. · WWW 2015

LINE learns representations by modeling first-order and second-order network proximity. It is designed for large information networks and supports weighted as well as directed graphs.

![](https://www.google.com/s2/favicons?domain=https://newtraell.cs.uchicago.edu\&sz=32)

newtraell.cs.uchicago.edu

+1

It provides an edge-proximity-oriented comparison with Fodiwalk's combination of local attraction and nonlocal repulsion.

Implementation 

## 5. ProNE

Zhang et al. · IJCAI 2019

ProNE combines sparse matrix factorization with spectral propagation to construct network embeddings. Its implementation provides both Python and optimized C++ versions.

![](https://www.google.com/s2/favicons?domain=https://www.frontiersin.org\&sz=32)

Modern Hopfield Networks for graph embedding

+1

It offers a comparison between Fodiwalk's iterative force-directed construction and a scalable factorization-based approach.

Implementation 

## 6. NetSMF

Qiu et al. · The Web Conference 2019

NetSMF uses sparse matrix factorization to approximate network relationships that would otherwise involve expensive dense representations. Its formulation builds on the matrix-factorization interpretation of random-walk embeddings.

![](https://www.google.com/s2/favicons?domain=https://academic.oup.com\&sz=32)

Oxford Academic

+1

This comparison can investigate how Fodiwalk's sampled, force-based representation differs from a sparse factorization of network proximity.

Implementation 

## 7. Laplacian Eigenmaps

Belkin and Niyogi · 2003

Laplacian Eigenmaps constructs low-dimensional representations from the spectral properties of a graph Laplacian. Its mathematical formulation is directly relevant to the degree-normalized attraction component of Fodiwalk.

![](https://www.google.com/s2/favicons?domain=https://researchers.mq.edu.au\&sz=32)

Macquarie University

This baseline provides a way to compare the geometry obtained from a Laplacian-based representation with that obtained by combining neighborhood smoothing and repulsive forces.

A sparse eigensolver can be used to implement this baseline for manageable graph sizes.

## 8. Landmark shortest-path MDS

Graph-distance-oriented reference baseline

Classical multidimensional scaling can construct embeddings from pairwise distances. Related geodesic-distance approaches, such as Isomap, use shortest-path distances to represent longer-range geometry.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

For our benchmark, I propose a landmark-based approximation: select a subset of nodes, compute shortest-path distances from those landmarks, and use the resulting distance information to construct a low-dimensional representation.

This is an explicitly specified experimental baseline, not a claim that one particular landmark implementation is the canonical Isomap algorithm.

It would provide a useful comparison for Fodiwalk's graph-distance preservation, especially because the landmark baseline has direct access to shortest-path information that Fodiwalk estimates through walks.

## 9. GOSH

Akyildiz, Aljundi, and Kaya · 2020

GOSH uses graph coarsening and GPU-based embedding computation to address large-scale graph representation learning.

![](https://www.google.com/s2/favicons?domain=https://anvithpothula.github.io\&sz=32)

Graph Neural Networks

+1

This method is particularly relevant to the scalability experiments. It gives us a comparison against a graph embedding system designed to operate on large graphs under hardware constraints.

Implementation 

### Additional candidate: GraRep

Optional comparison for smaller graphs

GraRep learns representations from higher-order graph relationships and explicitly incorporates information beyond direct adjacency.

![](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

I would include GraRep in the smaller-graph experiments if computational resources permit. Its use of higher-order structural information provides a useful comparison with Fodiwalk's walk-derived nonlocal interactions.

For very large graphs, it may be more practical to use the sparse factorization approach represented by NetSMF.

# 3. Revised tables for the Markdown manuscript

I would make the following additions to our experimental section.

The tables below preserve the existing Fodiwalk, DeepWalk, and node2vec rows and add the proposed comparison methods.

All unmeasured values are marked `TBD`. No numerical performance values have been invented.

## Table 3. Experimental configuration of embedding methods

The previous Table 3 described only Fodiwalk's parameters. I suggest retaining that table as Table 3a and adding the following comparison-method table as Table 3b. |
Method | Main parameters to report | Embedding dimension | Implementation |
| --- | --- | --- | --- |
| Fodiwalk | Walk parameters, force coefficients, learning rate, epochs, update strategy | 128 | Fodiwalk |
| DeepWalk | Walk count, length, window, epochs | 128 | Existing baseline |
| node2vec | Walk count, length, window, p,qp,qp,q, epochs | 128 | Existing baseline |
| Force2Vec | Force model, iterations, batch size, negative samples, learning rate | 128 | Official implementation |
| tForce2Vec | Force model, iterations, batch size, negative samples, learning rate | 128 | Force2Vec option 5 |
| rForce2Vec | Walk configuration, iterations, batch size, negative samples, learning rate | 128 | Force2Vec option 7 |
| LINE | Proximity order, samples, learning rate, epochs | 128 | LINE implementation |
| ProNE | Factorization and spectral-propagation parameters | 128 | ProNE implementation |
| NetSMF | Sparsification parameters, factorization rank, approximation settings | 128 | NetSMF implementation |
| Laplacian Eigenmaps | Laplacian type, eigensolver, eigenvector selection | 128 | Sparse spectral implementation |
| Landmark shortest-path MDS | Landmark count, landmark selection, projection method | 128 | Experimental implementation |
| GOSH | Coarsening parameters, negative samples, epochs, learning rate | 128 | GOSH implementation |
| GraRep (optional) | Maximum transition order, dimension allocation, factorization parameters | 128 | GraRep implementation | The 128-dimensional value is the proposed common setting for the main benchmark, not a claim that all these methods have already been run at that dimension.

### Instructions for completing Table 3b

Use the published implementations where practical. For each method, save the precise configuration, source-code version or commit, embedding seed, runtime environment, and any changes made to the original implementation.

Use the same embedding dimension for the main comparison. For methods that concatenate several representations, allocate dimensions so that the final output dimension is 128 rather than assigning 128 dimensions to every intermediate representation.

For Force2Vec, record the exact implementation option because different force models should not be treated as equivalent. For Laplacian Eigenmaps, document the handling of zero eigenvalues and disconnected graph components. For landmark MDS, specify whether the selected shortest-path distances are exact or approximate.

## Table 5. Main geometric and graph reconstruction results

This is the revised version of the main evaluation table.

I suggest creating one instance of this table for each dataset rather than combining all datasets into one enormous table.

Table 5 — Template

## Embedding quality on \(DATASET\)

Dimension: 128 · Report mean ± standard deviation across embedding seeds

<table class="_6IUVGW_Table" data-d-column-sizing="auto" data-d-dividers="" style="table-layout: auto;"><tbody><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-has-width="" data-d-valign="start" style="width: 29%;"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-weight="medium">Method</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-weight="medium">Spearman ρ</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-weight="medium">Hop R²</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-weight="medium">Recall@10</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" data-d-weight="medium">LP AUC</p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Fodiwalk</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">DeepWalk</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">node2vec</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Force2Vec</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">tForce2Vec</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">rForce2Vec</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">LINE</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">ProNE</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">NetSMF</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Laplacian Eigenmaps</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">Landmark shortest-path MDS</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">GOSH</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start">GraRep (optional)</td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td><td data-d-component="table-cell" data-d-valign="start"><p class="w6asjq_TextBase _85PZeG_Text" data-d-component="text" style="color: var(--color-text-secondary);">TBD</p></td></tr></tbody></table>

Unmeasured results are left blank until the corresponding embedding and evaluation runs are completed.

Generate separate versions for Cora, PubMed, WordNet, and every additional graph included in the final experimental campaign.

The existing Fodiwalk, DeepWalk, and node2vec measurements can be used to populate their corresponding rows after verifying that they were produced under the same evaluation protocol.

### Instructions for generating Table 5

Use identical graph data and node indexing for every method. Generate embeddings at the same dimension, preferably using the same number of embedding seeds.

Evaluate the embeddings using the unified evaluator already developed in the repository.

For graph-distance measurements, use the same sampled node pairs and true shortest-path targets for every method. Compute Spearman correlation and hop-distance regression using an identical evaluation procedure.

Report both local and longer-range structural measurements, since a method that places direct neighbors close together does not necessarily preserve longer graph distances.

Keep edge reconstruction separate from held-out link prediction. If the embedding was trained using every original graph edge, do not present the resulting edge-classification score as a strictly held-out link-prediction result.

For every unavailable result, report the reason: unsupported graph size, memory exhaustion, timeout, incompatible input requirements, or an incomplete run.

## Table 6. Controlled Fodiwalk ablation

I would leave the main purpose of Table 6 unchanged.

This table should evaluate the contribution of Fodiwalk's own mechanisms, rather than compare unrelated algorithms.

The current ablation design—observed walk gaps, constant nonlocal coefficients, shuffled coefficients, and exact shortest-path coefficients—remains appropriate.

The expanded method comparison does not replace these controlled experiments.

They answer different questions: the comparison measures the behavior of different embedding approaches, while the ablation investigates the role of specific Fodiwalk components.

## New Table 7. Runtime and memory comparison

Because Fodiwalk is intended to avoid exhaustive pairwise interactions and all-pairs shortest-path computation, I would add a separate table for computational performance.

This table should include the entire pipeline, not just the embedding iteration loop. |
Method | Preprocessing / augmentation (s) | Embedding (s) | Total time (s) | Peak RAM (GB) | Peak GPU memory (GB) |
| --- | --- | --- | --- | --- | --- |
| Fodiwalk | TBD | TBD | TBD | TBD | TBD |
| DeepWalk | TBD | TBD | TBD | TBD | TBD |
| node2vec | TBD | TBD | TBD | TBD | TBD |
| Force2Vec | TBD | TBD | TBD | TBD | TBD |
| tForce2Vec | TBD | TBD | TBD | TBD | TBD |
| rForce2Vec | TBD | TBD | TBD | TBD | TBD |
| LINE | TBD | TBD | TBD | TBD | TBD |
| ProNE | TBD | TBD | TBD | TBD | TBD |
| NetSMF | TBD | TBD | TBD | TBD | TBD |
| Laplacian Eigenmaps | TBD | TBD | TBD | TBD | TBD |
| Landmark shortest-path MDS | TBD | TBD | TBD | TBD | TBD |
| GOSH | TBD | TBD | TBD | TBD | TBD |
| GraRep (optional) | TBD | TBD | TBD | TBD | TBD | ### Instructions for generating Table 7

Use the same machine where possible. Record the CPU, GPU, available RAM, thread count, and software environment.

For every method, measure total wall-clock time from the beginning of graph-specific preprocessing until the final embedding has been generated.

For Fodiwalk, report augmentation and embedding times separately. For DeepWalk and node2vec, include walk generation and representation training. For factorization and spectral methods, include matrix construction and factorization or eigensolver time.

Report peak resident system memory and peak GPU memory separately.

For CPU-only methods, enter `N/A` in the GPU-memory column rather than zero.

A method that cannot complete within the available memory or time budget should be recorded as incomplete, not omitted from the comparison.

I also suggest repeating this table for several graph sizes to show how computational requirements change with the number of nodes and edges.

# 4. A dedicated graph-distance preservation experiment

I would add one further experiment because graph-distance fidelity is central to Fodiwalk's scientific motivation.

The main results table summarizes the overall relationship between embedding distance and graph distance. However, it does not show how that relationship changes as nodes become more separated in the input graph.

For example, an embedding might represent nodes at distances 2–4 accurately while losing the distinction between nodes at distances 8–12.

A dedicated distance-stratified experiment would reveal this behavior.

## New Table 8. Distance preservation by graph-distance range |
Method | 2–3 hops | 4–5 hops | 6–8 hops | 9–12 hops | 13+ hops |
| --- | --- | --- | --- | --- | --- |
| Fodiwalk | TBD | TBD | TBD | TBD | TBD |
| DeepWalk | TBD | TBD | TBD | TBD | TBD |
| node2vec | TBD | TBD | TBD | TBD | TBD |
| Force2Vec | TBD | TBD | TBD | TBD | TBD |
| tForce2Vec | TBD | TBD | TBD | TBD | TBD |
| rForce2Vec | TBD | TBD | TBD | TBD | TBD |
| LINE | TBD | TBD | TBD | TBD | TBD |
| ProNE | TBD | TBD | TBD | TBD | TBD |
| NetSMF | TBD | TBD | TBD | TBD | TBD |
| Laplacian Eigenmaps | TBD | TBD | TBD | TBD | TBD |
| Landmark shortest-path MDS | TBD | TBD | TBD | TBD | TBD |
| GOSH | TBD | TBD | TBD | TBD | TBD |
| GraRep (optional) | TBD | TBD | TBD | TBD | TBD | ### How to generate this table

For every dataset, sample reachable node pairs stratified by their true shortest-path distance. Use the same sampled pairs for every method.

The columns above are provisional distance bins. Adapt them to the graph's actual diameter and path-length distribution.

Because each bin contains a narrow range of graph distances, do not use within-bin Spearman correlation as the principal measure. Instead, report a scale-aligned distance error or another clearly defined distance-distortion metric.

One possible procedure is to fit a single distance-calibration function on a common training-pair set for each embedding, then evaluate the absolute prediction error within each distance bin using a disjoint test-pair set.

Fit the calibration once per embedding and apply it across all bins. Do not independently rescale each bin, since doing so could conceal differences in the global geometry.

Report the number of sampled pairs in every bin.

The corresponding figure should plot the true graph distance against the calibrated embedding distance, with the error distribution or confidence interval shown for each distance range.

This would allow us to investigate whether Fodiwalk captures longer-range graph separation or primarily preserves relatively local structural relationships.

# 5. Recommended experimental scope

Running all nine additional configurations on every graph may require substantial computation. I would organize the experiments into three groups according to the scientific question.

Main geometric comparison

Fodiwalk, DeepWalk, node2vec, Force2Vec, tForce2Vec, rForce2Vec, LINE, ProNE, NetSMF, and Laplacian Eigenmaps.

These methods provide coverage of force-directed, random-walk, proximity-based, matrix-factorization, and spectral representations.

Use Cora, PubMed, and WordNet initially, with identical embedding dimensions and evaluation protocols.

Graph-distance reference experiment

Fodiwalk and landmark shortest-path MDS, alongside selected methods from the main comparison.

This experiment examines the geometry produced by Fodiwalk against a representation constructed using explicit graph-distance information.

Run it first on smaller graphs, where exact shortest-path computations and controlled evaluations are practical.

Large-graph scalability comparison

Fodiwalk, DeepWalk, node2vec, Force2Vec, LINE, ProNE, NetSMF, and GOSH.

Use larger graphs to measure total runtime, preprocessing cost, peak memory, embedding quality, and whether each method can complete within the specified hardware budget.

Include the remaining methods only where their implementations support the graph size and available resources.

## Final recommendation

The most important addition is the Force2Vec family, particularly rForce2Vec, because these methods provide direct comparisons with the force-directed and walk-based aspects of Fodiwalk.

Laplacian Eigenmaps provides a useful comparison for the degree-normalized attraction mechanism, while landmark shortest-path MDS provides a distance-oriented reference for the geometric fidelity experiments.

ProNE, NetSMF, and GOSH broaden the evaluation to include scalable embedding methods based on different computational principles.

I would use the expanded tables to report results across these distinct methodological families, while keeping Fodiwalk's own force function fixed throughout the main experiments. This will make it possible to evaluate the proposed framework without turning the paper into a comparison among different Fodiwalk force functions.
