# IDEA: embed the structure first, freeze it, then fill in the detail

Status: **a future project, not built.** Recorded 2026-08-29 so it can be
picked up later. Owner's idea; the measurements below are the first
feasibility probe.

## The idea

Embed the major nodes of a very large graph FIRST. Freeze them. They then
hold the STRUCTURE of the graph. Then embed the rest -- within each node
cluster or community -- against that fixed frame.

    stage 1   the major nodes            -> embed -> FREEZE
    stage 2   the next tier / a community-> embed against the frozen frame
    ...       and so on

**This is for very large graphs.** It is not aimed at cora or pubmed;
those are only where it gets tested.

## Why it might work

The whole graph relaxes at once today, and at a million nodes that is slow
to settle: the streaming campaign of 2026-08-28 found that NOTHING had
converged at 200 epochs -- every configuration was still gaining 0.12 to
0.22 hop R2 from 50 to 200 epochs
(`experiments/fodiwalk-streaming/FINDINGS.md` section 10). Epochs are a
bigger lever than the optimiser right now, and they are the cost this idea
attacks.

A frozen frame gives every later node a stable reference instead of a
moving one. The same reasoning is why multilevel graph layout works
(Walshaw's multilevel force-directed method; HARP, AAAI 2018; MILE;
GraphZoom).

**This is NOT a memory optimisation.** `Z` for every node is still needed
eventually, and the frozen nodes must stay readable because they are the
attractors. What it buys is initialisation and convergence: the same
quality in fewer epochs. State the pitch that way.

## The objection that does NOT apply, and why

A first probe measured the INDUCED SUBGRAPH OF `A` on the top 1% of nodes
by degree, on cora, and it looked fatal: 27 nodes, 13 edges, 17 connected
components, 13 of the nodes isolated, 0.2% of the graph's edges. cora is
also degree-DISASSORTATIVE (r = -0.066), so hubs attach to leaves and not
to each other -- which is typical of citation and social graphs.

**That measurement was of the wrong graph.** `fodiwalk` never embeds `A`.
It embeds the AUGMENTED pair set, and a walk from one hub reaches another
hub THROUGH the leaves between them. Measured on cora, walks over the full
graph, a pair kept when both ends are in the selected set:

| top k% | nodes | `A`: edges / comps / isolated | AUGMENTED: pairs / comps / isolated |
| --- | --- | --- | --- |
| 1% | 27 | 13 / 17 / 13 | **141 / 1 / 0** |
| 2% | 54 | 43 / 21 / 13 | 377 / 2 / 0 |
| 5% | 135 | 148 / 29 / 21 | **1,749 / 1 / 0** |
| 10% | 271 | 381 / 41 / 32 | 5,438 / 3 / 1 |
| 20% | 542 | 1,052 / 18 / 13 | 16,374 / 4 / 1 |

**The top 1% is ONE connected component in the augmented graph, with
nothing isolated.** The augmentation supplies exactly the connectivity the
raw subgraph lacks, which is what the owner said it would. Selection by
degree is therefore viable, and no k-core or connector machinery is needed
to make the frame connected.

(The residual components at 10% and 20% are not a defect of the method:
cora itself has 78 connected components.)

## The low-hanging fruit, to try FIRST

**Exclude the leaves, embed, freeze, then bring the leaves back.** One
split, two stages, nothing to tune. It answers the only question that
matters at this stage -- does freezing a frame and adding nodes against it
give an embedding as good as a flat run -- without any community
detection.

cora:

| set | nodes | edges | components | largest |
| --- | --- | --- | --- | --- |
| full | 2,708 | 5,278 | 78 | 2,485 |
| leaves removed ONCE | 2,223 (82.1%) | 4,850 | 21 | 2,131 |
| 2-core (leaves peeled to a fixpoint) | 2,136 (78.9%) | 4,768 | 16 | 2,051 |

cora has 485 degree-1 nodes, 17.9% of it. Removing them once creates 65
new leaves, so "remove once" and "peel to a fixpoint" are different sets;
either is a fair first split, but say which one was used.

**Measure the leaf fraction on a large graph before assuming this is a
small saving.** cora keeps 82% of its nodes, but a power-law graph has far
more leaves, and that is where the idea is aimed.

## Open questions, in the order they should be settled

1. **Hard freeze or soft?** The multilevel literature REFINES -- every
   node stays free at each level and the coarse layout is only an
   INITIALISATION. A hard freeze makes an early error permanent, and the
   first stage is the one built on the least information. A smaller
   learning rate for settled nodes, rather than zero, is the safer first
   try. Test both.
2. **How many stages?** 1% increments mean 100 stages, each paying its own
   walks and setup. Geometric steps (1, 2, 4, 8 ...) give about log(n)
   stages. The natural k-core levels are another option and there are few
   of them -- cora has 4.
3. **How small can stage 1 be?** 1% of cora is 27 nodes, and ANY 27 points
   embed exactly in 26 dimensions. At `dim = 64` such a stage is
   underdetermined and would freeze an arbitrary configuration. Require
   `nodes >> dim` before freezing anything.
4. **Selection rule.** Degree works (the table above). k-core is denser
   per node in the RAW graph, but that advantage may vanish once the
   augmentation is applied -- it was measured on `A`, not on the pair set.
   Re-measure before choosing.
5. **What does a later stage walk over?** The full graph, or only the
   unfrozen part? Walking the full graph keeps the frozen nodes as
   attractors, which is the point; walking only the new nodes would lose
   the frame.
6. **Which force does a frozen node exert?** It attracts and repels but
   does not move. The degree divisor `inv_deg` of a frozen row is then
   never used -- check that the kernel's row batching still holds when a
   row contributes force but takes none.

## How to judge it

Against a FLAT run at equal TOTAL epochs, on the same graph and seed, with
accuracy, f1, auc and hop R2 from `evaluator`. The claim to test is
"reaches the same quality in fewer epochs", so the x-axis is epochs (or
wall time) and not the final number alone.

Provenance: owner's idea, 2026-08-29. Feasibility probe run the same day;
the augmented-connectivity table above is the result that makes it worth
building. Related: `experiments/fodiwalk-streaming/FINDINGS.md`,
`dev-docs/CATALOG.md` section 29.
