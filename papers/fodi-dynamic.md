Q:
 
Using fodiwalk for dynamic graphs. Find the potentials. What methods should it be compared against? What datasets can be used? What are the performance criteria metrics?
 
Consider the knowledge base and the repo.
 
---
 
A:
 
I read the repo again. There is no dynamic-graph or temporal code in it. The words "dynamic", "temporal" and "incremental" appear only in refactor notes. So this is new work, and the design is open.
 
## 1. The potentials
 
Four properties of fodiwalk fit a dynamic graph. Three of them already exist in the code.
 
**The method has no loss function.** The embedding is the equilibrium of a force field. When the graph changes, the force field changes. The old layout stays a legal start state for the new field. A method with an objective must define a new objective and train again. Fodiwalk only continues to relax. This is the core argument for the paper.
 
**The engine already resumes.** `ForceDirected.embed()` keeps `end_epoch` and advances it on each call. `Z` persists on the model, or a caller passes it in. A warm start needs no new engine. The hook is there.
 
**`D` is row-local.** In `nbr_walk`, row `u` holds only what the walks from `u` found. `D` is not symmetric, and the force law reads one row. So a changed edge touches a bounded set of rows. You rebuild those rows only. A method with a global objective has no such property.
 
**The walk budget is per node and constant.** The cost of one row is `walks * walk_len`. It does not follow the hub degree. In a dynamic graph, the hubs are the nodes that change most often. The k-ball method of `fodined` fails there, and a full node2vec re-walk is also costly. This is the clearest place to win.
 
There is a fifth potential, and it is the strongest one for this project. **Incremental operation can remove the G3 memory failure.** Gate G3 fails at 1,134,890 nodes, at 4374 MB against a 2600 MB limit. The augmentation stage holds 3583 MB of that peak. An incremental update never builds the full augmentation at one time. Its working set is the affected rows only. So the dynamic setting turns the open weakness of the project into the claim of the paper.
 
## 2. What blocks the potentials
 
Say these in the paper. They are real.
 
| Problem | Why it is hard |
| --- | --- |
| Deletion | A walk record keeps no provenance. You cannot subtract the pairs that a removed edge produced. You must re-walk the affected rows, or accept stale pairs and decay them. |
| New nodes | A new node has no position. Random init in a dense region gives a long relaxation. Init at the mean of the known neighbours is the obvious first rule to test. |
| Global structures | The far-pair sampler uses `deg^alpha`, and the landmarks are global. Both go stale when the degree distribution moves. They need a refresh schedule. |
| Restart policy | Small updates accumulate error. You must decide when to rebuild from zero. This is a result on its own, not a detail. |
 
## 3. Methods to compare against
 
Group A and B are required. Group C protects the paper.
 
**A. Cost and quality anchors**
 
- Fodiwalk, full recompute at each snapshot. This is the essential ablation. It isolates the value of the warm start.
- node2vec and DeepWalk, retrained at each snapshot. Both are already in `experiments/other-ge/` and `experiments/large-graph-node2vec/`, so the cost to add them is low.
- EdgeBank. It only memorizes past edges. It is now a standard control, and many methods do not beat it.
 
**B. The direct family: incremental and walk-based**
 
- CTDNE. It uses time-respecting walks. It is both a baseline and a design source, because a time filter in `walks.py` is a small change.
- DynGEM. It also warm starts from the last state, so it tests the same idea with a different engine.
- dyngraph2vec (dynAE, dynRNN, dynAERNN).
- DynamicTriad.
- TIMERS and DHPE. TIMERS decides when to restart, which is the same question as the restart policy above.
- EvoNRL, or another walk-maintenance method, if a working implementation exists.
 
**C. Temporal graph networks, for credibility**
 
- TGN, TGAT, JODIE, DyRep. Add CAW or DyGFormer if space allows.
- Take care with the protocol. These methods train a decoder. Fodiwalk is unsupervised and structural. Freeze the embedding and use one downstream protocol for all methods. This is the same rule that made `evaluator` freeze its six protocols.
 
## 4. Datasets
 
Start at tier 1. Do not skip tier 3, because tier 3 carries the claim.
 
**Tier 1, small, for fast iteration**
 
- HEP-Th and HEP-Ph citation growth. Insertion only, so deletion does not confuse the first result.
- CollegeMsg (UCI message).
- Enron email.
- AS-733 and AS-Oregon. These have insertions and deletions.
- Bitcoin OTC and Bitcoin Alpha.
 
**Tier 2, the standard suite, for comparison against group C**
 
- The TGB sets: tgbl-wiki, tgbl-review, tgbl-coin, tgbl-comment. TGB fixes the split and the negative sampler. That solves the same problem that the six frozen protocols solve, so it fits how this project already works.
- The DyGLib set: Wikipedia, Reddit, MOOC, LastFM. Note that three of these are bipartite. Test the degree normalization on one of them first.
 
**Tier 3, scale, to attack G3**
 
- sx-stackoverflow, about 2.6M nodes.
- wiki-talk temporal, about 1.1M nodes. It is close in size to com_youtube, so the current record gives a reference point.
- Reddit hyperlinks.
 
**A synthetic control.** Build a stochastic block model. Merge two communities at a known time, then split them again. This gives a ground truth for "did the layout track the change". No real dataset gives that. It is cheap and it makes a strong figure.
 
## 5. Performance criteria
 
Four families. Report cost and quality together, never alone.
 
**Quality at each snapshot**
 
- Link prediction AUC and AP, under one frozen protocol.
- Spearman ρ and Kendall τ-b between the embedded distance and the hop distance. Use them, not hop R2. A model-free score is the only score that stays comparable across a time series, because a per-snapshot model does not stay constant.
 
**Temporal task quality**
 
- Future link prediction. Train to snapshot `t`, predict edges at `t+1`. Report AP and MRR.
- Report random negatives and historical negatives apart. Historical negatives separate structure from memory.
- Report new edges and recurring edges apart. Fodiwalk should be strong on new edges and weak on recurring edges. State this before a reviewer does.
- Use an inductive split, where one or both nodes are new. This tests the new-node rule.
 
**Stability**
 
- Drift: the mean of `||Z_t[u] - Z_(t-1)[u]||` over the nodes that did not change.
- The same value after an optimal Procrustes rotation. The difference between the two numbers is the rotation that a retrained baseline needs, and that fodiwalk does not need.
- Neighbour stability: the Jaccard index of the top-k nearest neighbour set, between `t-1` and `t`.
- A frozen embedding scores perfectly on all three and fails on quality. So plot stability against AP. Do not report stability alone.
 
**Cost**
 
- Update time for each event and for each snapshot. Split it into walk time, `D` build time and relax time.
- Peak RSS of the update, not of the full run. This is the G3 number.
- The count of rows touched for each update. Plot it against the degree of the two endpoints. The expected figure is a flat line for fodiwalk and a rising line for the k-ball method. This is the main figure of the paper.
- Epochs to reconverge, at a fixed threshold. `ForceDirected.Th(dZ)` already gives this.
- Speedup against full recompute at equal quality. This is the headline number.
 
**Staleness**
 
- Update for `k` steps with no restart, and measure the fall in AP. The curve gives the restart schedule.
 
## Suggested first step
 
Use HEP-Th, insertions only, in 10 snapshots. Compare three runs: full recompute of fodiwalk, warm-started fodiwalk with re-walk of the affected rows only, and node2vec retrained each time. Measure AP, ρ, rows touched, and peak RSS. That single table tests the whole idea, and it needs no new force law.
 
Do you want me to write this as a research plan document, or to start with the design of the incremental `D` update?
 
---
 