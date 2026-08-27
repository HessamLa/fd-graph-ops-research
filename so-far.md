Wed Aug 26 10:27:01 PM PDT 2026

The progress so far is the following

This research is a walk-based force-directed graph embedding. We don't use k-ball or shell sets here. We walk graph walk methods, such as those employed in DeepWalk and node2vec, to find the set of candid nodes for each target node.

NOTE: fodiwalk is same as fdwalk. From now on we use fodiwalk (*Fo*rce *Di*rected *Walk*) as a more descriptive, and somewhat unique name.

## Graph augmentation:

After loading the graph, we perform walks, and for each node u, create a set of other nodes W(u): the set of nodes visited on the walk windows of u.
[at the bottom of this file list all the walk method and their parameters as experimented under experiments/fdwalk]

## Graph embedding:

We have separated the graph embedding into a package in ``forcedirected/``. All sessions must import this package, instead of rewriting the whole class.

The, dz_u (gradient of node embedding of u) will be calculated with respect to the node embedding of others nodes v ∈ N(u) ∪ W(u)

To calculate dz_u, we need to define (1) force function, (2) optimizer, (3) parameters such as learning rate (constant vs. diminishing) or 

1. Force functions are in close tandem with walk methods and in general with graph augmentation policies. All the parameters and values that are consumed in a force function are generated during graph augmentation, such as the W(.) sets and walk-related values per each node in node in a W(.) set.

2. Optimizer functions determine how quick an embedding converges. Some optimizers work better with some force functions.
[at the bottom of this file please list the set of all optimizer functions explored in experiments/fdwalk]

3. Some parameters control how the embedding algorithm functions and converges. These parametes are of varying nature and use. For example, a constant or decaying learning rate can affect the rate and quality of convergence. Another example is the drop strategy, which determines which rows of columns of the gradient calculation to avoid. One drop strategies work row-wise, skipping a random set of nodes per epoch. Another strategy zeros out random (row,column) values in the gradient block. Another strategy which is **pending an experiment** skips a random set of columns during each epoch.

Another integral part of graph embedding is the sell-C-σ method, used for memory optimization. This method is a core functionality of the current ForceDirected implementation and, therefore, is placed under ``forcedirected/`` package.

## Evaluation:

To have a unified evaluation, we are using the ``evaluator/`` package. All agents must import this module for evaluation tasks. If an update is required, or if there is bug report, it must be relayed to the ``evaluator`` agent in ``.claude/agents/evaluator.md``.


## Experiments summary
[Write a very short summary of the experiment results found under experiments/fdwalk/. No verbosity. Very short sentences. Let the tables and results talk.]
