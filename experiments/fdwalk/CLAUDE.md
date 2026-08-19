## Experiments

For all experiments, also show the following data and metrics whenever available.
- Maximum memory utilization
- Runtime (per stage or per phase)

The following are the metrics that we are interested in:
- Link predictio (Accuracy/F1 score/AUC)
- Memory usage of various stages and phases
- Duration of various stages and phases

Use small-scale datasets for quick revision and verification. Use very large datasets for actual performance report. 

For a more elaborate experiment, do the node embeddings with the following configurations:
- Various embedding methods: Node2vec, various fodined models (the best aug polics(s) and/or schedule and/or functions etc)
- Various dimensions (128, 16, 32, 64, in this order)
- Various very large scale datasets

Run an elaborate experiment only if a model has passed all gates and performs well on one very large-scale graph.

Maintain a list of TODOs in the file [TODOs.md](TODOs.md)

**IMPORTANT**:
For all the subsequent experiments, try the following gradient optimizer functions:
- No optimizer: ``Z = Z + lr · dZ``
- Velocity: ``v = eta · dZ + (1-eta) · v0`` and  ``Z = Z + lr · v`` and ``v0 = v``
- SGD
- Adam
- Stochastic Quasi-Newton (SQN)

## Documentation

- Keep a timestamped log under ./log/ of the communication, your thoughts and actions. Rotate to a new timestamped file when one grows large.
- Never delete from any document. State the reason and rationale, then append. If something is removed, that removal must be recorded in the log.
- Timestamp all documentation and log entries.
- A gain of 1.5% or less is noise.
- Update the [FINDINGS.md](FINDINGS.md) with each new and meaningful finding. These can be the effect of new policies or methods on the results, when the effect is considerable or enhances/worsens a metric of interest.
- Catalog: Keep a catalog of named entities — augmentation policy, force function, gradient optimizer, gradient update schedule, normalization method, and so on. For each entity: an elaborate description of what it is and how it works, plus a code snippet where applicable, plus provenance — a link to the corresponding code block or file. In the documents, keep a catalog of names of entities such as method, policy, schedule, etc. For each entity, provide an elaborate description and what it is and/or how it works, along with a code snippet if applicable. These entities can be things suchas augmentation policy, gradient optimizer function, gradient update schedule (row-wise, cell-wise, etc), force function, normalization method, etc. Provenance: Include links to the corresponding code  block or file. 

**IMPORTANT — parallel runs, by the size of the graph.**
Resources here are limited, and the two metrics this project cares about
most are runtime and peak memory. A concurrent run makes both of them
meaningless for BOTH runs. The rule therefore depends on the size:

- **Small graphs: parallel is allowed.** Cora (2,708) and PubMed (19,717)
  serve mostly as smoke tests — they verify that a code path works and that
  a quality metric moves in the expected direction. Their time and memory
  are dominated by a fixed ~1 GB of JAX and Python and are not a
  performance report in any case.
- **Medium and large graphs: NEVER in parallel, with anything.** One run at
  a time, alone on the machine, including alone against a smoke test. This
  is where the performance report comes from and where a shared machine
  destroys it.
- The boundary used in practice, unless told otherwise: **small is up to
  about 50,000 nodes.** Thus Cora and PubMed are small; the 150,000-node
  BFS ball, roadNet-CA at 200,000, and com_youtube at 1,134,890 are not.

A quality metric (AUC, F1, accuracy, hop R2, final `||dZ||`) is
deterministic given the seed and is NEVER affected by a shared machine.
Only time and memory are. Thus a contaminated run keeps its quality numbers
and loses its time and memory: mark them, and re-measure alone.

Report maximum memory use and runtime per stage or phase for every experiment.

