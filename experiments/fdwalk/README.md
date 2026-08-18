# fdwalk

Created: 2026-08-15. Last updated: 2026-08-17 22:55 PDT.

Random walks make the pairs of the augmented graph, and a property of the
walk makes the weight of each pair.

## Why

The augmentation of fodined decides how large the graph can be. The two
policies of today both let the GRAPH decide the size of `D`. A hub with
28,754 neighbours puts 2.5 G pairs in the 2-hop ball of com_youtube, and
53 G at k=3. No machine here holds that.

A random walk has a budget: `r` walks of `l` steps from each node. The
budget does not change when a hub appears. node2vec embeds the same 1.13M
node graph in 12.7 minutes on this machine, and that is the evidence.

fdwalk keeps the force law of fodined and it replaces only the augmentation.

## The documents

| File | Content |
| --- | --- |
| `PLAN.md` | the design, the 7 hypotheses, the variants, the gates, and the rules to stop |
| `CATALOG.md` | the catalog of the named entities: force functions, augmentation policies, weights, optimizers, layout, protocol. Added 2026-08-17. |
| `log/` | the timestamped work log, in four files: `2026-08-15T231648`, `2026-08-16T0020`, `2026-08-17T043000` (written after the fact, to close a gap), `2026-08-17T225332`. The file names sort chronologically. |
| `FINDINGS.md` | the results. It starts with the baselines and the probe that started the work |
| `REFERENCES.md` | the citations, with a note on what each one gives us |

Read PLAN.md first. The criteria in it are fixed before a run, and
FINDINGS.md records what happened against them.

## The state

As of 2026-08-16 23:25 PDT.

| Gate | Graph | Result |
| --- | --- | --- |
| G1 | Cora, 2,708 | **passed**. 30 runs. `min_gap` wins, `mean_gap` and `pmi` collapse. |
| G2 | PubMed, 19,717 | **passed**. 9 runs. R2 0.549 against 0.080 for node2vec. |
| G3 | com_youtube, 1,134,890 | **failed**. 3 attempts, all inside the augmentation. |
| Axis C | the update rules | not started. It opens on a variant that passes G3. |

The best variant is `--pairs walk --weight min_gap`. It gives an AUC of
0.9970 on Cora and 0.9953 on PubMed, and it reads the hop distance 2 to 7
times better than node2vec on the same walks.

The failure of G3 is a defect of the code of this experiment, and the three
defects are named in FINDINGS.md. One constant (`prune_factor`) stands
between the third attempt and a fourth. The rule of PLAN.md stops the gate
before that constant changes, thus a fourth attempt is a new decision.

Read FINDINGS.md for the numbers, and PLAN.md for the criteria that they
are measured against.

## The order of the work

1. Cora (2,708 nodes), gate G1. It is fast, thus every variant starts here.
2. PubMed (19,717 nodes), gate G2. Only a variant that passed G1.
3. com_youtube (1,134,890 nodes), gate G3. Only a variant that passed G2.
   This is the graph that no version of fodined has finished.

roadNet-CA (1,965,206 nodes) is the contrast graph. It has no hubs, thus the
old policy works on it, and a good result there proves nothing about the
scale.

## To run one experiment

Not possible yet. When the harness exists, the form will be:

```
.venv/bin/python experiments/fdwalk/bench_fdwalk.py --graph cora \
    --pairs walk --weight min_gap --optim plain --seed 42
```

Every run writes one log, and the log is never edited by hand.
