# Off-branch: a constant repulsion, and a fixed pair budget

Created: 2026-08-16 01:50 PDT. Last updated: 2026-08-16 23:25 PDT.
Runs made 2026-08-16 between 01:10 and 01:50 PDT. One seed (42), because this is an off-branch check and
not a gate.

## What was asked

Two changes at the same time.

**The force law.**

```
h == 1:  F = Fa + Fr,  Fr = -k3 * h * exp(-k4 * x)     (unchanged)
h  > 1:  F = Fr,       Fr = -k3 * h                    (no decay with x)
```

**The augmentation.** Keep every original edge at `h = 1`, and add exactly
`n * log10(n)` pairs with `h >= 2`, at 50% `h = 2`, 25% `h = 3`, and 25%
`h >= 4`. Thus `|D| = |E| + |V| * log10(|V|)`.

Run the ball on Cora and PubMed, and the walk on Cora, PubMed, and
com_youtube (1,134,890 nodes).

## One decision that the specification forces

The size formula counts every pair, thus it leaves NO room for the random
far pairs that the old policy adds at the weight 100. **This branch has no
far pairs and no sentinel.** Every `h` in `D` is a real hop distance or a
real walk gap.

The bucket `h >= 4` must therefore hold real pairs, thus the ball runs at
`k = 4` and the walk at `window = 5`. A ball at `k = 3` holds nothing at 4,
and the policy would silently become 50/25/0.

## The augmentation is exactly the size that was asked for

| Graph | \|E\| | n·log10(n) | expected `D.nnz` (directed) | measured |
| --- | --- | --- | --- | --- |
| Cora | 5,278 | 9,295 | 29,146 | 29,148 |
| PubMed | 44,324 | 84,680 | 258,008 | 258,008 |
| com_youtube | 2,987,624 | 6,871,706 | 19,718,660 | 19,718,658 |

Every bucket filled its share on every graph. The 1.13M run:

```
bucket h=2 : 5,567,647 available, 3,435,853 asked, 3,435,853 taken
bucket h=3 : 4,819,944 available, 1,717,926 asked, 1,717,926 taken
bucket h>=4: 2,020,990 available, 1,717,926 asked, 1,717,926 taken
```

## The results

Against the main line, which is the force law of the package and the
per-node cap with far pairs.

| Graph | Source | AUC | R2 dist | R2 vec | final \|\|dZ\|\| |
| --- | --- | --- | --- | --- | --- |
| Cora | ball | 0.9889 | 0.044 | 0.463 | 0.92 |
| Cora | walk | 0.9898 | 0.073 | 0.564 | 0.97 |
| PubMed | ball | 0.9894 | 0.050 | 0.160 | 1.43 |
| PubMed | walk | 0.9908 | 0.018 | 0.065 | 1.52 |
| com_youtube | walk | 0.9026 | 0.005 | -0.332 | 5.88 |
| *Cora, main line* | *walk* | *0.9970* | *0.655* | *0.813* | *0.36* |
| *PubMed, main line* | *walk* | *0.9953* | *0.549* | *0.515* | *0.23* |
| *com_youtube, main line* | *walk* | *0.8988* | *0.433* | *0.184* | *1.53* |

**The link prediction does not move.** Every difference is inside 2%, thus
noise by the rule of PLAN.md section 5.

**The hop distance collapses.** 0.655 to 0.073 on Cora, 0.549 to 0.018 on
PubMed, and 0.433 to 0.005 at 1.13M nodes. An R2 of 0.005 means that the
model reads nothing: it is the mean of the train set.

`R2 vec` at com_youtube is **-0.332**, thus WORSE than predicting the mean.
The 64 numbers of `|Z[u] - Z[v]|` mislead the model there.

## Which change caused it

Two changes arrived together, thus the four combinations were run on Cora.

| Force | Policy | `D.nnz` | AUC | R2 dist | final \|\|dZ\|\| |
| --- | --- | --- | --- | --- | --- |
| v1 | cap | 75,334 | 0.9958 | **0.616** | 0.27 |
| v1 | buckets | 29,148 | 0.9957 | **0.169** | 0.21 |
| v2 | cap | 75,334 | 0.9905 | 0.584 | **250.90** |
| v2 | buckets | 29,148 | 0.9898 | **0.073** | 0.97 |

**The augmentation policy is the larger cause.** The force law of the
package with the bucket policy already falls from 0.616 to 0.169, thus the
policy alone costs 73%.

**The force law costs more inside the new policy than outside it.** With
the far pairs present it costs 5% (0.616 -> 0.584). Inside the bucket
policy it costs 57% (0.169 -> 0.073).

**And `v2` with the far pairs does not converge.** `||dZ||` ends at 250.90,
against 0.27 for the same policy with `v1`. The reason is arithmetic: a far
pair carries `h = 100`, thus `Fr = -k3 * h = -1000`, and that push never
decays with the distance. The layout cannot reach an equilibrium. This
combination is not what the specification asks for -- the specification has
no far pairs -- and it is in this table only to separate the two changes.

## Why the bucket policy costs the geometry

The far pairs at `h = 100` were 25% of `D`, and they were the only term
that said "most of the graph is far away". The bucket policy replaces them
with pairs at `h = 2..5`, which are all LOCAL.

The hop regression measures whether the distance in the embedding can give
the distance in the graph, over pairs at 2 to 17 hops. An augmentation that
holds nothing beyond 5 hops gives the layout no information about that
range, thus the embedding distance saturates and the R2 falls to zero.

The link prediction does not fall, and that agrees: it asks about pairs at
1 hop, and the `h = 1` shell is complete in both policies.

This is the same lesson as G1 finding 2, from the other side. The pair set
decides the link prediction, and the weight -- and the RANGE of the weights
-- decides the geometry.

## A correction to an earlier claim of this project

The G3 report says that the low AUC of the 1.13M run (0.8988) comes from
the adjacency: the walks recovered only 67% of the edges at `h = 1`.

This branch tests that, because the bucket policy adds EVERY original edge:
the histogram of the 1.13M run shows `{1: 5,975,248, ...}`, which is 100%
of the directed edges.

**The AUC moved from 0.8988 to 0.9026.** That is 0.4%, thus noise.

The claim is therefore not supported. The comparison is confounded, because
the force law and the policy also changed, and a clean test needs `v1` with
the bucket policy at 1.13M nodes. Until that runs, the honest statement is:
restoring the adjacency in full did NOT repair the link prediction at
1.13M nodes, and the cause of the 0.90 AUC is unknown.

## Cost

| Graph | augmentation | embedding | peak RSS |
| --- | --- | --- | --- |
| Cora | 0.4 s | 7.4 s | 768 MB |
| PubMed | 3.8 s | 19.3 s | 1206 MB |
| com_youtube | 1141 s | 448 s | 3333 MB |

The augmentation of the 1.13M graph needs 19 minutes, against 9 for the
main line. `window = 5` is the reason: it gives 448M raw pairs, where
`window = 3` gives 283M.

The peak RSS of 3333 MB is above the 2.6 GB of the gate. The run had no
guard, because it is off-branch.

## The deviations

| Deviation | Why |
| --- | --- |
| `--cap 16` on com_youtube | the accumulator needs a bound at 1.13M nodes. It does NOT change `D`, because the bucket budget fixes the size. It only sets how many candidates the buckets draw from. |
| `dim 64`, `epochs 500`, 8 chunks at com_youtube | the same limits as G3 |
| one seed | an off-branch check, not a gate |

## What this says

1. The specification builds exactly the graph that it describes, on every
   graph including 1.13M nodes.
2. Neither change repairs the link prediction.
3. Both changes cost the hop-distance geometry, and the augmentation
   policy costs more of it than the force law.
4. The far pairs are not a detail of the old policy. They carry the only
   long-range information that the embedding gets, and removing them
   removes the global scale.
