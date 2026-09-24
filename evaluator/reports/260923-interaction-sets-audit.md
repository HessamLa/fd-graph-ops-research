# Interaction sets: theory against the code

**Built:** 2026-09-23. **Scope:** the set of nodes that acts on node `u` in
one force step, and the force formula over that set. Code read:
`fodiwalk/augment_graph/`, `fodiwalk/embed/forces.py`,
`fodiwalk/embed/degrees.py`, `forcedirected/sell_c_sigma.py`,
`fodiwalk/config.py`, `experiments/embeddings/make_embedding.py`.
Measurements: cora and citeseer, the `Config` defaults (`walks=10`,
`walk_len=20`, `window=5`, `cap=16`, `weight=min_gap`, `far_weight=100`,
seed 42). Script: kept in the session scratchpad, not in the repo.

## The theory under test

For node `u`:

- `N(u) = {v : d(u,v) = 1}`, with `h_uv = 1`.
- `W(u)` = the nodes visited by the random walks that START at `u`.
  `h_uv` = the minimum walk gap from `u` to `v` over those walks.
- `Far(u)` = random nodes that are in neither `N(u)` nor `W(u)`.

```
F(u) = sum over v in N(u) ∪ W(u) ∪ Far(u) of
       ( fa(||z_v - z_u||, h_uv) + fr(||z_v - z_u||, h_uv) ) * (z_v - z_u) / ||z_v - z_u||
```

## Summary

The code has THREE pair policies, and each one builds a different set.
None of them builds the theory set exactly.

| set | `walk` (Config default) | `walk_edges` (ICLR headline) | `nbr_walk` (make_embedding default) |
|---|---|---|---|
| `N(u)` | edges a walk crossed; can miss some | complete, `h = 1` | complete, `h = 1` |
| `W(u)` | window co-occurrence in ANY walk, capped, symmetric | same as `walk` | walks FROM `u`, all steps, uncapped — **matches the theory** |
| `h` on `W(u)` | min gap over all walks, `2..5`; 98–99 % exact | same | min first-visit step from `u`, `2..19`; 12–16 % exact |
| `Far(u)` | global pool of `n·log10 n` pairs, `h = 100` | same | **none** (`far = 0`) |
| attraction on | `N(u)` only | `N(u)` only | `N(u)` only |

Row make-up (cora): `walk_edges` rows are 14 % `N`, 61 % `W`, 25 % `Far`;
`nbr_walk` rows are 5 % `N`, 95 % `W`, 0 % `Far`.

The two policies fail the theory in opposite ways. `nbr_walk` builds the
right `W(u)` but has no `Far(u)` and a loose `h`. `walk_edges` has a tight
`h` and a `Far(u)`, but its `W(u)` is a different set.

## 1. Discrepancies in the sets

### D1. Under `walk` and `walk_edges`, `W(u)` is not "walks that start at `u`"

`walk_pair_stats` starts `walks` walks at EVERY node
(`walks.py:183`) and pairs any two nodes that sit within `window` steps of
each other in ANY walk (`walks.py:111-118`). So `v` enters `W(u)` when a
walk that started at some third node `c` passes `u` and then `v`. The key
is `min(u,v)·n + max(u,v)`, so the set is symmetric: `v ∈ W(u)` exactly
when `u ∈ W(v)`. The theory set is anchored at `u` and is not symmetric.

Measured overlap with one draw of the theory set (walks from `u`, the same
budget): only **62 %** of the code's `W(u)` entries on cora, **71 %** on
citeseer, are nodes that a walk from `u` reached. (The theory set is itself
random, so this is an estimate, not an exact count.)

### D2. The window cuts `W(u)` to 5 steps

Only pairs within `window = 5` positions are made. A walk of `walk_len = 20`
visits positions 6–19 too, and those nodes never pair with the start. So
`h` on `W(u)` runs `2..5`, never higher. The theory set holds every visited
node.

### D3. The cap cuts `W(u)` to ~16 partners, ranked by visit count

`cap_per_node` keeps each node's 16 most-visited pairs, and a pair survives
if EITHER endpoint keeps it (`pairs.py:116`, called at
`policy_walk.py:140`). On cora 181,423 distinct walk pairs shrink to
28,372 — **84 % dropped**.

| graph | code `|W(u)|` mean | theory `|W(u)|` mean |
|---|---|---|
| cora | 17.1 | 77.1 |
| citeseer | 12.4 | 39.1 |

The cap ranks by COUNT, not by `h`, so it keeps the short pairs: the kept
`h` histogram on cora is `{2: 28118, 3: 14276, 4: 2770, 5: 1030}`. The
"either endpoint" rule also makes hub rows wide: the top 1 % by degree hold
142 entries per row on cora, the rest 27.

### D4. `h` on `W(u)` is the gap over ALL walks, not over walks from `u`

Under `walk`/`walk_edges`, `h_uv` = the smallest gap at which `u` and `v`
co-occur in any walk. This is a different estimator from the theory, and a
much TIGHTER one: `h` equals the true hop **98.1 %** of the time on cora
and **98.9 %** on citeseer, and is never below it. Many walks cross each
pair, and the cap keeps the frequent (short) pairs.

### D5. `N(u)` is not guaranteed under the default `walk` policy

`walk` stores an edge only if a walk crossed it. Measured: **3 of 5,278**
edges missing on cora, **1 of 4,552** on citeseer. `walk_edges` adds every
missing edge at `h = 1` (`policy_walk.py:58`) and `nbr_walk` adds every
edge in `with_all_neighbours`, so both give the full `N(u)`.

Latent case, 0 observed: `walk_edges` adds an edge only when its key is
ABSENT from the walk pairs. An edge that a walk saw only at gap 2
(`u → x → v`, never `u → v` directly) keeps `h = 2`. Nothing resets it to 1.
It did not happen on cora or citeseer at the default budget.

### D6. `Far(u)` is a global pool, not a per-node draw

`finish_cap` draws `n·log10 n` pairs in total, uniform over node pairs
(`far_bias = 0`), and stores each in both directions
(`policy_walk.py:106-108`). The count per node is therefore random: mean
6.9 on cora, and 4 nodes get none. Far pairs are rejected against the
near `D`, so they avoid the CODE's `N ∪ W` — which is the theory's rule,
but applied to the code sets. 2.4 % (cora) and 0.8 % (citeseer) of far
pairs are nodes that the theory `W(u)` would hold.

### D7. `h` on `Far(u)` is a constant 100, not a distance

Every far pair gets `h = far_weight = 100` (`policy_walk.py:77`). The true
hop of those pairs has median **6** on cora and **9** on citeseer, and
**15 %** (cora) / **60 %** (citeseer) of them join two different
components (no path at all). The theory leaves `h_uv` on `Far(u)` open.
The code's choice makes a far pair's repulsion coefficient 20× the largest
`W(u)` coefficient (5). A landmark distance exists as an option
(`spec.landmarks > 0`) and is off by default.

### D8. `nbr_walk` has NO `Far(u)` by default

`Config.far = 0`, and `nbr_walk` adds far pairs only when `spec.far > 0`
(`policy_nbr_walk.py:138`). The `n·log10 n` default applies to
`walk`/`walk_edges` only (`policy_walk.py:106`). `make_embedding.py` has
no `--far` flag and defaults `--pairs` to `nbr_walk`. **So every `nbr_walk`
run in the store, including the `fdhop nbr_walk` rows of
`260917-iclr2027-preliminary.md`, ran with `Far(u) = ∅`.** That is a
likely part of why `nbr_walk` scores low on rho and high on recall@10:
nothing pushes the far graph away.

### D9. `nbr_walk` builds the theory `W(u)`, but its `h` is loose

`walk_rows` anchors the walk at the row (`walks.py:258-264`), uses every
step, and has no window and no cap by default. This IS the theory set.
But `h` = the first step at which a walk from `u` reached `v`, and a random
walk is not a shortest path. On cora the kept `h` is almost FLAT from 2 to
19 (about 11,000 entries at each value), and `h` equals the true hop only
**12.4 %** (cora) / **15.5 %** (citeseer) of the time; the rest overshoot.
Under `nbr_walk` the `h` plane mostly records the walk step, not the
distance.

### D10. `nbr_walk` ignores the weight rule

`h = stats.mn` always (`policy_nbr_walk.py:119`); `weights.RULES` is never
called. Already recorded in `260917-iclr2027-preliminary.md` §8, defect 1.

### D11. Isolated nodes get only `Far(u)`

A walk from a degree-0 node stays in place and self-pairs are dropped, so
`N = W = ∅`. Under `walk`/`walk_edges` such a node still gets far pairs,
which gives it a row with no `h = 1` entry and trips invariant I5 (48 nodes
on citeseer; the run needs `--deg-source A`). Under `nbr_walk` (no far
pairs) its row is empty and it never moves.

### D12. Direction of `D`

`walk`/`walk_edges` store a symmetric `D`; `nbr_walk` a directed one (row
`u` = what `u` found), which matches the theory. When far pairs are added
to `nbr_walk`, they go in symmetrically.

## 2. Discrepancies in the force formula

### F1. Attraction acts on `N(u)` only

In `fdhop`, `fdhop2`, `fdhop_min` and `fdlinear`, `Fa` is non-zero only
where `h ≤ 1` (`forces.py:228` for `fdhop`). On `W(u)` and `Far(u)` the
theory's `fa(·, h_uv)` is zero in code; those pairs only repel. Only
`fdhop_all` and `fdhop_all_freq` attract at every `h`.

### F2. Repulsion grows with `h`

`fdhop`: `Fr = −kr·h·exp(−k4·x)` for `h ≥ 2`, and a flat `−kr` (no decay) at
`h = 1`. So a pair repels harder the farther the code says it is: a far
pair (`h = 100`) at the same embedding distance pushes 20× harder than an
`h = 5` walk pair.

### F3. The sum is divided by `|N(u)|`, not by the size of the set

Every law ends with `(Fa + Fr) / deg`, and `deg` = the count of `h == 1`
entries of the row (`forces.py:80`), or the graph degree with
`--deg-source A` (`degrees.py:59`). The sum runs over `N ∪ W ∪ Far`, but
the divisor counts `N` only. The theory formula has no divisor.

### F4. Direction and sign match

The kernel computes `Zdiff = z_v − z_u`, `x = ||Zdiff||`, and adds
`Zdiff · F_mag / x`, with an exact 0 at `x = 0`
(`sell_c_sigma.py:494-505`). This is the theory's unit vector. A positive
magnitude moves `u` toward `v`.

## 3. What matches

- The direction term and its sign (F4).
- `N(u)` with `h = 1`, under `walk_edges` and `nbr_walk` (D5).
- `W(u)` as an anchored, directed set, under `nbr_walk` (D9, D12).
- `Far(u)` disjoint from the stored near pairs, under `walk`/`walk_edges`
  (D6).
- `h` as a minimum walk gap, an upper bound of the hop distance, never
  below it (D4, D9).

## 4. Choices the owner must make

These are options, not changes made. Nothing in the code was edited.

1. **Which `W(u)` the paper means.** Either state that `W(u)` is the
   window-co-occurrence set (what `walk_edges` builds), or run `nbr_walk`
   for the theory set.
2. **Far pairs for `nbr_walk`.** Expose `--far` in `make_embedding.py` and
   set it to `n·log10 n`, or state in the paper that the `nbr_walk` rows
   have no `Far(u)`. The current `nbr_walk` numbers cannot test the
   three-set theory.
3. **`h` on `Far(u)`.** Keep the constant 100, or turn on the landmark
   distance (`spec.landmarks`) that already exists.
4. **The divisor.** Add `1/|N(u)|` to the theory formula, or change the
   code.
5. **The latent `walk_edges` edge at `h > 1` (D5).** A one-line fix sets
   every graph edge to `h = 1` whether or not the walks saw it first.
