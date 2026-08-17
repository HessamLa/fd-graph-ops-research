# 2026-08-08 — NetworKit: evaluated, rejected

## How this started, and the mistake worth recording

`TODOS.md` §3 originally contained a NetworKit section I had written from
the module index: a table mapping its offerings onto our needs, a list of
caveats, and three "install and benchmark" action items. It looked like
research. It was not — it deferred the actual decision.

The user cut through it:

> Did you research about Networkit? Installation is not a problem. The main
> question is whether we'll gain benefits using it!

Correct. Installation was never the obstacle; I had answered a question
nobody asked. Installed `networkit==11.2.1` and measured.

## The question I should have asked first

**Is augmentation even worth optimizing?** It runs *once*; the embedding
loop runs every epoch. Measured at n=20,000, radius 2, d=64:

```
augmentation (one time) : 0.43 s
embedding, steady state : 292.5 ms/epoch

  100 epochs -> augmentation is 1.4% of runtime
 1000 epochs -> augmentation is 0.1% of runtime
 2000 epochs -> augmentation is 0.1% of runtime
```

An *infinitely fast* augmenter saves 0.1% of a realistic run. Amdahl's law
nearly settles it before any benchmark. I only reached for this framing
after being forced to justify the line of inquiry.

## Job B — bounded radius (`sparse_hops`), the path used at scale

| n | r | current | NetworKit | |
|---|---|---|---|---|
| 2,000 | 2 | 0.02 s | 0.37 s | 18x slower |
| 5,000 | 2 | 0.06 s | 2.20 s | 37x slower |
| 10,000 | 2 | 0.11 s | 9.41 s | **85x slower** |

Two attempts, deliberately, to avoid condemning it on a strawman:

1. One `nk.distance.BFS` per source — 30–100x slower, but the loop is in
   Python so it forfeits OpenMP entirely. Not a fair test.
2. Checked the API surface and found `SPSP(G, sources)`, which runs every
   source **inside C++ in parallel**. Still 85x slower.

The reason is structural and no amount of threading fixes it: **radius 2
needs exactly two sparse matrix products** and never computes a distance it
will not keep. NetworKit computes *every* pairwise distance and discards
everything past the radius — asymptotically more work.

`SPSP` also materializes a dense `(n, n)` result: 400 MB at n=10k, 80 GB at
n=100k. That reintroduces the exact wall this package exists to avoid.

## Job A — all-pairs (`hopfill`)

| n | scipy | NetworKit APSP (+conversion) | |
|---|---|---|---|
| 1,000 | 0.22 s | 0.14 s | 1.6x faster |
| 2,000 | 0.91 s | 0.46 s | 2.0x faster |
| 4,000 | 3.85 s | 1.79 s | 2.2x faster |
| 8,000 | 18.12 s | 7.92 s | **2.3x faster** |

A genuine win. Distances agree exactly. Conversion is negligible (0.04 s at
n=8k) — and I built the NetworKit graph from CSR arrays rather than via
`nxadapter.nx2nk`, which loops in Python, so as not to stack the deck
against it.

It still does not matter, for two independent reasons:

1. **It does not move the memory wall.** Peak RSS 3.9 GB at n=8,000 — both
   sides are `O(n^2)`. NetworKit makes mode 1 *faster*, not *feasible*, and
   feasibility is the actual problem.
2. **Amdahl.** 2.3x on 0.1% of runtime = 0.24 s saved out of 293.

## Verdict

**Do not adopt.** Not as a speed play, under any current workload.

## What would reverse this

Not speed — a capability the current stack lacks entirely:

- `networkit.community` (label propagation, modularity) for the
  **inter-community augmentation policy** (`TODOS.md` §1.4). scipy has no
  equivalent and writing one is real work. This is the only live reason.
- `networkit.centrality` for hub selection in the hub policy.
- `PrunedLandmarkLabeling` *if* a future policy needs many repeated
  point-to-point distance queries. Current ones do not.

Revisit only when building the community-based policy, and adopt it for the
community detection alone — never to replace a distance computation.

## Housekeeping

I deleted the original speculative subsection from `TODOS.md` rather than
leaving it beside the measured verdict. Stale guesses adjacent to real
numbers are worse than no notes.

`networkit` remains installed in the venv but **nothing in the package
imports it**. Keep it for the community-detection path, or
`pip uninstall networkit`.

## Lesson

I wrote a plausible-sounding evaluation plan and called it research. The
question to apply to my own writeups, before someone else has to: *which of
these numbers did I actually measure?*
