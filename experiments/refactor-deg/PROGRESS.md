# Moving `1/deg(u)` from the kernel into the force laws

Started 2026-09-09. Owner's instruction: remove `1/deg(u)` from the places
it does not belong, put it in the places it does, one force function at a
time, with verification. Target bit-identical; a relative difference of
1e-6 counts as identical.

Repo state at the start: `5c46a98`, clean, tagged `fodiwalk-pre-refactor2`.

## Why

The model is

```
dz_u = F_u / deg(u),    F_u = SUM_{v in S(u)} f_uv(||z_uv||) * unit(z_uv)
```

`1/deg(u)` is an averaging coefficient. It decides whether a row converges
or diverges, and the right denominator is the size of the set the law sums
over. Until now the SELL-C-sigma kernel applied one divisor to every law
([sell_c_sigma.py:485](../../forcedirected/sell_c_sigma.py#L485)) and no law
could disagree.

That is a real defect and not only a tidiness question. `degrees_from_D`
counts the `h == 1` entries of the row. `fdhop` attracts at `h == 1` and
nowhere else, so that count is its correct denominator. `fdhop_all`
attracts at EVERY `h`: on cora it sums about 7x more terms and divides by
the same small count. That is why it went non-finite at `k1 = 0.999,
k2 = 1.0`.

## The arithmetic

Division is linear and distributes over the sum:

```
F_u / deg(u) = SUM_v [ f_uv(x) / deg(u) ] * unit(z_uv)
```

So a per-CELL divide inside the law is the same number as a per-ROW divide
after the sum, up to float32 rounding order. Bit-identical is expected only
where `1/deg(u)` is a power of two, because then the multiply changes the
exponent and leaves the mantissa alone. Everywhere else a difference near
the float32 epsilon (~1.2e-7) is the correct outcome, not a defect.

## The gate

`dz_oracle.py`. 8 law cases x 2 graphs (tiny 60 nodes, cora 2708), one full
jitted kernel pass each, fixed `Z`, fixed seed. It calls the jitted step
directly and never `Fodiwalk.forces`, because `forces` applies a random
drop.

Proven sensitive before use: perturbing `kr` inside `fdhop` by one part in
1e5 moved dZ by 4.56e-5 relative on cora and 2.84e-5 on tiny, both flagged,
exit 1. That is 46x above the 1e-6 tolerance.

`fdhop_nodeg` is the built-in control: `no_deg_norm` sets every degree to 1,
so `inv_deg` is exactly 1.0 and the multiply is exact. That case must stay
BIT-IDENTICAL through every milestone. If it ever moves, the move is wrong.

## Scope

The owner ruled on 2026-09-09 that `fodined` and `fdwalk` are replaced by
`fodiwalk` and are out of scope. Four experiment scripts drop off the work
list with them: `bench_shortest_path_gemsec.py`, `bench_large_graphs.py`,
`drop_strategies/bench_drop_strategies.py` and `fdwalk/bench_fdwalk.py`.
All four already fail to import (`fodined`, and nine dead names in the
`fdwalk` bench), verified by resolving their imports, so nothing that runs
today is left behind.

In scope: `fodiwalk/`, `forcedirected/`, and one experiment.

`experiments/fodiwalk-streaming/bench_stream.py` is the one outside caller
that still runs. It is a hand-written streaming copy of the kernel: it
calls the law at `:301` and applies its own divisor at `:304`. It gets the
same treatment as `sell_c_sigma.py`, in the same milestone as `fdlinear`.
Left alone it would double-divide and every force would become `1/deg^2`.

`experiments/fodiwalk/verify_baseline.py` imports cleanly but only compares
function identity (`fw.force_fn is forces.fdlinear`) and never calls a law,
so the signature change does not reach it.

## The end state

`forcedirected/sell_c_sigma.py` holds NO division, NO `kernel_divides`
flag, and `forces.self_deg` is deleted. The kernel builds `inv_deg` and
hands it to the law; what the law does with it is the law's business. The
scaffold below exists only to move one law at a time and is removed at M9.

## Milestones

| # | what | dZ result | gate |
| - | ---- | --------- | ---- |
| M0 | `dz_oracle.py`, baseline captured | 16/16 bit-identical on a no-op re-run | n/a |
| M1 | 4th positional arg `inv_deg` | 16/16 bit-identical | REJECTED by the owner: bad signature, bad name |
| M2-M8 | seven laws, one at a time, `params["node_degree"]` | 16/16 within 1e-6, worst 7.16e-07 | pass |
| M9 | kernel divides nothing; `deg_ext` holds DEGREES | -- | pass |

DESIGN, as the owner set it: `force_fn(x, planes, params)` keeps its three
arguments. The degree rides in `params["node_degree"]`, like `k1`. Every law
ends with `averaged(F, params)`. `forcedirected/sell_c_sigma.py` takes no
reciprocal and performs no division.

## Per-step result, all seven laws

16 cases, 2 graphs. **16/16 within 1e-6, 0 over. Worst 7.16e-07**
(cora/fdlinear). The two `no_deg_norm` controls are BIT-IDENTICAL, because a
degree of 1 makes the multiply exact -- the check that the plumbing is right
and not merely close.

Bit-identical is not reachable for a real degree, and the reason is
structural: the old code computed `(SUM terms) * (1/deg)` once per ROW; a law
computes `f/deg` once per CELL. Float32 multiplication does not distribute
over a sum.

## What 2000 epochs does to that

`test_parity.py` pins p1 (200 ep) and p2 (2000 ep) at `abs=5e-5`. After the
move, p1 loses only `r2_dist` (5.374e-04 against a 5e-04 tolerance) and p2
loses all four.

THE CAUSE IS THE POSITION OF THE MULTIPLY, and it was measured, not assumed.
An isolation control -- pre-refactor code with the SAME reciprocal array
moved from after the row sum to inside it, nothing else changed --
reproduces the same per-step numbers (tiny/fdlinear 5.9406e-07, identical to
4 significant figures) and breaks the same pins, on four of five metrics
FURTHER from the pins than this refactor.

A second control was VOID and is recorded so nobody repeats it:
`F_mag * (1.0/x_safe)` against `F_mag / x_safe` is BIT-IDENTICAL on all 16
cases. XLA folds them. It perturbs nothing, so its passing the pins proved
nothing.

## Why: the divergence curve

Same config, both trees, `Z` every 25 epochs, `||Z_new - Z_old|| / ||Z_old||`:

| epoch | 25 | 50 | 100 | 200 | 400 | 1000 | 2000 |
| ----- | -- | -- | --- | --- | --- | ---- | ---- |
| rel   | 1.32e-06 | 1.23e-06 | 3.21e-05 | 2.57e-03 | 1.13e-01 | 6.59e-02 | 9.14e-02 |

Three regimes. Seeded at 1.32e-06, about 11x float32 epsilon. Exponential
over epochs 50-400, fitted **lambda = 0.0330 per epoch** -- doubling every 21
epochs, x91,800 over that window. Then FLAT from epoch 500: mean 7.54e-02,
trend +2.8e-05 per epoch. The two runs sit on one attractor at different
points, and the distance between them stops growing because the attractor
has a finite width.

The mechanism is cancellation. At convergence attraction and repulsion
nearly balance, so `F_u` is a small residual of large opposing terms, and
rounding error is set by the size of the TERMS. Over 200k random 8-cell rows
the worst relative gap between the two orderings was 3.4e-03, not 1e-07.

**The coordinates diverge 7.5 percent. The quality does not.** `acc` moved by
one test pair in 1056, `auc` by 0.028 percent. Many configurations score
equally well and the two runs found different ones.

CONSEQUENCE FOR THE PINS: a 2000-epoch pin at `abs=5e-5` is pinning the exact
trajectory of a chaotic system. It holds only while the arithmetic is
bit-for-bit identical. That fragility is pre-existing; this refactor exposed
it. A JAX upgrade or a different GPU would do the same -- an inference from
lambda, NOT a measurement.

## Old against new, 24 paired configs

Every >=50-epoch fodiwalk config of `data_cache/embeddings/` on cora, pubmed
and wordnet, re-run with the new implementation, flags generated from each
stored `config.json` so the walks are bit-identical and only the embedding
arithmetic differs. All 24 succeeded, 41 minutes, sequential.

**16 of 24 identical to four decimals on accuracy, f1 AND auc.**

| metric | mean abs d | median abs d | max abs d | new better |
| ------ | ---------- | ------------ | --------- | ---------- |
| acc    | 0.00011    | 0.00000      | 0.00047   | 4/24 |
| f1     | 0.00010    | 0.00000      | 0.00048   | 4/24 |
| auc    | 0.00004    | 0.00000      | 0.00054   | 7/24 |
| rf_r2  | 0.00219    | 0.00001      | 0.04120   | 15/24 |

Mean SIGNED delta on accuracy: -0.000016. No direction, which is what
re-rounding looks like and not what a bias looks like. The largest accuracy
shift is half a test pair out of 1056. Runtime total 1885.8s -> 1923.9s,
ratio 1.020, with per-config swings both ways -- too noisy to call a
slowdown.

The full table is `agentic-log/00.master-agent/artifacts/comparison-table.txt`.

NOT COVERED: com_youtube, 6 configs, old timings sum to 17.8 h sequential.
`fdunit` and `fdunit_freq`, whose laws were rewound out of the code.

## Unverified claims

* `cora/fdhop/10x20x5/q=1` lost 0.041 of `rf_r2`, 5.5 percent relative and an
  order of magnitude above every other `rf_r2` delta. The run-to-run spread of
  `rf_r2` at a fixed config HAS NOT BEEN MEASURED, so whether 0.041 is inside
  normal variance is unknown. Treat that cell as unexplained.
* That a JAX or GPU change would break the p1/p2 pins the same way. Inferred
  from lambda = 0.033, not measured.
* The 2 percent runtime ratio. Wall-clock on a shared machine, single sample
  per config.

## Design: the migration scaffold

The kernel cannot divide for a law that already divided for itself, or every
force becomes `1/deg^2`. During the migration it reads a mark:

```python
kernel_divides = not getattr(force_fn, "applies_inv_deg", False)
```

`force_fn` is static in the `functools.partial`, so this runs at trace time
and costs nothing per step. `forces.self_deg` sets the mark. Both the branch
and the helper are deleted when every law carries the mark.

## Unverified claims

Recorded here so they are not mistaken for measurements. See the bottom of
this file for the current list.
