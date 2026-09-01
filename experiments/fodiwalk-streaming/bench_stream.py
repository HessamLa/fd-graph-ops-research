#!/bin/env python3
"""bench_stream.py -- node2vec's memory strategy, on fodiwalk's physics.

THE QUESTION. `fodiwalk` stores every pair a walk finds, in `D`, and keeps
it for the whole run: 77,257,655 entries at com_youtube, about 2.5 GB of
live arrays. node2vec stores NO pair. It walks, uses a pair once, and
throws it away; its walks sit on disk and its memory holds only the model.
Same walks, opposite fate (`REPORT_1M.md`).

This script asks what fodiwalk costs if it does the same thing.

WHAT IT DOES. For a BATCH of nodes at a time:

    walk from those nodes only          5 walks x 20 steps for each
    reduce to three vectors             partner id, min hop `h`, frequency
    add the neighbours of those nodes   at h = 1
    compute dZ for those rows           the fdlinear law, verbatim
    update Z for those rows             at once, then drop everything else

`D` is never built. Nothing of size `nnz` is ever live. The peak is one
batch of walks plus `Z` itself.

WHY A BATCH AND NOT ONE NODE. The rule asked for is per node: walk, fuse,
take the gradient, move on. One node at a time in Python costs 1.13M
interpreter round trips for each epoch, which measures Python and not the
idea. A batch is the same rule applied to `batch` nodes at once, it keeps
the memory bound (the batch is the unit that is live), and `--batch 1`
runs the literal per-node form for anyone who wants to see it.

WHAT IS DIFFERENT FROM `fodiwalk`, and it is not a detail:

1. **The walks are fresh in every epoch.** There is no `D` to reuse, so
   each epoch draws its own sample. `fodiwalk` relaxes against ONE fixed
   `D` for every epoch. This is the stochastic form of the same objective,
   and it is what node2vec does over its own corpus.
2. **The update is immediate.** Row `u` moves before row `u + 1` is
   computed, so a batch reads a `Z` that earlier batches of the same epoch
   already changed. `fodiwalk` computes the whole `dZ` against one frozen
   `Z` and applies it once for each epoch. That is Gauss-Seidel against
   Jacobi; they do not give the same trajectory.

Both differences are the POINT of the experiment, not defects of it. The
report says what they cost.

The physics is imported, never copied: `fodiwalk.embed.forces.fdlinear`
and `fodiwalk.augment_graph.walks.uniform_walks`. The projection around
the law -- `Zdiff / x`, the degree division, the `x == 0` guard -- is
reproduced from `forcedirected.sell_c_sigma.step` and is marked where it
is.

Run from the repository root:

    .venv/bin/python experiments/fodiwalk/bench_stream.py \
        --graph com_youtube --dim 64 --epochs 5
"""
from __future__ import annotations

import argparse
import functools
import resource
import threading
import time

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--graph", default="com_youtube")
ap.add_argument("--max-nodes", type=int, default=0)
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--walks", type=int, default=5)
ap.add_argument("--walk-len", type=int, default=20)
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--batch", type=int, default=4096,
                help="nodes per step. 1 is the literal per-node rule")
ap.add_argument("--lr", type=float, default=1.0)
ap.add_argument("--optim", default="plain",
                choices=("plain", "velocity", "nesterov", "adam", "fa2",
                         "sgd", "momentum"))
ap.add_argument("--lr-decay", default="const", choices=("const", "linear"),
                help="linear: lr_t = lr * (1 - epoch/epochs), the schedule "
                     "of fodiwalk/model.py. It never reaches 0, so the "
                     "last epoch still moves")
ap.add_argument("--eta", type=float, default=0.3, help="velocity")
ap.add_argument("--sgd-frac", type=float, default=0.5, help="sgd")
ap.add_argument("--b1", type=float, default=0.9, help="adam")
ap.add_argument("--b2", type=float, default=0.999, help="adam")
ap.add_argument("--k-s", type=float, default=0.1, help="fa2")
ap.add_argument("--k-max", type=float, default=10.0, help="fa2")
ap.add_argument("--beta", type=float, default=0.9,
                help="momentum: the decay of the accumulator")
ap.add_argument("--alpha", type=float, default=1.0,
                help="momentum: the gain ON dZ. `m = beta*m + alpha*dZ`. "
                     "The DC gain of the rule is alpha/(1-beta), so the "
                     "EFFECTIVE learning rate is lr*alpha/(1-beta). "
                     "alpha=1, beta=0.9 is the classic rule and its gain "
                     "is 10; alpha+beta=1 gives gain 1 and IS `velocity`")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--device", default="cpu")
ap.add_argument("--score", action="store_true",
                help="score Z with `evaluator` when the run ends")
ap.add_argument("--protocol", default="n2v1m")
ap.add_argument("--lp-max-pairs", type=int, default=50_000,
                help="total link-prediction pairs. 50,000 matches the "
                     "archived node2vec run of "
                     "experiments/large-graph-node2vec/, which passed "
                     "--lp-pairs 25000 and doubled it")
ap.add_argument("--tag", default="", help="name for the log line")
ap.add_argument("--out", default="")
args = ap.parse_args()

import os
# JAX names the CUDA backend "cuda"; "gpu" is not a backend name.
os.environ.setdefault("JAX_PLATFORMS",
                      {"gpu": "cuda"}.get(args.device, args.device))
import jax
import jax.numpy as jnp

from fodiwalk.augment_graph.walks import uniform_walks
from fodiwalk.embed.forces import fdlinear
from fodiwalk.make_graph import load

# ---------------------------------------------------------------- memory
_peak = [0.0]
_stop = [False]


def rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / 2 ** 20


def _sample():
    while not _stop[0]:
        _peak[0] = max(_peak[0], rss_mb())
        time.sleep(0.005)


threading.Thread(target=_sample, daemon=True).start()


def log(msg):
    print(f"[stream] {msg}", flush=True)


# ---------------------------------------------------------------- the rows
def rows_of(A, nodes, n, n_walks, walk_len, rng):
    """The three vectors of each node in `nodes`, fused over its own walks.

    Returns `(u_local, v, h, freq)`, flat and grouped by `u_local`, where
    `u_local` indexes `nodes` and not the graph. This is the whole of what
    row `u` needs, and it is thrown away as soon as the row has moved.
    """
    b = nodes.size
    starts = np.repeat(nodes, n_walks)
    w = uniform_walks(A, starts, walk_len, rng)          # (b*n_walks, len)

    # column t of a walk is t steps from its start
    local = np.repeat(np.arange(b, dtype=np.int64), n_walks)
    src_l = np.repeat(local, walk_len - 1)
    dst = w[:, 1:].ravel()
    gap = np.tile(np.arange(1, walk_len, dtype=np.int32), w.shape[0])
    keep = dst != np.repeat(starts, walk_len - 1)        # a walk can stay
    src_l, dst, gap = src_l[keep], dst[keep], gap[keep]

    # every neighbour of the row, at h = 1. `nbr_walk`'s own rule.
    deg = np.diff(A.indptr)[nodes]
    e_local = np.repeat(np.arange(b, dtype=np.int64), deg)
    e_dst = np.concatenate([A.indices[A.indptr[u]:A.indptr[u + 1]]
                            for u in nodes]) if b else np.empty(0, np.int64)

    src_l = np.concatenate([src_l, e_local])
    dst = np.concatenate([dst, e_dst.astype(np.int64)])
    gap = np.concatenate([gap, np.ones(e_dst.size, np.int32)])

    # fuse: one entry for each (row, partner), h = the first step that
    # reached it, freq = how many times the row reached it
    key = src_l * n + dst
    order = np.argsort(key, kind="stable")
    key, src_l, dst, gap = key[order], src_l[order], dst[order], gap[order]
    first = np.ones(key.size, dtype=bool)
    first[1:] = key[1:] != key[:-1]
    idx = np.flatnonzero(first)
    h = np.minimum.reduceat(gap, idx) if idx.size else gap
    freq = np.diff(np.append(idx, key.size)).astype(np.int32)
    return src_l[idx], dst[idx], h.astype(np.float32), freq.astype(np.float32)


# ---------------------------------------------------------------- the force
PARAMS = {"k1": 0.999, "k4": 0.01, "kr": 1.0, "sign": -1.0}


# ---------------------------------------------------------------------------
# The update rules, for a BATCH OF ROWS.
#
# `forcedirected/optim.py` updates the WHOLE of `Z` at once and starts each
# state array from `None`. Streaming touches one batch, so a state array is
# `(n, d)` and only the batch's rows move. That changes the FIRST touch of a
# row, and the rules differ in whether it matters:
#
#   momentum/nesterov  `m = beta*m + dZ` gives `dZ` from a zero state, which
#                      is what `None` gives. Exact.
#   adam               `m = b1*m + (1-b1)*dZ` gives `(1-b1)*dZ` from zero,
#                      which is what `None` gives. Exact. `t = epoch + 1`
#                      because a row is touched once per epoch.
#   velocity/fa2       `None` gives `dZ` itself, and a zero state does NOT.
#                      A `seen` mask marks a row's first touch.
#
# Every rule keeps the reference's parameter names and defaults.
# ---------------------------------------------------------------------------
def make_state(rule, n, d):
    """The `(n, d)` arrays a rule holds, plus its first-touch mask."""
    k = {"plain": 0, "sgd": 0, "momentum": 1, "nesterov": 1, "velocity": 1,
         "fa2": 1, "adam": 2}[rule]
    st = {f"a{i}": jnp.zeros((n, d), dtype=jnp.float32) for i in range(k)}
    if rule in ("velocity", "fa2"):
        st["seen"] = jnp.zeros((n,), dtype=bool)
    return st


def apply_update(rule, Z, st, jn, F, lr, epoch, a):
    """`Z` and the state, after the rows `jn` take one step of `rule`."""
    if rule == "plain":
        return Z.at[jn].add(lr * F), st

    if rule == "sgd":
        # the reference draws ONE mask over every row for the epoch, then
        # applies it; drawing it whole and indexing keeps that.
        key = jax.random.PRNGKey(a.seed + epoch)
        keep = jax.random.bernoulli(key, a.sgd_frac, (Z.shape[0], 1))
        return Z.at[jn].add(lr * jnp.where(keep[jn], F, 0.0)), st

    if rule == "momentum":
        m = a.beta * st["a0"][jn] + a.alpha * F
        st["a0"] = st["a0"].at[jn].set(m)
        return Z.at[jn].add(lr * m), st

    if rule == "nesterov":
        m = a.beta * st["a0"][jn] + F
        st["a0"] = st["a0"].at[jn].set(m)
        return Z.at[jn].add(lr * (F + a.beta * m)), st

    if rule == "velocity":
        first = st["seen"][jn][:, None]
        v = jnp.where(first, a.eta * F + (1.0 - a.eta) * st["a0"][jn], F)
        st["a0"] = st["a0"].at[jn].set(v)
        st["seen"] = st["seen"].at[jn].set(True)
        return Z.at[jn].add(lr * v), st

    if rule == "adam":
        m = a.b1 * st["a0"][jn] + (1 - a.b1) * F
        v = a.b2 * st["a1"][jn] + (1 - a.b2) * F ** 2
        st["a0"] = st["a0"].at[jn].set(m)
        st["a1"] = st["a1"].at[jn].set(v)
        t = epoch + 1
        mh = m / (1 - a.b1 ** t)
        vh = v / (1 - a.b2 ** t)
        return Z.at[jn].add(lr * mh / (jnp.sqrt(vh) + 1e-8)), st

    if rule == "fa2":
        first = st["seen"][jn][:, None]
        prev = st["a0"][jn]
        swing = jnp.linalg.norm(F - prev, axis=1, keepdims=True)
        tract = jnp.linalg.norm(F + prev, axis=1, keepdims=True) / 2.0
        speed = jnp.minimum(a.k_s * tract / (1.0 + jnp.sqrt(swing)), a.k_max)
        step = jnp.where(first, speed * F, F)   # first touch: plain step
        st["a0"] = st["a0"].at[jn].set(F)
        st["seen"] = st["seen"].at[jn].set(True)
        return Z.at[jn].add(lr * step), st

    raise ValueError(rule)


# Pair counts differ from batch to batch, and a new shape means a new
# compile. Rounding the pair axis up to a multiple of this keeps the number
# of distinct shapes small: a handful of compiles for the whole run instead
# of one for every batch.
PAD_TO = 1 << 16


def pad_to(a, m, fill):
    """`a` grown to length `m` with `fill`. `m >= a.size` by construction."""
    out = np.full(m, fill, dtype=a.dtype)
    out[:a.size] = a
    return out


@functools.partial(jax.jit, static_argnums=(7,))
def row_forces(Z, u_glob, u_loc, v, h, freq, inv_deg, b):
    """`dZ` for `b` rows, from a flat pair list. The projection of `step`.

    Reproduced from `forcedirected.sell_c_sigma.step`: the law gives a
    MAGNITUDE along `u -> v`, this divides by `x` to project it onto
    `Zdiff`, guards `x == 0`, sums over the partners of a row, and divides
    by the degree. A pad entry carries `h = 0`, thus `fdlinear`'s `live`
    test zeroes it, exactly as a pad cell of the plan is zeroed.
    """
    Zu = Z[u_glob]
    Zdiff = Z[v] - Zu                                    # (m, d)
    x = jnp.sqrt(jnp.sum(Zdiff * Zdiff, axis=-1))        # (m,)
    x_safe = jnp.where(x == 0, 1.0, x)
    mag = fdlinear(x, (h, freq), PARAMS)                 # the law, verbatim
    scale = jnp.where(x == 0, 0.0, mag / x_safe)
    F = jax.ops.segment_sum(Zdiff * scale[:, None], u_loc, num_segments=b)
    return F * inv_deg[:, None]


def main():
    t0 = time.perf_counter()
    A, n = load(args.graph, max_nodes=args.max_nodes, seed=args.seed)
    t_load = time.perf_counter() - t0
    log(f"{args.graph}: n={n:,} nodes, {A.nnz // 2:,} undirected edges "
        f"({t_load:.1f}s, RSS {rss_mb():.0f} MB)")
    log(f"streaming: {args.walks} walks x {args.walk_len} steps per node, "
        f"batch={args.batch}, dim={args.dim}, {args.epochs} epochs, "
        f"lr={args.lr}. D is NEVER built.")

    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    Z = jax.random.normal(key, (n, args.dim), dtype=jnp.float32)
    log(f"Z allocated: {n:,} x {args.dim} float32 = "
        f"{n * args.dim * 4 / 2 ** 20:.0f} MB (RSS {rss_mb():.0f} MB)")

    # `m = beta*m + alpha*dZ`, held for every row, updated for the batch
    # only. The DC gain is alpha/(1-beta): at alpha=1, beta=0.9 that is 10,
    # thus the classic rule multiplies the step by ten and diverges at
    # lr = 1.0. See the effective-lr note in the module docstring.
    # The DC gain of each rule. `nesterov` accumulates like momentum at
    # alpha = 1, so its gain is 1/(1-beta) too. `adam`, `fa2` and `sqn`
    # rescale adaptively and have no fixed gain -- reported as 1.0 and
    # marked, not claimed.
    gain = {"momentum": args.alpha / (1.0 - args.beta),
            "nesterov": 1.0 / (1.0 - args.beta)}.get(args.optim, 1.0)
    eff = args.lr * gain
    log(f"optim={args.optim}"
        + (f" alpha={args.alpha} beta={args.beta} (alpha+beta="
           f"{args.alpha + args.beta:.2f})" if args.optim == "momentum" else "")
        + f", lr={args.lr}, DC gain={gain:.4f}, EFFECTIVE lr={eff:.4f}"
        + f"  -> predicted {'STABLE' if eff <= 1.0 else 'DIVERGENT'}"
          f" (the edge is 1.0)")
    STATE = make_state(args.optim, n, args.dim)

    t_embed0 = time.perf_counter()
    pairs_seen = 0
    diverged = False
    for ep in range(args.epochs):
        te = time.perf_counter()
        # lr_t = lr * (1 - epoch/epochs). At the last epoch this is
        # lr/epochs and not 0, so the final step still moves.
        lr_t = (args.lr * (1.0 - ep / max(1, args.epochs))
                if args.lr_decay == "linear" else args.lr)
        order = np.arange(n, dtype=np.int64)
        ep_pairs = 0
        for s in range(0, n, args.batch):
            nodes = order[s:s + args.batch]
            b = nodes.size
            u_loc, v, h, freq = rows_of(A, nodes, n, args.walks,
                                        args.walk_len, rng)
            ep_pairs += u_loc.size
            # the degree divisor: the h == 1 entries of the row, as
            # `embed.forces.degrees_from_D` counts them
            deg = np.bincount(u_loc[h == 1], minlength=b).astype(np.float32)
            inv_deg = np.where(deg == 0, 0.0, 1.0 / np.maximum(deg, 1.0))
            # pad the pair axis to a bucket boundary. A pad entry carries
            # h = 0, thus `fdlinear`'s `live` test zeroes it, and its
            # partner is its own row, thus `Zdiff` is 0 too -- the same
            # double safety the plan's pad cells have.
            m = ((u_loc.size + PAD_TO - 1) // PAD_TO) * PAD_TO
            u_g = pad_to(nodes[u_loc], m, nodes[0])
            F = row_forces(Z, jnp.asarray(u_g),
                           jnp.asarray(pad_to(u_loc, m, 0)),
                           jnp.asarray(pad_to(v, m, nodes[0])),
                           jnp.asarray(pad_to(h, m, 0.0)),
                           jnp.asarray(pad_to(freq, m, 1.0)),
                           jnp.asarray(inv_deg), b)
            # THE IMMEDIATE UPDATE. Row u moves before row u+1 is computed.
            Z, STATE = apply_update(args.optim, Z, STATE,
                                    jnp.asarray(nodes), F, lr_t, ep, args)
        pairs_seen += ep_pairs
        if not bool(jnp.all(jnp.isfinite(Z))):
            log(f"DIVERGED at epoch {ep + 1}: Z holds "
                f"{int(jnp.sum(~jnp.isfinite(Z))):,} non-finite values")
            diverged = True
            break
        log(f"epoch {ep + 1}/{args.epochs}: {ep_pairs:,} pairs "
            f"({time.perf_counter() - te:.1f}s, RSS {rss_mb():.0f} MB, "
            f"peak {_peak[0]:.0f} MB)")
    t_embed = time.perf_counter() - t_embed0

    log(f"TOTAL {t_load + t_embed:.1f}s (load {t_load:.1f} + "
        f"embed {t_embed:.1f}), {pairs_seen:,} pairs used and discarded")
    dz = float(jnp.mean(jnp.linalg.norm(lr_t * F, axis=1))) if not diverged \
        else float("nan")
    zmax = float(jnp.max(jnp.abs(Z))) if not diverged else float("nan")
    log(f"peak RSS after embed: {_peak[0]:.0f} MB")
    log(f"dz (mean row-norm of the last update) = {dz:.6f},  "
        f"max|Z| = {zmax:.3f},  diverged={diverged}")
    scores = {}
    if args.score and not diverged:
        import evaluator
        t = time.perf_counter()
        Znp = np.asarray(Z)
        lp = evaluator.link_prediction(A, Znp, protocol=args.protocol,
                                       seed=args.seed,
                                       max_pairs=args.lp_max_pairs)
        t_lp = time.perf_counter() - t
        log(f"lp: {lp.sizes['pairs']:,} pairs, "
            f"protocol_modified={lp.protocol_modified} ({t_lp:.1f}s)")
        for k, v in lp.scores.items():
            log(f"  {k:10s}: {v:.4f}")
        scores.update({k: v for k, v in lp.scores.items()})
        t = time.perf_counter()
        da = evaluator.dist_approx(A, Znp, protocol=args.protocol,
                                   seed=args.seed)
        t_da = time.perf_counter() - t
        log(f"da: {da.sizes['n_pairs']:,} pairs, "
            f"protocol_modified={da.protocol_modified} ({t_da:.1f}s)")
        for m, sc in da.scores.items():
            log(f"  {m:9s} r2={sc['r2']:+.4f} mae={sc['mae']:.3f} "
                f"rmse={sc['rmse']:.3f} exact={sc.get('exact', 0):.3f}")
        for m, sc in da.scores.items():
            scores[f"{m}_r2"] = sc["r2"]
            scores[f"{m}_mae"] = sc["mae"]
        scores["t_lp"] = t_lp
        scores["t_da"] = t_da
        _peak[0] = max(_peak[0], rss_mb())

    # the sampler runs THROUGH scoring: the hop-distance BFS is
    # part of the run and part of its peak.
    _stop[0] = True
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    log(f"peak RSS {_peak[0]:.0f} MB (statm), {ru:.0f} MB (ru_maxrss)"
        f" -- embedding AND scoring")

    line = (f"STREAM\tgraph={args.graph}\tn={n}\tdim={args.dim}\t"
            f"walks={args.walks}\twalk_len={args.walk_len}\t"
            f"batch={args.batch}\tepochs={args.epochs}\t"
            f"t_load={t_load:.1f}\tt_embed={t_embed:.1f}\t"
            f"t_total={t_load + t_embed:.1f}\tpairs={pairs_seen}\t"
            f"peak_rss={_peak[0]:.0f}\tedges={A.nnz // 2}\t"
            f"lr={args.lr}\tseed={args.seed}\tdevice={args.device}\t"
            f"tag={args.tag or args.graph}\toptim={args.optim}\t"
            f"alpha={args.alpha}\tbeta={args.beta}\tgain={gain:.4f}\t"
            f"eff_lr={eff:.4f}\tdz={dz:.6f}\tdiverged={int(diverged)}\t"
            f"lr_decay={args.lr_decay}\tmaxZ={zmax:.4g}"
            + "".join(f"\t{k}={v:.4f}" if isinstance(v, float) else
                      f"\t{k}={v}" for k, v in scores.items()))
    log(line)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")
    np.save(f"/tmp/stream_Z_{args.graph}_{args.dim}.npy", np.asarray(Z))
    log(f"Z saved to /tmp/stream_Z_{args.graph}_{args.dim}.npy")


main()
