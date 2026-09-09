"""dz_oracle.py -- the before/after gate for moving `1/deg(u)` into the laws.

The kernel divides the row sum of the force by `deg(u)`
(`forcedirected/sell_c_sigma.py:485`). That divisor is force-law policy, so
it moves into `fodiwalk/embed/forces.py`. Division is linear and it
distributes over the sum, thus the answer must not change:

    F_u / deg(u) = SUM_v [ f_uv(x) / deg(u) ] * unit(z_uv)

Only the float32 rounding ORDER changes. This script measures how much.

Use
---
    .venv/bin/python experiments/refactor-deg/dz_oracle.py --save
    ... edit a law ...
    .venv/bin/python experiments/refactor-deg/dz_oracle.py --check

`--save` writes one `dZ` for each (graph, law) case to
`data_cache/refactor-deg/dz_baseline.npz`. `--check` recomputes and prints
the max absolute and max relative difference of each case. It exits 1 when
a case is over `--rtol` (default 1e-6, the tolerance the owner set).

It calls the JITTED STEP DIRECTLY and never `Fodiwalk.forces`, because
`forces` applies `drop_steady_rate`, which is random.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp

from fodiwalk import Fodiwalk

OUT = "data_cache/refactor-deg/dz_baseline.npz"

# Every law of the registry. `fdlinear_fused` is reached through
# `fuse_planes`, not through `force`, thus it carries its own row.
CASES = [
    dict(tag="fdlinear", force="fdlinear"),
    dict(tag="fdlinear_fused", force="fdlinear", fuse_planes=True),
    dict(tag="fdhop", force="fdhop"),
    dict(tag="fdhop2", force="fdhop2"),
    dict(tag="fdhop_min", force="fdhop_min"),
    dict(tag="fdhop_all", force="fdhop_all", k2=2.0),
    dict(tag="fdhop_all_freq", force="fdhop_all_freq", k1=1.0, k2=6.0),
    # `no_deg_norm` sends a degree of 1 to every row, thus the divisor is
    # the identity. It must stay the identity after the move.
    dict(tag="fdhop_nodeg", force="fdhop", no_deg_norm=True),
]


def tiny_graph(n: int = 60, seed: int = 0):
    """A ring plus random chords: connected, and the degrees differ."""
    rng = np.random.default_rng(seed)
    r = np.arange(n)
    src = np.concatenate([r, r, rng.integers(0, n, 3 * n)])
    dst = np.concatenate([(r + 1) % n, (r + 7) % n, rng.integers(0, n, 3 * n)])
    keep = src != dst
    src, dst = src[keep], dst[keep]
    A = sp.csr_matrix((np.ones(src.size, np.float64), (src, dst)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64).tocsr()
    A.sort_indices()
    return A


def cora():
    from fodiwalk.make_graph.datasets import read_edges, to_csr
    A, _n = to_csr(read_edges("cora"))
    return A


GRAPHS = {"tiny": tiny_graph, "cora": cora}


def one_dz(A, case, d: int = 128, seed: int = 7):
    """`dZ` of ONE full kernel pass, for one law, on a fixed `Z`.

    The walk runs for each case with the same seed, thus `D` and `freq` are
    the same for every law and only the physics differs.
    """
    kw = {k: v for k, v in case.items() if k != "tag"}
    fw = Fodiwalk(n_dim=d, lr=0.999, seed=seed, **kw)
    fw.augment_graph(A)
    n = A.shape[0]
    Z = jax.random.normal(jax.random.PRNGKey(11), (n, d), dtype=jnp.float32)
    dZ = fw.steps[0](Z, fw.plans[0], fw.inv_deg_ext, fw.params)
    return np.asarray(jax.device_get(dZ)), fw


def collect(graphs, cases):
    out = {}
    for gname in graphs:
        A = GRAPHS[gname]()
        for case in cases:
            key = f"{gname}::{case['tag']}"
            dZ, _ = one_dz(A, case)
            out[key] = dZ
            fin = np.isfinite(dZ).all()
            print(f"  {key:34s} shape={dZ.shape} "
                  f"|dZ|max={np.abs(dZ).max():.6e} finite={fin}")
            if not fin:
                print(f"  !! {key} is NOT finite -- the case is unusable")
    return out


def compare(new: dict, ref: dict, rtol: float) -> int:
    """Print one line for each case. Return the count of cases over `rtol`."""
    bad, exact = 0, 0
    keys = sorted(set(new) | set(ref))
    print(f"{'case':36s} {'max abs':>12s} {'max rel':>12s}  verdict")
    for k in keys:
        if k not in new or k not in ref:
            print(f"{k:36s} {'--':>12s} {'--':>12s}  MISSING")
            bad += 1
            continue
        a, b = new[k], ref[k]
        if a.shape != b.shape:
            print(f"{k:36s} {'--':>12s} {'--':>12s}  SHAPE {a.shape}!={b.shape}")
            bad += 1
            continue
        d = np.abs(a - b)
        # The denominator is the reference row-force magnitude, not the
        # element: an element near 0 inside a large row is not evidence.
        scale = np.maximum(np.abs(b).max(axis=1, keepdims=True), 1e-30)
        rel = float((d / scale).max())
        amax = float(d.max())
        if np.array_equal(a, b):
            verdict, exact = "BIT-IDENTICAL", exact + 1
        elif rel <= rtol:
            verdict = f"ok (<= {rtol:g})"
        else:
            verdict, bad = "OVER TOLERANCE", bad + 1
        print(f"{k:36s} {amax:12.4e} {rel:12.4e}  {verdict}")
    print(f"\n{exact}/{len(keys)} bit-identical, {bad} over tolerance")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--graphs", default="tiny,cora")
    ap.add_argument("--cases", default="",
                    help="comma-separated tags; default is every case")
    a = ap.parse_args()

    graphs = [g for g in a.graphs.split(",") if g]
    tags = [t for t in a.cases.split(",") if t]
    cases = [c for c in CASES if not tags or c["tag"] in tags]

    print(f"graphs={graphs} cases={[c['tag'] for c in cases]}")
    new = collect(graphs, cases)

    if a.save:
        np.savez(a.out, **new)
        print(f"\nsaved {len(new)} cases -> {a.out}")
        return 0
    if a.check:
        ref = dict(np.load(a.out))
        ref = {k: v for k, v in ref.items() if k in new}
        print()
        bad = compare(new, ref, a.rtol)
        return 1 if bad else 0
    print("nothing to do: pass --save or --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
