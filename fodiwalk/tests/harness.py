"""harness.py -- one recorded run, end to end, on the package.

`run(**cfg)` does what `experiments/fdwalk/bench_fdwalk.py` does, in the
same ORDER and with the same generator, and it gives back the fields of the
`RESULT` line. `test_parity.py` compares those fields to the numbers that
`experiments/fdwalk/RESULTS.md` and `FINDINGS.md` recorded.

THE ORDER OF THE GENERATOR IS PART OF THE PARITY. The script makes ONE
`np.random.default_rng(seed)` and it passes that one generator to the
augmentation, then to the link prediction, then to the hop sample. A run
that makes a second generator, or that calls them in another order, gives
different pairs and a different number -- and the difference would look
like a defect of the physics. `Fodiwalk` keeps that generator as
`self.rng`, thus this harness passes `fw.rng` to the evaluation, after the
augmentation has advanced it.
"""
from __future__ import annotations

import time

import numpy as np

from fodiwalk import Fodiwalk
from fodiwalk.augment_graph import pairs as PR
from fodiwalk.misc.evaluation import link_prediction, hop_sample, task_hop


def run(A, n, *, dim=64, epochs=200, lr=1.0, seed=42, optim="plain",
        lp_pairs=50_000, hop_sources=200, hop_pairs=20_000, hop_min=2,
        batch_count=None, verbosity=0, **cfg):
    """One run: augment, embed, then the two measurements.

    Returns the fields of the `RESULT` line as a dict. A diverged run
    returns `nan` for every measurement and `diverged=True` -- it does NOT
    raise, which is I7.
    """
    fw = Fodiwalk(n_dim=dim, lr=lr, seed=seed, optim=optim,
                  verbosity=verbosity, **cfg)
    chunks = fw.cfg.chunks if batch_count is None else batch_count

    t0 = time.time()
    fw.embed(A, epochs=epochs, batch_count=chunks)
    Z = fw.get_embeddings()
    t_embed = time.time() - t0

    out = {"n": n, "dnnz": int(fw.D.nnz), "t_embed": t_embed,
           "dz": float(fw.Th(fw.dZ)), "diverged": bool(fw.diverged),
           "plan": fw.plan_stats, "info": fw.info}
    if fw.diverged:
        out.update(acc=np.nan, f1=np.nan, auc=np.nan, r2_dist=np.nan,
                   mae_dist=np.nan, r2_vec=np.nan, dz=np.nan)
        return out

    scores, lp = link_prediction(Z, A, n, lp_pairs, fw.rng, seed)
    # `gap_csr` measures H2, the walk gap against the true distance. Every
    # policy is walk-based, thus every run has one.
    gap_csr = PR.to_csr(fw.stats["key"], fw.stats["mn"].astype(np.float64), n)
    u, v, d, h2 = hop_sample(A, n, fw.rng, hop_sources, hop_pairs, gap_csr)
    keep = d >= hop_min                 # a neighbour is the easy case
    u, v, d = u[keep], v[keep], d[keep]

    best = {}
    for feature in ("distance", "vector"):
        for name, mae, mre, rmse, r2, ex, secs in task_hop(
                Z, u, v, d, seed, feature):
            if name == "MLP":
                best[feature] = (r2, mae)

    out.update(acc=scores["accuracy"], f1=scores["f1-score"],
               auc=scores["auc"], r2_dist=best["distance"][0],
               mae_dist=best["distance"][1], r2_vec=best["vector"][0],
               lp_pairs=lp["pairs"], hop_pairs=int(u.size))
    return out


def result_line(out) -> str:
    """The fields a log holds, in one line. For a report, and for a diff."""
    keys = ("n", "dnnz", "dz", "acc", "f1", "auc", "r2_dist", "mae_dist",
            "r2_vec", "diverged")
    return "\t".join(f"{k}={out[k]:.4f}" if isinstance(out.get(k), float)
                     else f"{k}={out.get(k)}" for k in keys)
