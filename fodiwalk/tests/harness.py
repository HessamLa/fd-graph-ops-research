"""harness.py -- one recorded run, end to end, on the package.

`run(**cfg)` does what `archive/fdwalk/bench_fdwalk.py` does, in the
same ORDER and with the same generator, and it gives back the fields of the
`RESULT` line. `test_parity.py` compares those fields to the numbers that
`archive/fdwalk/RESULTS.md` and `FINDINGS.md` recorded.

THE ORDER OF THE GENERATOR IS PART OF THE PARITY. The script makes ONE
`np.random.default_rng(seed)` and it passes that one generator to the
augmentation, then to the link prediction, then to the hop sample. A run
that makes a second generator, or that calls them in another order, gives
different pairs and a different number -- and the difference would look
like a defect of the physics. `Fodiwalk` keeps that generator as
`self.rng`, thus this harness passes `fw.rng` to the evaluation, after the
augmentation has advanced it.

THE SCORES COME FROM `evaluator/`, which is the one scoring package of
this repository. The protocols `fodiwalk_dist` and `fodiwalk_vec` hold
the settings that `fodiwalk/misc/evaluation.py` used, thus the numbers do
not move. Checked on cora, at the settings of `ab.py` and at the settings
of the protocol: every score, every count and the ten draws taken after
the run are identical on both paths.
"""
from __future__ import annotations

import time

import numpy as np

import evaluator as ev
from fodiwalk import Fodiwalk


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

    # The graph and the embedding are already in the memory, thus they go
    # to `evaluator` as the loaded forms and no file is read again.
    graph = (A, n, {"source": "<memory>", "n": n,
                    "n_edges": int(A.nnz // 2)})
    emb = (Z, {"source": "<memory>"})
    lp = ev.link_prediction(graph, emb, protocol="fodiwalk_dist", seed=seed,
                            rng=fw.rng, max_pairs=lp_pairs)

    # ONE hop sample, read two ways. The state of the generator is put
    # back before the second protocol, thus both feature widths score the
    # SAME pairs -- which is what one `hop_sample` and two `task_hop`
    # calls did -- and the generator ends where one sample leaves it.
    hop_state = fw.rng.bit_generator.state
    best, hop_pairs_kept = {}, 0
    for feature, protocol in (("distance", "fodiwalk_dist"),
                              ("vector", "fodiwalk_vec")):
        fw.rng.bit_generator.state = hop_state
        hop = ev.dist_approx(graph, emb, protocol=protocol, seed=seed,
                             rng=fw.rng, n_sources=hop_sources,
                             n_pairs=hop_pairs, min_hop=hop_min)
        best[feature] = (hop.scores["mlp"]["r2"], hop.scores["mlp"]["mae"])
        hop_pairs_kept = int(hop.sizes["n_pairs"])

    out.update(acc=lp.scores["accuracy"], f1_score=lp.scores["f1_score"],
               auc=lp.scores["auc"], r2_dist=best["distance"][0],
               mae_dist=best["distance"][1], r2_vec=best["vector"][0],
               lp_pairs=int(lp.sizes["pairs"]), hop_pairs=hop_pairs_kept)
    return out


def result_line(out) -> str:
    """The fields a log holds, in one line. For a report, and for a diff."""
    keys = ("n", "dnnz", "dz", "acc", "f1_score", "auc", "r2_dist", "mae_dist",
            "r2_vec", "diverged")
    return "\t".join(f"{k}={out[k]:.4f}" if isinstance(out.get(k), float)
                     else f"{k}={out.get(k)}" for k in keys)
