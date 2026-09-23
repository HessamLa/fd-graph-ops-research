#!/bin/env python3
"""Score one embedding and save it in the shared store.

Every method under `other-methods/` calls `save_scored` with its `Z`. This
gives ONE scoring path and ONE record layout, so a number from a new method
compares to a fodiwalk, deepwalk or node2vec number with no adjustment.

The record matches what `experiments/embeddings/make_embedding.py` writes
(same `dataset`, `run`, `optimizer`, `metrics` keys), thus the existing
score scripts under `agentic-log/recall-and-rho/` and the report builder
`other-methods/score_report.py` read it with no change.

The graph loads through `fodiwalk.make_graph.load`, so row `i` of `Z` is
the same node for every method. Scoring uses the `fodiwalk_dist` protocol
and `lp_max_pairs=50_000`, the SAME as this session's fodiwalk and baseline
runs -- one report, one protocol.

Import discipline: numpy, plus `evaluator` and `fodiwalk.make_graph` at
call time (not import time), so a runner can build `Z` in an environment
where those are absent and hand it here in the shared `.venv`.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

ROOT = "/home/h/gm/fd-graph-ops-research"
STORE = os.path.join(ROOT, "data_cache", "embeddings")
PROTOCOL = "fodiwalk_dist"
LP_MAX_PAIRS = 50_000


def load_graph(graph, max_nodes=0, seed=42):
    """The graph as a symmetric CSR, node ids `0..n-1`. One numbering for
    every method (`fodiwalk.make_graph.load`)."""
    import sys
    sys.path.insert(0, ROOT)
    from fodiwalk.make_graph import load
    return load(graph, max_nodes, seed)


def save_scored(Z, *, graph, method, seed, A=None, n=None,
                seconds=0.0, peak_rss_mb=0.0, params=None, notes=""):
    """Score `Z` and write `Z.npy` + `config.json` under
    `data_cache/embeddings/<graph>/<dim>/<name>/`.

    `method`  the label a table groups on. It is written to BOTH
              `method.name` and `optimizer.rule`, so the old scripts (which
              group on `optimizer.rule`) and the report builder (which reads
              `method.name`) agree.
    `A`, `n`  pass them if the caller already has the graph, to skip a
              reload; else this loads it.

    Returns the run directory. Raises on a non-finite `Z`: a record that
    cannot be scored is never stored.
    """
    import sys
    sys.path.insert(0, ROOT)
    import evaluator

    Z = np.asarray(Z, dtype=np.float32)
    if not np.isfinite(Z).all():
        raise RuntimeError(
            f"{method}/{graph}/seed{seed}: Z holds non-finite values; "
            f"nothing stored.")
    if A is None or n is None:
        A, n = load_graph(graph, seed=seed)

    lp = evaluator.link_prediction(A, Z, protocol=PROTOCOL, seed=seed,
                                   max_pairs=LP_MAX_PAIRS)
    da = evaluator.dist_approx(A, Z, protocol=PROTOCOL, seed=seed)
    rename = {"f1": "f1_score", "f1-score": "f1_score"}
    metrics = {"protocol": PROTOCOL,
               "protocol_modified": bool(lp.protocol_modified
                                         or da.protocol_modified),
               "lp_max_pairs": LP_MAX_PAIRS}
    metrics.update({rename.get(k, k): float(v) for k, v in lp.scores.items()})
    for m, sc in da.scores.items():
        metrics[f"{m}_r2"] = float(sc["r2"])
        metrics[f"{m}_mae"] = float(sc["mae"])

    dim = int(Z.shape[1])
    now = time.gmtime()
    cfg = {
        "schema_version": "1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", now),
        "datetime": time.strftime("%y%m%d-%H%M%S", now),
        "dataset": {"name": graph, "loader": "fodiwalk.make_graph.load",
                    "n": int(n), "edges": int(A.nnz // 2),
                    "avg_degree": round(float(A.nnz / n), 4), "max_nodes": 0},
        "n_dim": dim,
        # `optimizer.rule` carries the method label ONLY so the older score
        # scripts, which group on that key, keep working. It is not a real
        # optimiser; a baseline method has no fodiwalk optimiser.
        "method": {"name": method, "entry": "other-methods/",
                   "package": (params or {}).get("package", ""),
                   "version": (params or {}).get("version", "")},
        "optimizer": {"rule": method, "lr_decay": "n/a", "lr": 0.0},
        "run": {"epochs": (params or {}).get("epochs", 0), "seed": int(seed),
                "device": "cpu"},
        "environment": {"peak_rss_mb": round(peak_rss_mb, 1),
                        "seconds": round(seconds, 1)},
        "params": params or {},
        "notes": notes,
        "metrics": metrics,
    }
    blob = json.dumps(cfg, sort_keys=True).encode()
    h = hashlib.sha256(blob).hexdigest()[:8]
    name = f"{cfg['datetime']}-{method}-{h}"
    out = os.path.join(STORE, graph, str(dim), name)
    os.makedirs(out, exist_ok=True)
    zp = os.path.join(out, "Z.npy")
    np.save(zp, Z)
    cfg["outputs"] = {"file": "Z.npy", "shape": list(Z.shape),
                      "dtype": str(Z.dtype),
                      "sha256": hashlib.sha256(open(zp, "rb").read()).hexdigest()}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    line = "  ".join(f"{k}={metrics[k]:.4f}" for k in
                     ("accuracy", "auc", "f1_score", "rf_r2")
                     if k in metrics)
    print(f"[{method}] {graph} seed={seed} dim={dim}  {line}", flush=True)
    print(f"[{method}] stored {os.path.relpath(out, ROOT)}", flush=True)
    return out
