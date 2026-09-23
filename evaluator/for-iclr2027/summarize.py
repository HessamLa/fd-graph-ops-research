"""Group the ICLR 2027 runs into cells and report mean and std over seeds.

Usage:
    .venv/bin/python data_cache/evaluator/for-iclr2027/summarize.py [--csv FILE]

A cell is one method setting on one graph at one dimension. Its seeds are the
repeats. The owner's rule: 11 seeds for a small or medium graph, 3 for a very
large one (com_youtube and bigger). A cell with fewer seeds is printed with
its real count, never silently.

Blow-up flag: FROZEN at 10x on 2026-09-21 and not to be moved again.

Unit: per CELL, not per seed. The cell's median row norm (median over its
seeds of each run's mean row norm) against the median of the reference
cell, which is the same graph, dim and force setting under
plain / const / lr 0.999. `ratio` records it and `ratio_max` records the
worst single seed, so a cell whose median sits under the line while one
seed sits over it can still be seen.

The threshold was set on cora, where it is safe: at dim 64 the largest
healthy ratio is 2.08x and the smallest blown ratio 34.6x, so anything from
3x to 30x gives the same split. It is NOT safe to call the split
threshold-insensitive in general: on pubmed at dim 64, sqn lr 0.02 seed 56
reaches 18.9x while momentum lr 0.5 reaches 26 to 30x, so 10x and 20x
disagree there (Metrologist, measured 2026-09-21).

The threshold is frozen rather than tuned because the pubmed norms were
read on 2026-09-17, before any cut-off existed, so no one can now choose
one blind on that graph. Moving it would be a data-driven choice.

The mark is a LABEL, not a filter. Every marked cell keeps its row and its
rho, and nothing is dropped from any mean or ranking. If a marked cell is
ever excluded from a number, the threshold becomes a data-selection knob
and then needs pre-registration and the effect shown both ways.
"""
import csv, json, sys
from pathlib import Path

import numpy as np

REPO = Path("/home/h/gnn/fd-graph-embedding/fdmap")
STORE = REPO / "experiments/for-iclr2027/embeddings"
BLOWUP = 10.0


def cell_of(cfg):
    """A short label for the method setting, without the seed."""
    m = cfg["method"]["name"]
    o = cfg.get("optimizer", {})
    # The walk budget and the window must be in the label: Runner sweeps
    # both for the baselines, and one label for all of them merges cells
    # that differ by a factor of four in sampled steps per node.
    if m.startswith("node2vec") or m == "deepwalk":
        w = cfg.get("walker", {})
        budget = f"{w.get('walks')}x{w.get('walk_len')} win{w.get('window')}"
        name = "deepwalk" if m == "deepwalk" else f"node2vec q={w.get('q')}"
        return f"{name} {budget}", "-"
    w, f = cfg.get("walker", {}), cfg.get("force", {})
    p = f.get("params", {})
    bits = [f["law"], w["name"], w.get("weight", "?"),
            f"{w.get('walks')}x{w.get('walk_len')}"]
    for k in ("k2", "k4", "kr"):
        if p.get(k) is not None:
            bits.append(f"{k}={p[k]:g}")
    if w.get("q") not in (None, 1.0):
        bits.append(f"q={w['q']:g}")
    rule = f"{o.get('rule')}/{o.get('lr_decay')}/lr={o.get('lr'):g}"
    return " ".join(bits), rule


def load():
    rows = []
    for ev_path in sorted(STORE.glob("*/*/*/evaluation.json")):
        run = ev_path.parent
        cfg = json.loads((run / "config.json").read_text())
        ev = json.loads(ev_path.read_text())
        cell, rule = cell_of(cfg)
        Z = np.load(run / "Z.npy", mmap_mode="r")
        rows.append({
            "graph": cfg["dataset"]["name"], "n": cfg["dataset"]["n"],
            "dim": cfg["n_dim"], "cell": cell, "optim": rule,
            "seed": cfg["run"]["seed"], "run": run.name,
            "norm": float(np.linalg.norm(np.asarray(Z), axis=1).mean()),
            "acc": ev["link_prediction"]["accuracy"],
            "f1_score": ev["link_prediction"]["f1_score"],
            "auc": ev["link_prediction"]["auc"],
            "rec10": ev["knn"]["neighbour_recall_at_10"],
            "rho": ev["rank"]["spearman"],
            "tau_b": ev["rank"]["kendall_tau_b"],
            "stress1": ev["stress"]["stress1"],
            "rf_r2": ev["dist_approx"]["rf"]["r2"],
            "rf_mae": ev["dist_approx"]["rf"]["mae"],
            "ckpts": len(ev.get("checkpoints", {})),
        })
    return rows


METRICS = ("acc", "f1_score", "auc", "rec10", "rho", "tau_b", "stress1", "rf_r2", "rf_mae")


def summarize(rows):
    cells = {}
    for r in rows:
        cells.setdefault((r["graph"], r["dim"], r["cell"], r["optim"]), []).append(r)
    # The reference is the SAME force setting under plain / const / lr 0.999.
    # A norm must not be compared across force settings: k4 = 0.01 gives a
    # norm near 236 on cora d128 with no instability, because weak decay
    # spreads the layout. Only the optimizer changes within a reference group.
    ref = {}
    for (g, d, c, o), v in cells.items():
        if o == "plain/const/lr=0.999":
            ref[(g, d, c)] = float(np.median([x["norm"] for x in v]))
    out = []
    for (g, d, c, o), v in sorted(cells.items()):
        rec = {"graph": g, "dim": d, "cell": c, "optim": o, "seeds": len(v),
               "n": v[0]["n"], "ckpts": max(x["ckpts"] for x in v)}
        base = ref.get((g, d, c))
        norm = float(np.median([x["norm"] for x in v]))
        rec["norm"] = norm
        # ratio: the cell against its reference. ratio_max: the worst single
        # seed, which shows a cell sitting inside the disputed band.
        rec["ratio"] = round(norm / base, 2) if base else ""
        rec["ratio_max"] = round(max(x["norm"] for x in v) / base, 2) if base else ""
        rec["blown_up"] = bool(base and norm > BLOWUP * base)
        for m in METRICS:
            a = np.array([x[m] for x in v], dtype=float)
            rec[m + "_mean"], rec[m + "_std"] = float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0
        out.append(rec)
    return out


def pm(rec, m, nd=3):
    return f"{rec[m + '_mean']:.{nd}f} ± {rec[m + '_std']:.{nd}f}"


def main():
    rows = load()
    cells = summarize(rows)
    if "--csv" in sys.argv:
        path = sys.argv[sys.argv.index("--csv") + 1]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(cells[0].keys()))
            w.writeheader()
            w.writerows(cells)
        print(f"{len(cells)} cells -> {path}")
    for c in cells:
        print(f"{c['graph']:9} d{c['dim']:<4}{c['seeds']:3} seeds  {c['cell']:44} "
              f"{c['optim']:26} rho {pm(c, 'rho')}  auc {pm(c, 'auc')}"
              f"{'  BLOWN UP' if c['blown_up'] else ''}")


main()
