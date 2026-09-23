"""Print the markdown tables of the preliminary report from cells.csv."""
import csv, sys

ROWS = list(csv.DictReader(open("data_cache/evaluator/for-iclr2027/cells.csv")))
# The optimizer study uses one force setting; every other cell must be kept
# out or the plain/lr=0.999 row picks up the nbr_walk cell instead.
STUDY = "fdhop walk_edges min_gap 10x20 k4=1 kr=1"
COLS = [("rho", "Spearman rho", 3), ("rf_r2", "hop R2 (rf)", 3),
        ("rec10", "recall@10", 3), ("auc", "LP AUC", 4),
        ("f1_score", "LP f1_score", 4)]


def pick(graph, dim, keep):
    out = [r for r in ROWS if r["graph"] == graph and r["dim"] == str(dim) and keep(r)]
    return sorted(out, key=lambda r: -float(r["rho_mean"]))


def table(rows, label="cell"):
    head = "| " + label + " | seeds | " + " | ".join(c[1] for c in COLS) + " |"
    print(head)
    print("|" + "---|" * (len(COLS) + 2))
    for r in rows:
        name = r["cell"] if label == "cell" else r["optim"]
        flag = " **blown up**" if r["blown_up"] == "True" else ""
        cells = " | ".join(f"{float(r[c + '_mean']):.{nd}f} ± {float(r[c + '_std']):.{nd}f}"
                           for c, _, nd in COLS)
        print(f"| {name}{flag} | {r['seeds']} | {cells} |")
    print()


# The baselines' headline budget: deepwalk 80x40 window 10, node2vec 10x80
# window 10. Other budgets are the matched-budget study, table "budget".
MAIN = ("deepwalk 80x40 win10", "node2vec q=0.5 10x80 win10",
        "node2vec q=1.0 10x80 win10", "node2vec q=2.0 10x80 win10",
        "fdlinear walk_edges min_gap 10x20 k4=0.01 kr=1",
        "fdhop walk_edges min_gap 10x20 k4=1 kr=1",
        "fdhop nbr_walk min_gap 10x20 k4=1 kr=1")
LAWS = ("fdhop", "fdhop2", "fdhop_min", "fdhop_all ", "fdhop_all_freq")

GRAPHS = [g for g in ("cora", "pubmed", "wordnet", "com_youtube",
                      "roadnet_ca", "ncbi_taxonomy", "as_skitter")
          if any(r["graph"] == g for r in ROWS)]

what = sys.argv[1]
if what == "main":
    for g in GRAPHS:
        print(f"**{g}, dim 128**\n")
        table(pick(g, 128, lambda r: r["cell"] in MAIN))
elif what == "dims":
    for g in GRAPHS:
        print(f"**{g}**\n")
        print("| dim | " + " | ".join(("deepwalk", "node2vec q=0.5", "fdhop walk_edges",
                                       "fdhop nbr_walk")) + " |")
        print("|" + "---|" * 5)
        for d in (128, 64, 32, 16):
            vals = []
            for c in ("deepwalk 80x40 win10", "node2vec q=0.5 10x80 win10",
                      "fdhop walk_edges min_gap 10x20 k4=1 kr=1",
                      "fdhop nbr_walk min_gap 10x20 k4=1 kr=1"):
                # Dim 64 also holds the optimizer study, so the cell name
                # alone matches many rows. Keep the plain reference only.
                m = [r for r in pick(g, d, lambda r: r["cell"] == c and r["optim"]
                                     in ("-", "plain/const/lr=0.999"))]
                vals.append(f"{float(m[0]['rho_mean']):.3f} ± {float(m[0]['rho_std']):.3f}"
                            + (f" ({m[0]['seeds']})" if m[0]["seeds"] != "11" else "")
                            if m else "-")
            print(f"| {d} | " + " | ".join(vals) + " |")
        print()
elif what == "laws":
    for g in GRAPHS:
        print(f"**{g}, dim 128, walk_edges min_gap, plain**\n")
        table(pick(g, 128, lambda r: any(r["cell"].startswith(l) for l in LAWS)
                   and "10x20" in r["cell"] and "k4=1 kr=1" in r["cell"] or
                   r["cell"].startswith("fdhop_all")))
elif what == "support":
    print("**cora, dim 128: walk policy and weight rule**\n")
    table(pick("cora", 128, lambda r: r["cell"].startswith("fdhop ") and "10x20" in r["cell"]
               and "k4=1 kr=1" in r["cell"] and "_pq" not in r["cell"]))
    print("**cora, dim 128: walk budget (walks x length), walk_edges min_gap**\n")
    table(pick("cora", 128, lambda r: r["cell"].startswith("fdhop walk_edges min_gap")
               and "k4=1 kr=1" in r["cell"]))
elif what == "budget":
    for g in GRAPHS:
        rows = [r for r in ROWS if r["graph"] == g and r["dim"] == "128"
                and (r["cell"].startswith("deepwalk") or r["cell"].startswith("node2vec q=1.0"))
                and "win5" in r["cell"]]
        if not rows:
            continue
        print(f"**{g}, dim 128, window 5: the same walk budget for both**\n")
        print("| walks x length | steps per node | deepwalk | node2vec q=1 | difference |")
        print("|" + "---|" * 5)
        def budget_key(b):
            a, c = b.split("x")
            return int(a) * int(c)
        budgets = sorted({r["cell"].split()[-2] for r in rows}, key=budget_key)
        for b in budgets:
            got = {}
            for name in ("deepwalk", "node2vec q=1.0"):
                m = [r for r in rows if r["cell"] == f"{name} {b} win5"]
                if m:
                    got[name] = (float(m[0]["rho_mean"]), float(m[0]["rho_std"]))
            if len(got) < 2:
                continue
            (d, ds), (nv, ns) = got["deepwalk"], got["node2vec q=1.0"]
            print(f"| {b} | {budget_key(b)} | {d:.3f} ± {ds:.3f} | {nv:.3f} ± {ns:.3f} "
                  f"| {d - nv:+.3f} |")
        print()
elif what == "weights":
    import math
    print("| graph | min_gap | flat | difference | gap / seed spread |")
    print("|" + "---|" * 5)
    for g in GRAPHS:
        got = {}
        for w in ("min_gap", "flat"):
            m = [r for r in ROWS if r["graph"] == g and r["dim"] == "128"
                 and r["cell"] == f"fdhop walk_edges {w} 10x20 k4=1 kr=1"]
            if m:
                got[w] = (float(m[0]["rho_mean"]), float(m[0]["rho_std"]), m[0]["seeds"])
        if len(got) < 2:
            continue
        (a1, s1, n1), (a2, s2, n2) = got["min_gap"], got["flat"]
        d = a1 - a2
        print(f"| {g} | {a1:.3f} ± {s1:.3f} ({n1}) | {a2:.3f} ± {s2:.3f} ({n2}) "
              f"| {d:+.3f} | {abs(d) / math.hypot(s1, s2):.1f} |")
    print()
elif what == "force_params":
    for g in ("cora", "pubmed"):
        print(f"**{g}, dim 128: k4 and kr, walk_edges min_gap 10x20**\n")
        table(pick(g, 128, lambda r: r["cell"].startswith(("fdhop walk_edges min_gap 10x20 k4",
                                                           "fdlinear walk_edges min_gap 10x20 k4"))))
elif what == "optim":
    for g in ("cora", "pubmed"):
        print(f"**{g}, dim 64, fdhop walk_edges min_gap, 11 seeds**\n")
        table(pick(g, 64, lambda r: r["seeds"] == "11" and r["cell"] == STUDY), label="optimizer")
elif what == "lrmap":
    for g in ("cora", "pubmed"):
        print(f"**{g}, dim 64: rho by rule and learning rate (const)**\n")
        lrs = ["0.999", "0.5", "0.2", "0.099", "0.05", "0.02"]
        print("| rule | " + " | ".join("lr " + x for x in lrs) + " |")
        print("|" + "---|" * (len(lrs) + 1))
        band = []
        for rule in ("plain", "velocity", "sqn", "adam", "momentum", "nesterov"):
            cells = []
            for lr in lrs:
                m = [r for r in ROWS if r["graph"] == g and r["dim"] == "64"
                     and r["cell"] == STUDY
                     and r["optim"] == f"{rule}/const/lr={lr}"]
                if not m:
                    cells.append("-")
                    continue
                r = m[0]
                ratio = float(r["ratio"] or 0)
                rmax = float(r["ratio_max"] or 0)
                txt = f"{float(r['rho_mean']):.3f} ± {float(r['rho_std']):.3f} ({r['seeds']})"
                if r["blown_up"] == "True":
                    txt = "✗ " + txt
                # Under 2x no threshold anyone would pick can be crossed, so
                # the ratio is printed only where it could matter.
                if ratio >= 2:
                    txt += f" {ratio:.3g}x"
                # A cell whose median clears the line while one seed does not.
                if ratio <= 10 < rmax:
                    band.append(f"{rule} lr {lr} (cell {ratio:.3g}x, worst seed {rmax:.3g}x)")
                cells.append(txt)
            print(f"| {rule} | " + " | ".join(cells) + " |")
        print()
        if band:
            print(f"Inside the disputed band on {g}, NOT marked because the cell "
                  f"median is under 10x, although a single seed is over it: "
                  + "; ".join(band) + ". The mark is per cell, not per seed.\n")
