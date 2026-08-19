#!/bin/env python3
"""dashboard.py -- the state of every planned run: pending, running, done.

It derives the PLAN in Python the same way the driver scripts derive it in
bash, thus a run that no script will ever start does not appear, and a run
that a script will start appears as `pending` before its log exists.

Classification of one point:
    done      the log holds a RESULT line
    diverged  the log holds `diverged=1` (a RESULT, and a real measurement)
    failed    the log exists, has no RESULT, and has not changed recently
    running   the log exists, has no RESULT, and changed in the last 180 s
    pending   no log

    .venv/bin/python experiments/fdwalk/dashboard.py [--html OUT] [--json OUT]
"""
from __future__ import annotations
import glob, json, os, re, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FRESH = 180.0

OPTIMS = ["plain", "velocity", "sgd", "sqn", "momentum", "nesterov",
          "adam", "fa2"]
SEEDS = [42, 56, 88]
PQ = [("1.0", "1.0"), ("1.0", "0.5"), ("1.0", "2.0"),
      ("0.5", "1.0"), ("2.0", "1.0")]
BCELLS = ["cap_sym", "cap_dir", "buckets_sym", "buckets_dir"]


def plan():
    """[(stage, graph, subdir, name)] for every point any driver will run."""
    p = []
    for o in OPTIMS:                       # Cora grid A: the first lr scan
        for lr in ("1.0", "0.1", "0.01"):
            p.append(("A lr scan", "cora", "grid_cora", f"A_lrscan_{o}_lr{lr}"))
    for c in BCELLS:                       # Cora grid B
        for s in SEEDS:
            p.append(("B cap/buckets x sym/dir", "cora", "grid_cora", f"B_{c}_s{s}"))
    for v in ("buckets_nofar", "buckets_far"):
        for s in SEEDS:
            p.append(("C buckets far/budget", "cora", "grid_cora", f"C_{v}_s{s}"))
    for bt in ("5000", "20000", "80000"):
        p.append(("C buckets far/budget", "cora", "grid_cora", f"C_budget_{bt}"))
    for pp, q in PQ:                       # Cora grid D
        for s in SEEDS:
            p.append(("D second-order walk", "cora", "grid_cora", f"D_p{pp}_q{q}_s{s}"))
    for o in OPTIMS:                       # the lr ladder
        for lr in ("0.999", "0.9", "0.1"):
            p.append(("LR ladder (constant)", "cora", "lrladder", f"const_{o}_lr{lr}"))
        for lr in ("0.999", "0.9"):
            p.append(("LR ladder (decay)", "cora", "lrladder", f"decay_{o}_lr{lr}"))
    # PubMed: grid A carries only the Cora survivors, thus it is discovered
    # from disk rather than predicted here.
    for c in BCELLS:
        for s in SEEDS:
            p.append(("PubMed B", "pubmed", "grid_pubmed", f"B_{c}_s{s}"))
    for v in ("buckets_nofar", "buckets_far"):
        for s in SEEDS:
            p.append(("PubMed C", "pubmed", "grid_pubmed", f"C_{v}_s{s}"))
    for pp, q in PQ:
        for s in SEEDS:
            p.append(("PubMed D", "pubmed", "grid_pubmed", f"D_p{pp}_q{q}_s{s}"))
    return p


def classify(path):
    if not os.path.exists(path):
        return "pending", {}
    txt = open(path, errors="ignore").read()
    m = re.search(r"RESULT\t(.*)", txt)
    if m:
        kv = dict(x.split("=", 1) for x in m.group(1).split("\t") if "=" in x)
        return ("diverged" if kv.get("diverged") == "1" else "done"), kv
    age = time.time() - os.path.getmtime(path)
    return ("running" if age < FRESH else "failed"), {}


def collect():
    seen, rows = set(), []
    for stage, graph, sub, name in plan():
        path = os.path.join(RES, sub, name + ".log")
        seen.add(path)
        state, kv = classify(path)
        rows.append(dict(stage=stage, graph=graph, name=name,
                         state=state, kv=kv))
    # anything on disk that the plan did not predict (PubMed grid A, and
    # any ad-hoc run) is reported too, never silently dropped.
    for sub, graph in (("grid_pubmed", "pubmed"), ("lrdecay", "cora"),
                       ("optim", "cora"), ("newsmoke", "cora"),
                       ("fused", "com_youtube"), ("mixmatch", "com_youtube")):
        for f in sorted(glob.glob(os.path.join(RES, sub, "*.log"))):
            if f in seen:
                continue
            state, kv = classify(f)
            rows.append(dict(stage=f"other ({sub})", graph=graph,
                             name=os.path.basename(f)[:-4],
                             state=state, kv=kv))
    return rows


def main():
    rows = collect()
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": rows}
    j = os.path.join(ROOT, "dashboard.json")
    json.dump(out, open(j, "w"), indent=1)
    from collections import Counter
    c = Counter(r["state"] for r in rows)
    ref = 0
    if "--refresh" in sys.argv:
        ref = int(sys.argv[sys.argv.index("--refresh") + 1])
    out_html = os.path.join(ROOT, "dashboard.html")
    if "--html" in sys.argv:
        out_html = sys.argv[sys.argv.index("--html") + 1]
    open(out_html, "w").write(render(rows, ref))
    print(" ".join(f"{k}={v}" for k, v in sorted(c.items())),
          f"total={len(rows)} -> {out_html}")




# ---------------------------------------------------------------------------
# The HTML view
# ---------------------------------------------------------------------------
STAGE_ORDER = [
    "A lr scan", "LR ladder (constant)", "LR ladder (decay)",
    "B cap/buckets x sym/dir", "C buckets far/budget", "D second-order walk",
    "PubMed B", "PubMed C", "PubMed D",
]
STATES = [("done", "done"), ("diverged", "diverged"), ("running", "running"),
          ("failed", "failed"), ("pending", "pending")]

CSS = """
:root{
 --ground:#f6f7f7; --panel:#fff; --panel-2:#eef1f1;
 --ink:#12171a; --ink-2:#4a565c; --ink-3:#7d8b91; --rule:#dde3e4;
 --accent:#0d7d78;
 --ok:#2f7d4f; --run:#b07503; --div:#a8402c; --fail:#7c2d22; --pend:#9aa7ac;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ground:#0d1114; --panel:#151b1f; --panel-2:#1b2328;
 --ink:#e8eded; --ink-2:#9fb0b5; --ink-3:#6b7b81; --rule:#232c31;
 --accent:#3fb8ae;
 --ok:#4fa871; --run:#d9a02a; --div:#d4674f; --fail:#b8503c; --pend:#55646a;
}}
:root[data-theme="dark"]{
 --ground:#0d1114; --panel:#151b1f; --panel-2:#1b2328;
 --ink:#e8eded; --ink-2:#9fb0b5; --ink-3:#6b7b81; --rule:#232c31;
 --accent:#3fb8ae;
 --ok:#4fa871; --run:#d9a02a; --div:#d4674f; --fail:#b8503c; --pend:#55646a;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;
 font-family:Archivo,-apple-system,Segoe UI,sans-serif;
 font-size:15px;line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding:34px 22px 70px}
h1{font-size:26px;font-weight:700;letter-spacing:-.02em;margin:0 0 4px;
 text-wrap:balance}
.sub{color:var(--ink-2);font-size:14px;margin:0 0 26px}
.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;
 font-variant-numeric:tabular-nums}
.lab{font-size:11px;letter-spacing:.09em;text-transform:uppercase;
 color:var(--ink-3);font-weight:600}
.rail{display:flex;height:16px;border-radius:3px;overflow:hidden;
 background:var(--panel-2);margin:10px 0 12px}
.rail>span{display:block}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:30px}
.legend div{display:flex;align-items:center;gap:7px;font-size:13px}
.dot{width:9px;height:9px;border-radius:2px;flex:none}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(268px,1fr));
 gap:13px;margin-bottom:34px}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:7px;
 padding:14px 15px}
.card h3{margin:0;font-size:14.5px;font-weight:600;letter-spacing:-.01em}
.card .n{font-size:12px;color:var(--ink-3);margin-top:2px}
.card .rail{height:7px;margin:11px 0 8px}
.pct{font-size:22px;font-weight:700;letter-spacing:-.02em}
.stage-i{color:var(--accent);font-weight:700;font-size:12px;
 letter-spacing:.06em}
.tablewrap{overflow-x:auto;border:1px solid var(--rule);border-radius:7px;
 background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;
 color:var(--ink-3);font-weight:600;padding:10px 11px;
 border-bottom:1px solid var(--rule);white-space:nowrap;
 background:var(--panel);position:sticky;top:0}
td{padding:8px 11px;border-bottom:1px solid var(--rule);white-space:nowrap}
tr:last-child td{border-bottom:0}
td.num{text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace;
 font-variant-numeric:tabular-nums}
.chip{display:inline-block;padding:2px 8px;border-radius:11px;font-size:11px;
 font-weight:600;letter-spacing:.02em;color:#fff}
.stripe{border-left:3px solid var(--rule)}
h2{font-size:17px;font-weight:700;letter-spacing:-.015em;margin:34px 0 12px}
.note{color:var(--ink-2);font-size:13px;margin:0 0 16px;max-width:66ch}
"""


def _c(state):
    return {"done": "var(--ok)", "running": "var(--run)",
            "diverged": "var(--div)", "failed": "var(--fail)",
            "pending": "var(--pend)"}[state]


def _rail(counts, total):
    if not total:
        return '<div class="rail"></div>'
    seg = "".join(
        f'<span style="width:{counts.get(s,0)/total*100:.4f}%;'
        f'background:{_c(s)}"></span>'
        for s, _ in STATES if counts.get(s))
    return f'<div class="rail">{seg}</div>'


def render(rows, refresh=0):
    from collections import Counter, defaultdict
    total = len(rows)
    all_c = Counter(r["state"] for r in rows)
    by = defaultdict(list)
    for r in rows:
        by[r["stage"]].append(r)
    order = [s for s in STAGE_ORDER if s in by] + \
            sorted(s for s in by if s not in STAGE_ORDER)

    h = ['<title>fdwalk Run Board</title>',
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Archivo:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600'
         '&display=swap">']
    if refresh:
        h.append(f'<meta http-equiv="refresh" content="{refresh}">')
    h.append(f"<style>{CSS}</style>")
    h.append('<div class="wrap">')
    h.append("<h1>fdwalk Run Board</h1>")
    done = all_c.get("done", 0) + all_c.get("diverged", 0)
    h.append(f'<p class="sub">{done} of {total} planned runs measured &middot; '
             f'generated {time.strftime("%Y-%m-%d %H:%M:%S")} &middot; '
             'a diverged run is a measurement, not a failure</p>')
    h.append(_rail(all_c, total))
    h.append('<div class="legend">')
    for s, name in STATES:
        h.append(f'<div><span class="dot" style="background:{_c(s)}"></span>'
                 f'<span class="lab">{name}</span>'
                 f'<span class="mono">{all_c.get(s,0)}</span></div>')
    h.append("</div>")

    h.append('<h2>Stages</h2>')
    h.append('<p class="note">The pipeline runs in order: every Cora grid, '
             'then PubMed carrying only the parameter values that did not '
             'diverge, then com_youtube at 1.13M nodes excluding whatever '
             'PubMed flags as poor. Medium and large graphs never share the '
             'machine.</p>')
    h.append('<div class="cards">')
    for i, st in enumerate(order, 1):
        rr = by[st]
        c = Counter(x["state"] for x in rr)
        fin = c.get("done", 0) + c.get("diverged", 0)
        h.append('<div class="card">')
        h.append(f'<div class="stage-i">STAGE {i:02d}</div>')
        h.append(f"<h3>{st}</h3>")
        h.append(f'<div class="n mono">{rr[0]["graph"]}</div>')
        h.append(_rail(c, len(rr)))
        h.append(f'<div class="pct mono">{fin*100//max(1,len(rr))}%'
                 f'<span class="lab" style="margin-left:8px">'
                 f'{fin}/{len(rr)}</span></div>')
        h.append("</div>")
    h.append("</div>")

    h.append("<h2>Every run</h2>")
    h.append('<p class="note">Peak memory and stage runtimes are shown for '
             'every completed run. On Cora and PubMed they are indicative '
             'only &mdash; a fixed ~1&nbsp;GB of JAX and Python dominates the '
             'peak, so D.nnz is the better size proxy at that scale.</p>')
    h.append('<div class="tablewrap"><table><thead><tr>')
    for c in ("run", "graph", "state", "AUC", "hop R²", "‖dZ‖",
              "D.nnz", "aug s", "embed s", "peak MB"):
        cls = ' class="num"' if c not in ("run", "graph", "state") else ""
        h.append(f"<th{cls}>{c}</th>")
    h.append("</tr></thead><tbody>")
    rank = {s: i for i, (s, _) in enumerate(STATES)}
    for r in sorted(rows, key=lambda r: (rank.get(r["state"], 9), r["name"])):
        kv, st = r["kv"], r["state"]
        h.append(f'<tr class="stripe" style="border-left-color:{_c(st)}">')
        h.append(f'<td class="mono">{r["name"]}</td><td>{r["graph"]}</td>')
        h.append(f'<td><span class="chip" style="background:{_c(st)}">'
                 f"{st}</span></td>")
        for k in ("auc", "r2_dist", "dz", "dnnz", "t_aug", "t_embed", "rss"):
            v = kv.get(k, "")
            if v == "nan":
                v = "&mdash;"
            h.append(f'<td class="num">{v}</td>')
        h.append("</tr>")
    h.append("</tbody></table></div></div>")
    return "\n".join(h)


if __name__ == "__main__":
    main()
