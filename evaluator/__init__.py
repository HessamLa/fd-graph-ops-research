#!/bin/env python3
"""evaluator -- one measurement of the quality of a graph embedding.

Four files of this repository each measured an embedding a different way,
thus a number of one was not comparable to a number of another. This
package ends the duplication and it makes the disagreement EXPLICIT: each
recorded baseline is a NAMED PROTOCOL (`config.PROTOCOLS`), and each
result record carries the name.

    import evaluator as ev
    rep = ev.evaluate("cora", "emb/cora.npy",
                      tasks=("link_prediction", "dist_approx"),
                      protocol="otherge", seed=42, tags={"method": "n2v"})
    print(rep.table());  rep.write("runs/otherge.jsonl")

THE PROTOCOL RULE. A keyword left at `None` takes the value of the
protocol. A keyword given EXPLICITLY overrides it and it sets
`protocol_modified` on the record, thus a changed number cannot claim the
name of a recorded baseline (PRD invariant I2). `table()` marks such a row
with a `*`.

IMPORT COST (PRD invariant I6). This import must cost less than 1.0 s and
it must pull in NO `sklearn`, NO `networkit` and NO `gensim`. The task
functions import sklearn inside their bodies, `hops` imports networkit
inside `_pll`, and `io` imports the dataset registry inside
`_load_registry`. Do not lift any of those to a module top.
"""
from __future__ import annotations

from .config import PROTOCOLS
from .guards import DegenerateEvaluation
from .io import load_embedding, load_graph
from .report import Report, TaskResult, env, now_iso
from .tasks import TASKS, dist_approx, keywords, link_prediction
from .tasks import load_embedding as _load_embedding
from .tasks import load_graph as _load_graph

__all__ = [
    "link_prediction", "dist_approx", "evaluate",
    "Report", "TaskResult",
    "load_graph", "load_embedding",
    "PROTOCOLS", "TASKS", "DegenerateEvaluation",
]

__version__ = "1.0"


def evaluate(graph, Z, *, tasks=("link_prediction",), protocol="default",
             seed=42, tags=None, strict=True, **task_kw) -> Report:
    """Run several tasks over one graph and one embedding. Returns a
    `Report`.

    The graph and the embedding are read ONE TIME and the loaded objects
    go to every task, thus a two-task run does not pay the load twice.

    Each task gets its own generator, `np.random.default_rng(seed)`. That
    reproduces the frozen references, each of which makes the generator at
    its first line, thus a two-task report holds the same numbers as two
    separate parity runs.

    `**task_kw` overrides a protocol setting. A key goes to EVERY selected
    task that declares it -- `test_size` reaches both tasks, `metric`
    reaches `dist_approx` alone. A key that NO selected task declares
    raises, thus a typed keyword never disappears in silence.
    """
    names = tuple(tasks)
    unknown = [t for t in names if t not in TASKS]
    if unknown:
        raise ValueError(
            f"evaluate: unknown task {', '.join(unknown)}; the tasks are "
            f"{', '.join(sorted(TASKS))}.")

    allowed = {k for t in names for k in keywords(t)}
    bad = sorted(set(task_kw) - allowed)
    if bad:
        raise ValueError(
            f"evaluate: the selected tasks ({', '.join(names)}) hold no "
            f"keyword {', '.join(bad)}; the keywords are "
            f"{', '.join(sorted(allowed))}.")

    # The tuple-aware loaders of `tasks`: a `(A, n, info)` triple or a
    # `(Z, info)` pair passes through, thus the tasks below do NOT read
    # the graph again.
    A, n, ginfo = _load_graph(graph, seed=seed)
    Z, zinfo = _load_embedding(Z, n)

    results, modified = {}, False
    for name in names:
        kw = {k: v for k, v in task_kw.items() if k in keywords(name)}
        res = TASKS[name]((A, n, ginfo), (Z, zinfo), protocol=protocol,
                          seed=seed, strict=strict, **kw)
        modified = modified or bool(getattr(res, "protocol_modified", False))
        results[name] = res

    return Report(created=now_iso(), protocol=protocol,
                  protocol_modified=modified, seed=seed, graph=ginfo,
                  embedding=zinfo, env=env(), tags=dict(tags or {}),
                  tasks=results)
