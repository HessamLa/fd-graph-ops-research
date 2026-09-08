#!/bin/env python3
"""evaluator.cli -- the command-line surface of `evaluator` (PRD section 6).

    .venv/bin/python -m evaluator GRAPH EMBEDDING [tasks] [options]

This module holds NO measurement. It parses arguments, builds each task's
keyword dict, calls `evaluate()`, and prints. PRD invariant I7: the CLI and
the API reach the SAME task functions. Graph and embedding are read ONE
time and the loaded objects pass into every call, thus a two-task run does
not pay the load twice.

ONE `evaluate()` CALL PER TASK. `link_prediction` and `dist_approx` share
the field names `feature` and `test_size`, and `evaluate()` routes a
keyword to EVERY selected task that declares it. One call for both tasks
would therefore apply one `--lp-feature` to both, against section 6.2:
"each option carries the task prefix, thus two tasks cannot collide."

THE PAREN SUGAR (section 6.5). `--task "name(key=value, ...)"`, parsed by
`parse_call`. A value reads as int, then float, then bool, then string. An
unknown key raises and names the valid ones, read from
`evaluator.keywords(name)` -- the task signature itself, thus the list
cannot drift from a renamed field and a new task needs no change here.

An explicit flag wins over the sugar: the sugar dict of a task is built
first, and every explicit `--lp-*` / `--da-*` flag then overwrites its
key (section 6.5, "an explicit flag wins over the sugar").

ERROR SHAPING (PRD B12.5, the "must not" of the block brief). Every
exception raised while building or running the request is caught in
`main` and printed to stderr as `error: message`, with exit code 1 --
never a traceback. A `DegenerateEvaluation` (a guard, `evaluator.guards`)
already carries its check id in the message, thus the same handler
satisfies "print the guard id and its message."
"""
from __future__ import annotations

import argparse
import json
import sys

from . import (
    DegenerateEvaluation, PROTOCOLS, TASKS, evaluate, keywords,
    load_embedding, load_graph,
)

__all__ = ["main", "build_parser", "parse_call", "run"]

# The canonical task order. `--all` and a mixed `-lp -da` run both print
# and record in this order, regardless of the order the flags were typed.
_TASK_ORDER = ("link_prediction", "dist_approx")

# task name -> {cli flag dest: cfg field name}. The map is how a prefixed
# flag reaches only ITS OWN task (section 6.2): `--lp-feature` is read out
# of the `link_prediction` row and it can never touch `dist_approx`.
_TASK_FLAGS = {
    "link_prediction": {
        "lp_max_pairs": "max_pairs",
        "lp_test_size": "test_size",
        "lp_neg_ratio": "neg_ratio",
        "lp_feature": "feature",
        "lp_estimators": "n_estimators",
    },
    "dist_approx": {
        "da_pairs": "n_pairs",
        "da_sources": "n_sources",
        "da_min_hop": "min_hop",
        "da_feature": "feature",
        "da_hops": "hops",
        "da_models": "models",
    },
}


def _csv_tuple(text: str) -> tuple:
    """`"baseline,rf,mlp"` -> `("baseline", "rf", "mlp")`. For `--da-models`."""
    return tuple(x.strip() for x in text.split(",") if x.strip())


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of PRD section 6. One place, so `--help`
    (B12.1) and `main` read the same flags."""
    ap = argparse.ArgumentParser(
        prog="python -m evaluator",
        description="Score a graph embedding: link prediction, hop-distance "
                    "approximation. A named protocol keeps a number "
                    "comparable to the baseline that used the same "
                    "settings, and to no other.")
    ap.add_argument("graph", help="a path (.txt/.edges/.npz/.mtx) or a "
                     "registry name (cora, pubmed, wordnet, ...)")
    ap.add_argument("embedding", help="a path (.npy/.npz/.csv/.tsv)")

    tasks = ap.add_argument_group("tasks")
    tasks.add_argument("--link-prediction", "-lp", action="store_true",
                       help="run the link prediction task")
    tasks.add_argument("--dist-approx", "-da", action="store_true",
                       help="run the hop distance approximation task")
    tasks.add_argument("--all", action="store_true",
                       help="run every registered task")
    tasks.add_argument("--task", action="append", default=[],
                       metavar="CALL",
                       help="sugar: 'link_prediction(test_size=0.2)' or "
                            "'dist_approx(n_pairs=20000, feature=distance)'. "
                            "repeatable. An explicit flag wins over this.")

    lp = ap.add_argument_group("link prediction options")
    lp.add_argument("--lp-max-pairs", type=int, default=None,
                    help="positives and negatives together")
    lp.add_argument("--lp-test-size", type=float, default=None,
                    help="the test fraction (0.2)")
    lp.add_argument("--lp-neg-ratio", type=float, default=None,
                    help="negatives for each positive (1.0)")
    lp.add_argument("--lp-feature", default=None,
                    choices=("hadamard", "l1", "concat", "avg"),
                    help="hadamard | l1 | concat | avg")
    lp.add_argument("--lp-estimators", type=int, default=None,
                    help="the trees of the forest")

    da = ap.add_argument_group("dist approx options")
    da.add_argument("--da-pairs", type=int, default=None,
                    help="the pairs to sample")
    da.add_argument("--da-sources", type=int, default=None,
                    help="draw the first node from N sources only. 0 = all")
    da.add_argument("--da-min-hop", type=int, default=None,
                    help="drop the pairs closer than this (2)")
    da.add_argument("--da-feature", default=None,
                    choices=("distance", "vector"),
                    help="distance (1 column) | vector (n_dim columns)")
    da.add_argument("--da-hops", default=None,
                    choices=("auto", "pll", "bfs", "landmark"),
                    help="the ground-truth backend")
    da.add_argument("--da-models", type=_csv_tuple, default=None,
                    metavar="LIST", help="comma list, e.g. baseline,rf,mlp")

    g = ap.add_argument_group("global options")
    g.add_argument("--protocol", default="default", choices=sorted(PROTOCOLS),
                   help="the named baseline (default: %(default)s)")
    g.add_argument("--metric", default=None,
                   choices=("euclidean", "poincare", "cosine", "dot"),
                   help="the distance of dist_approx's 'distance' feature")
    g.add_argument("--seed", type=int, default=42, help="42")
    g.add_argument("--max-nodes", type=int, default=0,
                   help="truncate a registry graph (0 = no truncation)")
    g.add_argument("--results", default=None, metavar="PATH",
                   help=".json writes one record; .jsonl appends one line")
    g.add_argument("--tag", action="append", default=[], metavar="K=V",
                   help="repeatable. Goes into the record as provenance")
    g.add_argument("--no-strict", action="store_true",
                   help="a guard writes a warning and does not raise")
    g.add_argument("--quiet", action="store_true",
                   help="the record only, no table")
    g.add_argument("--json", action="store_true",
                   help="write the whole record as JSON to stdout, and "
                        "nothing else. Implies --quiet.")
    return ap


# ---------------------------------------------------------------------------
# The paren sugar
# ---------------------------------------------------------------------------
def _parse_value(text: str):
    """`text` as int, else float, else bool (`true`/`false`), else the
    string itself -- the order section 6.5 writes down."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    return text


def parse_call(text: str) -> tuple:
    """`"name(key=value, ...)"` -> `(name, {key: value})`.

    Raises `ValueError` when `text` is not `NAME(...)`, when `name` is not
    a registered task, or when a key is not one of `name`'s settings --
    the message then names the keys that ARE valid, read from
    `evaluator.keywords(name)` and never a hand-kept list, so it cannot
    drift when a config field is renamed.
    """
    raw = text.strip()
    if "(" not in raw or not raw.endswith(")"):
        raise ValueError(
            f"--task: {text!r} is not NAME(key=value, ...); "
            f"the tasks are {', '.join(sorted(TASKS))}.")
    name, _, body = raw.partition("(")
    name = name.strip()
    body = body[:-1].strip()          # drop the closing ')'
    if name not in TASKS:
        raise ValueError(
            f"--task: unknown task {name!r}; "
            f"the tasks are {', '.join(sorted(TASKS))}.")

    kwargs = {}
    if body:
        for part in body.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(
                    f"--task: {part!r} is not key=value, in {text!r}.")
            key, _, value = part.partition("=")
            kwargs[key.strip()] = _parse_value(value.strip())

    valid = set(keywords(name))
    bad = sorted(set(kwargs) - valid)
    if bad:
        raise ValueError(
            f"--task: {name} has no key {', '.join(bad)}; "
            f"the keys of {name} are {', '.join(sorted(valid))}.")
    return name, kwargs


# ---------------------------------------------------------------------------
# Assembling the per-task keyword dicts
# ---------------------------------------------------------------------------
def _task_kwargs(args, task_names) -> dict:
    """`{task_name: {cfg_field: value}}`, sugar first, flags overwrite.

    A flag reaches only ITS OWN task -- `--lp-feature` never touches
    `dist_approx`'s `feature` -- because each is read out of
    `_TASK_FLAGS[task_name]`, and not out of one shared namespace. That is
    what keeps two tasks from colliding on a field name they share
    (section 6.2).
    """
    out = {name: {} for name in task_names}
    for call in args.task:
        name, kw = parse_call(call)
        if name in out:
            out[name].update(kw)
    for name in task_names:
        for flag_dest, cfg_field in _TASK_FLAGS[name].items():
            value = getattr(args, flag_dest)
            if value is not None:
                out[name][cfg_field] = value
    if "dist_approx" in out and args.metric is not None:
        out["dist_approx"]["metric"] = args.metric
    return out


def _selected_tasks(args) -> list:
    """The task names to run, in the canonical order, regardless of the
    order the flags were typed. `--task` also selects its task (section
    6.5: the sugar "is equal to the flag form")."""
    chosen = set()
    if args.all:
        chosen.update(TASKS)
    if args.link_prediction:
        chosen.add("link_prediction")
    if args.dist_approx:
        chosen.add("dist_approx")
    for call in args.task:
        name, _ = parse_call(call)
        chosen.add(name)
    return [t for t in _TASK_ORDER if t in chosen]


def _tags(pairs) -> dict:
    """`["k=v", ...]` -> `{"k": "v", ...}`. A repeated key keeps the last
    value, in the order `--tag` was given."""
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--tag: {p!r} is not K=V.")
        k, _, v = p.partition("=")
        out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(args):
    """Load once, call `evaluate()` once for each selected task, and
    return the merged `Report`. No printing here -- `main` shapes the
    exit code and the output, so this stays callable (and testable)
    without a subprocess.
    """
    task_names = _selected_tasks(args)
    if not task_names:
        raise ValueError(
            "no task selected. Give --link-prediction/-lp, "
            "--dist-approx/-da, --all, or --task NAME(...).")

    strict = not args.no_strict
    tags = _tags(args.tag)
    kwargs = _task_kwargs(args, task_names)

    # Read the graph and the embedding ONE TIME. `--max-nodes` reaches
    # only `load_graph`, because `evaluate()`'s own pass-through loader
    # (`evaluator/tasks/__init__.py`) does not forward it.
    A, n, ginfo = load_graph(args.graph, max_nodes=args.max_nodes,
                             seed=args.seed)
    Z, zinfo = load_embedding(args.embedding, n)

    reports = [
        evaluate((A, n, ginfo), (Z, zinfo), tasks=(name,),
                protocol=args.protocol, seed=args.seed, tags=tags,
                strict=strict, **kwargs[name])
        for name in task_names
    ]
    report = reports[0]
    for extra in reports[1:]:
        report.tasks.update(extra.tasks)
        report.protocol_modified = (report.protocol_modified
                                    or extra.protocol_modified)
    return report


def main(argv=None) -> int:
    """The entry point of `python -m evaluator`. Returns the exit code;
    it never lets an exception reach the caller as a traceback (the
    "must not" of PRD block B12)."""
    ap = build_parser()
    args = ap.parse_args(argv)
    quiet = args.quiet or args.json

    try:
        report = run(args)
    except DegenerateEvaluation as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:                                # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.results:
        report.write(args.results)

    if args.json:
        print(json.dumps(report.to_dict()))
    elif not quiet:
        print(report.table())
    return 0


if __name__ == "__main__":
    sys.exit(main())
