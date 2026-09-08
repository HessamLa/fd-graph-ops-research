#!/bin/env python3
"""evaluator.tasks -- the task registry, and the helpers that every task shares.

A task is ONE file plus ONE registry line (PRD goal G2).

    @register("my_task")
    def my_task(graph, Z, *, protocol="default", seed=42, rng=None,
                strict=True, ...):
        ...
        return TaskResult(...)

`evaluate()` reads the rest of the signature with `keywords()` and it
routes a caller keyword to each task that declares it.

IMPORT DISCIPLINE, and it is a requirement and not a style (PRD invariant
I6). `import evaluator` must cost less than 1.0 s and it must pull in no
`sklearn`. `evaluator.scoring` imports `sklearn.metrics` at its top, thus
a task module imports `scoring` and every sklearn estimator INSIDE its
function body. A top-level `from .. import scoring` in a task file breaks
the budget for the whole package.
"""
from __future__ import annotations

import dataclasses as _dataclasses
import inspect

import numpy as np

from .. import io as _io

__all__ = ["TASKS", "register", "keywords", "resolve_rng",
           "load_graph", "load_embedding", "cfg_record",
           "link_prediction", "dist_approx"]

# name -> callable. A task registers itself when its module imports.
TASKS: dict = {}

# The keywords that EVERY task holds. `keywords()` removes them, thus
# `evaluate()` routes only the settings of the task.
COMMON = ("graph", "Z", "protocol", "seed", "rng", "strict")


def register(name: str):
    """Register a task function under `name`. Returns the function.

    A duplicate name raises: two tasks under one name would make a record
    ambiguous about which code produced it.
    """
    def deco(fn):
        if name in TASKS:
            raise ValueError(
                f"register: the task name {name!r} is taken by "
                f"{TASKS[name].__module__}.{TASKS[name].__name__}.")
        TASKS[name] = fn
        return fn
    return deco


def keywords(name: str) -> tuple:
    """The setting keywords of the task `name`, without `COMMON`.
    `evaluate()` reads this to route `**task_kw`.
    """
    fn = TASKS[name]
    return tuple(p for p in inspect.signature(fn).parameters
                 if p not in COMMON)


def resolve_rng(rng, seed):
    """The generator of a task run. `rng` when the caller gave one, else
    `np.random.default_rng(seed)`.

    This is the ONE place that a generator comes from, and it sits at the
    API boundary and not inside a measurement step (PRD invariant I1). The
    fallback reproduces the first line of every frozen reference, thus a
    parity run needs no `rng` argument.

    `seed` and `rng` stay separate: `seed` goes to sklearn
    (`random_state`) and `rng` goes to the sampling. Both reach the record.
    """
    return np.random.default_rng(seed) if rng is None else rng


def load_graph(graph, seed=42):
    """`(A, n, info)` from whatever the caller passed.

    A `(A, n, info)` TRIPLE passes through untouched. That is how
    `evaluate()` reads a graph ONE time: a second `io.load_graph` would
    symmetrize and copy the whole matrix again, which costs the peak
    memory at 1.13M nodes. Anything else goes to `io.load_graph`.
    """
    if isinstance(graph, tuple) and len(graph) == 3:
        return graph
    return _io.load_graph(graph, seed=seed)


def load_embedding(Z, n=None):
    """`(Z, info)` from whatever the caller passed. A `(Z, info)` PAIR
    passes through untouched, for the reason of `load_graph` above."""
    if isinstance(Z, tuple) and len(Z) == 2:
        return Z
    return _io.load_embedding(Z, n)


def cfg_record(cfg, protocol, modified):
    """The `cfg` dict of a `TaskResult`: every dataclass field, plus
    `protocol` and `protocol_modified`. `report.Report` carries the same
    two at the top of the record; a `TaskResult` read alone, out of a
    `.jsonl` line, must carry them too (PRD invariant I2).
    """
    d = _dataclasses.asdict(cfg)
    d["protocol"] = protocol
    d["protocol_modified"] = bool(modified)
    return d


# The registry lines. They come last: each module imports `register` from
# this one.
from .link_prediction import link_prediction     # noqa: E402
from .dist_approx import dist_approx             # noqa: E402
