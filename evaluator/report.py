#!/bin/env python3
"""evaluator.report -- `TaskResult`, `Report`, the JSON record.

One task run gives one `TaskResult`. One `evaluate()` call gives one
`Report`, which holds every `TaskResult` plus the provenance: the
protocol, the seed, the graph and embedding `info` of `io.py`, the
library versions, and the caller's tags. `Report.to_dict()` is the JSON
schema of the project (`schema = "evaluator/1"`); `write()` delegates the
`.json` / `.jsonl` choice to `io.write_report`; `table()` prints the
`=== summary ===` block of `experiments/other-ge/bench_other_ge.py`.

`.io` is the one package import at the top, and it closes no cycle:
`io.py` never imports `report`. The heavy libraries are read for their
versions INSIDE `env()`, each import guarded, thus `import evaluator`
stays free of them.
"""
from __future__ import annotations

import dataclasses
import datetime
import platform
from typing import Any

from . import io as _io

__all__ = ["TaskResult", "Report", "env", "now_iso"]


# ---------------------------------------------------------------------------
# JSON casting
# ---------------------------------------------------------------------------
def _to_jsonable(x: Any) -> Any:
    """Recursively cast `x` into `dict` / `list` / `str` / `int` / `float`
    / `bool` / `None`.

    The numpy check runs FIRST, ahead of the builtin `isinstance` test.
    `np.float64` subclasses Python `float` (and `np.bool_` behaves like
    `bool`), thus an `isinstance(x, float)` test done first lets an
    `np.float64` pass straight through UNCONVERTED. It still satisfies
    `json.dumps`, because a float subclass serializes fine, so the defect
    is invisible until a stricter consumer (`type(v) is float`, a
    type-dispatching reader, `orjson`) meets it. sklearn's scorers return
    `np.float64` for nearly every metric, thus this ordering is the one
    fact this function must get right.

    `type(x).__module__ == "numpy"` catches every numpy scalar and array
    by name, with no `import numpy` here. A numpy array's `.tolist()`
    already returns pure Python, thus that branch needs no recursion.
    """
    if type(x).__module__ == "numpy":
        return x.tolist() if hasattr(x, "tolist") else x.item()
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if hasattr(x, "tolist"):          # a non-numpy array-like
        return _to_jsonable(x.tolist())
    if hasattr(x, "item"):            # a non-numpy scalar-like
        return x.item()
    return x


# ---------------------------------------------------------------------------
# The environment record
# ---------------------------------------------------------------------------
def now_iso() -> str:
    """The current UTC time, ISO-8601, seconds precision. For `created`."""
    return datetime.datetime.now(datetime.timezone.utc) \
        .isoformat(timespec="seconds")


def env() -> dict:
    """Versions of every library that can take part in a run. `numpy`,
    `scipy`, `sklearn` and `networkit` each import inside a `try`, thus an
    absent optional library records `None` and never raises.
    """
    out = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "sklearn", "networkit"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", None)
        except ImportError:
            out[name] = None
    return out


# ---------------------------------------------------------------------------
# The records
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class TaskResult:
    """One task's outcome: its resolved config, its scores, its sizes.

    `cfg` is the resolved `LPCfg` or `DACfg` as a dict, so a reader sees
    exactly which settings produced `scores`. `scores` is
    `{metric: value}` for a single-model task, or
    `{model: {metric: value}}` when a task runs several models.
    `warnings` holds the guard ids that `strict=False` downgraded from a
    raise to a warning.
    """

    task: str
    cfg: dict
    scores: dict
    sizes: dict
    seconds: float
    warnings: list = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return _to_jsonable(dataclasses.asdict(self))


@dataclasses.dataclass(kw_only=True)
class Report:
    """The whole run: every `TaskResult`, plus the provenance.

    `kw_only=True` (Python >= 3.10) because `schema` carries a default and
    every other field does not; a plain dataclass forbids a non-default
    field after a default one, and `schema` is written first.

    `protocol_modified` is `True` when any task keyword overrode the named
    protocol (`config.override`); `table()` marks such a row with a `*`,
    because its numbers must not be compared against a recorded baseline
    (PRD invariant I2).
    """

    created: str
    protocol: str
    protocol_modified: bool
    seed: int
    graph: dict
    embedding: dict
    env: dict
    tags: dict
    tasks: dict
    schema: str = "evaluator/1"

    def to_dict(self) -> dict:
        """The whole record as JSON-safe types: every numpy scalar and array
        is cast away, thus `json.dumps` always accepts the result.
        """
        d = dataclasses.asdict(self)
        return _to_jsonable(d)

    def write(self, path) -> None:
        """Append or write the record. `io.write_report` picks `.jsonl`
        (append one line) or `.json` (one object) from the suffix of `path`.
        """
        _io.write_report(self, path)

    def table(self) -> str:
        """One line for each task. A `protocol_modified` record is marked with
        a leading `*`, because its numbers are not comparable to a recorded
        baseline.
        """
        mark = "*" if self.protocol_modified else " "
        lines = [f"{mark} protocol={self.protocol} seed={self.seed}"]
        for name, result in self.tasks.items():
            scores = result.scores if isinstance(result, TaskResult) \
                else result["scores"]
            seconds = result.seconds if isinstance(result, TaskResult) \
                else result["seconds"]
            flat = _flatten_scores(scores)
            score_str = " ".join(f"{k}={_fmt_score(v)}" for k, v in flat.items())
            lines.append(f"{mark} {name:>16s} {seconds:>7.1f}s  {score_str}")
        return "\n".join(lines)


def _fmt_score(v) -> str:
    """`v` as a 4-decimal float, or its `repr` when it is not a real
    number. A printer must not crash on a value that `to_dict()` can
    serialize.
    """
    try:
        return f"{v:.4f}"
    except (TypeError, ValueError):
        return repr(v)


def _flatten_scores(scores: dict) -> dict:
    """`{metric: value}` unchanged. `{model: {metric: value}}` becomes
    `{model.metric: value}`, so `table()` prints one line either way.
    """
    flat = {}
    for k, v in scores.items():
        if isinstance(v, dict):
            for mk, mv in v.items():
                flat[f"{k}.{mk}"] = mv
        else:
            flat[k] = v
    return flat
