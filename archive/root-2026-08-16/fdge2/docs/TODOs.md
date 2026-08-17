# fdge2 TODOs

Nice-to-haves that aren't blocking anything in RPD.md, tracked here so they
don't get lost.

- **Replace hand-rolled `verbosity` int + raw `print` with the `logging`
  module.** `core/force_directed.py`'s `embed()` currently builds its
  per-epoch status line across three separate `if self.verbosity >= N`
  blocks using `print(..., end="")`, closed off by a final `print("")` --
  the partial line only exists as terminal state (not an inspectable
  value) and is fragile to any other code writing to stdout mid-line
  (e.g. a callback). `logging` (`logger.debug`/`logger.info`, etc.)
  separates *what* to report from *where it goes* (console, file, both)
  and lets callers control verbosity externally instead of baking an
  `int` threshold into `__init__`. Not urgent -- `embed()` is currently a
  faithful port of the legacy loop, not a redesign.
