"""fdge2.validation -- cross-cutting correctness + perf/memory harnesses.

Not a pipeline stage (RPD.md Sec 4.5): this package validates the
*assembled* pipeline (``fdge2.models.ReferenceFDModel``, which wires
``graph_building`` + ``graph_augmenting`` + ``embedding`` together), plus
regression parity against the legacy ``fdge_numba`` implementation and
perf/memory benchmarks. Per RPD.md Sec 5, ``validation/`` is the one
package explicitly allowed to import across all of ``core``,
``graph_building``, ``graph_augmenting``, ``embedding``, and even the legacy
``fdge_numba`` package -- that's its job.

Contents
--------
* ``test_pipeline_behavior.py``  -- T4.1: whole-pipeline behavioral tests
  against ``ReferenceFDModel`` (adapted from
  ``fdge_numba/test_fd_numba.py``).
* ``test_regression_parity.py``  -- T4.2: epoch-by-epoch numeric parity
  vs. ``fdge_numba.forcedirected_numba.model_204_shell.FDModel``
  (RPD.md success metric #2).
* ``bench_embedding_perf.py``    -- T4.3: Numba-vs-vectorized-NumPy speed
  benchmark (success metric #6) and peak-RSS memory scaling check
  (success metric #5). Runnable script, not a pytest suite.
"""
