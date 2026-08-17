# 2026-08-08 — JAX profiling: what exists, what we want

## The ask

> Include JAX profiling. Using an argument, the profiler should be turned
> on/off. Memory utilization, I/O, CPU/GPU/TPU utilization, etc are the
> metrics we are interested in.

Status: **researched and verified, not implemented.** The design below is
recorded in `TODOS.md` §2.

## API surface

Read the JAX profiling and device-memory-profiling docs, then **verified
every API actually exists in the installed JAX** rather than trusting the
docs page. Environment: JAX 0.6.2, one CUDA device.

| API | Present | Use here |
|---|---|---|
| `jax.profiler.start_trace(dir, profiler_options=...)` / `stop_trace()` | yes | explicit on/off — maps cleanly to a flag |
| `jax.profiler.trace(dir)` context manager | yes | wrap the epoch loop |
| `jax.profiler.ProfileOptions()` | yes | `host_tracer_level` 0–3, `device_tracer_level` 0–1, `python_tracer_level` 0–1 |
| `jax.profiler.StepTraceAnnotation` | yes | label each epoch as a step → per-epoch breakdown |
| `jax.profiler.annotate_function` | yes | label plan build / kernel / drop separately |
| `jax.profiler.save_device_memory_profile("mem.prof")` | yes | device memory, viewed with `pprof --web` |
| `jax.live_arrays()` | yes | cheap always-on leak check, no pprof needed |

## Things that will bite during implementation

- **Async dispatch.** Call `.block_until_ready()` before stopping a trace or
  saving a memory profile, or the trace measures dispatch rather than
  execution.
- **`jit` is opaque to the memory profiler.** Everything inside `_step` is
  attributed to `_step` as a whole. To attribute *within* it, use
  `annotate_function` / `TraceAnnotation` — the memory profile will not do
  it.
- **Compile time dominates.** `validation/bench_embedding_perf.py` already
  goes to some trouble to separate epoch-1 compile cost from steady state
  (its module docstring explains why: epoch 1 ran ~150x longer even with
  the engine pre-warmed, because XLA also compiles `embed()`'s own bare
  array ops). Profiling must preserve that split or the trace will be
  mostly XLA compilation.
- **Diffing beats absolute numbers** for leak-hunting:
  `pprof --web --diff_base a.prof b.prof`.
- **Viewing**: `tensorboard --logdir=<dir>`, or `xprof --port 8791 <dir>`;
  `jax.profiler.trace(..., create_perfetto_link=True)` for Perfetto.

## Proposed shape

`profile` flag on `ForceDirected.embed`, off by default, zero cost when off
(e.g. `embed(..., profile="/tmp/fdge-trace")`). `StepTraceAnnotation` per
epoch; `annotate_function` on plan build, kernel, and drop; optional
`save_device_memory_profile` at `train_end`.

## The metric that actually matters

**Report peak device memory in the benchmark output, alongside Mcells/s.**
That is the number the complexity work is really about — see the
[complexity thread](2026-08-08-complexity-modes-and-oom.md). Throughput is
already measured; memory is not.

## Environment caveat worth knowing

The dev GPU here is a 2 GB GTX 950, shared with a Jupyter kernel that holds
~1.5 GB. CUDA init fails intermittently with OOM during this work;
`JAX_PLATFORMS=cpu` is the workaround. Device-memory headroom is far
tighter than host headroom on this machine, which makes device profiling
more useful here than it would be on a larger card — and also makes it
easy to mistake contention for a regression.
