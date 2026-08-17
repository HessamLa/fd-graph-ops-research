# The root of the project, moved here on 2026-08-16 23:50 PDT

## Why

A clean-up. The active work is `fodined/` (the package), `experiments/`
(the measurements), and `CLAUDE.md` (the instructions). Everything else in
the root moved here.

## Why a subdirectory, and not `archive/` itself

`archive/` already held `fdmap_bucketed_bench_jax.py` and
`fdmap_bucketed_bench_torch.py`, and the root held files with the SAME
names and DIFFERENT content (13089 against 12907 bytes, and 15463 against
14730). A flat move would overwrite the older copies and destroy them.
Both pairs are now kept: the older ones stay in `archive/`, and the newer
ones are here.

## What stayed in the root

| Kept | Why |
| --- | --- |
| `CLAUDE.md` | the instructions of the project |
| `fodined/` | the package |
| `experiments/` | the measurements, and the logs |
| `.venv/` | the interpreter that `CLAUDE.md` names |
| `data_cache/` | the graphs that `fodined` and `experiments` read |
| `.claude/` | the settings of the assistant |
| `archive/` | this directory |

## What moved: 49 entries

The origin packages (`fdge2/`, `fdge_jax/`, `fdge_jax_sell_c_sigma/`,
`fdge_numba/`), the single-file benchmarks (`fdmap_*.py`), the loaders
(`dataset_loaders_*.py`), the outputs (`outputs_*/`, `embedding_*.html`,
`embedding_motion*.gif`, `embeddings*.npz`), the documents (`docs/`,
`dev_docs/`, `papers/`, `README.md`), the probes, the tests, and
`data_cache_BKUP/`.

`fdge_jax_sell_c_sigma/modular.py` is the ORIGIN of `fodined/modular.py`.
The file in `fodined/` is the one that runs.

## One reference that this move breaks

Nothing in `fodined/` or `experiments/` IMPORTS anything that moved. The
check was a grep for an import of every moved module, and the only match
was the text "from fdge2" inside a docstring, which is prose.

But the docstrings of `fodined/` point to two documents that are now here:

| The docstring says | The document is now at |
| --- | --- |
| `docs/DESIGN.md` | `archive/root-2026-08-16/docs/DESIGN.md` |
| `dev_docs/fdmap_engine_design_notes.md` | `archive/root-2026-08-16/dev_docs/fdmap_engine_design_notes.md` |

The docstrings were NOT edited, because they are a record of what the code
said. Read the paths through this table.
