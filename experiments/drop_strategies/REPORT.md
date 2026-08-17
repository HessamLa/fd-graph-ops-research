# Drop strategies: is it faster to NOT compute the dropped rows?

Date: 2026-08-13
Script: `bench_drop_strategies.py`
Raw output: `results_2026-08-13.log`

## The question

`fodined/embedding/drop.py` zeroes about half of the rows of `dZ` **after**
the kernel computed all of them. Half of the arithmetic goes to the waste
bin. The question is:

> If the engine does not compute the dropped rows, is it faster, and does it
> use less memory?

## The answer

**Yes on PubMed, and no on Cora.** The size of the graph controls the
answer.

| Graph | Best variant | Speed | Memory | Quality |
| --- | --- | --- | --- | --- |
| Cora, 2,708 nodes | `mask_in_kernel` | 1.63x | equal | equal |
| PubMed, 19,717 nodes | `row_subset` | 1.62x | 15% less | equal |

On PubMed a true subset of the rows gives 1.62x more epochs each second,
and it uses 49 MB in place of 58 MB. It is also **1.55x faster than a step
with no drop at all**. Thus the step really becomes less expensive.

On Cora the same variant is SLOWER than a simple mask, because the gather
costs more than the arithmetic that it removes.

---

## Method

### The variants

Each variant is a different version of the `_step` kernel of fodined. The
force law and the plan do not change.

| Variant | What it does |
| --- | --- |
| `none` | No drop. The reference for a full step. |
| `rows_after` | The current fodined. It computes every row, then it zeroes about half of the rows. **This is the baseline.** |
| `cells_after` | The second strategy of `drop_steady_rate`. It zeroes single cells. |
| `mask_in_kernel` | It gives the keep mask to the kernel, and it multiplies before the scatter-add. This removes the WRITE of a dropped row, but not its arithmetic. |
| `row_subset` | A true subset. `lax.top_k` on random values takes a fixed fraction of the row slots of each batch, thus the shape stays fixed and one compile is sufficient. |
| `batch_subset` | It drops whole BATCHES, and not single rows. The gather is cheap, but the plan sorts the rows by width, thus this drops nodes of a similar degree together. |

### The setup

* Graphs: Cora (2,708 nodes) and PubMed (19,717 nodes).
* Seeds: 42, 56, and 88. The seed controls the augmentation sample, the
  initial `Z`, and the drop mask.
* 1000 epochs, `n_dim=128`, `drop_rate=0.5`.
* The hyperparameters are those of `fodined/modular.py`.
* GPU: GeForce GTX 950 with 2 GB.

### Three decisions that make the measurement correct

1. **Each run has its own process.** JAX holds the memory of a device for
   the life of a process. A shared process gives the peak of all the
   variants together, and not of one variant.
2. **`XLA_PYTHON_CLIENT_PREALLOCATE=false`.** The default of JAX takes 75%
   of the device at the start. Without this setting every measurement of
   the memory gives the same number.
3. **The table shows the drop that really occurred.** A variant can be fast
   because it does nothing. The columns `0rows` and `0cells` count the zero
   rows and the zero cells of `dZ` after the last epoch. See finding 7.

### The quality measure

Speed without quality is not a result. Each run also does the
link-prediction task of `fodined/modular.py` and reports the AUC, and it
reports the final `||dZ||`.

---

## Results

Mean of the three seeds. The speed is relative to `rows_after`.

### Cora, 2,708 nodes (the plan has 12 batches)

| Variant | epochs/s | speedup | peak GPU | rows dropped | AUC |
| --- | --- | --- | --- | --- | --- |
| `none` | 1800.9 | 1.60x | 7 MB | 0% | 0.9916 ± 0.001 |
| `rows_after` | 1126.7 | 1.00x | 6 MB | 51% | 0.9934 ± 0.000 |
| `cells_after` | 1463.7 | 1.30x | 8 MB | 0% (52% cells) | 0.9932 ± 0.001 |
| **`mask_in_kernel`** | **1838.2** | **1.63x** | 7 MB | 51% | 0.9934 ± 0.000 |
| `row_subset` | 1531.7 | 1.36x | 7 MB | 50% | 0.9923 ± 0.001 |
| `batch_subset` | 1847.9 | 1.64x | 7 MB | **0% (no-op)** | 0.9916 ± 0.001 |

### PubMed, 19,717 nodes (the plan has 27 batches)

| Variant | epochs/s | speedup | peak GPU | rows dropped | AUC |
| --- | --- | --- | --- | --- | --- |
| `none` | 183.7 | 1.05x | 49 MB | 0% | 0.9894 ± 0.000 |
| `rows_after` | 175.5 | 1.00x | 58 MB | 50% | 0.9914 ± 0.000 |
| `cells_after` | 164.2 | 0.94x | 60 MB | 0% (50% cells) | 0.9918 ± 0.000 |
| `mask_in_kernel` | 182.8 | 1.04x | 50 MB | 50% | 0.9914 ± 0.000 |
| **`row_subset`** | **284.1** | **1.62x** | **49 MB** | 50% | 0.9917 ± 0.000 |
| `batch_subset` | 284.9 | 1.62x | 50 MB | 46% | 0.9912 ± 0.000 |

---

## Findings

### 1. The answer changes with the size of the graph

`row_subset` gives 1.62x on PubMed and only 1.36x on Cora. On Cora it is
17% SLOWER than the simple `mask_in_kernel`.

Cora has 2,708 nodes and 128 dimensions. That work is too small for the
GPU, thus the time is the time to start the kernels, and not the time to
compute. Less arithmetic therefore gives nothing. PubMed is 7 times larger,
the GPU is busy, and the removed work becomes visible.

### 2. A true subset is a real win at the larger size

On PubMed `row_subset` gives 284.1 epochs/s against 175.5. It uses 49 MB
against 58 MB, which is 15% less. The AUC is 0.9917 against 0.9914, thus
there is no cost in quality.

It is also faster than `none` (183.7 epochs/s). A step that computes half of
the rows is 1.55x faster than a step that computes all of them. This is the
direct answer to the question of this experiment.

### 3. The gather can cost more than the arithmetic

Cora shows this clearly. `mask_in_kernel` keeps all the arithmetic and only
removes the write, and it gives 1.63x. `row_subset` removes half of the
arithmetic, and it gives 1.36x. The difference is the gather.

Do not assume that less arithmetic gives less time. This kernel is limited
by memory bandwidth.

### 4. `mask_in_kernel` is free, and it is exactly equal

`mask_in_kernel` gives the same final `||dZ||` and the same AUC as
`rows_after` on both graphs and all three seeds. The values agree to four
decimal places, because the two variants compute the same mask from the same
key.

It is never slower: 1.63x on Cora and 1.04x on PubMed. Thus it is a strict
improvement of the current code.

This equality also tests the harness. If it were not equal, the harness
would be wrong.

### 5. `cells_after` costs more and gives nothing

It is the only variant that is slower than the baseline (0.94x on PubMed),
and it uses the most memory (60 MB). The cause is the mask: it needs
`(n, d)` random values, and a row drop needs only `(n,)`.

Its AUC is equal to `rows_after`. Thus there is at present no measured
reason to prefer cells. Both strategies work, and the research can use both,
but not for speed.

### 6. The drop is a stabilizer, and not only a regularizer

Turn the drop off, and the layout gets worse:

* Cora: `none` ends with `||dZ||` = 1022, and `rows_after` ends with 0.28.
  Without the drop the layout oscillates. It does not settle.
* PubMed: `none` has the LOWEST AUC of all the variants (0.9894 against
  0.9914).

Thus the drop is not a cost that a faster engine should remove. It improves
the result.

### 7. A no-op can look like the best variant

`batch_subset` was the fastest variant on Cora at 1.64x. It dropped
**nothing**. Each rung of the Cora plan holds one batch, thus
`max(1, round(1 x 0.5))` keeps that batch.

The `0rows` column found this. Without that column the report would
recommend a variant that does no work.

On PubMed `batch_subset` works, but it drops 46% and not 50%, and the plan
sorts the rows by width, thus it drops nodes of a similar degree together.
It is not faster than `row_subset`. Therefore `row_subset` is better.

### 8. `row_subset` has a large compile time

`row_subset` needs 6 s to 11 s to compile, and the other variants need 1.5 s
to 3.2 s. The `top_k` and the gather of each rung cause this. It is a
one-time cost. It has no importance for 2000 epochs, but it has importance
for a short run.

---

## Conclusion

1. **Use `mask_in_kernel` always.** It gives the same numbers as the current
   code, and it is never slower. On Cora it is 1.63x faster.
2. **Use `row_subset` for a large graph.** Above approximately 10,000 nodes
   it gives 1.62x and 15% less memory, and the quality does not change.
3. **Keep the drop.** It stabilizes the layout and it improves the AUC. Do
   not remove it to gain speed.
4. **Do not use `cells_after` for speed.** It is slower and it uses more
   memory. Keep it for the research question that it answers, which is a
   different one.
5. **Do not use `batch_subset`.** It gives the same speed as `row_subset`,
   its drop is correlated with the degree, and it silently does nothing on a
   small graph.

Before `row_subset` becomes the default, note that it is not exactly the
same regularizer:

* It drops a fixed COUNT in each batch. It does not drop each row with an
  independent probability. The rate is exact, and not steady.
* A hub row that the plan divided into more than one virtual row can lose
  one part and keep another. That node then moves with a PART of its force.
  A row drop gives all or nothing.

---

## Limits of this experiment

* **Two graphs only.** The crossover point is between 2,708 and 19,717
  nodes. These two points do not find it. A sweep of the size is necessary.
* **One GPU.** A GeForce GTX 950 with 2 GB. A larger GPU moves the crossover
  up, because a larger graph is then necessary to fill it.
* **One quality measure.** The AUC of one link-prediction task. A different
  task can react differently to a change of the regularizer.
* **1000 epochs.** A longer run can show a difference in the quality that
  1000 epochs hide.
* `cells_after` and `row_subset` are different regularizers, thus a
  comparison of their AUC is not a comparison of two implementations of one
  method.

## How to repeat this experiment

```bash
# From the repo root (fdmap/). The full matrix needs about 30 minutes.
.venv/bin/python experiments/drop_strategies/bench_drop_strategies.py --epochs 1000

# One graph, and a fast check.
.venv/bin/python experiments/drop_strategies/bench_drop_strategies.py \
    --graphs cora --seeds 42 --epochs 200
```

`_cache/` holds the augmented `D` of each graph and seed. Delete a file to
build it again. The hop distances come from Pruned Landmark Labeling,
because a dense SciPy matrix for PubMed is 3.1 GB. See
`experiments/bench_shortest_path_gemsec.py`.
