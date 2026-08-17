"""fdge_jax_sell_c_sigma.validation -- whole-pipeline behavioral tests + perf benchmark.

Kernel-level correctness (bucketed kernel vs. a naive dense oracle, and vs.
the sibling ``fdge_jax`` flat-edge-list engine) is already covered by
``embedding/test_sell_c_sigma.py``; augmentation correctness (vs. an
independent scipy/networkx oracle) by
``graph_augmenting/test_*_smoke.py``. What's left, and what this package
checks, is that the **assembled** ``fdge_jax_sell_c_sigma.models`` pipeline
behaves correctly end-to-end (``test_pipeline_behavior.py``), and how fast
the bucketed engine actually runs on the large, power-law-degree graphs it
was built for (``bench_embedding_perf.py``).
"""
