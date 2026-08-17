"""fdge_jax.validation -- whole-pipeline behavioral + regression-parity tests.

Kernel-level correctness (JAX kernel vs. a naive dense oracle) is already
covered by ``embedding/test_shell_force.py``; augmentation correctness (vs.
an independent scipy/networkx oracle) by ``graph_augmenting/test_*_smoke.py``.
What's left, and what this package checks, is that the **assembled**
``fdge_jax.models`` pipeline behaves correctly end-to-end, and that it
tracks fdge2's (Numba) physics -- not bit-for-bit (different summation
order, different RNG stream shape), but the same trajectory to float64-ish
precision. See ``docs/DESIGN.md``'s "Numeric parity expectation".
"""
