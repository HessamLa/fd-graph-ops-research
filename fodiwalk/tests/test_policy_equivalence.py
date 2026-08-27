"""test_policy_equivalence.py -- the new stage 2 EQUALS the old methods.

The policies moved from `Fodiwalk._build_D*` into module functions of
`augment_graph`. This file proves the move changed no number: for every one
of the 16 configurations of `golden.AUGMENT_CASES`, the old method and
`policies.build` give the same `D`, the same `freq`, the same statistics,
the same `info` counts -- BYTE for byte -- and they leave the generator in
the same state.

The generator matters as much as the matrix. ONE `np.random.default_rng`
flows through the walks, the bucket sample, the far draw and the landmarks,
thus a moved call or one extra draw changes every pair after it and looks
like a defect of the physics. The last assertion draws from both generators
and compares.

The file RETIRES ITSELF: `importorskip` skips it once `fodiwalk.fodiwalk`
is deleted and only the new code remains.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("fodiwalk.fodiwalk")

from fodiwalk.augment_graph import policies
from fodiwalk.augment_graph.result import AugmentSpec
from fodiwalk.fodiwalk import Fodiwalk
from fodiwalk.tests import golden

SEED = 42


def csr_equal(a, b):
    """Two CSR matrices, entry for entry and dtype for dtype."""
    return (a.shape == b.shape
            and np.array_equal(a.indptr, b.indptr)
            and np.array_equal(a.indices, b.indices)
            and np.array_equal(a.data, b.data)
            and a.data.dtype == b.data.dtype)


@pytest.mark.parametrize("name", sorted(golden.AUGMENT_CASES))
def test_policy_equals_old_method(name, cora):
    A, n = cora
    case = golden.AUGMENT_CASES[name]

    old = Fodiwalk(n_dim=64, seed=SEED, **case)
    D_old = old._build_D(A)

    rng = np.random.default_rng(SEED)      # as `ForceDirected.__init__` makes it
    aug = policies.build(A, n, AugmentSpec.from_config(old.cfg), rng)

    assert csr_equal(aug.D, D_old), f"{name}: D differs"
    assert np.array_equal(aug.D.indptr, D_old.indptr)
    assert np.array_equal(aug.D.indices, D_old.indices)
    assert np.array_equal(aug.D.data, D_old.data)

    if old.freq is None:
        assert aug.freq is None
    else:
        assert np.array_equal(aug.freq, old.freq), f"{name}: freq differs"
        assert aug.freq.dtype == old.freq.dtype

    assert set(aug.stats) == set(old.stats), f"{name}: stats keys differ"
    for key, ref in old.stats.items():
        got = aug.stats[key]
        if sp.issparse(ref):
            assert csr_equal(got, ref), f"{name}: stats[{key!r}] differs"
        elif isinstance(ref, np.ndarray):
            assert np.array_equal(got, ref), f"{name}: stats[{key!r}] differs"
            assert got.dtype == ref.dtype, f"{name}: stats[{key!r}] dtype"
        else:
            assert got == ref, f"{name}: stats[{key!r}] differs"

    assert set(aug.info) == set(old.info), f"{name}: info keys differ"
    for key, ref in old.info.items():
        if key.startswith("t_"):
            continue                        # a wall clock, the KEY is the gate
        assert aug.info[key] == ref, f"{name}: info[{key!r}] differs"

    # The generator state IS part of the result: the link prediction and the
    # hop sample draw from the same one, after this stage.
    assert np.array_equal(old.rng.integers(0, 2 ** 31, 8),
                          rng.integers(0, 2 ** 31, 8)), \
        f"{name}: the generator is in another state"


def test_unknown_pairs_raises():
    """An unknown name raises the message the old dispatch raised."""
    with pytest.raises(ValueError, match="Unknown pairs policy"):
        policies.build(None, 0, AugmentSpec(pairs="ball"), None)
