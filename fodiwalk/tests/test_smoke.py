"""test_smoke.py -- the criteria of PRD B4 and B8, on a small graph.

The engine, the callbacks, the batching, the update-rule seam, and the
class that binds them. Every test here runs in seconds; the recorded runs
are in `test_parity.py`.
"""
from __future__ import annotations

import numpy as np
import pytest

from fodiwalk import Fodiwalk
from forcedirected import Callback_Base, ForceDirected
from fodiwalk.embed.plan_contract import PlaneContractError
from fodiwalk.augment_graph import walks as W


class Recorder(Callback_Base):
    """Every event, in the order it arrives."""

    def __init__(self):
        self.events = []

    def on_train_begin(self, model, **kw): self.events.append("train_begin")
    def on_train_end(self, model, **kw): self.events.append("train_end")
    def on_epoch_begin(self, model, **kw): self.events.append("epoch_begin")
    def on_epoch_end(self, model, **kw): self.events.append("epoch_end")
    def on_batch_begin(self, model, **kw): self.events.append("batch_begin")
    def on_batch_end(self, model, **kw): self.events.append("batch_end")


def _model(**kw):
    """A small model: the defaults of a test, and `kw` wins over them."""
    base = dict(n_dim=8, seed=42, verbosity=0, walks=3, walk_len=8,
                pairs="nbr_walk", weight="min_gap", force="fdlinear",
                random_drop_rate=0.0)
    base.update(kw)
    return Fodiwalk(**base)


# ===========================================================================
# B4 -- the engine, `forcedirected/force_directed.py`
# ===========================================================================
def test_b4_callback_event_order(tiny):
    """B4.1. epochs = 3, batch_count = 2."""
    A, n = tiny
    fw = _model()
    rec = Recorder()
    fw.attach_callback(rec)
    fw.embed(A, epochs=3, batch_count=2)
    per_epoch = ["epoch_begin", "batch_begin", "batch_end",
                 "batch_begin", "batch_end", "epoch_end"]
    assert rec.events == ["train_begin"] + per_epoch * 3 + ["train_end"]


def test_b4_attach_callback_rejects_anything_else(tiny):
    fw = _model()
    with pytest.raises(TypeError):
        fw.attach_callback(object())
    with pytest.raises(ValueError, match="Unknown callback event"):
        fw.notify_callback("no_such_event")


def test_b4_Th_is_the_mean_row_norm():
    """B4.2."""
    rng = np.random.default_rng(0)
    dZ = rng.normal(size=(50, 4)).astype(np.float32)
    want = float(np.linalg.norm(dZ, axis=-1).mean())
    assert ForceDirected.Th(dZ) == pytest.approx(want, rel=1e-6)


def test_b4_batching_is_invariant(tiny):
    """B4.3. `batch_count = k` gives the `dZ` of `batch_count = 1`.

    With the drop OFF. The drop draws ONE key for each batch, thus it is a
    different regularizer at a different batch count -- by design, and not
    a defect of the batching.
    """
    A, n = tiny
    out = []
    for bc in (1, 2, 5):
        fw = _model()
        fw.embed(A, epochs=4, batch_count=bc)
        out.append(np.asarray(fw.dZ))
    assert np.allclose(out[0], out[1], atol=1e-5)
    assert np.allclose(out[0], out[2], atol=1e-5)


def test_b4_updateZ_dispatches_through_optim_and_defaults_to_plain(tiny):
    """B4.4."""
    from forcedirected import optim
    A, n = tiny
    fw = _model()
    assert fw.rule is optim.RULES["plain"]
    assert fw.rule_name == "plain"

    # `plain` IS `Z = Z + lr * dZ`, thus the dispatch changed no number.
    fw.embed(A, epochs=3)
    Z_plain = fw.get_embeddings()
    fw2 = _model()
    fw2.rule = None                    # the pre-refactor one-liner path
    fw2.embed(A, epochs=3)
    assert np.allclose(Z_plain, np.asarray(fw2.Z))

    fw3 = _model(optim="adam")
    assert fw3.rule is optim.RULES["adam"]
    fw3.embed(A, epochs=3)
    assert not np.allclose(Z_plain, np.asarray(fw3.Z))


def test_b4_every_rule_runs_and_reports(tiny):
    """B7.1, on the small graph. A rule returns a finite `Z`, or the run
    RECORDS a divergence (I7). Neither raises."""
    from forcedirected import optim
    A, n = tiny
    for name in optim.RULES:
        fw = _model(optim=name, lr=0.1)
        Z = np.asarray(fw.embed(A, epochs=20))
        assert np.isfinite(Z).all() or fw.diverged, name


def test_i7_a_non_finite_Z_is_recorded_and_returned(tiny):
    """I7. Seven runs of the Cora grid of 2026-08-18 died inside sklearn
    with "Input X contains NaN", thus a real measurement -- this rate
    diverges -- was lost as a stack trace."""
    A, n = tiny
    fw = _model(fdlinear_sign=+1.0, k4=5.0)      # the literal exp(+k4*x)
    Z = np.asarray(fw.embed(A, epochs=60, lr=1e3))
    assert fw.diverged is True
    assert fw.diverged_epoch is not None
    assert not np.isfinite(Z).all()
    assert np.asarray(fw.get_embeddings()).shape == (n, 8)


# ===========================================================================
# B8 -- class Fodiwalk
# ===========================================================================
def test_b8_graph_walk_reproduces_walk_rows_for_the_same_seed(tiny):
    """B8.1."""
    A, n = tiny
    fw = _model()
    got = fw.graph_walk(A, n, np.random.default_rng(42))
    want = W.walk_rows(A, n, 3, 8, np.random.default_rng(42),
                       walker=W.make_walker(A, n, 1.0, 1.0))
    assert set(got) == set(want)
    for k in want:
        if isinstance(want[k], np.ndarray):
            assert np.array_equal(got[k], want[k]), k
        else:
            assert got[k] == want[k], k


def test_b8_augment_graph_builds_the_planes_from_the_registry(tiny):
    """B8.1. The plane list follows the registry, in its order."""
    from fodiwalk.embed.forces import planes_of
    A, n = tiny
    for force, fuse_planes, want_len in (("fdlinear", False, 2),
                                         ("fdlinear", True, 1)):
        fw = _model(force=force, fuse_planes=fuse_planes)
        D = fw.augment_graph(A)
        planes = fw._build_planes(D)
        assert len(planes) == want_len == len(planes_of(fw.law))
        for p in planes:
            assert p.shape == (D.nnz,)


def test_b8_a_missing_freq_raises_and_never_falls_back(tiny):
    """B8.2. THE 2026-08-18 DEFECT: the script's `if` chain gave `fdlinear`
    the planes of another law, thus it read a coefficient plane as `h`.
    The run went to NaN at 2000 epochs under an `fdlinear` label, and
    nothing raised."""
    A, n = tiny
    fw = _model(force="fdlinear")
    D = fw.augment_graph(A)             # this one builds a freq

    bare = _model(force="fdlinear").set_D(D)          # and this one does not
    with pytest.raises(PlaneContractError, match="freq"):
        bare.augment_graph(A)

    bare_fused = _model(force="fdlinear", fuse_planes=True).set_D(D)
    with pytest.raises(PlaneContractError, match="`w`|freq"):
        bare_fused.augment_graph(A)


def test_b8_every_policy_builds_a_freq_for_fdlinear(tiny):
    """The same defect, from the other side: the `buckets` branch returned
    no `freq` until 2026-08-18."""
    A, n = tiny
    for pairs in ("walk", "walk_edges", "nbr_walk"):
        for policy in ("cap", "buckets"):
            fw = _model(pairs=pairs, policy=policy, force="fdlinear",
                        cap=8, bucket_total=200)
            D = fw.augment_graph(A)
            assert fw.freq is not None and fw.freq.shape == (D.nnz,), (
                pairs, policy)


def test_b8_make_graph_is_a_stub_and_fit_does_not_exist(tiny):
    """B8.3. `fit` was removed entirely (2026-08-21): `make_graph` is a
    stub, thus a `fit` chaining it into `embed` promised a stage that was
    never real."""
    A, n = tiny
    fw = _model()
    assert fw.make_graph(A) is A
    assert not hasattr(fw, "fit")


def test_b8_an_unknown_option_is_refused():
    with pytest.raises(TypeError, match="Unknown Fodiwalk option"):
        Fodiwalk(n_dim=8, no_such_knob=1)


def test_b8_the_chunked_plan_gives_the_whole_plan_back(tiny):
    """I6. A chunk is a ROW RANGE, thus the parts are additive and the
    chunked run gives the run of one plan."""
    A, n = tiny
    fw1 = _model()
    fw1.embed(A, epochs=5, batch_count=1)
    fw4 = _model(chunks=4)
    fw4.embed(A, epochs=5, batch_count=4)
    assert fw4.plan_stats["chunks"] == 4
    assert fw4.plan_stats["cells"] == fw1.plan_stats["cells"]
    assert np.allclose(np.asarray(fw1.Z), np.asarray(fw4.Z), atol=1e-4)
