# %%
"""fdmap_jax_auxdim-2.py -- fdmap_jax_sell_c_sigma-2.py, retargeted at the
auxiliary-dimension model instead of the plain SELL-C-sigma model.

Same pipeline as `fdmap_jax_sell_c_sigma-2.py` (build a kNN/MST graph ->
force-directed embed -> scatter + motion gif + interactive html), run once
per dataset, but the embedding step uses `EuclideanAuxDimModel` from
`model_fdmap_auxdim` instead of `EuclideanDistanceModel`: it embeds in
`d + aux` dimensions and anneals the `aux` auxiliary dimensions toward zero
over training, to give the layout "extra room" to route around topological
obstructions early on before collapsing back to a genuinely `d`-dimensional
embedding. This script demonstrates that model at the "2+2 dims" config
(`D=2, AUX_D=2, DECAY_SCHEDULE="linear"`); it is a demo/test harness, not
documentation for the idea -- see `model_fdmap_auxdim/idea.md` for the
design rationale, the alternative decay schedules, and the known risks
(tug-of-war between forces and the shrink step, k1..k4 scale interaction,
aux init scale, compute cost).

The loaders for the datasets below live in three sibling modules (split so
they could be developed independently without merge conflicts), reused here
completely unchanged from the reference script:

  dataset_loaders_synthetic.py -- swiss_roll, s_curve, 3d_clusters
      (sklearn generators, no download)
  dataset_loaders_vision.py    -- mnist, fashion_mnist, cifar10, cifar100
      (torchvision, real downloads, cached under ./data_cache/)
  dataset_loaders_exotic.py    -- coil20, coil100, elephant, scrna_seq
      (Columbia CAVE COIL zips, a parametric/point-cloud elephant, and
      scanpy's real 10x PBMC 3k dataset; also cached under ./data_cache/)

Every loader returns the same 5-key dict -- {"data", "target", "name",
"thumbnail_shape", "cmap"} -- which is exactly what `run_pipeline` below
needs, so adding a 12th dataset later is "write one more loader function",
not "learn a new pipeline".

Run: python fdmap_jax_auxdim-2.py   (from the repo root, fdmap/)

Env overrides (handy for iterating on one dataset at a time instead of all
of them, since a full pass -- graph build + N epochs + gif + html, times
several datasets -- is genuinely slow):
  FDMAP_DATASETS   comma-separated subset of names, e.g. "swiss_roll,mnist"
  FDMAP_N_SAMPLES  override every loader's n_samples (default: loader's own)
  FDMAP_EPOCHS     override EPOCHS below
  FDMAP_D          override D below (real embedding dimensionality)
  FDMAP_AUX        override AUX_D below (auxiliary scaffolding dimensions)
  FDMAP_DECAY      override DECAY_SCHEDULE below (one of model_fdmap_auxdim's
                   DECAY_SCHEDULES, e.g. "linear", "cosine", "sigmoid")
"""
import os
import subprocess
import sys
import traceback

import numpy as np
import matplotlib.pyplot as plt

from fdge_jax_sell_c_sigma import graph_building
from fdge_jax_sell_c_sigma.core import Callback_Base
from model_fdmap_auxdim import EuclideanAuxDimModel

import dataset_loaders_synthetic as dl_synth
import dataset_loaders_vision as dl_vision
import dataset_loaders_exotic as dl_exotic

# %%
## ---------------------------------
## Config
## ---------------------------------
# 2000 epochs / snapshot-every-10 (the single-dataset script's defaults)
# makes a ~200-frame gif per dataset; times several datasets that's an hour
# of animation rendering alone. Trade fidelity for "all of them actually
# finish".
EPOCHS = int(os.environ.get("FDMAP_EPOCHS", 4000))
SNAPSHOT_EVERY = 10
K_NEIGHBORS = 15
OUT_DIR = "outputs_auxdim_multi"

# The "2+2 dims" configuration this script exists to demonstrate: embed in
# D + AUX_D dims, anneal the AUX_D ones to ~0 by the end of training, return only
# the leading D dims from get_embeddings(). See model_fdmap_auxdim/idea.md
# for why (and DECAY_SCHEDULES in model_fdmap_auxdim/model.py for the other
# schedule choices besides "linear").
D = int(os.environ.get("FDMAP_D", 2))
AUX_D = int(os.environ.get("FDMAP_AUX", 2))
# DECAY_SCHEDULE = os.environ.get("FDMAP_DECAY", "linear")
DECAY_SCHEDULE = os.environ.get("FDMAP_DECAY", "linear_sine")

# MST edge construction is a dense O(n^2) distance matrix (see
# graph_building/building_blocks.py:mst_edges) -- fine into the low
# thousands, not for a raw 60k-row CIFAR-100 train split. Loaders already
# default to a capped n_samples; FDMAP_N_SAMPLES overrides all of them at
# once for quick smoke tests.
_env_n = os.environ.get("FDMAP_N_SAMPLES")
N_SAMPLES = int(_env_n) if _env_n else None

# name -> (loader, kwargs). kwargs stay empty so each loader's own default
# n_samples is used, unless FDMAP_N_SAMPLES overrides it below.
DATASET_REGISTRY = {
    "swiss_roll":    (dl_synth.load_swiss_roll, {}),
    "s_curve":       (dl_synth.load_s_curve, {}),
    "3d_clusters":   (dl_synth.load_3d_clusters, {}),
    "mnist":         (dl_vision.load_mnist, {}),
    "fashion_mnist": (dl_vision.load_fashion_mnist, {}),
    # "cifar10":       (dl_vision.load_cifar10, {}),
    # "cifar100":      (dl_vision.load_cifar100, {}),
    "coil20":        (dl_exotic.load_coil20, {}),
    "coil100":       (dl_exotic.load_coil100, {}),
    "elephant":      (dl_exotic.load_elephant, {}),
    "scrna_seq":     (dl_exotic.load_scrna_seq, {}),
}

_selected = os.environ.get("FDMAP_DATASETS")
DATASET_NAMES = (
    [n.strip() for n in _selected.split(",") if n.strip()]
    if _selected else list(DATASET_REGISTRY)
)


# %%
## ---------------------------------
## Generic pipeline (graph build -> embed -> scatter/gif/html)
## ---------------------------------
class _Rec(Callback_Base):
    """Per-run snapshot recorder -- one instance per dataset, not a shared
    module-level dict, so consecutive datasets in the loop below don't
    bleed epoch snapshots into each other.

    Snapshots use ``model.get_embeddings()`` rather than raw ``model.Z``
    (the reference script's choice, safe there because
    ``EuclideanDistanceModel.n_dim == d`` so the two are identical). Here
    ``model.Z`` is ``(n, d + aux)`` -- the aux scaffolding columns are still
    live, un-annealed state -- while ``get_embeddings()`` is
    ``EuclideanAuxDimModel``'s own override that slices to the real ``(n,
    d)`` columns. render_gif_html.py's animation expects one 2D position per
    node, so recording raw ``Z`` here would silently hand it ``(n, d+aux)``
    frames instead and break `nx.draw`'s position dict."""

    def __init__(self, name):
        self.name = name
        self.snapshots = {}

    def on_epoch_end(self, model, **kw):
        epoch = kw["epoch"]
        if epoch % 20 == 0:
            print(f"[{self.name}] epoch {epoch:4d} "
                  f"||dZ|| avg: {np.linalg.norm(model.dZ, axis=1).mean():.3f}")
        if epoch % SNAPSHOT_EVERY == 0:
            self.snapshots[epoch] = model.get_embeddings().copy()

def run_pipeline(name, data, target, thumbnail_shape=None, cmap=None,
                  k_neighbors=K_NEIGHBORS, epochs=EPOCHS, out_dir=OUT_DIR):
    """Build graph -> force-directed embed -> save scatter/gif/html for one
    dataset. Mirrors fdmap_jax_sell_c_sigma-2.py's run_pipeline, swapping in
    EuclideanAuxDimModel(d=D, aux=AUX_D, decay_schedule=DECAY_SCHEDULE) for the
    plain EuclideanDistanceModel."""
    dataset_dir = os.path.join(out_dir, name)
    os.makedirs(dataset_dir, exist_ok=True)

    data = np.asarray(data, dtype=np.float64)
    target = np.asarray(target)
    n = data.shape[0]
    k = min(k_neighbors, n - 1)

    print(f"[{name}] building graph: n={n}, d={data.shape[1]}, k={k}")
    Gx = graph_building.make_graph(
        data, strategy="union_mst_nndescent", node_labels=target, k_neighbors=k)

    node_list = list(Gx.nodes())
    row_of = {node: i for i, node in enumerate(node_list)}
    assert node_list == list(range(n)), \
        f"[{name}] Gx node keys are not 0..n-1; embedding rows won't line up"

    rec = _Rec(name)
    fdobj = EuclideanAuxDimModel(d=D, aux=AUX_D, decay_schedule=DECAY_SCHEDULE,
                                  verbosity=0, seed=42)
    fdobj.attach_callback(rec)

    print(f"[{name}] embedding for {epochs} epochs (d={D}, aux={AUX_D}, "
          f"decay={DECAY_SCHEDULE})")
    fdobj.embed(Gx, epochs=epochs, verbosity=0)
    embedding = fdobj.get_embeddings()

    # one savez call, all snapshots as separate keys -- see the single-
    # dataset script's on_train_end docstring for why repeated np.savez
    # calls on the same handle would corrupt the archive.
    np.savez(os.path.join(dataset_dir, "embeddings.npz"),
              **{f"epoch_{e:04d}": Z for e, Z in rec.snapshots.items()})

    # --- scatter -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    is_numeric_target = np.issubdtype(target.dtype, np.number)
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                     c=target if is_numeric_target else
                     np.unique(target, return_inverse=True)[1],
                     cmap="Spectral", s=8, alpha=0.9)
    ax.set_title(f"fdmap embedding -- {name} (aux-dim, d={D}+aux={AUX_D}, {DECAY_SCHEDULE})")
    fig.savefig(os.path.join(dataset_dir, "embedding_scatter.png"), dpi=150)
    plt.close(fig)

    # --- hand off gif + interactive html to a detached subprocess ------
    # These are cheap CPU/IO work (matplotlib animation, a bokeh page) but
    # slow in wall-clock terms (~200 animated frames, thumbnail encoding);
    # none of it needs the graph-building/JAX machinery above, so it runs
    # as its own `python render_gif_html.py <job> &`-style process instead
    # of blocking run_pipeline from moving on to the next dataset. Note the
    # snapshots saved above (and the embeddings.npz format read by
    # render_gif_html.py) are already sliced to (n, D) by get_embeddings()/
    # rec.snapshots -- render_gif_html.py itself needs no changes.
    job_path = os.path.join(dataset_dir, "render_input.npz")
    np.savez(
        job_path,
        name=np.array(name),
        dataset_dir=np.array(dataset_dir),
        data=data,
        target=target,
        edges=np.array(list(Gx.edges()), dtype=np.int64),
        embedding=embedding,
        # npz can't store None -- empty array is the "absent" sentinel,
        # decoded back to None on the render side.
        thumbnail_shape=np.array(thumbnail_shape if thumbnail_shape is not None else [], dtype=np.int64),
        cmap=np.array(cmap) if cmap is not None else np.array([], dtype="U1"),
    )
    render_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_gif_html.py")
    subprocess.Popen(
        [sys.executable, render_script, job_path],
        stdout=open(os.path.join(dataset_dir, "render.log"), "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,  # survives this script's own exit/os._exit
    )

    print(f"[{name}] wrote {dataset_dir}/{{embeddings.npz,embedding_scatter.png}}; "
          f"gif+html rendering in the background (see {dataset_dir}/render.log)")
    return embedding


# %%
## ---------------------------------
## Run all datasets
## ---------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for name in DATASET_NAMES:
        loader, kwargs = DATASET_REGISTRY[name]
        if N_SAMPLES is not None:
            kwargs = {**kwargs, "n_samples": N_SAMPLES}
        try:
            print(f"\n=== {name} ===")
            d = loader(**kwargs)
            embedding = run_pipeline(
                d["name"], d["data"], d["target"],
                thumbnail_shape=d["thumbnail_shape"], cmap=d["cmap"],
            )
            results[name] = embedding
            print(f"[{name}] embedding shape: {embedding.shape}, "
                  f"finite: {np.isfinite(embedding).all()}")
        except Exception:
            print(f"[{name}] FAILED, skipping:")
            traceback.print_exc()

    print(f"\n[fdmap_jax_auxdim-2] completed {len(results)}/{len(DATASET_NAMES)} "
          f"datasets: {sorted(results)}")

    # torch (imported by dataset_loaders_vision) and jax's CUDA plugin each
    # own a CUDA context; having both loaded in one process is fine at
    # runtime but their native teardown races at interpreter exit and can
    # segfault/double-free *after* all work above is already done and on
    # disk. Skip Python's normal shutdown instead of chasing that race.
    sys.stdout.flush()
    os._exit(0)
