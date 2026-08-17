# %%
"""fdmap_jax_sell_c_sigma-2.py -- fdmap_jax_sell_c_sigma.py, over many datasets.

Same pipeline as `fdmap_jax_sell_c_sigma.py` (build a kNN/MST graph -> force-
directed embed to 2D on the SELL-C-sigma bucketed engine -> scatter + motion
gif + interactive html), run once per dataset instead of once for iris/digits.

The loaders for the 11 datasets below live in three sibling modules (split
so they could be developed independently without merge conflicts):

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

Run: python fdmap_jax_sell_c_sigma-2.py   (from the repo root, fdmap/)

Env overrides (handy for iterating on one dataset at a time instead of all
11, since a full pass -- graph build + N epochs + gif + html, times 11 --
is genuinely slow):
  FDMAP_DATASETS   comma-separated subset of names, e.g. "swiss_roll,mnist"
  FDMAP_N_SAMPLES  override every loader's n_samples (default: loader's own)
  FDMAP_EPOCHS     override EPOCHS below
"""
import base64
import os
import traceback
from io import BytesIO

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fdge_jax_sell_c_sigma import graph_building
from fdge_jax_sell_c_sigma.core import Callback_Base
from fdge_jax_sell_c_sigma.models import EuclideanDistanceModel

import dataset_loaders_synthetic as dl_synth
import dataset_loaders_vision as dl_vision
import dataset_loaders_exotic as dl_exotic

# %%
## ---------------------------------
## Config
## ---------------------------------
# 2000 epochs / snapshot-every-10 (the single-dataset script's defaults)
# makes a ~200-frame gif per dataset; times 11 datasets that's an hour of
# animation rendering alone. Trade fidelity for "all 11 actually finish".
EPOCHS = int(os.environ.get("FDMAP_EPOCHS", 1000))
SNAPSHOT_EVERY = 20
K_NEIGHBORS = 9
OUT_DIR = "outputs_sell_c_sigma_multi"

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
    "cifar10":       (dl_vision.load_cifar10, {}),
    "cifar100":      (dl_vision.load_cifar100, {}),
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
    bleed epoch snapshots into each other."""

    def __init__(self, name):
        self.name = name
        self.snapshots = {}

    def on_epoch_end(self, model, **kw):
        epoch = kw["epoch"]
        if epoch % 20 == 0:
            print(f"[{self.name}] epoch {epoch:4d} "
                  f"||dZ|| avg: {np.linalg.norm(model.dZ, axis=1).mean():.3f}")
        if epoch % SNAPSHOT_EVERY == 0:
            self.snapshots[epoch] = model.Z.copy()


def _thumb_data_uri(row, thumbnail_shape, cmap, vmin, vmax):
    """Render one row of `data` as a base64 PNG data URI for hover tooltips."""
    img = np.asarray(row).reshape(thumbnail_shape)
    buf = BytesIO()
    plt.imsave(buf, img, cmap=cmap, vmin=vmin, vmax=vmax, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run_pipeline(name, data, target, thumbnail_shape=None, cmap=None,
                  k_neighbors=K_NEIGHBORS, epochs=EPOCHS, out_dir=OUT_DIR):
    """Build graph -> force-directed embed -> save scatter/gif/html for one
    dataset. Mirrors fdmap_jax_sell_c_sigma.py's cell-by-cell workflow,
    parameterized by dataset instead of hardcoded to iris/digits."""
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
    fdobj = EuclideanDistanceModel(n_dim=2, verbosity=0, seed=42)
    fdobj.attach_callback(rec)

    print(f"[{name}] embedding for {epochs} epochs")
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
    ax.set_title(f"fdmap embedding -- {name} (SELL-C-sigma)")
    fig.savefig(os.path.join(dataset_dir, "embedding_scatter.png"), dpi=150)
    plt.close(fig)

    # --- motion gif ----------------------------------------------------
    epoch_keys = sorted(rec.snapshots, key=int)
    fig, ax = plt.subplots(figsize=(6, 6))

    def _draw_frame(i):
        ax.clear()
        Z_i = rec.snapshots[epoch_keys[i]]
        pos = dict(zip(Gx.nodes(), Z_i))
        nx.draw(Gx, pos=pos, ax=ax, node_size=8, width=0.05,
                node_color=target if is_numeric_target else
                np.unique(target, return_inverse=True)[1], cmap="Spectral")
        ax.set_title(f"{name} -- epoch {epoch_keys[i]}")

    ani = animation.FuncAnimation(fig, _draw_frame, frames=len(epoch_keys), interval=200)
    ani.save(os.path.join(dataset_dir, "embedding_motion.gif"),
              writer=animation.PillowWriter(fps=5))
    plt.close(fig)

    # --- interactive html -----------------------------------------------
    from bokeh.plotting import figure, save, output_file
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.transform import factor_cmap
    from bokeh.palettes import Spectral10
    from bokeh.resources import INLINE

    labels = [str(t) for t in target[[row_of[n_] for n_ in node_list]]]

    src_dict = dict(
        x=embedding[:, 0], y=embedding[:, 1],
        label=labels, idx=list(range(len(node_list))),
    )
    if thumbnail_shape is not None:
        vmin, vmax = float(data.min()), float(data.max())
        src_dict["img"] = [
            _thumb_data_uri(data[row_of[n_]], thumbnail_shape, cmap, vmin, vmax)
            for n_ in node_list
        ]
    nodes_src = ColumnDataSource(src_dict)

    edge_xs, edge_ys = [], []
    for u, v in Gx.edges():
        i, j = row_of[u], row_of[v]
        edge_xs.append([embedding[i, 0], embedding[j, 0]])
        edge_ys.append([embedding[i, 1], embedding[j, 1]])
    edges_src = ColumnDataSource(dict(xs=edge_xs, ys=edge_ys))

    p = figure(width=900, height=900,
               title=f"fdmap embedding ({name}) -- SELL-C-sigma bucketed engine",
               tools="pan,wheel_zoom,box_zoom,reset,save",
               active_scroll="wheel_zoom", match_aspect=True)
    p.grid.visible = False
    p.axis.visible = False
    p.multi_line("xs", "ys", source=edges_src,
                 line_color="#999999", line_width=0.3, line_alpha=0.25)

    n_factors = sorted(set(labels))
    palette = Spectral10 if len(n_factors) <= 10 else "Turbo256"
    node_r = p.scatter("x", "y", source=nodes_src, size=7, line_color=None,
                        fill_color=factor_cmap("label", palette=palette, factors=n_factors),
                        legend_field="label" if len(n_factors) <= 20 else "")
    if len(n_factors) <= 20:
        p.legend.title = name
        p.legend.click_policy = "hide"

    tooltip = '<div style="padding:4px">'
    if thumbnail_shape is not None:
        tooltip += ('<img src="@img" style="width:96px;height:96px;'
                    'image-rendering:pixelated">')
    tooltip += '<div><b>label:</b> @label</div><div><b>node:</b> @idx</div></div>'
    p.add_tools(HoverTool(renderers=[node_r], tooltips=tooltip))

    html_path = os.path.join(dataset_dir, "embedding_interactive.html")
    output_file(html_path, title=f"fdmap embedding ({name})", mode="inline")
    save(p, resources=INLINE)

    print(f"[{name}] wrote {dataset_dir}/{{embeddings.npz,embedding_scatter.png,"
          f"embedding_motion.gif,embedding_interactive.html}}")
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

    print(f"\n[fdmap_jax_sell_c_sigma-2] completed {len(results)}/{len(DATASET_NAMES)} "
          f"datasets: {sorted(results)}")
