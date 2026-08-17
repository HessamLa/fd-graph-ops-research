# %%
"""fdmap_jax_sell_c_sigma.py -- fdmap_jax.py, run through the bucketed engine.

Identical workflow to `fdmap_jax.py` (load digits -> build a kNN/MST graph ->
force-directed embed to 2D -> gif + interactive html), the ONLY difference
is the import: `fdge_jax_sell_c_sigma` instead of `fdge_jax`, so `embed()`
runs on the SELL-C-sigma bucketed/padded engine instead of the flat
edge-list + segment_sum one (see fdge_jax_sell_c_sigma/docs/DESIGN.md).
Same physics, same `EuclideanDistanceModel`, same output shape -- this
script exists to demonstrate that the new engine is a drop-in replacement,
not a different pipeline to learn.

Run: python fdmap_jax_sell_c_sigma.py   (from the repo root, fdmap/)
"""
import numpy as np
import networkx as nx
from sklearn.datasets import load_digits, load_iris
import matplotlib.pyplot as plt

# %%

data, target = load_digits(return_X_y=True)
data, target = load_iris(return_X_y=True)

# %%
## ---------------------------------
## Graph Building
## ---------------------------------
from fdge_jax_sell_c_sigma import graph_building

Gx = graph_building.make_graph(
    data, strategy="union_mst_nndescent", node_labels=target, k_neighbors=9)

nx.draw(Gx, node_size=1, width=0.05, node_color=target, cmap='Spectral')
# %%
## ---------------------------------
## Graph Embedding
## ---------------------------------
from fdge_jax_sell_c_sigma.core import Callback_Base
from fdge_jax_sell_c_sigma.models import ReferenceFDModel, EuclideanDistanceModel


_snapshots = {}  # epoch -> (n, n_dim) embedding snapshot, filled every 10 epochs


class Rec(Callback_Base):
    def on_epoch_end(self, model, **kw):
        """Callback to print epoch info and stash a snapshot every 10 epochs."""
        print(f"epoch {kw['epoch']:4d} ||dZ|| avg: {np.linalg.norm(model.dZ, axis=1).mean():.3f}")
        if kw['epoch'] % 10 == 0:  # snapshot every 10 epochs
            _snapshots[kw['epoch']] = model.Z.copy()

    def on_train_end(self, model, **kw):
        # One np.savez call, many named arrays -- NOT repeated np.savez(f, ...)
        # calls on a file opened in "ab" mode. Each savez call writes an
        # independent zip archive; concatenating them corrupts the offsets,
        # so np.load on the result raises BadZipFile. Writing once here with
        # all snapshots as separate keys is what actually produces a valid,
        # multi-snapshot embeddings file.
        np.savez("embeddings_sell_c_sigma.npz",
                 **{f"epoch_{e:04d}": Z for e, Z in _snapshots.items()})
        print(f"[fdmap_jax_sell_c_sigma] saved {len(_snapshots)} snapshots "
              f"to embeddings_sell_c_sigma.npz")


# fdobj = ReferenceFDModel(n_dim=2, random_drop_rate=0.0, verbosity=0, seed=0)
# fdobj = ReferenceFDModel(n_dim=2, verbosity=0, seed=42)
fdobj = EuclideanDistanceModel(n_dim=2, verbosity=0, seed=42)


DIM = 2
AUX_DIM = 1
fdobj = EuclideanDistanceModel(n_dim=DIM+AUX_DIM, verbosity=0, seed=42)

# import math
# def updateZ(self, lr: float | None = None) -> None:
#     """Apply the assembled step ``self.dZ`` to ``self.Z``.

#     ``Z = Z + lr * dZ`` -- no in-place ``+=``, ``jnp.ndarray`` is
#     immutable. Momentum is not applied here (it lives in
#     ``updateGradient``); this is a thin override seam for a custom
#     integrator if a researcher wants one.
#     """
#     if lr is None:
#         lr = self.lr
#     t, T = self.epoch, self.epochs
#     c = (1-t/T) * 0.5 * ( 1 + math.sin( t * math.pi/4) )
#     dZ = self.dZ.copy()  # copy to avoid mutating the original dZ in-place
#     dZ[:, DIM:] *= c  # anneal the auxiliary dimensions to ~0 over time
#     self.Z = self.Z + dZ# anneal the auxiliary dimensions to ~0 over time

# fdobj.updateZ = updateZ.__get__(fdobj)  # bind the method to fdobj

fdobj.attach_callback(Rec())
# %%
fdobj.embed(Gx, epochs=2000, verbosity=3)
embedding = fdobj.get_embeddings()[:, :DIM]  # slice off the auxiliary dimensions for the final embedding

plt.scatter(embedding[:,0], embedding[:,1], c=target, cmap='Spectral', s=8, alpha=0.9)
plt.legend(np.unique(target))
plt.show()

# generate a motion gif
import matplotlib.animation as animation

snapshots = np.load("embeddings_sell_c_sigma.npz")
epoch_keys = sorted(snapshots.files, key=lambda k: int(k.split("_")[1]))

fig, ax = plt.subplots(figsize=(6, 6))


def _draw_frame(i):
    ax.clear()
    Z_i = snapshots[epoch_keys[i]]
    pos = dict(zip(Gx.nodes(), Z_i))
    nx.draw(Gx, pos=pos, ax=ax, node_size=8, width=0.05,
            node_color=target, cmap="Spectral")
    ax.set_title(epoch_keys[i].replace("_", " "))


ani = animation.FuncAnimation(fig, _draw_frame, frames=len(epoch_keys), interval=200)
ani.save("embedding_motion_sell_c_sigma.gif", writer=animation.PillowWriter(fps=5))
plt.close(fig)
print(f"[fdmap_jax_sell_c_sigma] saved motion gif with {len(epoch_keys)} frames "
      f"to embedding_motion_sell_c_sigma.gif")

# %%
# to interactive html.
# nodes positioned by the embedding, colored by target label;
# hovering a node shows its original 8x8 digit image.
import base64
from io import BytesIO

from bokeh.plotting import figure, save, output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral10
from bokeh.resources import INLINE

# --- node order guard -------------------------------------------------
# Everything below (and the gif cell above) assumes row i of `embedding`
# describes node i of Gx. Make that explicit instead of trusting the
# iteration order of Gx.nodes(): a silent mismatch here does not raise,
# it just produces a plausible-looking but wrong picture.
node_list = list(Gx.nodes())
row_of = {n: i for i, n in enumerate(node_list)}
assert node_list == list(range(len(data))), \
    "Gx node keys are not 0..n-1; embedding rows cannot be matched by position"

# labels: prefer whatever make_graph stored on the graph, fall back to target
labels = [str(Gx.nodes[n].get("label", target[row_of[n]])) for n in node_list]

# --- 8x8 digit thumbnails as base64 PNG data URIs ---------------------
def _thumb(row):
    buf = BytesIO()
    plt.imsave(buf, row.reshape(8, 8), cmap="gray_r",
               vmin=0, vmax=16, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

imgs = [_thumb(data[row_of[n]]) for n in node_list]

# --- data sources ------------------------------------------------------
nodes_src = ColumnDataSource(dict(
    x=embedding[:, 0], y=embedding[:, 1],
    label=labels, img=imgs, idx=list(range(len(node_list))),
))

# one multi_line glyph for all edges, drawn once from the final embedding
edge_xs, edge_ys = [], []
for u, v in Gx.edges():
    i, j = row_of[u], row_of[v]
    edge_xs.append([embedding[i, 0], embedding[j, 0]])
    edge_ys.append([embedding[i, 1], embedding[j, 1]])
edges_src = ColumnDataSource(dict(xs=edge_xs, ys=edge_ys))

# --- figure ------------------------------------------------------------
p = figure(width=900, height=900,
           title="fdmap embedding (digits) -- SELL-C-sigma bucketed engine",
           tools="pan,wheel_zoom,box_zoom,reset,save",
           active_scroll="wheel_zoom", match_aspect=True)
p.grid.visible = False
p.axis.visible = False

p.multi_line("xs", "ys", source=edges_src,
             line_color="#999999", line_width=0.3, line_alpha=0.25)

node_r = p.scatter("x", "y", source=nodes_src, size=7,
                   line_color=None,
                   fill_color=factor_cmap("label", palette=Spectral10,
                                          factors=sorted(set(labels))),
                   legend_field="label")
p.legend.title = "digit"
p.legend.click_policy = "hide"

# hover on nodes only -- attaching it to the edge glyph as well makes the
# tooltip flicker as the cursor crosses line segments
p.add_tools(HoverTool(renderers=[node_r], tooltips="""
    <div style="padding:4px">
      <img src="@img" style="width:96px;height:96px;image-rendering:pixelated">
      <div><b>digit:</b> @label</div>
      <div><b>node:</b> @idx</div>
    </div>
"""))

output_file("embedding_interactive_sell_c_sigma.html",
            title="fdmap embedding (SELL-C-sigma)", mode="inline")
save(p, resources=INLINE)   # INLINE -> single self-contained file, works offline
print("[fdmap_jax_sell_c_sigma] wrote embedding_interactive_sell_c_sigma.html")



# %%
if __name__ == "__main__":
    print(f"[fdmap_jax_sell_c_sigma] embedding shape: {embedding.shape}, "
          f"finite: {np.isfinite(embedding).all()}")

    import os
    out_path = os.environ.get("FDMAP_SELL_C_SIGMA_OUT")
    if out_path:
        np.savez(out_path, embedding=embedding, target=target,
                 edges=np.array(list(Gx.edges())))
        print(f"[fdmap_jax_sell_c_sigma] saved embedding + graph to {out_path}")



