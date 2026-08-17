"""render_gif_html.py -- motion gif + interactive html for one dataset run.

Split out of fdmap_jax_sell_c_sigma-2.py's run_pipeline() so the expensive
part (graph build + N epochs of force-directed relaxation) doesn't have to
wait on the cheap-but-slow part (animating ~200 frames to gif, building a
bokeh page). run_pipeline spawns this as `python render_gif_html.py <job.npz> &`
right after the scatter plot is saved, then moves straight on to the next
dataset instead of blocking on this.

Takes one argument: path to a `render_input.npz` written by run_pipeline
into the dataset's output dir, containing everything needed to reconstruct
the graph and re-render -- run_pipeline's own `embeddings.npz` (epoch
snapshots) is read directly rather than duplicated into the job file.
"""
import base64
import os
import sys
from io import BytesIO

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def _thumb_data_uri(row, thumbnail_shape, cmap, vmin, vmax):
    """Render one row of `data` as a base64 PNG data URI for hover tooltips."""
    img = np.asarray(row).reshape(thumbnail_shape)
    buf = BytesIO()
    plt.imsave(buf, img, cmap=cmap, vmin=vmin, vmax=vmax, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def render(job_path, interval=100):
    job = np.load(job_path, allow_pickle=False)
    name = str(job["name"])
    dataset_dir = str(job["dataset_dir"])
    data = job["data"]
    target = job["target"]
    edges = job["edges"]
    embedding = job["embedding"]
    # sentinel-decode: empty array -> None (npz can't store None directly)
    thumbnail_shape = tuple(job["thumbnail_shape"]) if job["thumbnail_shape"].size else None
    cmap = str(job["cmap"]) if job["cmap"].size else None

    n = data.shape[0]
    Gx = nx.Graph()
    Gx.add_nodes_from(range(n))
    Gx.add_edges_from(edges.tolist())
    node_list = list(Gx.nodes())
    row_of = {node: i for i, node in enumerate(node_list)}

    snaps = np.load(os.path.join(dataset_dir, "embeddings.npz"))
    epoch_keys = sorted(snaps.files, key=lambda k: int(k.split("_")[1]))
    is_numeric_target = np.issubdtype(target.dtype, np.number)
    is_continuous = np.issubdtype(target.dtype, np.floating)

    # --- motion gif ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    def _draw_frame(i):
        ax.clear()
        Z_i = snaps[epoch_keys[i]]
        pos = dict(zip(Gx.nodes(), Z_i))
        nx.draw(Gx, pos=pos, ax=ax, node_size=8, width=0.05,
                node_color=target if is_numeric_target else
                np.unique(target, return_inverse=True)[1], cmap="Spectral")
        ax.set_title(f"{name} -- epoch {epoch_keys[i]}")

    ani = animation.FuncAnimation(fig, _draw_frame, frames=len(epoch_keys), interval=interval)
    ani.save(os.path.join(dataset_dir, "embedding_motion.gif"),
              writer=animation.PillowWriter(fps=5))
    plt.close(fig)

    # --- interactive html -----------------------------------------------
    from bokeh.plotting import figure, save, output_file
    from bokeh.models import ColumnDataSource, HoverTool, ColorBar
    from bokeh.transform import factor_cmap, linear_cmap
    from bokeh.palettes import Spectral10, Turbo256
    from bokeh.resources import INLINE

    labels = [str(t) for t in target[[row_of[n_] for n_ in node_list]]]

    src_dict = dict(
        x=embedding[:, 0], y=embedding[:, 1],
        label=labels, idx=list(range(len(node_list))),
    )
    if is_continuous:
        src_dict["value"] = target[[row_of[n_] for n_ in node_list]].astype(float)
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

    if is_continuous:
        mapper = linear_cmap("value", palette=Turbo256,
                              low=float(target.min()), high=float(target.max()))
        node_r = p.scatter("x", "y", source=nodes_src, size=7, line_color=None,
                            fill_color=mapper)
        p.add_layout(ColorBar(color_mapper=mapper["transform"], title=name), "right")
    else:
        n_factors = sorted(set(labels))
        palette = Spectral10 if len(n_factors) <= 10 else Turbo256
        scatter_kwargs = dict(size=7, line_color=None,
                               fill_color=factor_cmap("label", palette=palette, factors=n_factors))
        if len(n_factors) <= 20:
            scatter_kwargs["legend_field"] = "label"
        node_r = p.scatter("x", "y", source=nodes_src, **scatter_kwargs)
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

    os.remove(job_path)
    print(f"[{name}] wrote {dataset_dir}/{{embedding_motion.gif,embedding_interactive.html}}")


if __name__ == "__main__":
    render(sys.argv[1])
