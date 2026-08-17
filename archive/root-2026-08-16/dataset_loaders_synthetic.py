import numpy as np
from sklearn.datasets import make_swiss_roll, make_s_curve, make_blobs


def load_swiss_roll(n_samples=1500, random_state=0) -> dict:
    data, t = make_swiss_roll(n_samples=n_samples, random_state=random_state)
    return {
        "data": data.astype(np.float64),
        "target": t.astype(np.float64),
        "name": "swiss_roll",
        "thumbnail_shape": None,
        "cmap": None,
    }


def load_s_curve(n_samples=1500, random_state=0) -> dict:
    data, t = make_s_curve(n_samples=n_samples, random_state=random_state)
    return {
        "data": data.astype(np.float64),
        "target": t.astype(np.float64),
        "name": "s_curve",
        "thumbnail_shape": None,
        "cmap": None,
    }


def load_3d_clusters(n_samples=1500, random_state=0) -> dict:
    # cluster_std=1.5 with default center_box (-10, 10) keeps the 6 clusters
    # visually distinct while still overlapping enough for MST/kNN edges to
    # connect them into one graph rather than isolated components.
    data, y = make_blobs(
        n_samples=n_samples,
        n_features=3,
        centers=6,
        cluster_std=1.5,
        random_state=random_state,
    )
    return {
        "data": data.astype(np.float64),
        "target": y.astype(np.int64),
        "name": "3d_clusters",
        "thumbnail_shape": None,
        "cmap": None,
    }
