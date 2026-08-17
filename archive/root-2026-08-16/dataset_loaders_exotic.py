"""Loaders for the four "hard" datasets: COIL-20, COIL-100, elephant, scRNA-seq.

Every function returns {"data", "target", "name", "thumbnail_shape", "cmap"}
so the integration script can loop over all dataset loaders uniformly.
"""
import io
import os
import re
import tarfile
import urllib.request
import zipfile

import numpy as np
from PIL import Image

_CACHE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

_COIL20_URL = "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
_COIL100_URL = "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.tar.gz"
_PBMC3K_CACHE_SUBDIR = "scrna_seq"

_COIL_NAME_RE = re.compile(r"obj(\d+)__\d+\.(?:png|ppm)$")


def _download(url, dest_path):
    if os.path.exists(dest_path):
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".part"
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, dest_path)


def _subsample(data, target, n_samples, random_state):
    n = len(data) if n_samples is None else min(n_samples, len(data))
    idx = np.random.default_rng(random_state).choice(len(data), size=n, replace=False)
    return data[idx], target[idx]


def _iter_coil_archive(archive_path):
    """Yields (obj_id (0-indexed), HxW[x3] uint8 array) for every rotation image.

    Streams the archive in one sequential pass (tar.gz is only cheap to read
    forward once) instead of building a random-access index, which matters
    for COIL-100's ~260MB gzip stream.
    """
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            for name in sorted(zf.namelist()):
                m = _COIL_NAME_RE.search(name)
                if not m:
                    continue
                img = Image.open(io.BytesIO(zf.read(name)))
                yield int(m.group(1)) - 1, np.asarray(img, dtype=np.uint8)
    else:
        with tarfile.open(archive_path, "r|gz") as tf:
            for member in tf:
                m = _COIL_NAME_RE.search(member.name)
                if not m or not member.isfile():
                    continue
                img = Image.open(io.BytesIO(tf.extractfile(member).read()))
                yield int(m.group(1)) - 1, np.asarray(img, dtype=np.uint8)


def _load_coil(url, archive_filename, name, thumbnail_shape, cmap, n_samples, random_state):
    cache_dir = os.path.join(_CACHE_ROOT, name)
    archive_path = os.path.join(cache_dir, archive_filename)
    decoded_cache_path = os.path.join(cache_dir, "decoded.npz")

    # Decoding the archive (thousands of PIL.Image.open calls) dwarfs the
    # download itself and doesn't depend on n_samples, so cache the fully
    # decoded array once instead of re-streaming the tar/zip on every call.
    if os.path.exists(decoded_cache_path):
        cached = np.load(decoded_cache_path)
        data, target = cached["data"], cached["target"]
    else:
        _download(url, archive_path)
        images, obj_ids = [], []
        for obj_id, img in _iter_coil_archive(archive_path):
            images.append(img)
            obj_ids.append(obj_id)
        data = np.stack(images).reshape(len(images), -1).astype(np.float32) / 255.0
        target = np.asarray(obj_ids, dtype=np.int64)
        np.savez(decoded_cache_path, data=data, target=target)

    data, target = _subsample(data, target, n_samples, random_state)

    if not np.isfinite(data).all():
        raise RuntimeError(f"{name} contains non-finite pixel values")
    return {
        "data": data,
        "target": target,
        "name": name,
        "thumbnail_shape": thumbnail_shape,
        "cmap": cmap,
    }


def load_coil20(n_samples=1440, random_state=0) -> dict:
    return _load_coil(
        _COIL20_URL, "coil-20-proc.zip", "coil20", (128, 128), "gray_r", n_samples, random_state
    )


def load_coil100(n_samples=1500, random_state=0) -> dict:
    return _load_coil(
        _COIL100_URL, "coil-100.tar.gz", "coil100", (128, 128, 3), None, n_samples, random_state
    )


# "Drawing an Elephant with Four Complex Parameters" (Mayer, Khairy & Howard,
# Am. J. Phys. 78, 648 (2010)) -- no canonical downloadable elephant point
# cloud/mesh was found for manifold-learning demos, so this is a clearly
# synthetic fallback: the classic 2D Fourier silhouette, extruded to a thin
# 3D body via a small random z-thickness plus in-plane jitter.
_ELEPHANT_P = (50 - 30j, 18 + 8j, 12 - 10j, -14 - 60j, 40 + 20j)


def _elephant_fourier(t, coeffs):
    f = np.zeros_like(t)
    for k, c in enumerate(coeffs):
        f = f + c.real * np.cos(k * t) + c.imag * np.sin(k * t)
    return f


def _elephant_silhouette(t):
    p1, p2, p3, p4, _p5 = _ELEPHANT_P
    npar = 6
    cx = np.zeros(npar, dtype=complex)
    cy = np.zeros(npar, dtype=complex)
    cx[1] = p1.real * 1j
    cx[2] = p2.real * 1j
    cx[3] = p3.real
    cx[5] = p4.real
    cy[1] = p4.imag + p1.imag * 1j
    cy[2] = p2.imag * 1j
    cy[3] = p3.imag * 1j
    x = _elephant_fourier(t, cx)
    y = _elephant_fourier(t, cy)
    return x, y


def load_elephant(n_samples=1500, random_state=0) -> dict:
    rng = np.random.default_rng(random_state)
    n = 1500 if n_samples is None else n_samples
    t = rng.uniform(0, 2 * np.pi, size=n)
    x, y = _elephant_silhouette(t)
    thickness = 0.06 * (x.max() - x.min())
    z = rng.uniform(-thickness / 2, thickness / 2, size=n)
    jitter = rng.normal(scale=0.15, size=(n, 2))
    data = np.stack([y + jitter[:, 0], -x + jitter[:, 1], z], axis=1).astype(np.float32)

    if not np.isfinite(data).all():
        raise RuntimeError("elephant point cloud contains non-finite values")
    return {
        "data": data,
        "target": t.astype(np.float32),
        "name": "elephant",
        "thumbnail_shape": None,
        "cmap": None,
    }


def load_scrna_seq(n_samples=1500, random_state=0) -> dict:
    import scanpy as sc
    from sklearn.cluster import KMeans

    cache_dir = os.path.join(_CACHE_ROOT, _PBMC3K_CACHE_SUBDIR)
    os.makedirs(cache_dir, exist_ok=True)
    sc.settings.datasetdir = cache_dir

    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=50, svd_solver="arpack", random_state=random_state)

    data = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    # No leidenalg/igraph dependency: KMeans on the PCA space is a real,
    # defensible way to produce cluster-id labels for coloring the embedding.
    target = KMeans(n_clusters=8, random_state=random_state, n_init=10).fit_predict(data)
    data, target = _subsample(data, target.astype(np.int64), n_samples, random_state)

    if not np.isfinite(data).all():
        raise RuntimeError("scrna_seq PCA data contains non-finite values")
    return {
        "data": data,
        "target": target,
        "name": "scrna_seq",
        "thumbnail_shape": None,
        "cmap": None,
    }
