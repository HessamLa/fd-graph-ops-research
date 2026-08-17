import io
import os
import urllib.request
import uuid

import numpy as np

# Cache next to this module rather than os.getcwd() so loaders behave the same
# no matter which directory the integration script is launched from.
_CACHE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")

# OpenML fallbacks, used only if every other download route genuinely fails.
# CIFAR-100 has no comparable well-known OpenML mirror, hence the None.
_OPENML_NAMES = {
    "mnist": "mnist_784",
    "fashion_mnist": "Fashion-MNIST",
    "cifar10": "CIFAR_10",
    "cifar100": None,
}

# torchvision's CIFAR10/100 download straight from cs.toronto.edu, which is
# frequently throttled to ~100KB/s (a 161MB file can take 25+ minutes, or
# time out). uoft-cs's HuggingFace mirror serves the identical images from
# HF's CDN at ~30MB/s -- same University of Toronto source, just a much
# faster front door. Tried first; torchvision is still the fallback.
_HF_CIFAR_URLS = {
    "cifar10": "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet",
    "cifar100": "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet",
}


def _subsample(data, target, n_samples, random_state):
    n = min(n_samples, len(data))
    idx = np.random.default_rng(random_state).choice(len(data), size=n, replace=False)
    return data[idx], target[idx]


def _atomic_download(url, dest_path, timeout=120):
    """Download to a per-call-unique temp file, then os.replace into place.

    Never write straight to dest_path: a second concurrent download (another
    process, a retry, this same loader called twice) racing on the same path
    interleaves both writers' bytes into one corrupt file with no error from
    either side -- os.replace is atomic, so whichever writer finishes last
    simply wins instead of both mangling the target.
    """
    if os.path.exists(dest_path):
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = f"{dest_path}.{uuid.uuid4().hex}.part"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp_path, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(tmp_path, dest_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _from_huggingface(name):
    import pandas as pd
    from PIL import Image

    url = _HF_CIFAR_URLS[name]
    cache_dir = os.path.join(_CACHE_ROOT, name)
    parquet_path = os.path.join(cache_dir, "train.parquet")
    _atomic_download(url, parquet_path)

    df = pd.read_parquet(parquet_path)
    label_col = "fine_label" if name == "cifar100" else "label"
    images = np.stack([
        np.asarray(Image.open(io.BytesIO(row["bytes"])).convert("RGB"))
        for row in df["img"]
    ])
    target = df[label_col].to_numpy()
    return images.reshape(len(images), -1), target


def _from_torchvision(cls, name, **kwargs):
    import torchvision

    root = os.path.join(_CACHE_ROOT, name)
    os.makedirs(root, exist_ok=True)
    ds = getattr(torchvision.datasets, cls)(root=root, train=True, download=True, **kwargs)
    # MNIST-family exposes torch tensors, CIFAR-family a numpy array + list.
    data = np.asarray(ds.data)
    target = np.asarray(ds.targets)
    return data.reshape(len(data), -1), target


def _from_openml(name):
    from sklearn.datasets import fetch_openml

    openml_name = _OPENML_NAMES[name]
    if openml_name is None:
        raise RuntimeError(f"no OpenML fallback is available for {name}")
    bunch = fetch_openml(openml_name, version=1, as_frame=False, parser="auto")
    return np.asarray(bunch.data), np.asarray(bunch.target)


def _load(cls, name, thumbnail_shape, cmap, n_samples, random_state, **kwargs):
    errors = []
    data = target = None
    sources = ([_from_huggingface] if name in _HF_CIFAR_URLS else []) + [
        lambda name: _from_torchvision(cls, name, **kwargs),
        _from_openml,
    ]
    for source in sources:
        try:
            data, target = source(name)
            break
        except Exception as err:
            errors.append(f"{getattr(source, '__name__', source)}: {err!r}")
    if data is None:
        raise RuntimeError(f"could not download {name}: " + "; ".join(errors))

    data, target = _subsample(data, target, n_samples, random_state)
    data = (data.astype(np.float32) / 255.0)
    if not np.isfinite(data).all():
        raise RuntimeError(f"{name} contains non-finite pixel values")
    return {
        "data": data,
        "target": target.astype(np.int64),
        "name": name,
        "thumbnail_shape": thumbnail_shape,
        "cmap": cmap,
    }


def load_mnist(n_samples=1500, random_state=0) -> dict:
    return _load("MNIST", "mnist", (28, 28), "gray_r", n_samples, random_state)


def load_fashion_mnist(n_samples=1500, random_state=0) -> dict:
    return _load("FashionMNIST", "fashion_mnist", (28, 28), "gray_r", n_samples, random_state)


def load_cifar10(n_samples=1500, random_state=0) -> dict:
    return _load("CIFAR10", "cifar10", (32, 32, 3), None, n_samples, random_state)


def load_cifar100(n_samples=1500, random_state=0) -> dict:
    return _load("CIFAR100", "cifar100", (32, 32, 3), None, n_samples, random_state)
