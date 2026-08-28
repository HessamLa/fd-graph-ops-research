"""golden.py -- the byte-exact snapshot that a refactor must reproduce.

The parity gate of `test_parity.py` costs a GPU and ~18 minutes. This file
is the CHEAP gate for a code move: it hashes what the augmentation and the
plan build, on Cora, for one configuration of every branch of the pair
policies. A refactor that moves code without changing behaviour reproduces
every digest to the byte.

Two halves, because they have different determinism:

    augment   PURE NumPy. Exact on any machine and any backend. `D`, the
              planes, the degrees, the plan statistics, the `info` counts.
    embed     JAX. Exact on the CPU backend only -- a split hub row makes
              the GPU scatter-add order free (CATALOG.md section 15), thus
              the embed half RUNS ON THE CPU and skips elsewhere.

Use::

    .venv/bin/python -m fodiwalk.tests.golden --write   # record
    .venv/bin/python -m fodiwalk.tests.golden --check   # compare

`--write` refuses to overwrite unless `--force` is given: the recorded file
is the reference of the refactor, and a re-record after a change would
compare the change against itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

# The embed half is deterministic on the CPU backend only. This must run
# BEFORE `import jax`, thus it runs before the package import below.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

BASELINE = pathlib.Path(__file__).with_name("golden_baseline.json")

# One configuration for each branch of `_build_D`. The name says which
# branch it holds, thus a failing key names the code that moved.
AUGMENT_CASES = {
    "walk_cap_min_gap":      dict(pairs="walk", policy="cap",
                                  weight="min_gap"),
    "walk_cap_flat":         dict(pairs="walk", policy="cap", weight="flat"),
    "walk_edges_cap":        dict(pairs="walk_edges", policy="cap"),
    "walk_buckets":          dict(pairs="walk", policy="buckets"),
    "walk_buckets_far":      dict(pairs="walk", policy="buckets",
                                  far_with_buckets=True, far=2000),
    "walk_landmarks":        dict(pairs="walk", far=500, landmarks=8,
                                  far_max=16, far_scale=2.0),
    "walk_far_bias":         dict(pairs="walk", far=1500, far_bias=0.75),
    "nbr_walk":              dict(pairs="nbr_walk"),
    "nbr_walk_far":          dict(pairs="nbr_walk", far=2000),
    "nbr_walk_rowcap_lowdeg": dict(pairs="nbr_walk", row_cap=16,
                                   edge_rule="low_deg"),
    "nbr_walk_buckets":      dict(pairs="nbr_walk", policy="buckets"),
    "nbr_walk_freq_node":    dict(pairs="nbr_walk", freq_mode="node"),
    "nbr_walk_fused":        dict(pairs="nbr_walk", fuse_planes=True),
    "nbr_walk_chunks4":      dict(pairs="nbr_walk", chunks=4),
    "nbr_walk_no_deg_norm":  dict(pairs="nbr_walk", no_deg_norm=True),
    "nbr_walk_node2vec":     dict(pairs="nbr_walk", p=2.0, q=0.5),
}

# The embed half. Short runs: the digest catches a moved kernel, and a long
# run only costs time. `nbr_walk` holds every row under `k_max`, thus
# `n_split = 0` and the run is exact.
EMBED_CASES = {
    "nbr_walk_plain":   dict(pairs="nbr_walk", optim="plain", epochs=30),
    "nbr_walk_adam":    dict(pairs="nbr_walk", optim="adam", lr=0.1,
                             epochs=30),
    "walk_chunks2":     dict(pairs="walk", chunks=2, epochs=20),
    "nbr_walk_fused":   dict(pairs="nbr_walk", fuse_planes=True, epochs=30),
}

INFO_KEYS = ("raw_pairs", "unique_pairs", "capped_pairs", "near_nnz",
             "h1_entries", "edges_of_A", "edges_added", "far_pairs",
             "far_asked", "far_unreached", "prunes", "walk_order",
             "far_mean_deg")
PLAN_KEYS = ("cells", "n_split", "rungs", "pad_frac", "chunks",
             "rows_per_chunk", "k_max", "b_cells")


def digest(a) -> str:
    """The sha1 of the bytes of an array, with the dtype and the shape."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha1()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:16]


def _num(x):
    """A float or an int, as JSON holds it. `nan` is not comparable."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), 12)
    return x


def snapshot_augment(A, n, name: str, cfg: dict) -> dict:
    """Everything stage 2 builds, hashed. No embedding, thus no JAX kernel."""
    from fodiwalk import Fodiwalk
    from fodiwalk.embed.forces import planes_of

    fw = Fodiwalk(n_dim=64, seed=42, **cfg)
    D = fw.augment_graph(A)
    planes = fw._build_planes(D)
    degrees = fw._build_degrees(D, A)
    out = {
        "D_shape": list(D.shape), "D_nnz": int(D.nnz),
        "D_indptr": digest(D.indptr), "D_indices": digest(D.indices),
        "D_data": digest(D.data),
        "D_data_sum": _num(float(D.data.sum())),
        "law": fw.law, "planes": list(planes_of(fw.law)),
        "plane_digests": [digest(p) for p in planes],
        "degrees": digest(np.asarray(degrees)),
        "freq": None if fw.freq is None else digest(fw.freq),
        "stats_key": digest(fw.stats["key"]),
        "stats_mn": digest(fw.stats["mn"]),
        "stats_cnt": digest(fw.stats["cnt"]),
        "rng_state": digest(np.asarray(fw.rng.integers(0, 2 ** 31, 4))),
        "info": {k: _num(fw.info[k]) for k in INFO_KEYS if k in fw.info},
        "plan": {k: _num(fw.plan_stats[k]) for k in PLAN_KEYS
                 if k in fw.plan_stats},
    }
    return out


def snapshot_embed(A, n, name: str, cfg: dict) -> dict:
    """A short run on the CPU backend. `Z` as NUMBERS, and not as a digest.

    A digest of `Z` is too sharp a tool: the scatter-add of `_step` gives
    the same run a last-bit difference between two processes, thus a digest
    reports a defect that is not one. The numbers below are compared with
    `EMBED_RTOL`, which is 1e-4 -- five orders below what a wrong plane, a
    lost drop or the wrong update rule moves.
    """
    from fodiwalk import Fodiwalk

    cfg = dict(cfg)
    epochs = cfg.pop("epochs", 30)
    optim = cfg.pop("optim", "plain")
    lr = cfg.pop("lr", 1.0)
    fw = Fodiwalk(n_dim=32, seed=42, lr=lr, optim=optim, **cfg)
    fw.embed(A, epochs=epochs)
    Z = np.asarray(fw.get_embeddings(), dtype=np.float64)
    return {"Z_abs_sum": _num(float(np.abs(Z).sum())),
            "Z_sq_sum": _num(float((Z * Z).sum())),
            "Z_row_norm_max": _num(float(np.linalg.norm(Z, axis=1).max())),
            "Z_col0_sum": _num(float(Z[:, 0].sum())),
            "dz": _num(float(fw.Th(fw.dZ))),
            "diverged": bool(fw.diverged),
            "n_split": _num(fw.plan_stats.get("n_split"))}


def build(which: str = "both") -> dict:
    from fodiwalk.make_graph import load

    A, n = load("cora")
    out = {"dataset": "cora", "n": int(n), "A_nnz": int(A.nnz)}
    if which in ("both", "augment"):
        out["augment"] = {k: snapshot_augment(A, n, k, c)
                          for k, c in AUGMENT_CASES.items()}
    if which in ("both", "embed"):
        out["embed"] = {k: snapshot_embed(A, n, k, c)
                        for k, c in EMBED_CASES.items()}
    return out


EMBED_RTOL = 1e-4       # the CPU scatter-add spread. See `snapshot_embed`.


def compare(new: dict, ref: dict, path: str = "", rtol: float = 0.0) -> list:
    """Every leaf that differs, as `path: ref -> new`. Order does not matter.

    `rtol` is 0 -- BYTE EXACT -- for the augment half, which is pure NumPy.
    The embed half takes `EMBED_RTOL`.
    """
    bad = []
    if isinstance(ref, dict):
        if not isinstance(new, dict):
            return [f"{path}: {ref!r} -> {new!r}"]
        for k in sorted(set(ref) | set(new)):
            sub = EMBED_RTOL if k == "embed" else rtol
            if k not in ref:
                bad.append(f"{path}.{k}: ABSENT -> {new[k]!r}")
            elif k not in new:
                bad.append(f"{path}.{k}: {ref[k]!r} -> ABSENT")
            else:
                bad += compare(new[k], ref[k], f"{path}.{k}", sub)
        return bad
    if isinstance(ref, list):
        if list(new) != list(ref):
            bad.append(f"{path}: {ref!r} -> {new!r}")
        return bad
    if rtol and isinstance(ref, float) and isinstance(new, (int, float)):
        if abs(new - ref) <= rtol * max(abs(ref), 1e-12):
            return bad
    if new != ref:
        bad.append(f"{path}: {ref!r} -> {new!r}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--which", default="both",
                    choices=("both", "augment", "embed"))
    a = ap.parse_args(argv)

    snap = build(a.which)
    if a.write:
        if BASELINE.exists() and not a.force:
            print(f"{BASELINE} exists. --force to overwrite. A re-record "
                  f"after a change compares the change against itself.")
            return 2
        BASELINE.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
        print(f"wrote {BASELINE}")
        return 0

    ref = json.loads(BASELINE.read_text())
    if a.which != "both":
        ref = {k: v for k, v in ref.items()
               if k not in ("augment", "embed") or k == a.which}
        snap = {k: v for k, v in snap.items()
                if k not in ("augment", "embed") or k == a.which}
    bad = compare(snap, ref)
    if bad:
        print(f"GOLDEN MISMATCH, {len(bad)} leaves:")
        for line in bad[:80]:
            print("  " + line)
        return 1
    print(f"GOLDEN OK ({a.which})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
