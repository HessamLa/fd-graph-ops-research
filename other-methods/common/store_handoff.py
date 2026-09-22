#!/bin/env python3
"""Score an embedding built in another venv and save it to the store.

karateclub runs in its own `.venv-karate` (see
`other-methods/karateclub/run.py`). It writes `Z.npy` + `meta.json` to a
handoff directory. This script runs in the shared `.venv`, where `evaluator`
lives, reads that handoff and calls `common.store.save_scored`. So the
karate methods get the same scoring and record layout as every other method.

Usage (shared .venv):  store_handoff.py <handoff_dir>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join("/home/h/gm/fd-graph-ops-research", "other-methods"))
from common.store import save_scored


# report labels for the short emit names
LABEL = {"lapeig": "laplacian_eigenmaps"}


def main(handoff):
    meta = json.load(open(os.path.join(handoff, "meta.json")))
    Z = np.load(os.path.join(handoff, "Z.npy"))
    method = LABEL.get(meta["method"], meta["method"])
    save_scored(Z, graph=meta["graph"], method=method,
                seed=meta["seed"], seconds=meta.get("seconds", 0.0),
                peak_rss_mb=meta.get("peak_rss_mb", 0.0),
                params={"package": "karateclub"},
                notes="built in .venv-karate; scored in shared .venv")


if __name__ == "__main__":
    main(sys.argv[1])
