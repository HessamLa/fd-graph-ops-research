"""test_golden.py -- the code-move gate.

`golden.py` hashes what the augmentation and the plan build for one
configuration of every policy branch, and a short CPU embedding for four of
them. A refactor that only MOVES code reproduces every digest.

It runs in a SUBPROCESS, and it must: the embed half is exact on the CPU
backend only (CATALOG.md section 15), thus it needs `JAX_PLATFORMS=cpu`,
and `conftest.py` has already chosen the backend for this process.

A digest that changes is a behaviour change. When the change is INTENDED,
re-record with `--write --force` and say in the commit what moved and why.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_golden_augment_and_embed_reproduce():
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    p = subprocess.run([sys.executable, "-m", "fodiwalk.tests.golden",
                        "--check"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
