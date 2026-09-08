#!/bin/env python3
"""Test setup for `evaluator/tests`.

Three jobs, and no more:

1. Put the repository root on `sys.path`. The package is not installed
   (`pip install -e .` was never run), so a test run from any directory
   must still find `evaluator` and `fodiwalk`.
2. Register the `slow` marker and put the tests that carry it behind
   `--runslow`. P3 and P4 run on pubmed and they cost minutes; the
   default gate is cora only.
3. Print the measured differences at the end of the run. A parity test
   that prints only "passed" tells a reader nothing, and pytest hides the
   `print` of a test that passes unless the run adds `-s`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Filled by `record_diffs` in test_parity_v1.py: test name -> list of
# (value name, maximum absolute difference).
DIFFS: dict[str, list[tuple[str, float]]] = {}


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: runs on pubmed, minutes; needs --runslow")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run the pubmed parity tests P3 and P4")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="pubmed, minutes; pass --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def record_diffs(request):
    """Save `(value name, difference)` pairs under the running test's name.

    The summary hook below prints them, whether the test passed or failed.
    """
    def add(pairs):
        DIFFS.setdefault(request.node.name, []).extend(pairs)
    return add


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """The measured maximum absolute differences, one line per value."""
    if not DIFFS:
        return
    write = terminalreporter.write_line
    write("")
    write("parity -- maximum absolute difference, package against reference")
    for test in sorted(DIFFS):
        for name, diff in DIFFS[test]:
            write(f"  {test:<34} {name:<18} {diff:.3e}")
