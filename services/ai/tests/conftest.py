"""Shared pytest fixtures.

Adds ``services/ai`` to ``sys.path`` so the tests run without an editable install --
one less setup step between a fresh clone and a green test run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

AI_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_ROOT.parents[1]

if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from contextopti.index import build_graph  # noqa: E402


@pytest.fixture(scope="session")
def fixture_repo() -> Path:
    """Path to the toy cross-file repository used by M1-M5."""
    path = REPO_ROOT / "data" / "fixtures" / "toy_repo"
    assert path.is_dir(), "toy fixture repo missing at %s" % path
    return path


@pytest.fixture(scope="session")
def graph(fixture_repo: Path):
    """The code graph for the toy repo, built once per test session."""
    return build_graph(fixture_repo)
