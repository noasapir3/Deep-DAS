"""Shared fixtures. Tests run from anywhere: only the release's src/ is put on sys.path."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def release_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def processed_available() -> bool:
    return (ROOT / "data" / "processed" / "normal_corpus" / "meta.parquet").exists()
