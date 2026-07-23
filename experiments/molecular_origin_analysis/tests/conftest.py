"""Shared fixtures for the non-invasive molecular-origin analysis tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
for path in (str(PROJECT_ROOT),):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(scope="session")
def module_root() -> Path:
    return MODULE_ROOT


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config_path(module_root: Path) -> Path:
    return module_root / "config" / "analysis_config.yaml"
