"""Shared pytest fixtures for the HAR generalisation test suite."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.demo_data import generate_demo_dataset  # noqa: E402
from src.utils import ProjectPaths, load_config, set_seed  # noqa: E402

UCI_RAW_DIR = REPO_ROOT / "data" / "raw" / "UCI HAR Dataset"


@pytest.fixture(scope="session")
def repo_config() -> dict:
    return load_config(REPO_ROOT / "config" / "config.yaml")


@pytest.fixture(autouse=True)
def _seed_everything():
    set_seed(42)


@pytest.fixture
def config(repo_config, tmp_path) -> dict:
    """Config pointing at a temporary root with cheap model settings."""
    payload = copy.deepcopy(repo_config)
    payload["paths"]["root"] = str(tmp_path)
    payload["project"]["seed"] = 42
    payload["models"]["random_forest"]["n_estimators"] = 20
    payload["models"]["random_forest"]["selection_grid"] = {}
    payload["models"]["svm"]["selection_grid"] = {}
    payload["models"]["cnn"]["max_epochs"] = 2
    payload["models"]["cnn"]["patience"] = 2
    payload["models"]["cnn"]["batch_size"] = 64
    return payload


@pytest.fixture
def paths(config) -> ProjectPaths:
    return ProjectPaths.from_config(config, mode="research").ensure()


@pytest.fixture
def demo_dataset():
    """Small deterministic synthetic dataset used for fast pipeline tests."""
    dataset = generate_demo_dataset(seed=7, n_subjects=6, n_features=40)
    assert dataset.is_synthetic
    return dataset


@pytest.fixture
def demo_paths(config) -> ProjectPaths:
    return ProjectPaths.from_config(config, mode="demo").ensure()


@pytest.fixture
def uci_available() -> bool:
    return (UCI_RAW_DIR / "features.txt").exists()
