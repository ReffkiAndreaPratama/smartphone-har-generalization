"""Shared utilities: configuration, paths, seeding, logging and IO helpers.

Semua path di proyek ini diresolusi relatif terhadap root repository, bukan
terhadap current working directory, agar skrip dan dashboard konsisten.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

ACTIVITY_DISPLAY = {
    "WALKING": "Walking",
    "WALKING_UPSTAIRS": "Walking Upstairs",
    "WALKING_DOWNSTAIRS": "Walking Downstairs",
    "SITTING": "Sitting",
    "STANDING": "Standing",
    "LAYING": "Laying",
}

RAW_CHANNELS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

DEMO_NOTICE = (
    "DATA DEMO SINTETIS. Data ini dihasilkan oleh generator deterministik dan "
    "bukan data UCI HAR. Hasil dari data demo tidak boleh dilaporkan sebagai "
    "hasil penelitian."
)

PACKAGE_NAMES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "joblib",
    "PyYAML",
    "torch",
    "streamlit",
    "pytest",
]


class ProjectError(RuntimeError):
    """Base error for expected, user-facing project failures."""


class DatasetIntegrityError(ProjectError):
    """Raised when a dataset does not match the documented structure."""


class SubjectLeakageError(ProjectError):
    """Raised when a subject appears in both train and test partitions."""


class ArtifactUnavailableError(ProjectError):
    """Raised when a model/result artifact needed by the UI is missing."""


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved filesystem locations used by the pipeline."""

    root: Path
    data_raw: Path
    data_processed: Path
    data_demo: Path
    models: Path
    reports: Path
    figures: Path
    tables: Path
    results: Path
    predictions: Path

    @classmethod
    def from_config(cls, config: dict[str, Any], mode: str = "research") -> "ProjectPaths":
        raw_paths = config.get("paths", {})
        root = Path(raw_paths.get("root", PROJECT_ROOT))
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        root = root.resolve()

        def resolve(key: str, default: str) -> Path:
            value = Path(raw_paths.get(key, default))
            return value if value.is_absolute() else root / value

        paths = cls(
            root=root,
            data_raw=resolve("data_raw", "data/raw"),
            data_processed=resolve("data_processed", "data/processed"),
            data_demo=resolve("data_demo", "data/demo"),
            models=resolve("models", "models"),
            reports=resolve("reports", "reports"),
            figures=resolve("figures", "reports/figures"),
            tables=resolve("tables", "reports/tables"),
            results=resolve("results", "reports/results"),
            predictions=resolve("predictions", "reports/predictions"),
        )
        if mode == "demo":
            paths = replace(
                paths,
                models=paths.root / "models" / "demo",
                reports=paths.root / "reports" / "demo",
                figures=paths.root / "reports" / "demo" / "figures",
                tables=paths.root / "reports" / "demo" / "tables",
                results=paths.root / "reports" / "demo" / "results",
                predictions=paths.root / "reports" / "demo" / "predictions",
            )
        return paths

    def ensure(self) -> "ProjectPaths":
        for path in (
            self.data_raw,
            self.data_processed,
            self.data_demo,
            self.models,
            self.reports,
            self.figures,
            self.tables,
            self.results,
            self.predictions,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "data_raw": str(self.data_raw),
            "data_processed": str(self.data_processed),
            "data_demo": str(self.data_demo),
            "models": str(self.models),
            "reports": str(self.reports),
            "figures": str(self.figures),
            "tables": str(self.tables),
            "results": str(self.results),
            "predictions": str(self.predictions),
        }


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    """Load YAML config and resolve the path relative to the repository root."""
    config_path = Path(path)
    if not config_path.is_absolute():
        candidate = PROJECT_ROOT / config_path
        config_path = candidate if candidate.exists() else config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file tidak ditemukan: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ProjectError(f"Config harus berupa mapping YAML: {config_path}")
    config.setdefault("_config_path", str(config_path.resolve()))
    return config


def set_seed(seed: int) -> None:
    """Set Python, NumPy and (optionally) PyTorch seeds."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional for users who only run the sklearn pipeline
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch, "set_num_threads"):
            # Keep behaviour predictable on CPU-only machines.
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    except Exception:  # pragma: no cover - defensive
        pass


def get_logger(name: str = "har", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class Timer:
    """Context manager that records wall-clock duration in seconds."""

    def __init__(self) -> None:
        self.seconds: float = 0.0
        self._start: float | None = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start is not None:
            self.seconds = time.perf_counter() - self._start


def to_jsonable(value: Any) -> Any:
    """Recursively convert NumPy/path/dataclass values into JSON-safe types."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def save_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=indent, ensure_ascii=False)
    return destination


def read_json(path: str | Path, default: Any = None) -> Any:
    source = Path(path)
    if not source.exists():
        return default
    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return default


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def git_commit_hash(root: str | Path = PROJECT_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def environment_info() -> dict[str, Any]:
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "threads": torch.get_num_threads(),
        }
    except Exception:  # pragma: no cover
        torch_info = {"version": None}
    return {
        "timestamp": now_iso(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
        "torch": torch_info,
        "git_commit": git_commit_hash(),
    }


@contextmanager
def working_directory(path: str | Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).strip("_")
