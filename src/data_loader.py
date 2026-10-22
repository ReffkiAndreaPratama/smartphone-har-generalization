"""Dataset loading, verification, caching and summary for UCI HAR.

Sumber resmi:
    https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Dataset berisi fitur 561 dimensi yang sudah diekstrak oleh pembuat dataset
(window 2,56 detik, 50 Hz, overlap 50%). Pipeline ini **tidak** mengekstrak
ulang 561 fitur tersebut dari sinyal mentah; sinyal mentah 128x9 hanya dipakai
untuk model 1D-CNN. Lihat README bagian batasan.
"""

from __future__ import annotations

import logging
import os
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .utils import (
    ACTIVITY_NAMES,
    DEMO_NOTICE,
    RAW_CHANNELS,
    DatasetIntegrityError,
    ProjectPaths,
    get_logger,
    sha256_file,
)

UCI_FOLDER_NAME = "UCI HAR Dataset"
CACHE_VERSION = 2
UCI_CACHE_NAME = "uci_har_cache"
DEMO_CACHE_NAME = "demo_cache"

#: Lisensi mengikuti README dataset: sitasi wajib, penggunaan komersial dilarang.
UCI_LICENSE_NOTE = (
    "Dataset UCI HAR didistribusikan AS-IS. Penggunaan dalam publikasi wajib "
    "mensitasi Anguita et al. (2013). README dataset menyatakan penggunaan "
    "komersial dilarang."
)

UCI_CITATION = (
    "Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. "
    "Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using "
    "Smartphones. 21st European Symposium on Artificial Neural Networks, "
    "Computational Intelligence and Machine Learning (ESANN), 2013."
)


@dataclass
class HARDataset:
    """Container for feature windows, labels, subjects and optional raw signals."""

    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    split: np.ndarray
    feature_names: list[str]
    activity_names: list[str]
    raw: np.ndarray | None = None
    raw_channels: list[str] = field(default_factory=list)
    source: str = "uci_har"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.int64)
        self.subject = np.asarray(self.subject, dtype=np.int64)
        self.split = np.asarray(self.split).astype(str)
        if self.raw is not None:
            self.raw = np.asarray(self.raw, dtype=np.float32)

        if self.X.ndim != 2:
            raise DatasetIntegrityError(f"X harus 2D, ditemukan {self.X.shape}")
        n_samples = self.X.shape[0]
        if len(self.feature_names) != self.X.shape[1]:
            raise DatasetIntegrityError(
                f"Jumlah nama fitur ({len(self.feature_names)}) != kolom X ({self.X.shape[1]})"
            )
        for name, array in (("y", self.y), ("subject", self.subject), ("split", self.split)):
            if len(array) != n_samples:
                raise DatasetIntegrityError(
                    f"Panjang {name} ({len(array)}) != jumlah sampel ({n_samples})"
                )
        if self.y.size and (self.y.min() < 0 or self.y.max() >= len(self.activity_names)):
            raise DatasetIntegrityError("Label aktivitas di luar rentang yang didokumentasikan")
        if self.subject.size and self.subject.min() < 1:
            raise DatasetIntegrityError("ID subjek harus >= 1")
        if not np.isfinite(self.X).all():
            raise DatasetIntegrityError("X mengandung NaN atau nilai tak hingga")
        if self.raw is not None:
            if self.raw.ndim != 3 or self.raw.shape[0] != n_samples:
                raise DatasetIntegrityError(
                    f"raw harus berbentuk (n_samples, window, channel), ditemukan {self.raw.shape}"
                )
            if self.raw.shape[2] != len(self.raw_channels):
                raise DatasetIntegrityError("Jumlah channel raw tidak sesuai raw_channels")
            if not np.isfinite(self.raw).all():
                raise DatasetIntegrityError("Sinyal mentah mengandung NaN atau nilai tak hingga")
        self.metadata = dict(self.metadata or {})

    def __len__(self) -> int:
        return self.n_samples

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def subject_ids(self) -> list[int]:
        return sorted(int(value) for value in np.unique(self.subject))

    @property
    def is_synthetic(self) -> bool:
        return self.source == "synthetic_demo"

    @property
    def display_source(self) -> str:
        return "Demo sintetis (bukan data penelitian)" if self.is_synthetic else "UCI HAR (data nyata)"

    def subset(self, indices: Iterable[int], note: str | None = None) -> "HARDataset":
        index = np.asarray(list(indices), dtype=np.int64)
        metadata = dict(self.metadata)
        if note:
            metadata["subset_note"] = note
        return HARDataset(
            X=self.X[index],
            y=self.y[index],
            subject=self.subject[index],
            split=self.split[index],
            feature_names=list(self.feature_names),
            activity_names=list(self.activity_names),
            raw=None if self.raw is None else self.raw[index],
            raw_channels=list(self.raw_channels),
            source=self.source,
            metadata=metadata,
        )

    def split_indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)

    def subject_indices(self, subject_id: int) -> np.ndarray:
        return np.flatnonzero(self.subject == int(subject_id))

    def class_counts(self) -> dict[str, int]:
        counts = np.bincount(self.y, minlength=len(self.activity_names))
        return {name: int(counts[i]) for i, name in enumerate(self.activity_names)}


# ---------------------------------------------------------------------------
# Structure / verification
# ---------------------------------------------------------------------------
def _required_files(split: str) -> list[str]:
    return [
        f"{split}/X_{split}.txt",
        f"{split}/y_{split}.txt",
        f"{split}/subject_{split}.txt",
        *[f"{split}/Inertial Signals/{channel}_{split}.txt" for channel in RAW_CHANNELS],
    ]


def find_dataset_dir(raw_root: str | Path) -> Path | None:
    """Find the directory containing ``activity_labels.txt`` and train/test."""
    root = Path(raw_root)
    if not root.exists():
        return None
    candidates = [root / UCI_FOLDER_NAME, root]
    candidates.extend(sorted(path for path in root.iterdir() if path.is_dir()))
    for candidate in candidates:
        if (
            (candidate / "activity_labels.txt").exists()
            and (candidate / "features.txt").exists()
            and (candidate / "train" / "X_train.txt").exists()
            and (candidate / "test" / "X_test.txt").exists()
        ):
            return candidate.resolve()
    return None


def _read_feature_names(dataset_dir: Path) -> list[str]:
    frame = pd.read_csv(
        dataset_dir / "features.txt",
        sep=r"\s+",
        header=None,
        names=["id", "feature"],
        engine="python",
    )
    names = frame["feature"].astype(str).tolist()
    if len(names) != 561:
        raise DatasetIntegrityError(f"features.txt harus berisi 561 fitur, ditemukan {len(names)}")
    # UCI features.txt contains duplicate names for the bandsEnergy features.
    # Keep the original order but make names unique for safe DataFrame/CSV use.
    seen: dict[str, int] = {}
    unique_names: list[str] = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique_names.append(name if seen[name] == 1 else f"{name}__dup{seen[name]}")
    return unique_names


def _count_duplicate_feature_names(dataset_dir: Path) -> int:
    frame = pd.read_csv(
        dataset_dir / "features.txt",
        sep=r"\s+",
        header=None,
        names=["id", "feature"],
        engine="python",
    )
    return int(len(frame) - frame["feature"].nunique())


def _read_activity_mapping(dataset_dir: Path) -> tuple[list[str], dict[int, int]]:
    frame = pd.read_csv(
        dataset_dir / "activity_labels.txt",
        sep=r"\s+",
        header=None,
        names=["id", "activity"],
        engine="python",
    )
    frame = frame.sort_values("id")
    ids = [int(value) for value in frame["id"]]
    names = [str(value).upper() for value in frame["activity"]]
    if len(names) != 6 or ids != list(range(1, 7)):
        raise DatasetIntegrityError(f"activity_labels.txt tidak valid: {list(zip(ids, names))}")
    if names != ACTIVITY_NAMES:
        raise DatasetIntegrityError(
            f"Label aktivitas berbeda dari dokumentasi: {names} != {ACTIVITY_NAMES}"
        )
    mapping = {activity_id: index for index, activity_id in enumerate(ids)}
    return names, mapping


def verify_uci_structure(
    dataset_dir: str | Path, expected: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Read the dataset and return a machine-readable integrity report.

    Raises
    ------
    DatasetIntegrityError
        If required files are missing or labels/subjects are invalid.
    """
    directory = Path(dataset_dir)
    resolved = find_dataset_dir(directory)
    if resolved is not None:
        directory = resolved
    expected = expected or {}
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    required = ["activity_labels.txt", "features.txt"]
    for split in ("train", "test"):
        required.extend(_required_files(split))
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise DatasetIntegrityError(
            "File dataset tidak lengkap: " + ", ".join(missing[:8]) + ("..." if len(missing) > 8 else "")
        )

    features = _read_feature_names(directory)
    activities, _ = _read_activity_mapping(directory)
    checks["n_features"] = len(features)
    checks["activity_names"] = activities
    checks["n_duplicate_feature_names"] = _count_duplicate_feature_names(directory)
    checks["feature_names_unique"] = checks["n_duplicate_feature_names"] == 0
    if not checks["feature_names_unique"]:
        warnings.append(
            "features.txt UCI memiliki nama fitur duplikat; nama telah dibuat unik "
            "dengan sufiks __dupN tanpa mengubah urutan kolom."
        )

    subject_sets: dict[str, list[int]] = {}
    split_counts: dict[str, int] = {}
    label_counts: dict[str, dict[str, int]] = {}
    raw_shapes: dict[str, list[int]] = {}
    duplicate_rows: dict[str, int] = {}
    missing_values: dict[str, int] = {}

    for split in ("train", "test"):
        X = pd.read_csv(
            directory / split / f"X_{split}.txt",
            sep=r"\s+",
            header=None,
            engine="python",
            dtype=np.float32,
        )
        y = np.loadtxt(directory / split / f"y_{split}.txt", dtype=int)
        subject = np.loadtxt(directory / split / f"subject_{split}.txt", dtype=int)
        split_counts[split] = int(X.shape[0])

        if X.shape[1] != len(features):
            errors.append(f"X_{split} memiliki {X.shape[1]} kolom, seharusnya {len(features)}")
        if len(y) != X.shape[0] or len(subject) != X.shape[0]:
            errors.append(f"Panjang y/subject pada {split} tidak sama dengan X")
        unknown_labels = sorted(set(np.unique(y)) - set(range(1, len(activities) + 1)))
        if unknown_labels:
            errors.append(f"Label tidak dikenal pada {split}: {unknown_labels}")
        if np.isnan(X.to_numpy()).sum():
            errors.append(f"X_{split} mengandung NaN")
        if not np.isfinite(X.to_numpy()).all():
            errors.append(f"X_{split} mengandung nilai tak hingga")
        if X.duplicated().any():
            warnings.append(f"Terdapat {int(X.duplicated().sum())} baris duplikat pada X_{split}")

        subject_sets[split] = sorted(int(value) for value in np.unique(subject))
        label_counts[split] = {
            activities[label - 1]: int((y == label).sum()) for label in range(1, len(activities) + 1)
        }
        duplicate_rows[split] = int(X.duplicated().sum())
        missing_values[split] = int(X.isna().sum().sum())

        raw_arrays = []
        for channel in RAW_CHANNELS:
            raw = np.loadtxt(
                directory / split / "Inertial Signals" / f"{channel}_{split}.txt",
                dtype=np.float32,
            )
            raw_arrays.append(raw)
        raw = np.stack(raw_arrays, axis=-1)
        raw_shapes[split] = list(raw.shape)
        if raw.shape[0] != X.shape[0] or raw.shape[1] != 128:
            errors.append(f"Bentuk sinyal mentah {split} tidak valid: {raw.shape}")

    overlap = sorted(set(subject_sets["train"]).intersection(subject_sets["test"]))
    if overlap:
        errors.append(f"Subjek muncul pada train dan test resmi: {overlap}")

    checks.update(
        {
            "n_train": split_counts["train"],
            "n_test": split_counts["test"],
            "n_total": split_counts["train"] + split_counts["test"],
            "train_subjects": subject_sets["train"],
            "test_subjects": subject_sets["test"],
            "n_subjects": len(set(subject_sets["train"]) | set(subject_sets["test"])),
            "class_counts": label_counts,
            "raw_signal_shape": raw_shapes,
            "duplicate_rows": duplicate_rows,
            "missing_values": missing_values,
            "subject_overlap": overlap,
        }
    )

    expected_map = {
        "n_features": expected.get("n_features"),
        "n_subjects": expected.get("n_subjects"),
        "n_train": expected.get("n_train"),
        "n_test": expected.get("n_test"),
    }
    mismatches = {
        key: {"expected": value, "actual": checks[key]}
        for key, value in expected_map.items()
        if value is not None and checks[key] != value
    }
    if mismatches:
        errors.append(f"Jumlah tidak sesuai ekspektasi konfigurasi: {mismatches}")
    checks["expected_mismatches"] = mismatches

    if errors:
        raise DatasetIntegrityError("; ".join(errors))
    return {"ok": True, "checks": checks, "warnings": warnings, "errors": errors}


# ---------------------------------------------------------------------------
# Download / extract
# ---------------------------------------------------------------------------
def _download_file(url: str, destination: Path, logger: logging.Logger, timeout: int = 180) -> None:
    tmp = destination.with_suffix(destination.suffix + ".part")
    logger.info("Mengunduh dataset dari %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as response, tmp.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        last_report = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if downloaded - last_report >= 10 << 20:
                last_report = downloaded
                if total:
                    logger.info("  %.1f / %.1f MB", downloaded / 1e6, total / 1e6)
                else:
                    logger.info("  %.1f MB", downloaded / 1e6)
    os.replace(tmp, destination)


def download_uci_har(
    paths: ProjectPaths,
    config: dict[str, Any],
    force: bool = False,
    logger: logging.Logger | None = None,
) -> Path:
    """Download and extract UCI HAR into ``data/raw``, returning the dataset dir."""
    log = logger or get_logger("har.data")
    paths.ensure()

    existing = find_dataset_dir(paths.data_raw)
    if existing is not None and not force:
        log.info("Dataset UCI HAR sudah tersedia: %s", existing)
        return existing

    dataset_cfg = config.get("dataset", {})
    urls = [url for url in (dataset_cfg.get("primary_url"), dataset_cfg.get("mirror_url")) if url]
    if not urls:
        raise DatasetIntegrityError("Tidak ada URL dataset pada config")

    zip_path = paths.data_raw / "uci_har.zip"
    if zip_path.exists() and not force:
        try:
            with zipfile.ZipFile(zip_path) as archive:
                bad = archive.testzip()
            if bad is None:
                log.info("Menggunakan ZIP yang sudah ada: %s", zip_path)
            else:
                raise zipfile.BadZipFile(bad)
        except (zipfile.BadZipFile, OSError):
            log.warning("ZIP lama rusak, mengunduh ulang")
            zip_path.unlink(missing_ok=True)

    if not zip_path.exists():
        errors = []
        for url in urls:
            try:
                _download_file(url, zip_path, log)
                with zipfile.ZipFile(zip_path) as archive:
                    if archive.testzip() is not None:
                        raise zipfile.BadZipFile("CRC check gagal")
                break
            except Exception as exc:  # network/CRC problems -> try next mirror
                errors.append(f"{url}: {type(exc).__name__}: {exc}")
                zip_path.unlink(missing_ok=True)
                log.warning("Gagal mengunduh dari %s (%s)", url, exc)
        else:
            raise DatasetIntegrityError(
                "Gagal mengunduh dataset. Periksa koneksi atau unduh manual. " + " | ".join(errors)
            )

    log.info("Mengekstrak %s", zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(paths.data_raw)
    dataset_dir = find_dataset_dir(paths.data_raw)
    if dataset_dir is None:
        raise DatasetIntegrityError("ZIP terekstrak tetapi folder 'UCI HAR Dataset' tidak ditemukan")
    return dataset_dir


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _read_split(dataset_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = pd.read_csv(
        dataset_dir / split / f"X_{split}.txt",
        sep=r"\s+",
        header=None,
        engine="python",
        dtype=np.float32,
    ).to_numpy(dtype=np.float32)
    y = np.loadtxt(dataset_dir / split / f"y_{split}.txt", dtype=np.int64)
    subject = np.loadtxt(dataset_dir / split / f"subject_{split}.txt", dtype=np.int64)
    raw_channels = []
    for channel in RAW_CHANNELS:
        raw_channels.append(
            np.loadtxt(
                dataset_dir / split / "Inertial Signals" / f"{channel}_{split}.txt",
                dtype=np.float32,
            )
        )
    raw = np.stack(raw_channels, axis=-1).astype(np.float32)
    return X, y, subject, raw


def dataset_fingerprint(dataset_dir: str | Path) -> str:
    directory = Path(dataset_dir)
    parts = []
    for relative in ("features.txt", "train/X_train.txt", "test/X_test.txt"):
        path = directory / relative
        if path.exists():
            parts.append(sha256_file(path))
    return sha256_file_from_string("|".join(parts)) if parts else "unknown"


def sha256_file_from_string(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_uci_har(dataset_dir: str | Path, load_raw: bool = True) -> HARDataset:
    """Load the UCI HAR train/test splits, preserving the official partition."""
    directory = find_dataset_dir(dataset_dir) if Path(dataset_dir).is_dir() else None
    if directory is None:
        raise DatasetIntegrityError(f"Folder dataset UCI HAR tidak ditemukan di {dataset_dir}")

    feature_names = _read_feature_names(directory)
    activity_names, mapping = _read_activity_mapping(directory)

    matrices = []
    labels = []
    subjects = []
    splits = []
    raw_blocks = []
    raw_channels: list[str] = []
    for split in ("train", "test"):
        X, y_raw, subject, raw = _read_split(directory, split)
        if not set(np.unique(y_raw)).issubset({1, 2, 3, 4, 5, 6}):
            raise DatasetIntegrityError(f"Label aktivitas tidak valid pada split {split}")
        matrices.append(X)
        labels.append(np.asarray([mapping[int(value)] for value in y_raw], dtype=np.int64))
        subjects.append(subject)
        splits.append(np.full(len(y_raw), split, dtype="<U5"))
        raw_blocks.append(raw)
        raw_channels = list(RAW_CHANNELS)

    X = np.concatenate(matrices, axis=0)
    y = np.concatenate(labels)
    subject = np.concatenate(subjects)
    split = np.concatenate(splits)
    raw = np.concatenate(raw_blocks, axis=0) if load_raw else None

    ordered_names = list(activity_names)

    metadata = {
        "source": "uci_har",
        "is_synthetic": False,
        "dataset_name": "UCI Human Activity Recognition Using Smartphones",
        "dataset_version": "1.0",
        "repository_url": "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
        "doi": "10.24432/C54S4K",
        "citation": UCI_CITATION,
        "license": UCI_LICENSE_NOTE,
        "device": "Samsung Galaxy S II",
        "device_count": 1,
        "position": "waist-mounted",
        "sampling_rate_hz": 50,
        "window_size": 128,
        "window_seconds": 2.56,
        "overlap": 0.5,
        "activity_ids": [1, 2, 3, 4, 5, 6],
        "n_subjects": int(len(np.unique(subject))),
        "subject_age_range": "19-48",
        "official_split": "70% subjek train / 30% subjek test (subject-disjoint)",
        "feature_source": "precomputed_features",
        "n_duplicate_feature_names": _count_duplicate_feature_names(directory),
        "feature_name_note": (
            "Nama fitur duplikat pada features.txt UCI diberi sufiks __dupN agar unik; "
            "urutan kolom tidak berubah."
        ),
        "feature_note": (
            "561 fitur telah diekstrak oleh pembuat dataset dari sinyal 50 Hz. "
            "Pipeline ini tidak mengekstrak ulang fitur tersebut dari sinyal mentah."
        ),
        "fingerprint": dataset_fingerprint(directory),
        "dataset_dir": str(directory),
    }
    if raw is not None:
        metadata["raw_signal_note"] = (
            "Sinyal mentah 128x9 (body acc, body gyro, total acc) dipakai hanya oleh 1D-CNN."
        )

    return HARDataset(
        X=X,
        y=y,
        subject=subject,
        split=split,
        feature_names=feature_names,
        activity_names=ordered_names,
        raw=raw,
        raw_channels=raw_channels,
        source="uci_har",
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _cache_paths(paths: ProjectPaths, name: str) -> tuple[Path, Path]:
    return paths.data_processed / f"{name}.npz", paths.data_processed / f"{name}.meta.json"


def cache_dataset(dataset: HARDataset, paths: ProjectPaths, name: str | None = None) -> Path:
    from .utils import save_json

    cache_name = name or (DEMO_CACHE_NAME if dataset.is_synthetic else UCI_CACHE_NAME)
    npz_path, meta_path = _cache_paths(paths, cache_name)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "X": dataset.X.astype(np.float32),
        "y": dataset.y.astype(np.int64),
        "subject": dataset.subject.astype(np.int64),
        "split": dataset.split.astype("<U5"),
    }
    if dataset.raw is not None:
        arrays["raw"] = dataset.raw.astype(np.float32)
    np.savez_compressed(npz_path, **arrays)
    save_json(
        meta_path,
        {
            "cache_version": CACHE_VERSION,
            "source": dataset.source,
            "feature_names": dataset.feature_names,
            "activity_names": dataset.activity_names,
            "raw_channels": dataset.raw_channels,
            "metadata": dataset.metadata,
            "n_samples": dataset.n_samples,
            "n_features": dataset.n_features,
        },
    )
    return npz_path


def load_cached_dataset(paths: ProjectPaths, name: str) -> HARDataset | None:
    from .utils import read_json

    npz_path, meta_path = _cache_paths(paths, name)
    meta = read_json(meta_path)
    if not npz_path.exists() or not meta or meta.get("cache_version") != CACHE_VERSION:
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as arrays:
            dataset = HARDataset(
                X=arrays["X"],
                y=arrays["y"],
                subject=arrays["subject"],
                split=arrays["split"],
                feature_names=list(meta["feature_names"]),
                activity_names=list(meta["activity_names"]),
                raw=arrays["raw"] if "raw" in arrays.files else None,
                raw_channels=list(meta["raw_channels"]),
                source=str(meta["source"]),
                metadata=dict(meta["metadata"]),
            )
    except Exception:
        return None
    return dataset


def _cache_is_current(dataset: HARDataset, cached: HARDataset | None) -> bool:
    if cached is None:
        return False
    return (
        cached.source == dataset.source
        and cached.n_samples == dataset.n_samples
        and cached.n_features == dataset.n_features
        and cached.metadata.get("fingerprint") == dataset.metadata.get("fingerprint")
        and cached.metadata.get("seed") == dataset.metadata.get("seed")
    )


# ---------------------------------------------------------------------------
# Demo / high level loader
# ---------------------------------------------------------------------------
def load_demo_dataset(
    paths: ProjectPaths | None = None, config: dict[str, Any] | None = None
) -> HARDataset:
    from .demo_data import generate_demo_dataset

    seed = int((config or {}).get("project", {}).get("seed", 42))
    cached = load_cached_dataset(paths, DEMO_CACHE_NAME) if paths else None
    demo = generate_demo_dataset(seed=seed)
    if paths is not None and not _cache_is_current(demo, cached):
        cache_dataset(demo, paths)
    return demo


def load_dataset(
    config: dict[str, Any],
    paths: ProjectPaths | None = None,
    prefer_demo: bool = False,
    use_cache: bool = True,
    allow_demo_fallback: bool | None = None,
    logger: logging.Logger | None = None,
) -> HARDataset:
    """Load UCI HAR, falling back to a clearly-labelled synthetic demo set."""
    log = logger or get_logger("har.data")
    paths = paths or ProjectPaths.from_config(config)
    paths.ensure()
    dataset_cfg = config.get("dataset", {})
    if allow_demo_fallback is None:
        allow_demo_fallback = bool(dataset_cfg.get("allow_demo_fallback", True))

    if prefer_demo:
        log.warning(DEMO_NOTICE)
        return load_demo_dataset(paths, config)

    dataset_dir = find_dataset_dir(paths.data_raw)

    if use_cache:
        cached = load_cached_dataset(paths, UCI_CACHE_NAME)
        if cached is not None and not cached.is_synthetic:
            cache_note = paths.data_processed / f"{UCI_CACHE_NAME}.npz"
            if dataset_dir is None:
                # The raw folder may live on an external drive. The cache is
                # still self-describing (feature names, labels, metadata), so it
                # can be used explicitly for dashboards or tests.
                log.warning(
                    "Dataset mentah tidak ditemukan; memakai cache yang sudah ada (%s). "
                    "Jalankan `python scripts/prepare_data.py --download` untuk memverifikasi ulang.",
                    cache_note,
                )
                return cached
            try:
                current_fingerprint = dataset_fingerprint(dataset_dir)
            except OSError:
                current_fingerprint = None
            if current_fingerprint == cached.metadata.get("fingerprint"):
                log.info("Memuat dataset dari cache: %s", cache_note)
                return cached
            log.info(
                "Cache tidak cocok dengan dataset mentah saat ini; dataset akan dimuat ulang."
            )
    if dataset_dir is None:
        if allow_demo_fallback:
            log.warning("Dataset UCI HAR belum tersedia. Menggunakan data DEMO SINTETIS.")
            log.warning(DEMO_NOTICE)
            return load_demo_dataset(paths, config)
        raise DatasetIntegrityError(
            "Dataset UCI HAR belum tersedia. Jalankan `python scripts/prepare_data.py --download` "
            f"atau letakkan folder '{UCI_FOLDER_NAME}' di {paths.data_raw}."
        )

    log.info("Memuat dataset UCI HAR dari %s", dataset_dir)
    dataset = load_uci_har(dataset_dir, load_raw=True)
    if use_cache:
        cache_dataset(dataset, paths)
    return dataset


# ---------------------------------------------------------------------------
# Summary / integrity
# ---------------------------------------------------------------------------
def dataset_integrity(dataset: HARDataset) -> dict[str, Any]:
    duplicate_rows = 0
    if dataset.n_samples:
        try:
            duplicate_rows = int(dataset.n_samples - np.unique(dataset.X, axis=0).shape[0])
        except MemoryError:  # pragma: no cover - defensive
            duplicate_rows = -1
    counts = np.bincount(dataset.y, minlength=len(dataset.activity_names))
    return {
        "n_missing_values": int(np.isnan(dataset.X).sum()),
        "n_infinite_values": int((~np.isfinite(dataset.X)).sum()),
        "n_duplicate_rows": duplicate_rows,
        "n_duplicate_feature_names": int(
            dataset.metadata.get(
                "n_duplicate_feature_names",
                len(dataset.feature_names) - len(set(dataset.feature_names)),
            )
        ),
        "labels_valid": bool(
            np.all(dataset.y >= 0) and np.all(dataset.y < len(dataset.activity_names))
        ),
        "n_subjects": int(len(dataset.subject_ids)),
        "n_train": int((dataset.split == "train").sum()),
        "n_test": int((dataset.split == "test").sum()),
        "raw_signals_available": dataset.raw is not None,
        "raw_shape": None if dataset.raw is None else list(dataset.raw.shape),
        "class_counts": {name: int(counts[i]) for i, name in enumerate(dataset.activity_names)},
        "n_features": dataset.n_features,
        "all_features_finite": bool(np.isfinite(dataset.X).all()),
    }


def build_dataset_summary(
    dataset: HARDataset, expected: dict[str, Any] | None = None
) -> dict[str, Any]:
    from .feature_extraction import parse_feature_groups

    integrity = dataset_integrity(dataset)
    expected = expected or {}
    expected_checks = {}
    for key, expected_value in (
        ("n_features", expected.get("n_features")),
        ("n_subjects", expected.get("n_subjects")),
        ("n_train", expected.get("n_train")),
        ("n_test", expected.get("n_test")),
    ):
        actual_key = key
        actual = integrity.get(actual_key, dataset.n_features if key == "n_features" else None)
        if expected_value is not None:
            expected_checks[key] = {
                "expected": expected_value,
                "actual": actual,
                "ok": actual == expected_value,
            }

    total = max(dataset.n_samples, 1)
    class_distribution = []
    for index, name in enumerate(dataset.activity_names):
        count = int((dataset.y == index).sum())
        class_distribution.append(
            {
                "code": index,
                "activity": name,
                "display": name.replace("_", " ").title(),
                "n": count,
                "pct": round(100.0 * count / total, 2),
            }
        )

    by_split = {}
    for split in ("train", "test"):
        mask = dataset.split == split
        by_split[split] = {
            name: int((dataset.y[mask] == index).sum())
            for index, name in enumerate(dataset.activity_names)
        }

    subjects = []
    for subject_id in dataset.subject_ids:
        mask = dataset.subject == subject_id
        subjects.append(
            {
                "subject": int(subject_id),
                "split": str(dataset.split[mask][0]) if mask.any() else "unknown",
                "n_samples": int(mask.sum()),
                "activities": {
                    name: int((dataset.y[mask] == index).sum())
                    for index, name in enumerate(dataset.activity_names)
                },
            }
        )

    feature_groups = parse_feature_groups(dataset.feature_names)

    return {
        "source": dataset.source,
        "is_synthetic": dataset.is_synthetic,
        "source_label": dataset.display_source,
        "notice": DEMO_NOTICE if dataset.is_synthetic else None,
        "n_samples": dataset.n_samples,
        "n_features": dataset.n_features,
        "n_subjects": len(dataset.subject_ids),
        "n_train": integrity["n_train"],
        "n_test": integrity["n_test"],
        "class_distribution": class_distribution,
        "class_distribution_by_split": by_split,
        "subjects": subjects,
        "integrity": integrity,
        "expected_checks": expected_checks,
        "feature_groups": feature_groups,
        "metadata": dataset.metadata,
    }


def save_dataset_summary(summary: dict[str, Any], paths: ProjectPaths) -> Path:
    from .utils import save_json

    paths.ensure()
    save_json(paths.tables / "dataset_summary.json", summary)
    # ``data/processed`` is shared between research and demo modes. Never let a
    # synthetic summary overwrite the real-data summary used by the dashboard
    # fallback.
    if not summary.get("is_synthetic"):
        save_json(paths.data_processed / "dataset_summary.json", summary)
    return paths.tables / "dataset_summary.json"
