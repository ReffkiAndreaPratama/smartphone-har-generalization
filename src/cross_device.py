"""Optional cross-device experiment for self-collected smartphone data.

UCI HAR hanya berisi satu perangkat (Samsung Galaxy S II). Karena itu eksperimen
lintas perangkat **tidak** dijalankan pada dataset utama dan tidak ada klaim
generalisasi lintas perangkat dalam laporan. Modul ini menyediakan jalur
opsional apabila data multi-perangkat tersedia.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluate import aggregate_metrics, compute_metrics, per_subject_metrics
from .feature_extraction import extract_window_features
from .train import predict_sklearn, train_sklearn
from .utils import (
    ACTIVITY_NAMES,
    RAW_CHANNELS,
    DatasetIntegrityError,
    ProjectError,
    ProjectPaths,
    get_logger,
    now_iso,
    read_json,
    save_json,
)

REQUIRED_MANIFEST_COLUMNS = ["file_path", "subject_id", "device_id", "activity"]


class CrossDeviceDataUnavailable(ProjectError):
    pass


@dataclass
class MultiDeviceDataset:
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    device: np.ndarray
    raw: np.ndarray
    feature_names: list[str]
    activity_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.int64)
        self.subject = np.asarray(self.subject).astype(str)
        self.device = np.asarray(self.device).astype(str)
        self.raw = np.asarray(self.raw, dtype=np.float32)
        if self.X.ndim != 2 or self.raw.ndim != 3:
            raise DatasetIntegrityError("X harus 2D dan raw harus 3D")
        if len(self.X) != len(self.y) or len(self.X) != len(self.device):
            raise DatasetIntegrityError("Panjang X/y/device tidak konsisten")
        if not np.isfinite(self.X).all() or not np.isfinite(self.raw).all():
            raise DatasetIntegrityError("Data lintas perangkat mengandung NaN/tak hingga")
        if self.metadata.get("sampling_rate_hz") and self.raw.shape[1] != int(
            self.metadata.get("window_size", self.raw.shape[1])
        ):
            raise DatasetIntegrityError("Ukuran window tidak konsisten")

    @property
    def n_samples(self) -> int:
        return int(len(self.X))

    @property
    def device_ids(self) -> list[str]:
        return sorted(str(value) for value in np.unique(self.device))

    def subset(self, indices: np.ndarray) -> "MultiDeviceDataset":
        index = np.asarray(indices, dtype=np.int64)
        return MultiDeviceDataset(
            X=self.X[index],
            y=self.y[index],
            subject=self.subject[index],
            device=self.device[index],
            raw=self.raw[index],
            feature_names=list(self.feature_names),
            activity_names=list(self.activity_names),
            metadata=dict(self.metadata),
        )


def load_multi_device_dataset(manifest_path: str | Path) -> MultiDeviceDataset:
    """Load a manifest of raw 128x9 windows and extract a common feature set."""
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise CrossDeviceDataUnavailable(
            f"Manifest lintas perangkat tidak ditemukan: {manifest}. "
            "Lihat docs/cross_device_protocol.md untuk format yang diharapkan."
        )
    frame = pd.read_csv(manifest)
    missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetIntegrityError(f"Kolom manifest tidak lengkap: {missing}")
    if frame.empty:
        raise DatasetIntegrityError("Manifest lintas perangkat kosong")
    if frame["device_id"].nunique() < 2:
        raise DatasetIntegrityError(
            "Manifest harus memuat minimal dua device_id untuk eksperimen lintas perangkat"
        )
    if frame["subject_id"].nunique() < 2:
        raise DatasetIntegrityError("Manifest harus memuat minimal dua subjek")

    activity_names = list(ACTIVITY_NAMES)
    encoded_labels = {name: index for index, name in enumerate(activity_names)}
    labels = []
    features = []
    windows = []
    feature_names: list[str] | None = None
    sampling_rates = set()
    for row_index, row in frame.iterrows():
        activity = str(row["activity"]).upper()
        if activity not in encoded_labels:
            raise DatasetIntegrityError(
                f"Baris {row_index}: aktivitas '{activity}' tidak termasuk {activity_names}"
            )
        path = Path(str(row["file_path"]))
        if not path.is_absolute():
            path = manifest.parent / path
        if not path.exists():
            raise DatasetIntegrityError(f"File sinyal tidak ditemukan: {path}")
        if path.suffix.lower() == ".npy":
            window = np.load(path)
        elif path.suffix.lower() == ".csv":
            window = pd.read_csv(path, header=None).to_numpy(dtype=np.float32)
        else:
            raise DatasetIntegrityError(f"Format file tidak didukung: {path.suffix}")
        window = np.asarray(window, dtype=np.float32)
        if window.ndim != 2 or window.shape[1] != len(RAW_CHANNELS):
            raise DatasetIntegrityError(
                f"{path}: window harus berbentuk (128, 9), ditemukan {window.shape}"
            )
        rate = float(row.get("sampling_rate_hz", 50.0) or 50.0)
        sampling_rates.add(rate)
        names, vector = extract_window_features(window, sampling_rate_hz=rate)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise DatasetIntegrityError("Nama fitur tidak konsisten antar-file")
        features.append(vector)
        windows.append(window)
        labels.append(encoded_labels[activity])

    if len(sampling_rates) > 1:
        raise DatasetIntegrityError(
            f"Sampling rate berbeda antar-file: {sorted(sampling_rates)}. "
            "Resampling eksplisit diperlukan sebelum ekstraksi fitur."
        )

    metadata = {
        "source": "self_collected",
        "is_synthetic": False,
        "notice": "Data mandiri; bukan bagian dari dataset UCI HAR.",
        "manifest": str(manifest.resolve()),
        "devices": sorted(str(value) for value in frame["device_id"].unique()),
        "subjects": sorted(str(value) for value in frame["subject_id"].unique()),
        "window_size": int(np.asarray(windows).shape[1]),
        "sampling_rate_hz": float(sampling_rates.pop()),
        "feature_note": (
            "Fitur dihitung dari sinyal mentah dengan src.feature_extraction.extract_window_features; "
            "bukan 561 fitur UCI."
        ),
        "activity_ids": list(range(1, 7)),
        "n_features": len(feature_names or []),
    }
    return MultiDeviceDataset(
        X=np.vstack(features),
        y=np.asarray(labels, dtype=np.int64),
        subject=frame["subject_id"].astype(str).to_numpy(),
        device=frame["device_id"].astype(str).to_numpy(),
        raw=np.stack(windows).astype(np.float32),
        feature_names=list(feature_names or []),
        activity_names=activity_names,
        metadata=metadata,
    )


def run_cross_device_experiment(
    dataset: MultiDeviceDataset,
    config: dict[str, Any],
    paths: ProjectPaths,
    seed: int = 42,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Leave-one-device-out evaluation of RF and SVM."""
    log = logger or get_logger("har.cross_device")
    devices = dataset.device_ids
    if len(devices) < 2:
        raise CrossDeviceDataUnavailable("Minimal dua perangkat diperlukan")
    results = {
        "experiment": "C_cross_device",
        "title": "Generalisasi lintas perangkat (data mandiri)",
        "status": "complete",
        "protocol": "leave_one_device_out",
        "seed": int(seed),
        "created_at": now_iso(),
        "dataset": dataset.metadata,
        "models": {},
    }
    for model_name in ("random_forest", "svm"):
        folds = []
        for device in devices:
            train_idx = np.flatnonzero(dataset.device != device)
            test_idx = np.flatnonzero(dataset.device == device)
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            if set(dataset.device[train_idx]) & set(dataset.device[test_idx]):
                raise DatasetIntegrityError("Kebocoran perangkat antara train dan test")
            log.info("Cross-device %s: uji pada %s", model_name, device)
            training = train_sklearn(
                model_name,
                dataset.X[train_idx],
                dataset.y[train_idx],
                dataset.subject[train_idx],
                config,
                seed=seed,
                select=True,
                logger=log,
            )
            predictions, scores, _, predict_time = predict_sklearn(training.model, dataset.X[test_idx])
            metrics = compute_metrics(
                dataset.y[test_idx], predictions, dataset.activity_names, y_score=scores
            )
            folds.append(
                {
                    "fold": device,
                    "test_device": device,
                    "train_devices": sorted(set(dataset.device[train_idx]) - {device}),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "params": training.params,
                    "fit_time_s": round(training.fit_time_s, 4),
                    "predict_time_s": round(predict_time, 4),
                    "metrics": metrics,
                    "per_subject": per_subject_metrics(
                        dataset.y[test_idx],
                        predictions,
                        dataset.subject[test_idx],
                        dataset.activity_names,
                    ),
                }
            )
        results["models"][model_name] = {
            "folds": folds,
            "aggregate": aggregate_metrics([fold["metrics"] for fold in folds]),
        }
    save_json(paths.results / "cross_device.json", results)
    return results


def write_not_performed_result(
    paths: ProjectPaths,
    dataset_source: str = "uci_har",
    is_synthetic: bool = False,
) -> dict[str, Any]:
    """Write an explicit 'not performed' record instead of fabricating results."""
    existing = read_json(paths.results / "cross_device.json")
    if existing and existing.get("status") == "complete":
        return existing
    payload = {
        "experiment": "C_cross_device",
        "title": "Generalisasi lintas perangkat",
        "status": "not_performed",
        "reason": (
            "UCI HAR hanya menyediakan satu perangkat (Samsung Galaxy S II, "
            "waist-mounted). Tidak ada variasi perangkat yang dapat dipakai untuk "
            "menguji generalisasi lintas perangkat."
        ),
        "dataset_source": dataset_source,
        "is_synthetic": is_synthetic,
        "available_devices": ["Samsung Galaxy S II (satu perangkat)" if dataset_source == "uci_har" else "unknown"],
        "requirements": [
            "Minimal dua perangkat smartphone berbeda dengan sensor yang sama jenisnya.",
            "Protokol aktivitas identik (6 aktivitas, urutan dan durasi sama).",
            "Posisi pemasangan dan orientasi dicatat per sesi.",
            "Setiap perangkat memiliki data dari beberapa subjek (idealnya >= 5).",
            "Metadata perangkat: model, OS, vendor sensor, sampling rate, resolusi.",
            "Sinyal mentah tersedia dalam window 128 sampel x 9 kanal (50 Hz, overlap 50%).",
        ],
        "manifest_schema": {
            "columns": REQUIRED_MANIFEST_COLUMNS + ["sampling_rate_hz (opsional)"],
            "file_format": "CSV atau NPY berisi array (128, 9) dengan urutan kanal: " + ", ".join(RAW_CHANNELS),
            "example": "data/cross_device/manifest.csv",
        },
        "how_to_run": [
            "Letakkan manifest dan file sinyal di data/cross_device/.",
            "Jalankan: python scripts/run_experiments.py --stages cross-device",
            "Hasil akan ditulis ke reports/results/cross_device.json.",
        ],
        "documentation": "docs/cross_device_protocol.md",
        "created_at": now_iso(),
    }
    save_json(paths.results / "cross_device.json", payload)
    return payload
