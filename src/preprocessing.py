"""Preprocessing: validation, train-only scaling and subject normalisation.

Prinsip:
* Scaler global hanya di-``fit`` pada data latih.
* Normalisasi per-subjek memakai statistik subjek itu sendiri dan **tidak**
  memakai label. Pada skenario adaptasi, statistik subjek target dihitung dari
  data target tanpa label (transductive / unsupervised), dan hal ini
  dilaporkan eksplisit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .utils import ProjectError


class PreprocessingError(ProjectError):
    pass


def check_feature_matrix(
    X: np.ndarray, name: str = "X", strict: bool = True, n_features: int | None = None
) -> dict[str, Any]:
    values = np.asarray(X)
    if values.ndim != 2:
        raise PreprocessingError(f"{name} harus 2D, ditemukan shape {values.shape}")
    if n_features is not None and values.shape[1] != n_features:
        raise PreprocessingError(
            f"{name} memiliki {values.shape[1]} fitur, seharusnya {n_features}"
        )
    missing = int(np.isnan(values).sum())
    infinite = int(np.isinf(values).sum())
    report = {
        "name": name,
        "n_samples": int(values.shape[0]),
        "n_features": int(values.shape[1]),
        "n_missing": missing,
        "n_infinite": infinite,
        "min": float(np.nanmin(values)) if values.size else None,
        "max": float(np.nanmax(values)) if values.size else None,
        "finite": missing == 0 and infinite == 0,
    }
    if strict and not report["finite"]:
        raise PreprocessingError(
            f"{name} mengandung {missing} nilai kosong dan {infinite} nilai tak hingga"
        )
    return report


class FeatureStandardizer:
    """Z-score standardizer with explicit feature names and joblib persistence."""

    def __init__(self, feature_names: list[str] | None = None) -> None:
        self.feature_names = list(feature_names or [])
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray, y: Any = None) -> "FeatureStandardizer":
        values = np.asarray(X, dtype=np.float64)
        check_feature_matrix(values, strict=True)
        self.mean_ = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self.scale_ = scale
        self.n_features_in_ = values.shape[1]
        if not self.feature_names:
            self.feature_names = [f"feature_{index}" for index in range(values.shape[1])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise PreprocessingError("FeatureStandardizer belum di-fit")
        values = np.asarray(X, dtype=np.float64)
        check_feature_matrix(values, strict=True, n_features=self.n_features_in_)
        return ((values - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise PreprocessingError("FeatureStandardizer belum di-fit")
        return np.asarray(X, dtype=np.float64) * self.scale_ + self.mean_

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "FeatureStandardizer",
            "feature_names": self.feature_names,
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "scale": None if self.scale_ is None else self.scale_.tolist(),
            "n_features_in": self.n_features_in_,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureStandardizer":
        instance = cls(payload.get("feature_names"))
        if payload.get("mean") is not None:
            instance.mean_ = np.asarray(payload["mean"], dtype=np.float64)
            instance.scale_ = np.asarray(payload["scale"], dtype=np.float64)
            instance.n_features_in_ = int(payload["n_features_in"])
        return instance

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.to_dict(), destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "FeatureStandardizer":
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("type") != "FeatureStandardizer":
            raise PreprocessingError(f"File bukan FeatureStandardizer: {path}")
        return cls.from_dict(payload)


def fit_subject_statistics(
    X: np.ndarray, subjects: np.ndarray
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Compute per-subject mean/std without using labels."""
    values = np.asarray(X, dtype=np.float64)
    subject_values = np.asarray(subjects)
    if len(values) != len(subject_values):
        raise PreprocessingError("Panjang X dan subjects tidak sama")
    statistics: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for subject_id in np.unique(subject_values):
        mask = subject_values == subject_id
        subset = values[mask]
        mean = subset.mean(axis=0)
        scale = subset.std(axis=0)
        scale[scale < 1e-12] = 1.0
        statistics[int(subject_id)] = (mean, scale)
    return statistics


def transform_subject_statistics(
    X: np.ndarray,
    subjects: np.ndarray,
    statistics: dict[int, tuple[np.ndarray, np.ndarray]],
    fallback: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    subject_values = np.asarray(subjects)
    if len(values) != len(subject_values):
        raise PreprocessingError("Panjang X dan subjects tidak sama")
    output = np.empty_like(values, dtype=np.float32)
    for subject_id in np.unique(subject_values):
        mask = subject_values == subject_id
        if int(subject_id) in statistics:
            mean, scale = statistics[int(subject_id)]
        elif fallback is not None:
            mean, scale = fallback
        else:
            raise PreprocessingError(f"Statistik subjek {subject_id} tidak tersedia")
        output[mask] = ((values[mask] - mean) / scale).astype(np.float32)
    return output


def per_subject_standardize(
    X: np.ndarray,
    subjects: np.ndarray,
    reference_statistics: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """Normalise each subject using that subject's own (label-free) statistics."""
    statistics = reference_statistics or fit_subject_statistics(X, subjects)
    return transform_subject_statistics(X, subjects, statistics)


@dataclass
class RawSignalStandardizer:
    """Per-channel standardiser for ``(n, window, channel)`` raw signals."""

    mode: str = "global"
    channel_names: list[str] = field(default_factory=list)
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    subject_statistics_: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def _validate(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 3:
            raise PreprocessingError(f"Sinyal mentah harus 3D, ditemukan {values.shape}")
        if not np.isfinite(values).all():
            raise PreprocessingError("Sinyal mentah mengandung NaN/tak hingga")
        return values

    def fit(self, X: np.ndarray, subjects: np.ndarray | None = None) -> "RawSignalStandardizer":
        values = self._validate(X)
        self.mean_ = values.mean(axis=(0, 1))
        scale = values.std(axis=(0, 1))
        scale[scale < 1e-12] = 1.0
        self.scale_ = scale
        if self.mode == "per_subject":
            if subjects is None:
                raise PreprocessingError("subjects wajib untuk normalisasi per-subjek")
            subject_values = np.asarray(subjects)
            if len(subject_values) != len(values):
                raise PreprocessingError("Panjang subjects tidak sama dengan sinyal")
            self.subject_statistics_ = {}
            for subject_id in np.unique(subject_values):
                subset = values[subject_values == subject_id]
                mean = subset.mean(axis=(0, 1))
                std = subset.std(axis=(0, 1))
                std[std < 1e-12] = 1.0
                self.subject_statistics_[int(subject_id)] = (mean, std)
        return self

    def transform(self, X: np.ndarray, subjects: np.ndarray | None = None) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise PreprocessingError("RawSignalStandardizer belum di-fit")
        values = self._validate(X)
        if self.mode == "per_subject":
            if subjects is None:
                raise PreprocessingError("subjects wajib untuk normalisasi per-subjek")
            subject_values = np.asarray(subjects)
            output = np.empty_like(values, dtype=np.float32)
            for subject_id in np.unique(subject_values):
                mask = subject_values == subject_id
                mean, scale = self.subject_statistics_.get(int(subject_id), (self.mean_, self.scale_))
                output[mask] = ((values[mask] - mean) / scale).astype(np.float32)
            return output
        return ((values - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(
        self, X: np.ndarray, subjects: np.ndarray | None = None
    ) -> np.ndarray:
        return self.fit(X, subjects).transform(X, subjects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "RawSignalStandardizer",
            "mode": self.mode,
            "channel_names": self.channel_names,
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "scale": None if self.scale_ is None else self.scale_.tolist(),
            "subject_statistics": {
                str(subject): {"mean": mean.tolist(), "scale": scale.tolist()}
                for subject, (mean, scale) in self.subject_statistics_.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RawSignalStandardizer":
        instance = cls(
            mode=payload.get("mode", "global"),
            channel_names=list(payload.get("channel_names", [])),
        )
        if payload.get("mean") is not None:
            instance.mean_ = np.asarray(payload["mean"], dtype=np.float64)
            instance.scale_ = np.asarray(payload["scale"], dtype=np.float64)
        instance.subject_statistics_ = {
            int(subject): (
                np.asarray(stats["mean"], dtype=np.float64),
                np.asarray(stats["scale"], dtype=np.float64),
            )
            for subject, stats in payload.get("subject_statistics", {}).items()
        }
        return instance

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.to_dict(), destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "RawSignalStandardizer":
        payload = joblib.load(path)
        if not isinstance(payload, dict) or payload.get("type") != "RawSignalStandardizer":
            raise PreprocessingError(f"File bukan RawSignalStandardizer: {path}")
        return cls.from_dict(payload)


def save_preprocessor(preprocessor: Any, path: str | Path) -> Path:
    if not hasattr(preprocessor, "to_dict"):
        raise PreprocessingError("Preprocessor harus memiliki metode to_dict()")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor.to_dict(), destination)
    return destination


def load_preprocessor(path: str | Path) -> Any:
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise PreprocessingError(f"File preprocessor tidak valid: {path}")
    if payload.get("type") == "FeatureStandardizer":
        return FeatureStandardizer.from_dict(payload)
    if payload.get("type") == "RawSignalStandardizer":
        return RawSignalStandardizer.from_dict(payload)
    raise PreprocessingError(f"Tipe preprocessor tidak dikenal: {payload.get('type')}")
