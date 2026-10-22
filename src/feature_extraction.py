"""Feature metadata and signal windowing utilities.

Catatan metodologis
-------------------
Model Random Forest dan SVM pada penelitian ini memakai **561 fitur yang sudah
disediakan** oleh UCI HAR (diekstrak oleh pembuat dataset). Modul ini tidak
mengklaim mengekstrak ulang fitur tersebut. Fungsi ekstraksi di bagian bawah
hanya dipakai untuk protokol data mandiri / lintas perangkat, yang fiturnya
dihitung dari sinyal mentah dengan definisi yang eksplisit.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .utils import RAW_CHANNELS

WINDOW_SIZE = 128
SAMPLING_RATE_HZ = 50.0
OVERLAP = 0.5

_STATISTIC_PATTERN = re.compile(
    r"-(mean|std|mad|max|min|sma|energy|iqr|entropy|arCoeff|correlation|maxInds|"
    r"meanFreq|skewness|kurtosis)\(\)"
)


def parse_feature_groups(feature_names: list[str]) -> dict:
    """Summarise UCI HAR feature names by sensor, domain and statistic."""
    total = len(feature_names)
    if not feature_names or not any(name.startswith(("t", "f")) for name in feature_names):
        return {
            "total": total,
            "kind": "unknown",
            "sensor_counts": {},
            "domain_counts": {},
            "statistic_counts": {},
            "groups": [],
            "note": "Nama fitur tidak mengikuti pola UCI HAR (kemungkinan data demo).",
        }

    sensor_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {"time": 0, "frequency": 0, "other": 0}
    statistic_counts: dict[str, int] = {}
    for name in feature_names:
        domain = "time" if name.startswith("t") else "frequency" if name.startswith("f") else "other"
        domain_counts[domain] += 1
        if "BodyAcc" in name:
            sensor = "accelerometer_body"
        elif "GravityAcc" in name:
            sensor = "accelerometer_gravity"
        elif "BodyGyro" in name:
            sensor = "gyroscope_body"
        else:
            sensor = "other"
        sensor_counts[sensor] = sensor_counts.get(sensor, 0) + 1
        match = _STATISTIC_PATTERN.search(name)
        statistic = match.group(1) if match else "angle/other"
        statistic_counts[statistic] = statistic_counts.get(statistic, 0) + 1

    groups = [
        {"name": key, "n": value}
        for key, value in sorted(sensor_counts.items(), key=lambda item: -item[1])
    ]
    return {
        "total": total,
        "kind": "uci_har",
        "sensor_counts": sensor_counts,
        "domain_counts": domain_counts,
        "statistic_counts": dict(
            sorted(statistic_counts.items(), key=lambda item: -item[1])
        ),
        "groups": groups,
        "note": "561 fitur prakomputasi UCI HAR; bukan hasil ekstraksi ulang pipeline ini.",
    }


def load_inertial_signals(dataset_dir: str | Path, split: str) -> tuple[np.ndarray, list[str]]:
    """Load UCI HAR raw inertial windows as ``(n_windows, 128, 9)``."""
    directory = Path(dataset_dir) / split / "Inertial Signals"
    arrays = []
    for channel in RAW_CHANNELS:
        path = directory / f"{channel}_{split}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Sinyal mentah tidak ditemukan: {path}")
        arrays.append(np.loadtxt(path, dtype=np.float32))
    raw = np.stack(arrays, axis=-1)
    if raw.ndim != 3 or raw.shape[1] != WINDOW_SIZE:
        raise ValueError(f"Bentuk sinyal mentah tidak valid: {raw.shape}")
    return raw, list(RAW_CHANNELS)


def window_continuous_signal(
    signal: np.ndarray,
    window_size: int = WINDOW_SIZE,
    overlap: float = OVERLAP,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a continuous signal into overlapping windows.

    Parameters
    ----------
    signal:
        Array of shape ``(n_samples,)`` or ``(n_samples, n_channels)``.
    overlap:
        Fraction of overlap in ``[0, 1)``.

    Returns
    -------
    windows, start_indices
        ``windows`` has shape ``(n_windows, window_size, n_channels)``.
    """
    values = np.asarray(signal, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("signal harus 1D atau 2D (samples x channels)")
    if not 0 <= overlap < 1:
        raise ValueError("overlap harus berada pada rentang [0, 1)")
    if len(values) < window_size:
        return np.empty((0, window_size, values.shape[1]), dtype=np.float32), np.empty(0, dtype=int)
    stride = max(1, int(round(window_size * (1.0 - overlap))))
    starts = np.arange(0, len(values) - window_size + 1, stride, dtype=int)
    windows = np.stack([values[start : start + window_size] for start in starts])
    return windows.astype(np.float32), starts


def _spectral_features(channel: np.ndarray, sampling_rate_hz: float) -> tuple[float, float, float]:
    n = len(channel)
    centered = channel - float(np.mean(channel))
    spectrum = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
    total_energy = float(np.sum(spectrum**2))
    if total_energy <= 1e-12 or len(spectrum) < 2:
        return 0.0, 0.0, 0.0
    dominant = float(frequencies[int(np.argmax(spectrum[1:]) + 1)])
    probabilities = spectrum**2 / total_energy
    entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-12)) / np.log2(len(probabilities)))
    return dominant, total_energy, entropy


def extract_window_features(
    window: np.ndarray, sampling_rate_hz: float = SAMPLING_RATE_HZ
) -> tuple[list[str], np.ndarray]:
    """Extract an explicit, documented feature set from one raw window.

    Only used for self-collected / cross-device data. The UCI 561-feature
    models are *not* trained on these features.
    """
    from scipy.stats import kurtosis, skew

    values = np.asarray(window, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    names: list[str] = []
    features: list[float] = []
    for channel_index in range(values.shape[1]):
        channel = values[:, channel_index]
        prefix = f"ch{channel_index}"
        diffs = np.diff(channel)
        zero_crossings = float(np.sum(np.diff(np.signbit(channel)) != 0))
        dominant, energy, entropy = _spectral_features(channel, sampling_rate_hz)
        values_map = {
            "mean": float(np.mean(channel)),
            "std": float(np.std(channel)),
            "min": float(np.min(channel)),
            "max": float(np.max(channel)),
            "median": float(np.median(channel)),
            "rms": float(np.sqrt(np.mean(channel**2))),
            "iqr": float(np.percentile(channel, 75) - np.percentile(channel, 25)),
            "energy": float(np.sum(channel**2)),
            "skewness": float(skew(channel)) if np.std(channel) > 0 else 0.0,
            "kurtosis": float(kurtosis(channel)) if np.std(channel) > 0 else 0.0,
            "zero_crossings": zero_crossings,
            "diff_std": float(np.std(diffs)) if len(diffs) else 0.0,
            "dominant_freq_hz": dominant,
            "spectral_energy": energy,
            "spectral_entropy": entropy,
        }
        for key, value in values_map.items():
            names.append(f"{prefix}_{key}")
            features.append(value)
    return names, np.asarray(features, dtype=np.float32)


def extract_features_from_windows(
    windows: np.ndarray, channel_names: list[str] | None = None
) -> tuple[list[str], np.ndarray]:
    """Apply :func:`extract_window_features` to many windows.

    Returns ``(feature_names, matrix)`` where ``matrix`` has shape
    ``(n_windows, n_features)`` - the same ordering as
    :func:`extract_window_features`.
    """
    arrays = []
    names: list[str] | None = None
    for window in np.asarray(windows):
        window_names, vector = extract_window_features(window)
        if names is None:
            names = window_names
        arrays.append(vector)
    if not arrays:
        return names or [], np.empty((0, 0), dtype=np.float32)
    if channel_names:
        if len(channel_names) != np.asarray(windows).shape[-1]:
            raise ValueError("channel_names tidak sesuai jumlah channel")
    return names or [], np.vstack(arrays)
