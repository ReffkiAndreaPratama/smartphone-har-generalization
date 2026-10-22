"""Tests for feature extraction helpers, windowing and augmentation."""

from __future__ import annotations

import numpy as np
import pytest

from src.augmentation import augment_batch_numpy, augment_signal_window
from src.feature_extraction import (
    extract_features_from_windows,
    extract_window_features,
    parse_feature_groups,
    window_continuous_signal,
)
from src.utils import RAW_CHANNELS


def test_parse_feature_groups_for_uci_names():
    names = [
        "tBodyAccJerk-mean()",
        "tBodyGyroJerk-std()",
        "fBodyAcc-bandsEnergy()-mean()",
        "angle(Z,Y)",
    ]
    groups = parse_feature_groups(names)
    assert groups["kind"] == "uci_har"
    assert groups["domain_counts"]["time"] == 2
    assert groups["domain_counts"]["frequency"] == 1
    assert groups["domain_counts"]["other"] == 1
    assert groups["sensor_counts"]["accelerometer_body"] >= 1
    assert groups["sensor_counts"]["gyroscope_body"] >= 1


def test_parse_feature_groups_for_unknown_names():
    groups = parse_feature_groups(["demo_feature_001", "demo_feature_002"])
    assert groups["kind"] == "unknown"
    assert groups["sensor_counts"] == {}


def test_window_continuous_signal_shapes():
    signal = np.arange(1000, dtype=np.float32)
    windows, starts = window_continuous_signal(signal, window_size=128, overlap=0.5)
    assert windows.shape[1:] == (128, 1)
    assert len(starts) == len(windows)
    assert starts[0] == 0
    # 50% overlap -> stride 64
    assert starts[1] == 64


def test_window_continuous_signal_too_short():
    windows, starts = window_continuous_signal(np.arange(10, dtype=np.float32), window_size=128)
    assert windows.shape == (0, 128, 1)
    assert len(starts) == 0


def test_extract_window_features_is_deterministic():
    rng = np.random.default_rng(0)
    window = rng.normal(size=(128, len(RAW_CHANNELS))).astype(np.float32)
    names, vector = extract_window_features(window)
    assert len(names) == len(vector)
    assert len(names) == len(RAW_CHANNELS) * 15
    names2, vector2 = extract_window_features(window)
    assert names == names2
    assert np.allclose(vector, vector2)
    assert np.isfinite(vector).all()


def test_extract_features_from_windows():
    rng = np.random.default_rng(1)
    windows = rng.normal(size=(3, 128, len(RAW_CHANNELS))).astype(np.float32)
    names, matrix = extract_features_from_windows(windows)
    assert matrix.shape == (3, len(names))

def test_augmentation_changes_signal_but_keeps_shape():
    rng = np.random.default_rng(2)
    window = rng.normal(size=(128, len(RAW_CHANNELS))).astype(np.float32)
    augmented = augment_signal_window(window, rng, jitter_sigma=0.05, max_shift=4)
    assert augmented.shape == window.shape
    assert not np.allclose(augmented, window)


def test_augmentation_is_reproducible_with_seed():
    window = np.ones((128, len(RAW_CHANNELS)), dtype=np.float32)
    first = augment_signal_window(window, np.random.default_rng(11), max_shift=0)
    second = augment_signal_window(window, np.random.default_rng(11), max_shift=0)
    assert np.allclose(first, second)


def test_augment_batch_numpy():
    rng = np.random.default_rng(3)
    windows = rng.normal(size=(4, 128, len(RAW_CHANNELS))).astype(np.float32)
    augmented = augment_batch_numpy(windows, rng)
    assert augmented.shape == windows.shape
