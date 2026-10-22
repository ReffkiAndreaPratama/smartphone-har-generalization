"""Tests for preprocessing utilities (train-only scaling, subject statistics)."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import (
    FeatureStandardizer,
    PreprocessingError,
    RawSignalStandardizer,
    check_feature_matrix,
    fit_subject_statistics,
    per_subject_standardize,
    transform_subject_statistics,
)


def test_check_feature_matrix_rejects_non_finite():
    X = np.array([[1.0, 2.0], [np.nan, 3.0]])
    with pytest.raises(PreprocessingError):
        check_feature_matrix(X, strict=True)
    report = check_feature_matrix(X, strict=False)
    assert report["n_missing"] == 1
    assert report["finite"] is False


def test_feature_standardizer_fits_only_on_training_rows():
    X = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    scaler = FeatureStandardizer(["a", "b"]).fit(X[:2])
    transformed = scaler.transform(X)
    assert np.allclose(transformed[:2].mean(axis=0), 0.0, atol=1e-6)
    # The third row keeps its distance from the *training* statistics.
    assert transformed[2, 0] > 0
    assert scaler.n_features_in_ == 2
    assert scaler.feature_names == ["a", "b"]


def test_feature_standardizer_rejects_wrong_shape():
    scaler = FeatureStandardizer(["a", "b"]).fit(np.zeros((4, 2)))
    with pytest.raises(PreprocessingError):
        scaler.transform(np.zeros((2, 3)))


def test_feature_standardizer_roundtrip(tmp_path):
    X = np.random.default_rng(0).normal(size=(10, 4))
    scaler = FeatureStandardizer().fit(X)
    restored = FeatureStandardizer.from_dict(scaler.to_dict())
    assert np.allclose(scaler.transform(X), restored.transform(X))
    path = tmp_path / "scaler.joblib"
    scaler.save(path)
    loaded = FeatureStandardizer.load(path)
    assert np.allclose(loaded.transform(X), scaler.transform(X))


def test_subject_statistics_are_per_subject():
    X = np.array([[1.0, 1.0], [3.0, 3.0], [10.0, 20.0], [14.0, 24.0]])
    subjects = np.array([1, 1, 2, 2])
    statistics = fit_subject_statistics(X, subjects)
    assert set(statistics) == {1, 2}
    transformed = transform_subject_statistics(X, subjects, statistics)
    assert np.allclose(transformed[subjects == 1].mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(transformed[subjects == 2].mean(axis=0), 0.0, atol=1e-6)


def test_transform_subject_statistics_requires_statistics():
    with pytest.raises(PreprocessingError):
        transform_subject_statistics(np.zeros((2, 2)), np.array([1, 2]), {})


def test_per_subject_standardize_helper():
    X = np.array([[1.0], [3.0], [10.0], [14.0]])
    subjects = np.array([1, 1, 2, 2])
    transformed = per_subject_standardize(X, subjects)
    assert transformed.shape == X.shape
    assert np.allclose(transformed[subjects == 1].mean(), 0.0, atol=1e-6)


def test_raw_signal_standardizer_global_and_per_subject():
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(6, 128, 9)).astype(np.float32)
    subjects = np.array([1, 1, 1, 2, 2, 2])

    global_scaler = RawSignalStandardizer(mode="global", channel_names=[f"c{i}" for i in range(9)])
    global_scaler.fit(raw)
    global_values = global_scaler.transform(raw)
    assert global_values.shape == raw.shape
    assert np.allclose(global_values.mean(axis=(0, 1)), 0.0, atol=1e-4)

    per_subject = RawSignalStandardizer(mode="per_subject")
    per_subject.fit(raw, subjects)
    values = per_subject.transform(raw, subjects)
    assert np.allclose(values[subjects == 1].mean(axis=(0, 1)), 0.0, atol=1e-4)
    assert np.allclose(values[subjects == 2].mean(axis=(0, 1)), 0.0, atol=1e-4)


def test_raw_signal_standardizer_requires_subjects_for_per_subject():
    raw = np.zeros((2, 128, 9), dtype=np.float32)
    scaler = RawSignalStandardizer(mode="per_subject")
    with pytest.raises(PreprocessingError):
        scaler.fit(raw)


def test_raw_signal_standardizer_roundtrip(tmp_path):
    raw = np.random.default_rng(2).normal(size=(4, 128, 9)).astype(np.float32)
    subjects = np.array([1, 1, 2, 2])
    scaler = RawSignalStandardizer(mode="per_subject").fit(raw, subjects)
    path = tmp_path / "raw_scaler.joblib"
    scaler.save(path)
    loaded = RawSignalStandardizer.load(path)
    assert np.allclose(loaded.transform(raw, subjects), scaler.transform(raw, subjects))
