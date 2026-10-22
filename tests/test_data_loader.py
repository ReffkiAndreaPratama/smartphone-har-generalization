"""Tests for dataset loading, validation, caching and integrity reporting."""

from __future__ import annotations

import numpy as np
import pytest

from src.data_loader import (
    HARDataset,
    build_dataset_summary,
    cache_dataset,
    find_dataset_dir,
    load_cached_dataset,
    load_cached_dataset as _load_cached_dataset,
    load_demo_dataset,
    save_dataset_summary,
    verify_uci_structure,
)
from src.utils import DatasetIntegrityError, read_json, save_json


def _dataset(**overrides) -> HARDataset:
    base = dict(
        X=np.arange(12, dtype=np.float32).reshape(4, 3),
        y=np.array([0, 1, 2, 0], dtype=np.int64),
        subject=np.array([1, 1, 2, 2], dtype=np.int64),
        split=np.array(["train", "train", "test", "test"]),
        feature_names=["a", "b", "c"],
        activity_names=["A", "B", "C"],
    )
    base.update(overrides)
    return HARDataset(**base)


def test_dataset_rejects_feature_name_mismatch():
    with pytest.raises(DatasetIntegrityError):
        _dataset(feature_names=["a", "b"])


def test_dataset_rejects_nan_values():
    X = np.zeros((3, 2), dtype=np.float32)
    X[0, 0] = np.nan
    with pytest.raises(DatasetIntegrityError):
        _dataset(X=X, y=np.array([0, 1, 2]), subject=np.array([1, 1, 2]), split=np.array(["train"] * 3))


def test_dataset_rejects_out_of_range_labels():
    with pytest.raises(DatasetIntegrityError):
        _dataset(y=np.array([0, 1, 9, 0]))


def test_dataset_helpers():
    dataset = _dataset()
    assert len(dataset) == 4
    assert dataset.n_samples == 4
    assert dataset.n_features == 3
    assert dataset.subject_ids == [1, 2]
    assert dataset.class_counts() == {"A": 2, "B": 1, "C": 1}
    assert list(dataset.split_indices("test")) == [2, 3]
    assert list(dataset.subject_indices(1)) == [0, 1]
    assert dataset.is_synthetic is False
    assert "UCI HAR" in dataset.display_source


def test_dataset_subset_keeps_alignment():
    dataset = _dataset()
    subset = dataset.subset([0, 2], note="uji")
    assert subset.n_samples == 2
    assert list(subset.subject) == [1, 2]
    assert list(subset.split) == ["train", "test"]
    assert subset.metadata["subset_note"] == "uji"
    assert subset.feature_names == dataset.feature_names


def test_cache_roundtrip(demo_dataset, paths):
    cache_dataset(demo_dataset, paths)
    restored = load_cached_dataset(paths, "demo_cache")
    assert restored is not None
    assert restored.n_samples == demo_dataset.n_samples
    assert restored.n_features == demo_dataset.n_features
    assert list(restored.activity_names) == list(demo_dataset.activity_names)
    assert restored.raw is not None
    assert restored.raw.shape == demo_dataset.raw.shape
    assert restored.is_synthetic


def test_cache_is_ignored_when_version_mismatches(demo_dataset, paths):
    cache_dataset(demo_dataset, paths)
    meta_path = paths.data_processed / "demo_cache.meta.json"
    meta = read_json(meta_path)
    meta["cache_version"] = -1
    save_json(meta_path, meta)
    assert _load_cached_dataset(paths, "demo_cache") is None


def test_demo_dataset_is_labelled_synthetic(config, paths):
    dataset = load_demo_dataset(paths, config)
    assert dataset.is_synthetic
    assert dataset.metadata["notice"]
    summary = build_dataset_summary(dataset)
    assert summary["is_synthetic"] is True
    assert summary["notice"]
    assert summary["source_label"].lower().find("demo") >= 0


def test_demo_summary_does_not_overwrite_processed_summary(demo_dataset, paths):
    save_dataset_summary(build_dataset_summary(demo_dataset), paths)
    processed = paths.data_processed / "dataset_summary.json"
    assert not processed.exists()


def test_find_dataset_dir_locates_uci(uci_available):
    if not uci_available:
        pytest.skip("Dataset UCI HAR belum diunduh")
    from src.utils import PROJECT_ROOT

    directory = find_dataset_dir(PROJECT_ROOT / "data" / "raw")
    assert directory is not None
    assert (directory / "features.txt").exists()


def test_verify_uci_structure_detects_missing_files(tmp_path):
    with pytest.raises(DatasetIntegrityError):
        verify_uci_structure(tmp_path)


def test_build_dataset_summary_structure(demo_dataset):
    summary = build_dataset_summary(demo_dataset)
    assert summary["n_samples"] == demo_dataset.n_samples
    assert len(summary["class_distribution"]) == len(demo_dataset.activity_names)
    assert len(summary["subjects"]) == len(demo_dataset.subject_ids)
    assert summary["integrity"]["n_missing_values"] == 0
    assert summary["integrity"]["labels_valid"] is True
