"""Tests for split strategies and leakage guardrails."""

from __future__ import annotations

import numpy as np
import pytest

from src.split_strategy import (
    assert_indices_disjoint,
    assert_no_subject_overlap,
    calibration_split,
    group_kfold_folds,
    loso_folds,
    official_split_indices,
    select_subjects_evenly,
)
from src.utils import SubjectLeakageError


def test_official_split_has_no_subject_overlap(demo_dataset):
    train_idx, test_idx = official_split_indices(demo_dataset)
    assert len(train_idx) > 0 and len(test_idx) > 0
    assert set(demo_dataset.subject[train_idx]).isdisjoint(set(demo_dataset.subject[test_idx]))
    assert np.intersect1d(train_idx, test_idx).size == 0


def test_loso_tests_every_subject_exactly_once(demo_dataset):
    folds = loso_folds(demo_dataset)
    assert len(folds) == len(demo_dataset.subject_ids)
    tested = [fold.test_subjects[0] for fold in folds]
    assert sorted(tested) == demo_dataset.subject_ids
    for fold in folds:
        assert fold.test_subjects[0] not in fold.train_subjects
        assert np.intersect1d(fold.train_idx, fold.test_idx).size == 0
        assert len(fold.train_idx) + len(fold.test_idx) == demo_dataset.n_samples


def test_loso_max_folds_is_deterministic_and_evenly_spaced(demo_dataset):
    folds = loso_folds(demo_dataset, max_folds=3)
    assert len(folds) == 3
    again = loso_folds(demo_dataset, max_folds=3)
    assert [fold.test_subjects for fold in folds] == [fold.test_subjects for fold in again]


def test_select_subjects_evenly():
    assert select_subjects_evenly([1, 2, 3, 4], None) == [1, 2, 3, 4]
    assert select_subjects_evenly([1, 2, 3, 4, 5], 3) == [1, 3, 5]
    with pytest.raises(ValueError):
        select_subjects_evenly([1, 2, 3], 0)


def test_group_kfold_covers_all_subjects_once(demo_dataset):
    folds = group_kfold_folds(demo_dataset, n_splits=3, seed=42)
    tested = [subject for fold in folds for subject in fold.test_subjects]
    assert sorted(tested) == demo_dataset.subject_ids
    for fold in folds:
        assert set(fold.train_subjects).isdisjoint(set(fold.test_subjects))
        assert np.intersect1d(fold.train_idx, fold.test_idx).size == 0


def test_group_kfold_is_deterministic(demo_dataset):
    first = group_kfold_folds(demo_dataset, n_splits=3, seed=42)
    second = group_kfold_folds(demo_dataset, n_splits=3, seed=42)
    assert [fold.test_subjects for fold in first] == [fold.test_subjects for fold in second]


def test_group_kfold_rejects_too_many_splits(demo_dataset):
    with pytest.raises(ValueError):
        group_kfold_folds(demo_dataset, n_splits=len(demo_dataset.subject_ids) + 1)


def test_calibration_split_is_disjoint_and_complete(demo_dataset):
    indices = np.arange(demo_dataset.n_samples)
    calibration, evaluation = calibration_split(indices, demo_dataset.y, fraction=0.1, seed=5)
    assert np.intersect1d(calibration, evaluation).size == 0
    assert len(calibration) + len(evaluation) == len(indices)
    assert set(calibration).issubset(set(indices.tolist()))
    # Every activity keeps at least one evaluation row.
    for label in np.unique(demo_dataset.y):
        assert np.isin(demo_dataset.y[evaluation], label).any()


def test_calibration_split_rejects_invalid_fraction(demo_dataset):
    with pytest.raises(ValueError):
        calibration_split(np.arange(10), demo_dataset.y, fraction=1.5)


def test_leakage_assertions_raise():
    with pytest.raises(SubjectLeakageError):
        assert_no_subject_overlap([1, 2], [2, 3], "ctx")
    with pytest.raises(SubjectLeakageError):
        assert_indices_disjoint(np.array([1, 2]), np.array([2, 3]), "ctx")
