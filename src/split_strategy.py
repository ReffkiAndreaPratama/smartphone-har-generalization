"""Leakage-safe split strategies for subject-wise generalisation.

Semua fungsi di sini mengembalikan indeks baris pada dataset dan memvalidasi
bahwa tidak ada subjek yang muncul di kedua sisi partisi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .data_loader import HARDataset
from .utils import SubjectLeakageError


@dataclass
class Fold:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_subjects: list[int]
    test_subjects: list[int]

    def to_dict(self) -> dict:
        return {
            "fold": int(self.fold),
            "n_train": int(len(self.train_idx)),
            "n_test": int(len(self.test_idx)),
            "train_subjects": [int(value) for value in self.train_subjects],
            "test_subjects": [int(value) for value in self.test_subjects],
        }


def assert_no_subject_overlap(
    train_subjects: Iterable[int], test_subjects: Iterable[int], context: str = ""
) -> None:
    overlap = sorted(set(int(value) for value in train_subjects) & set(int(value) for value in test_subjects))
    if overlap:
        prefix = f"{context}: " if context else ""
        raise SubjectLeakageError(f"{prefix}subjek bocor antara train dan test: {overlap}")


def assert_indices_disjoint(train_idx: np.ndarray, test_idx: np.ndarray, context: str = "") -> None:
    overlap = np.intersect1d(np.asarray(train_idx), np.asarray(test_idx))
    if overlap.size:
        raise SubjectLeakageError(
            f"{context}: {overlap.size} indeks baris muncul pada train dan test"
        )


def official_split_indices(dataset: HARDataset) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.flatnonzero(dataset.split == "train")
    test_idx = np.flatnonzero(dataset.split == "test")
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Dataset tidak memiliki split train/test resmi")
    assert_no_subject_overlap(
        dataset.subject[train_idx], dataset.subject[test_idx], "official split"
    )
    assert_indices_disjoint(train_idx, test_idx, "official split")
    return train_idx, test_idx


def select_subjects_evenly(subjects: Iterable[int], max_folds: int | None) -> list[int]:
    ordered = sorted(int(value) for value in set(subjects))
    if max_folds is None or max_folds >= len(ordered):
        return ordered
    if max_folds < 1:
        raise ValueError("max_folds harus >= 1")
    positions = np.linspace(0, len(ordered) - 1, num=max_folds)
    return sorted({ordered[int(round(position))] for position in positions})


def loso_folds(dataset: HARDataset, max_folds: int | None = None) -> list[Fold]:
    """Leave-One-Subject-Out folds; every subject is tested exactly once."""
    folds: list[Fold] = []
    selected = select_subjects_evenly(dataset.subject_ids, max_folds)
    all_subjects = set(dataset.subject_ids)
    for fold_index, subject_id in enumerate(selected, start=1):
        test_idx = np.flatnonzero(dataset.subject == subject_id)
        train_idx = np.flatnonzero(dataset.subject != subject_id)
        train_subjects = sorted(all_subjects - {subject_id})
        assert_no_subject_overlap(train_subjects, [subject_id], f"LOSO fold {fold_index}")
        assert_indices_disjoint(train_idx, test_idx, f"LOSO fold {fold_index}")
        folds.append(
            Fold(
                fold=fold_index,
                train_idx=train_idx,
                test_idx=test_idx,
                train_subjects=train_subjects,
                test_subjects=[subject_id],
            )
        )
    return folds


def group_kfold_folds(
    dataset: HARDataset, n_splits: int = 6, seed: int = 42
) -> list[Fold]:
    """Deterministic balanced GroupKFold over subject identities.

    Subjects are shuffled with the configured seed and then assigned
    round-robin to folds. Every subject is tested exactly once.
    """
    subjects = np.asarray(dataset.subject_ids, dtype=np.int64)
    if n_splits < 2:
        raise ValueError("n_splits harus >= 2")
    if n_splits > len(subjects):
        raise ValueError(f"n_splits={n_splits} melebihi jumlah subjek={len(subjects)}")
    rng = np.random.default_rng(seed)
    shuffled = subjects[rng.permutation(len(subjects))]
    assignments: dict[int, list[int]] = {fold: [] for fold in range(n_splits)}
    for position, subject_id in enumerate(shuffled):
        assignments[position % n_splits].append(int(subject_id))

    folds: list[Fold] = []
    for fold_index in range(n_splits):
        test_subjects = sorted(assignments[fold_index])
        train_subjects = sorted(set(int(value) for value in subjects) - set(test_subjects))
        assert_no_subject_overlap(train_subjects, test_subjects, f"GroupKFold fold {fold_index + 1}")
        test_idx = np.flatnonzero(np.isin(dataset.subject, test_subjects))
        train_idx = np.flatnonzero(np.isin(dataset.subject, train_subjects))
        assert_indices_disjoint(train_idx, test_idx, f"GroupKFold fold {fold_index + 1}")
        folds.append(
            Fold(
                fold=fold_index + 1,
                train_idx=train_idx,
                test_idx=test_idx,
                train_subjects=train_subjects,
                test_subjects=test_subjects,
            )
        )
    return folds


def calibration_split(
    indices: np.ndarray,
    labels: np.ndarray,
    fraction: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split target-subject rows into labelled calibration and evaluation parts.

    The split is stratified per activity. At least one evaluation row per
    activity is always retained, and calibration rows are removed from the
    evaluation set to prevent leakage.
    """
    row_indices = np.asarray(indices, dtype=np.int64)
    row_labels = np.asarray(labels)[row_indices]
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction harus berada pada (0, 1)")
    if len(row_indices) == 0:
        raise ValueError("indices kosong")

    rng = np.random.default_rng(seed)
    calibration: list[int] = []
    evaluation: list[int] = []
    for label in np.unique(row_labels):
        class_indices = row_indices[row_labels == label].copy()
        rng.shuffle(class_indices)
        if len(class_indices) < 2:
            evaluation.extend(int(value) for value in class_indices)
            continue
        n_calibration = int(round(fraction * len(class_indices)))
        n_calibration = min(max(n_calibration, 1), len(class_indices) - 1)
        calibration.extend(int(value) for value in class_indices[:n_calibration])
        evaluation.extend(int(value) for value in class_indices[n_calibration:])

    calibration_idx = np.asarray(sorted(calibration), dtype=np.int64)
    evaluation_idx = np.asarray(sorted(evaluation), dtype=np.int64)
    assert_indices_disjoint(calibration_idx, evaluation_idx, "calibration split")
    if len(calibration_idx) + len(evaluation_idx) != len(row_indices):
        raise SubjectLeakageError("Calibration split tidak mencakup seluruh baris target")
    return calibration_idx, evaluation_idx


def folds_to_records(folds: list[Fold]) -> list[dict]:
    return [fold.to_dict() for fold in folds]
