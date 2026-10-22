"""Deterministic synthetic demo data.

Data demo hanya untuk *smoke test* pipeline dan eksplorasi dashboard ketika
dataset UCI belum tersedia. Data ini tidak boleh dipakai untuk menarik
kesimpulan penelitian. Semua artefak demo diberi penanda ``is_synthetic`` dan
disimpan di folder hasil terpisah.
"""

from __future__ import annotations

import numpy as np

from .utils import ACTIVITY_NAMES, DEMO_NOTICE, RAW_CHANNELS

DEMO_GENERATOR_VERSION = 1
DEFAULT_DEMO_SEED = 20240517
DEFAULT_DEMO_SUBJECTS = 12
WINDOW_SIZE = 128
SAMPLING_RATE_HZ = 50

_ACTIVITY_FREQUENCIES = {
    "WALKING": (1.05, 0.55),
    "WALKING_UPSTAIRS": (1.45, 0.70),
    "WALKING_DOWNSTAIRS": (1.75, 0.75),
    "SITTING": (0.18, 0.08),
    "STANDING": (0.12, 0.06),
    "LAYING": (0.08, 0.04),
}


def _make_raw_window(
    rng: np.random.Generator,
    activity_index: int,
    subject_phase: np.ndarray,
    subject_scale: np.ndarray,
) -> np.ndarray:
    """Create a (128, 9) synthetic inertial window with activity structure."""
    time = np.arange(WINDOW_SIZE) / SAMPLING_RATE_HZ
    activity = ACTIVITY_NAMES[activity_index]
    base_frequency, amplitude = _ACTIVITY_FREQUENCIES[activity]
    window = np.zeros((WINDOW_SIZE, len(RAW_CHANNELS)), dtype=np.float32)

    for channel_index in range(len(RAW_CHANNELS)):
        axis = channel_index % 3
        is_gyro = channel_index // 3 == 1
        frequency = base_frequency * (1.0 + 0.08 * axis)
        phase = subject_phase[channel_index]
        signal = subject_scale[channel_index] * amplitude * np.sin(
            2 * np.pi * frequency * time + phase
        )
        signal += 0.18 * np.sin(2 * np.pi * (2.1 * frequency) * time + phase / 2)
        if is_gyro:
            signal *= 0.55
        if activity in {"SITTING", "STANDING", "LAYING"}:
            signal *= 0.15
        window[:, channel_index] = signal + rng.normal(0, 0.035, size=WINDOW_SIZE)

    # Gravity component for static activities (total acceleration channels).
    gravity = {
        "SITTING": (0.0, 0.0, 0.98),
        "STANDING": (0.0, 0.02, 0.99),
        "LAYING": (0.0, 0.96, 0.05),
    }.get(activity)
    if gravity is not None:
        for axis in range(3):
            window[:, 6 + axis] += gravity[axis]
    return window


def generate_demo_dataset(
    seed: int = DEFAULT_DEMO_SEED,
    n_subjects: int = DEFAULT_DEMO_SUBJECTS,
    n_features: int = 561,
):
    """Generate a fully synthetic HAR dataset with subject/device shift.

    The generator is deterministic for a fixed seed. It produces a subject
    shift (domain shift) so that cross-subject evaluation is non-trivial, and
    it never claims to be the UCI HAR dataset.
    """
    from .data_loader import HARDataset  # local import to avoid a cycle

    rng = np.random.default_rng(seed)
    n_activities = len(ACTIVITY_NAMES)
    feature_names = [f"demo_feature_{index:03d}" for index in range(n_features)]

    # Class centroids plus subject-specific offsets and scaling.
    centers = rng.normal(0.0, 0.85, size=(n_activities, n_features)).astype(np.float32)
    subject_offsets = rng.normal(0.0, 0.45, size=(n_subjects, n_features)).astype(np.float32)
    subject_scales = rng.uniform(0.8, 1.2, size=(n_subjects, 1)).astype(np.float32)
    subject_phase = rng.uniform(0, 2 * np.pi, size=(n_subjects, len(RAW_CHANNELS)))
    subject_amplitude = rng.uniform(0.8, 1.2, size=(n_subjects, len(RAW_CHANNELS)))

    features = []
    labels = []
    subjects = []
    windows = []

    for subject_index in range(n_subjects):
        for activity_index in range(n_activities):
            n_windows = int(rng.integers(7, 12))
            noise = rng.normal(0, 0.75, size=(n_windows, n_features)).astype(np.float32)
            block = (
                centers[activity_index]
                + subject_offsets[subject_index]
                + subject_scales[subject_index] * noise
            ).astype(np.float32)
            features.append(block)
            labels.extend([activity_index] * n_windows)
            subjects.extend([subject_index + 1] * n_windows)
            for _ in range(n_windows):
                windows.append(
                    _make_raw_window(
                        rng,
                        activity_index,
                        subject_phase[subject_index],
                        subject_amplitude[subject_index],
                    )
                )

    X = np.concatenate(features, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    subject = np.asarray(subjects, dtype=np.int64)
    raw = np.stack(windows).astype(np.float32)

    n_train_subjects = max(1, int(round(n_subjects * 0.7)))
    split = np.where(subject <= n_train_subjects, "train", "test").astype(object)

    metadata = {
        "source": "synthetic_demo",
        "is_synthetic": True,
        "notice": DEMO_NOTICE,
        "generator": "src.demo_data.generate_demo_dataset",
        "generator_version": DEMO_GENERATOR_VERSION,
        "seed": int(seed),
        "n_subjects": int(n_subjects),
        "n_features": int(n_features),
        "device": "synthetic-demo-device",
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "window_size": WINDOW_SIZE,
        "window_seconds": WINDOW_SIZE / SAMPLING_RATE_HZ,
        "overlap": 0.5,
        "activity_ids": list(range(1, n_activities + 1)),
        "license": "Synthetic data generated for this repository. No real participants.",
    }

    return HARDataset(
        X=X,
        y=y,
        subject=subject,
        split=split,
        feature_names=feature_names,
        activity_names=list(ACTIVITY_NAMES),
        raw=raw,
        raw_channels=list(RAW_CHANNELS),
        source="synthetic_demo",
        metadata=metadata,
    )
