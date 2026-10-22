"""Tests for model factories, CNN architecture and training helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models import (
    CNNConfig,
    HARCNN,
    ModelError,
    TorchHARModel,
    make_sklearn_model,
    predict_scores,
)
from src.preprocessing import RawSignalStandardizer
from src.train import (
    load_cnn_model,
    predict_cnn,
    select_hyperparameters,
    select_validation_subjects,
    train_cnn,
    train_sklearn,
)


def test_make_sklearn_model_filters_parameters():
    rf = make_sklearn_model("random_forest", {"n_estimators": 10, "max_features": "sqrt", "bogus": 1}, seed=1)
    assert rf.n_estimators == 10
    assert rf.max_features == "sqrt"

    svm = make_sklearn_model("svm", {"C": 2.0, "kernel": "rbf"}, seed=1)
    assert svm.named_steps["svc"].C == 2.0
    assert svm.named_steps["svc"].probability is False

    with pytest.raises(ModelError):
        make_sklearn_model("unknown", {})


def test_predict_scores_kind():
    X = np.random.default_rng(0).normal(size=(20, 5))
    y = np.array([0, 1] * 10)
    rf = make_sklearn_model("random_forest", {"n_estimators": 10}, seed=1).fit(X, y)
    scores, kind = predict_scores(rf, X)
    assert kind == "probability"
    assert np.allclose(scores.sum(axis=1), 1.0)

    svm = make_sklearn_model("svm", {"C": 1.0}, seed=1).fit(X, y)
    scores, kind = predict_scores(svm, X)
    assert kind == "normalized_decision_score"
    assert np.allclose(scores.sum(axis=1), 1.0)


def test_cnn_forward_shape():
    model = HARCNN(in_channels=9, n_classes=6)
    output = model(torch.zeros(4, 9, 128))
    assert output.shape == (4, 6)


def test_cnn_config_from_config(repo_config):
    config = CNNConfig.from_config(repo_config, seed=7)
    assert config.seed == 7
    assert config.conv_channels == (32, 64, 128)
    assert config.in_channels == 9
    payload = config.to_dict()
    assert payload["conv_channels"] == [32, 64, 128]


def test_cnn_save_and_load(tmp_path):
    config = CNNConfig(max_epochs=1, batch_size=8)
    normalizer = RawSignalStandardizer(mode="global", channel_names=[f"c{i}" for i in range(9)])
    raw = np.random.default_rng(0).normal(size=(16, 128, 9)).astype(np.float32)
    normalizer.fit(raw)
    model = HARCNN(in_channels=9, n_classes=6)
    wrapper = TorchHARModel(
        model=model,
        config=config,
        normalizer=normalizer,
        class_names=["A", "B", "C", "D", "E", "F"],
        metadata={"n_train": 16},
    )
    path = tmp_path / "cnn.pt"
    wrapper.save(path)
    restored = load_cnn_model(path)
    original = wrapper.predict_proba(raw)
    reloaded = restored.predict_proba(raw)
    assert np.allclose(original, reloaded, atol=1e-6)


def test_select_validation_subjects_is_subject_disjoint():
    subjects = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    train_idx = np.arange(8)
    training, validation, val_subjects = select_validation_subjects(train_idx, subjects, fraction=0.5, seed=1)
    assert np.intersect1d(training, validation).size == 0
    assert set(val_subjects).isdisjoint(set(subjects[training].tolist()))
    assert len(training) + len(validation) == 8


def test_select_hyperparameters_returns_best_config(demo_dataset, config):
    payload = select_hyperparameters(
        "random_forest",
        demo_dataset.X,
        demo_dataset.y,
        demo_dataset.subject,
        {**config, "models": {**config["models"], "random_forest": {**config["models"]["random_forest"], "selection_grid": {"min_samples_leaf": [1, 3]}}}},
        seed=42,
    )
    assert payload["method"].startswith("GroupKFold")
    assert payload["selected"]["min_samples_leaf"] in {1, 3}
    assert len(payload["cv_results"]) == 2


def test_train_sklearn_returns_timing(demo_dataset, config):
    result = train_sklearn(
        "random_forest",
        demo_dataset.X,
        demo_dataset.y,
        demo_dataset.subject,
        config,
        seed=42,
        select=False,
    )
    assert result.fit_time_s >= 0
    assert result.n_train == demo_dataset.n_samples
    assert result.n_train_subjects == len(demo_dataset.subject_ids)


def test_train_cnn_produces_history_and_metadata(demo_dataset, config):
    cnn_config = CNNConfig.from_config(config, seed=42)
    wrapper = train_cnn(
        demo_dataset.raw,
        demo_dataset.y,
        demo_dataset.subject,
        np.arange(demo_dataset.n_samples),
        cnn_config,
        seed=42,
        class_names=demo_dataset.activity_names,
    )
    assert len(wrapper.history) >= 1
    assert wrapper.metadata["n_train"] > 0
    assert wrapper.metadata["n_validation"] > 0
    val_subjects = set(wrapper.metadata["val_subjects"])
    train_subjects = set(wrapper.metadata["train_subjects"])
    assert val_subjects.isdisjoint(train_subjects)
    predictions, probabilities, elapsed = predict_cnn(wrapper, demo_dataset.raw[:5])
    assert predictions.shape == (5,)
    assert probabilities.shape == (5, len(demo_dataset.activity_names))
    assert elapsed >= 0
