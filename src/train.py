"""Training routines for scikit-learn models and the PyTorch 1D-CNN."""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from .augmentation import augment_batch_torch
from .models import (
    CNNConfig,
    HARCNN,
    TorchHARModel,
    base_sklearn_params,
    make_sklearn_model,
    model_param_grid,
    predict_scores,
)
from .preprocessing import RawSignalStandardizer
from .utils import RAW_CHANNELS, get_logger

TORCH_MODEL_FILENAME = "cnn_model.pt"


# ---------------------------------------------------------------------------
# scikit-learn
# ---------------------------------------------------------------------------
def _param_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[key] for key in keys))]


def select_hyperparameters(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    config: dict[str, Any],
    seed: int = 42,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Select hyperparameters with GroupKFold on the supplied training data."""
    log = logger or get_logger("har.train")
    grid = model_param_grid(config, name)
    base = base_sklearn_params(config, name)
    if not grid:
        return {
            "method": "fixed (config)",
            "params": base,
            "cv_results": [],
            "n_splits": 0,
            "scoring": None,
        }

    n_subjects = len(np.unique(subjects))
    n_splits = min(int(config.get("models", {}).get(name, {}).get("inner_folds", 3)), n_subjects)
    if n_splits < 2:
        log.warning("Subjek latih < 2; pemilihan hyperparameter dilewati untuk %s", name)
        return {
            "method": "fixed (subjek latih < 2)",
            "params": base,
            "cv_results": [],
            "n_splits": 0,
            "scoring": "f1_macro",
        }

    splitter = GroupKFold(n_splits=n_splits)
    combinations = _param_combinations(grid)
    results: list[dict[str, Any]] = []
    for combination in combinations:
        scores = []
        for fold_index, (train_index, valid_index) in enumerate(
            splitter.split(X, y, groups=subjects), start=1
        ):
            params = {**base, **combination}
            estimator = make_sklearn_model(name, params, seed=seed + fold_index)
            estimator.fit(X[train_index], y[train_index])
            predictions = estimator.predict(X[valid_index])
            score = f1_score(
                y[valid_index],
                predictions,
                labels=sorted(np.unique(y[valid_index])),
                average="macro",
                zero_division=0,
            )
            scores.append(float(score))
        results.append(
            {
                "params": combination,
                "mean_macro_f1": float(np.mean(scores)),
                "std_macro_f1": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "fold_scores": scores,
            }
        )
        log.info(
            "  seleksi %s %s -> CV macro-F1 %.4f +/- %.4f",
            name,
            combination,
            results[-1]["mean_macro_f1"],
            results[-1]["std_macro_f1"],
        )

    best = max(results, key=lambda item: (item["mean_macro_f1"], -item["std_macro_f1"]))
    return {
        "method": f"GroupKFold(n_splits={n_splits}) pada subjek data latih",
        "params": {**base, **best["params"]},
        "selected": best["params"],
        "cv_results": results,
        "n_splits": n_splits,
        "scoring": "macro_f1",
        "n_training_subjects": n_subjects,
    }


@dataclass
class SklearnTrainingResult:
    model: Any
    params: dict[str, Any]
    selection: dict[str, Any]
    fit_time_s: float
    n_train: int
    n_train_subjects: int
    model_type: str = "sklearn"


def train_sklearn(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    subjects_train: np.ndarray,
    config: dict[str, Any],
    seed: int = 42,
    select: bool = True,
    logger: logging.Logger | None = None,
) -> SklearnTrainingResult:
    log = logger or get_logger("har.train")
    if select:
        selection = select_hyperparameters(name, X_train, y_train, subjects_train, config, seed, log)
    else:
        selection = {
            "method": "fixed (dari konfigurasi eksperimen)",
            "params": base_sklearn_params(config, name),
            "cv_results": [],
            "n_splits": 0,
            "scoring": None,
        }
    log.info("Melatih %s pada %d sampel / %d subjek", name, len(X_train), len(np.unique(subjects_train)))
    model = make_sklearn_model(
        name,
        selection["params"],
        seed=seed,
        n_jobs=int(config.get("project", {}).get("n_jobs", -1)),
    )
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start
    return SklearnTrainingResult(
        model=model,
        params=dict(selection["params"]),
        selection=selection,
        fit_time_s=float(fit_time),
        n_train=int(len(X_train)),
        n_train_subjects=int(len(np.unique(subjects_train))),
    )


def predict_sklearn(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, str, float]:
    start = time.perf_counter()
    predictions = np.asarray(model.predict(X))
    predict_time = time.perf_counter() - start
    try:
        scores, score_kind = predict_scores(model, X)
    except Exception:
        scores, score_kind = None, "unavailable"
    return predictions.astype(np.int64), scores, score_kind, float(predict_time)


def save_sklearn_model(path: str | Path, model: Any, metadata: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, destination)
    return destination


def load_sklearn_model(path: str | Path) -> tuple[Any, dict[str, Any]]:
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"File model tidak valid: {path}")
    return payload["model"], dict(payload.get("metadata", {}))


# ---------------------------------------------------------------------------
# PyTorch CNN
# ---------------------------------------------------------------------------
def _select_validation_indices(
    train_idx: np.ndarray,
    subjects: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    train_subjects = sorted(int(value) for value in np.unique(subjects[train_idx]))
    if len(train_subjects) < 2:
        # Fall back to a stratified row split when there is only one subject.
        rng = np.random.default_rng(seed)
        positions = rng.permutation(len(train_idx))
        n_val = max(1, int(round(len(train_idx) * fraction)))
        validation = np.sort(train_idx[positions[:n_val]])
        training = np.sort(train_idx[positions[n_val:]])
        return training, validation, train_subjects

    rng = np.random.default_rng(seed)
    n_val_subjects = max(1, int(round(len(train_subjects) * fraction)))
    n_val_subjects = min(n_val_subjects, len(train_subjects) - 1)
    chosen = sorted(rng.choice(train_subjects, size=n_val_subjects, replace=False).tolist())
    validation_mask = np.isin(subjects[train_idx], chosen)
    validation = train_idx[validation_mask]
    training = train_idx[~validation_mask]
    return np.sort(training), np.sort(validation), chosen


def select_validation_subjects(
    train_idx: np.ndarray,
    subjects: np.ndarray,
    fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Public helper for subject-disjoint CNN validation splits."""
    return _select_validation_indices(train_idx, subjects, fraction, seed)


def _validation_metrics(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            batch_x = x[start : start + 256]
            batch_y = y[start : start + 256]
            logits = model(batch_x)
            total_loss += float(loss_fn(logits, batch_y).item()) * len(batch_x)
            predictions.append(torch.argmax(logits, dim=1))
    pred = torch.cat(predictions).cpu().numpy()
    truth = y.cpu().numpy()
    accuracy = float(accuracy_score(truth, pred))
    present = sorted(int(value) for value in np.unique(truth))
    macro_f1 = float(
        f1_score(truth, pred, labels=present, average="macro", zero_division=0)
    )
    return total_loss / max(len(x), 1), accuracy, macro_f1


def train_cnn(
    X_raw: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    train_idx: np.ndarray,
    config: CNNConfig,
    seed: int = 42,
    val_idx: np.ndarray | None = None,
    normalizer: RawSignalStandardizer | None = None,
    logger: logging.Logger | None = None,
    class_names: list[str] | None = None,
) -> TorchHARModel:
    """Train the 1D-CNN with subject-disjoint early-stopping validation."""
    log = logger or get_logger("har.cnn")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train_idx = np.asarray(train_idx, dtype=np.int64)
    if val_idx is None:
        training_idx, validation_idx, val_subjects = _select_validation_indices(
            train_idx, subjects, config.val_subject_fraction, seed
        )
    else:
        validation_idx = np.asarray(val_idx, dtype=np.int64)
        training_idx = np.setdiff1d(train_idx, validation_idx)
        overlap = np.intersect1d(training_idx, validation_idx)
        if overlap.size:
            raise ValueError("val_idx tumpang tindih dengan train_idx")
        val_subjects = sorted(int(value) for value in np.unique(subjects[validation_idx]))

    if len(training_idx) == 0 or len(validation_idx) == 0:
        raise ValueError("Data latih/validasi CNN kosong setelah pemisahan")

    start_normalize = time.perf_counter()
    if normalizer is None:
        normalizer = RawSignalStandardizer(mode="global", channel_names=list(RAW_CHANNELS))
        normalizer.fit(X_raw[training_idx])
    elif normalizer.mean_ is None:
        raise ValueError("Normalizer yang diberikan belum di-fit")
    x_train = normalizer.transform(X_raw[training_idx], subjects[training_idx])
    x_valid = normalizer.transform(X_raw[validation_idx], subjects[validation_idx])
    normalize_time = time.perf_counter() - start_normalize

    # (n, window, channel) -> (n, channel, window)
    x_train_t = torch.from_numpy(np.transpose(x_train, (0, 2, 1)).copy())
    y_train_t = torch.from_numpy(np.asarray(y[training_idx], dtype=np.int64))
    x_valid_t = torch.from_numpy(np.transpose(x_valid, (0, 2, 1)).copy())
    y_valid_t = torch.from_numpy(np.asarray(y[validation_idx], dtype=np.int64))

    model = HARCNN(
        in_channels=config.in_channels,
        n_classes=config.n_classes,
        conv_channels=tuple(config.conv_channels),
        kernel_sizes=tuple(config.kernel_sizes),
        dropout=config.dropout,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    channel_std = np.std(x_train, axis=(0, 1)).astype(np.float32)
    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    augment = bool(config.augment)

    start_train = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        permutation = torch.from_numpy(rng.permutation(len(x_train_t)))
        total_loss = 0.0
        update_count = 0
        for start in range(0, len(permutation), config.batch_size):
            positions = permutation[start : start + config.batch_size]
            batch_x = x_train_t[positions]
            batch_y = y_train_t[positions]
            if augment:
                for _ in range(max(1, int(config.augment_copies))):
                    augmented_x = augment_batch_torch(batch_x, channel_std, rng)
                    optimizer.zero_grad()
                    loss = loss_fn(model(augmented_x), batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item()) * len(batch_y)
                    update_count += len(batch_y)
            else:
                optimizer.zero_grad()
                loss = loss_fn(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_y)
                update_count += len(batch_y)
        train_loss = total_loss / max(update_count, 1)
        valid_loss, valid_accuracy, valid_f1 = _validation_metrics(model, x_valid_t, y_valid_t)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(valid_loss, 6),
                "val_accuracy": round(valid_accuracy, 6),
                "val_macro_f1": round(valid_f1, 6),
            }
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == config.max_epochs:
            log.info(
                "  epoch %02d | train loss %.4f | val loss %.4f | val macro-F1 %.4f",
                epoch,
                train_loss,
                valid_loss,
                valid_f1,
            )
        if valid_f1 > best_score + 1e-5:
            best_score = valid_f1
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                log.info("  early stopping pada epoch %d", epoch)
                break

    fit_time = time.perf_counter() - start_train
    model.load_state_dict(best_state)
    model.eval()
    wrapper = TorchHARModel(
        model=model,
        config=config,
        normalizer=normalizer,
        class_names=list(class_names or []),
        metadata={
            "n_train": int(len(training_idx)),
            "n_validation": int(len(validation_idx)),
            "val_subjects": val_subjects,
            "seed": int(seed),
            "best_val_macro_f1": float(best_score),
            "normalize_time_s": round(normalize_time, 4),
            "fit_time_s": round(float(fit_time), 4),
            "n_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "augmentation": augment,
            "train_subjects": sorted(int(v) for v in np.unique(subjects[training_idx])),
        },
    )
    wrapper.history = history
    wrapper.best_epoch = best_epoch
    return wrapper


def predict_cnn(
    wrapper: TorchHARModel,
    X_raw: np.ndarray,
    subjects: np.ndarray | None = None,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    probabilities = wrapper.predict_proba(X_raw, subjects, batch_size=batch_size)
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    predict_time = time.perf_counter() - start
    return predictions, probabilities, float(predict_time)


def save_cnn_model(
    wrapper: TorchHARModel, path: str | Path, extra_metadata: dict[str, Any] | None = None
) -> Path:
    return wrapper.save(path, extra_metadata=extra_metadata)


def load_cnn_model(path: str | Path) -> TorchHARModel:
    return TorchHARModel.load(path)
